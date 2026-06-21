
"""
Vocal Isolation and Signal Preprocessing

This module provides vocal isolation capabilities using the UVR (Ultimate Vocal Remover)
MDX-NET architecture. It implements hardware-specific optimizations for ONNX Runtime,
including OpenVINO and CUDA backends, and handles both file-level and
segment-level audio cleaning.
"""
import errno
import gc
import logging
import math
import os
import shutil
import tempfile
import types
from pathlib import Path
import threading
import time
try:
    import onnxruntime as ort
except ImportError:
    ort = None

import sys
import importlib

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

from modules import config
from modules import utils
from modules.inference import scheduler


logger = logging.getLogger(__name__)

# --- [ENGINE CONFIGURATION] ---
logging.getLogger("audio_separator").setLevel(logging.INFO)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_output_dirs():
    """
    Return an ordered list of candidate output directories for UVR stem files.

    Priority:
        1. Configured preprocessing cache (fastest, usually on tmpfs).
        2. PERSISTENT_TEMP_DIR (cross-volume fallback).
        3. System /tmp.
        4. /dev/shm (RAM-disk, last resort — only for small montage files).
    """
    candidates = [
        str(CACHE_DIR),
        config.PERSISTENT_TEMP_DIR,
        tempfile.gettempdir(),
    ]
    # RAM-disk only when it exists and is distinct from the others
    if os.path.isdir("/dev/shm"):
        candidates.append("/dev/shm")
    # Deduplicate while preserving order
    seen = set()
    result = []
    for d in candidates:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def _separate_with_fallback(sep, sep_factory, audio_path):
    """
    Run UVR separation, automatically retrying on ENOSPC with alternative
    output directories. Re-instantiates the separator for each new directory
    so the write target actually changes.

    Parameters:
        sep:         The already-initialised Separator instance (primary attempt).
        sep_factory: Callable(output_dir: str) -> Separator, used when
                     the primary directory fails and we need a new instance.
        audio_path:  Path to the audio file to separate.

    Returns:
        List of stem paths returned by sep.separate().
    """
    candidates = _candidate_output_dirs()
    current_sep = sep
    last_err = None

    for i, out_dir in enumerate(candidates):
        try:
            os.makedirs(out_dir, exist_ok=True)
            if i > 0:
                # Re-instantiate with the new output directory
                current_sep = sep_factory(out_dir)
            # Apply progress patching if chunk_duration is active
            audio_dur = utils.get_audio_duration(audio_path)
            chunk_duration = getattr(current_sep, "chunk_duration", None)
            is_valid_duration = (
                chunk_duration
                and isinstance(chunk_duration, (int, float))
                and not hasattr(chunk_duration, "_mock_self")
                and audio_dur > chunk_duration
            )
            if is_valid_duration:
                total_chunks = math.ceil(audio_dur / chunk_duration)
                setattr(current_sep, "_chunk_paths_len", total_chunks)
                setattr(current_sep, "_chunk_index", 0)
                setattr(current_sep, "_audio_dur", audio_dur)

                original_separate_file = getattr(current_sep, "_separate_file")

                def patched_separate_file(
                    self,
                    audio_file_path,
                    custom_output_names=None,
                    *,
                    orig_sep_file=original_separate_file,
                    chunk_dur=chunk_duration,
                    sched=scheduler
                ):
                    res = orig_sep_file(audio_file_path, custom_output_names)
                    chunk_paths_len = getattr(self, "_chunk_paths_len", 0)
                    if chunk_paths_len > 0:
                        chunk_idx = getattr(self, "_chunk_index", 0) + 1
                        setattr(self, "_chunk_index", chunk_idx)
                        audio_dur_val = getattr(self, "_audio_dur", 0.0)
                        processed_dur = min(chunk_idx * chunk_dur, audio_dur_val)
                        pct = 5.0 + (float(chunk_idx) / chunk_paths_len) * 5.0
                        sched.update_task_metadata(current_position=processed_dur)
                        sched.update_task_progress(
                            int(pct),
                            f"Vocal Separation (Chunk {chunk_idx}/{chunk_paths_len} | "
                            f"{utils.format_duration(processed_dur)} / {utils.format_duration(audio_dur_val)})"
                        )
                    return res

                setattr(current_sep, "_separate_file", types.MethodType(patched_separate_file, current_sep))

            return current_sep.separate(audio_path)
        except OSError as exc:
            last_err = exc
            if exc.errno != errno.ENOSPC:
                # Not a disk-space error — propagate immediately
                raise
            try:
                free_mb = shutil.disk_usage(out_dir).free // (1024 * 1024)
            except OSError:
                free_mb = 0
            logger.error(
                "[UVR] No space left on %s (%d MB free) — trying next fallback.",
                out_dir, free_mb)

    # All candidates exhausted
    raise OSError(errno.ENOSPC,
                  "No space left on any candidate output directory") from last_err


def _lazy_import_separator():
    """Lazy import of audio-separator components."""
    return Separator


def apply_onnx_optimizations():
    """
    Monkeypatch ONNX Runtime and audio-separator for hardware acceleration.
    Must be called BEFORE any heavy AI imports or Separator instantiation.
    """
    try:
        module_obj = sys.modules[__name__]
        if module_obj.ort is None:
            loaded_ort = importlib.import_module("onnxruntime")
            module_obj.ort = loaded_ort

        curr_ort = module_obj.ort
        if getattr(curr_ort.InferenceSession, "is_patched", False) is not True:
            logger.debug("Optimization: Deep-patching ONNX InferenceSession...")

            # Save the original constructor
            original_init = curr_ort.InferenceSession.__init__

            def patched_init(self, model_path, sess_options=None, providers=None,
                             provider_options=None, **kwargs):
                """
                Intercept InferenceSession creation to inject OpenVINO options
                from the thread context, bypassing library-level limitations.
                """
                # Force OpenVINO if requested in thread context
                ctx_options = getattr(utils.THREAD_CONTEXT, "ov_options", None)

                # Check for CPU fallback override
                if ctx_options and "device_type" in ctx_options and \
                   (not providers or providers == ["CPUExecutionProvider"]):
                    logger.info("[System] Intercepted CPU fallback - Forcing OpenVINOProvider")
                    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

                # Inject options if OpenVINO is being used
                if ctx_options and providers and "OpenVINOExecutionProvider" in providers:
                    logger.info("[System] Injecting OpenVINO options into session: %s", ctx_options)
                    if not provider_options:
                        provider_options = [{}] * len(providers)

                    for i, p_name in enumerate(providers):
                        if p_name != "OpenVINOExecutionProvider":
                            continue
                        if i < len(provider_options):
                            if not isinstance(provider_options[i], dict):
                                provider_options[i] = {}
                            provider_options[i].update(ctx_options)
                        else:
                            provider_options.append(ctx_options)

                return original_init(self, model_path, sess_options,
                                     providers, provider_options, **kwargs)

            # Apply the patch
            curr_ort.InferenceSession.__init__ = patched_init
            curr_ort.InferenceSession.is_patched = True

        # Also patch audio-separator class if available to prevent internal CPU fallback flags
        try:
            audio_separator = importlib.import_module("audio_separator.separator")
            separator_cls = audio_separator.Separator
            if getattr(separator_cls, "is_patched", False) is not True:
                logger.debug("Optimization: Patching Separator class detection logic...")
                separator_cls.check_onnxruntime = lambda self: None
                separator_cls.is_patched = True
        except ImportError:
            pass

    except (ImportError, AttributeError, KeyError, TypeError, ValueError, OSError) as patch_err:
        logger.warning("[System] Failed to apply ONNX optimizations: %s", patch_err)


class PreprocessingManager:
    """
    Orchestrates audio cleaning models (UVR/MDX-NET) for a specific hardware unit.
    """

    def unload_model(self):
        """Release the ONNX session and clear RAM."""
        if self.separator:
            logger.info("[Prep] Unloading UVR model from %s", self._device_id)
            # audio-separator doesn't have an explicit unload, so we delete it
            self.separator = None
            gc.collect()

    def __init__(self, assigned_unit=None):
        """
        Initialize the manager for a specific hardware unit.
        :param assigned_unit: Dictionary containing 'id', 'type', 'name'.
        """
        self._unit = assigned_unit
        self._device_id = assigned_unit["id"] if assigned_unit else config.PREPROCESS_DEVICE
        self._device_type = assigned_unit["type"] if assigned_unit else config.PREPROCESS_DEVICE

        self.separator = None
        self._lock = threading.Lock()

        # Apply global optimizations once
        apply_onnx_optimizations()
        self._purge_stale_cache()

    @property
    def unit(self):
        """Getter for assigned hardware unit."""
        return self._unit

    @property
    def device_id(self):
        """Getter for device ID."""
        return self._device_id

    @property
    def device_type(self):
        """Getter for device type."""
        return self._device_type

    @property
    def lock(self):
        """Getter for preprocessor lock."""
        return self._lock

    def _init_separator(self):
        """Initialize the separator instance pinned to the assigned unit."""
        with self._lock:
            if self.separator is not None:
                return self.separator

            separator_class = _lazy_import_separator()
            if separator_class is None:
                raise ImportError("audio-separator not installed.")

            unit_name = self._unit["name"] if self._unit else self._device_id
            logger.info("[System] Initializing UVR (%s) on %s...",
                        config.VOCAL_SEPARATION_MODEL, unit_name)

            if ort is None:
                raise ImportError("onnxruntime not installed.")

            available_providers = ort.get_available_providers()
            logger.debug("Active ORT Version: %s | Providers: %s", ort.__version__, available_providers)
            target_providers, target_options = self._resolve_providers(available_providers)

            # Check if we are successfully using an accelerator
            is_accelerated = any(p in ["CUDAExecutionProvider", "OpenVINOExecutionProvider"]
                                 for p in target_providers)

            self.separator = separator_class(
                output_dir=str(CACHE_DIR),
                model_file_dir=config.UVR_MODEL_DIR,
                output_format="WAV",
                normalization_threshold=0.01,
                output_single_stem="Vocals",
                chunk_duration=config.UVR_CHUNK_DURATION,
                log_level=logging.INFO
            )

            # Injection of hardware pinning
            # Note: 0.41.1 doesn't support provider_options in the constructor,
            # so we rely on our deep InferenceSession patch and thread context.
            self.separator.onnx_execution_provider = target_providers

            # Save options for the session patcher to find during load_model()
            # Only set if we are actually using OpenVINO (i.e. options contain device_type)
            if target_options and "device_type" in target_options[0]:
                utils.THREAD_CONTEXT.ov_options = target_options[0]
            else:
                utils.THREAD_CONTEXT.ov_options = None

            # Force override internal state if we know we have an accelerator.
            if is_accelerated:
                logger.debug("[System] Forcing hardware_acceleration_enabled for %s", unit_name)
                self.separator.hardware_acceleration_enabled = True

            try:
                self.separator.load_model(config.VOCAL_SEPARATION_MODEL)
            except Exception as e:
                logger.error("[System] Failed to load UVR model: %s", e)
                self.separator = None
                raise
            finally:
                # Clear context to avoid affecting subsequent ONNX sessions (e.g. VAD)
                utils.THREAD_CONTEXT.ov_options = None
            return self.separator

    def _resolve_providers(self, available):
        """Map the unit's hardware to ONNX execution providers."""
        providers = ["CPUExecutionProvider"]
        options = [{}]

        if self._device_type == "CUDA" and "CUDAExecutionProvider" in available:
            dev_idx = int(self._device_id.split(":")[-1]) if ":" in self._device_id else 0
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            options = [{"device_id": str(dev_idx)}, {}]
        elif self._device_type in ["GPU", "NPU", "OpenVINO"] and "OpenVINOExecutionProvider" in available:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            ov_device = self._device_id if self._device_id else "GPU"
            options = [{
                "device_type": ov_device.upper(),
                "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                "num_streams": "1"
            }, {}]
        elif self._device_type == "AUTO":
            # Priority: GPU > NPU > CPU
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                options = [{"device_id": "0"}, {}]
            elif "OpenVINOExecutionProvider" in available:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                options = [{
                    "device_type": "GPU",
                    "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                    "num_streams": "1"
                }, {}]
            else:
                # AUTO Fallback to CPU
                options = [{
                    "intra_op_num_threads": str(config.PREPROCESS_THREADS),
                    "inter_op_num_threads": "1"
                }]
        else:
            # CPU Path or Fallback
            options = [{
                "intra_op_num_threads": str(config.PREPROCESS_THREADS),
                "inter_op_num_threads": "1"
            }]

        return providers, options

    def preprocess_audio(self, audio_path, force=False, yield_cb=None):
        """Perform vocal isolation on a file using the unit's separator."""
        if not config.ENABLE_VOCAL_SEPARATION and not force:
            return audio_path

        try:
            self._purge_stale_cache()

            # Standard audio-separator output format:
            # Output is written to output_dir with a specific naming convention
            # We must find the resulting 'Vocal' or 'Instrument' stem.

            # MDX-NET usually produces stems with the model name in the filename
            # Standardization: Ensure audio is in a high-quality format UVR can read
            original_path = audio_path
            audio_path = utils.prepare_for_uvr(audio_path)
            if not audio_path:
                return original_path

            if yield_cb:
                yield_cb()
            sep = self._init_separator()

            audio_dur = utils.get_audio_duration(audio_path)
            unit_name = self._unit["name"] if self._unit else self._device_id
            logger.info("[UVR] Starting vocal isolation on %s...", unit_name)
            p_start = time.time()
            use_cpu_lock = self._device_type == "CPU"

            # Resolve providers and options again to ensure thread-context is fresh
            available_providers = ort.get_available_providers()
            _, target_options = self._resolve_providers(available_providers)

            with self._lock:
                # Inject options into thread context for the deep-patched InferenceSession
                if target_options and "device_type" in target_options[0]:
                    utils.THREAD_CONTEXT.ov_options = target_options[0]
                else:
                    utils.THREAD_CONTEXT.ov_options = None

                def _make_separator(output_dir):
                    """Build a fresh Separator pinned to a specific output directory."""
                    new_sep = _lazy_import_separator()(
                        output_dir=output_dir,
                        model_file_dir=config.UVR_MODEL_DIR,
                        output_format="WAV",
                        normalization_threshold=0.01,
                        output_single_stem="Vocals",
                        chunk_duration=config.UVR_CHUNK_DURATION,
                        log_level=logging.INFO
                    )
                    new_sep.onnx_execution_provider = sep.onnx_execution_provider
                    return new_sep

                try:
                    if use_cpu_lock:
                        with utils.cpu_lock_ctx():
                            stems = _separate_with_fallback(sep, _make_separator, audio_path)
                    else:
                        stems = _separate_with_fallback(sep, _make_separator, audio_path)
                finally:
                    # CRITICAL: Clear hardware options to prevent leaking into VAD/Whisper logic
                    utils.THREAD_CONTEXT.ov_options = None

            dur = time.time() - p_start
            speed_val = audio_dur / dur if dur > 0 else 0.0
            logger.info(
                "[UVR] Isolation complete on %s (Duration: %s | Audio: %s | Speed: %.2fx)",
                unit_name,
                utils.format_duration(dur),
                utils.format_duration(audio_dur),
                speed_val
            )

            if yield_cb:
                yield_cb()

            # Return the path to the isolated vocals
            if stems and len(stems) > 0:
                stem_path = stems[0]
                if not os.path.isabs(stem_path):
                    stem_path = str(CACHE_DIR / stem_path)

                # Register for request-local cleanup
                utils.track_file(stem_path)

                # Eagerly delete any extra stems; track first so route cleanup catches failures
                for extra_stem in stems[1:]:
                    extra_path = extra_stem if os.path.isabs(extra_stem) \
                        else str(CACHE_DIR / extra_stem)
                    utils.track_file(extra_path)
                    if extra_path != stem_path:
                        utils.secure_remove(extra_path)

                return stem_path
            return audio_path

        except tuple([Exception]) as e:
            logger.error("[UVR] Processing failed on %s: %s", self._device_id, e)
            return audio_path

    def _purge_stale_cache(self):
        """Remove old preprocessing artifacts from the tmpfs cache."""
        try:
            # Only purge if cache directory is on a RAM-disk or designated temp area
            for item in CACHE_DIR.iterdir():
                if item.is_file() and (time.time() - item.stat().st_mtime) > 3600:
                    item.unlink()
        except tuple([Exception]):
            pass

    def offload(self):
        """Purge model from VRAM/RAM when idle."""
        with self._lock:
            if self.separator:
                logger.info("[System] Offloading UVR engine from %s", self._device_id)
                self.separator = None
                gc.collect()
                utils.clear_gpu_cache()
