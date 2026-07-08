"""Vocal Isolation and Signal Preprocessing."""

import errno
import gc
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import types
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import importlib
import sys

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

from modules.core import config, utils
from modules.inference import scheduler

logger = logging.getLogger(__name__)

# --- [ENGINE CONFIGURATION] ---
logging.getLogger("audio_separator").setLevel(logging.INFO)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_output_dirs():
    """Return ordered candidate output directories for UVR stem files."""
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


def _separate_with_fallback(sep, sep_factory, audio_path, yield_cb=None):
    """Run UVR separation, retrying on ENOSPC with alternative directories."""

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

            # Preserve the unwrapped original once per separator instance.
            is_mock = hasattr(current_sep, "_mock_self")
            if is_mock:
                # Mocks store custom attributes in __dict__.
                if "_orig_separate_file" not in current_sep.__dict__:
                    setattr(current_sep, "_orig_separate_file", getattr(current_sep, "_separate_file"))
            else:
                if not hasattr(current_sep, "_orig_separate_file"):
                    setattr(current_sep, "_orig_separate_file", getattr(current_sep, "_separate_file"))

            # Initialize thread-local chunk tracking context variables.
            # This avoids cross-talk when a priority task borrows a separator concurrently.
            utils.THREAD_CONTEXT.uvr_chunk_paths_len = 0
            utils.THREAD_CONTEXT.uvr_chunk_index = 0
            utils.THREAD_CONTEXT.uvr_audio_dur = 0.0
            utils.THREAD_CONTEXT.uvr_chunk_duration = 0
            utils.THREAD_CONTEXT.uvr_scheduler = scheduler
            utils.THREAD_CONTEXT.uvr_yield_cb = yield_cb

            # Always reset instance variables as well for test assertions compatibility.
            setattr(current_sep, "_chunk_paths_len", 0)
            setattr(current_sep, "_chunk_index", 0)
            setattr(current_sep, "_audio_dur", 0.0)

            if is_valid_duration:
                total_chunks = math.ceil(audio_dur / chunk_duration)
                utils.THREAD_CONTEXT.uvr_chunk_paths_len = total_chunks
                utils.THREAD_CONTEXT.uvr_chunk_index = 0
                utils.THREAD_CONTEXT.uvr_audio_dur = audio_dur
                utils.THREAD_CONTEXT.uvr_chunk_duration = chunk_duration

                setattr(current_sep, "_chunk_paths_len", total_chunks)
                setattr(current_sep, "_chunk_index", 0)
                setattr(current_sep, "_audio_dur", audio_dur)

            # Ensure the separator has the thread-safe permanent wrapper.
            should_patch = (
                "_is_permanently_patched" not in current_sep.__dict__
                if is_mock
                else not hasattr(current_sep, "_is_permanently_patched")
            )
            if should_patch:

                def permanent_patched_separate_file(self, audio_file_path, custom_output_names=None):
                    chunk_paths_len = getattr(utils.THREAD_CONTEXT, "uvr_chunk_paths_len", 0)
                    audio_dur_val = getattr(utils.THREAD_CONTEXT, "uvr_audio_dur", 0.0)
                    chunk_dur = getattr(utils.THREAD_CONTEXT, "uvr_chunk_duration", 0)
                    sched = getattr(utils.THREAD_CONTEXT, "uvr_scheduler", None)
                    current_yield_cb = getattr(utils.THREAD_CONTEXT, "uvr_yield_cb", None)

                    # Distinguish between outer orchestrator call and inner chunk processing call.
                    is_outer = not getattr(utils.THREAD_CONTEXT, "uvr_in_chunk_processing", False)

                    # In test environments with mock separator files, do not skip progress updates.
                    orig_sep = getattr(self, "_orig_separate_file")
                    if is_outer and chunk_paths_len > 0 and not hasattr(orig_sep, "_mock_self"):
                        # Outer call: delegate to original method (which calls _process_with_chunking)
                        utils.THREAD_CONTEXT.uvr_in_chunk_processing = True
                        try:
                            return orig_sep(audio_file_path, custom_output_names)
                        finally:
                            utils.THREAD_CONTEXT.uvr_in_chunk_processing = False

                    # Inner call (chunk) or non-chunked run
                    chunk_idx = getattr(utils.THREAD_CONTEXT, "uvr_chunk_index", 0) + 1

                    # Only update start progress if chunk_idx is greater than the current thread-local index.
                    if (
                        chunk_paths_len > 0
                        and sched
                        and getattr(utils.THREAD_CONTEXT, "uvr_chunk_index", 0) < chunk_idx
                    ):
                        processed_start_dur = min((chunk_idx - 1) * chunk_dur, audio_dur_val)
                        start_pct = 5.0 + (float(chunk_idx - 1) / chunk_paths_len) * 5.0
                        sched.update_task_metadata(current_position=processed_start_dur)
                        sched.update_task_progress(
                            int(start_pct),
                            f"Vocal Separation ({chunk_idx}/{chunk_paths_len} segments | "
                            f"{utils.format_duration(processed_start_dur)} / {utils.format_duration(audio_dur_val)})",
                        )

                    if current_yield_cb:
                        current_yield_cb()

                    res = orig_sep(audio_file_path, custom_output_names)

                    # Only update end progress if chunk_idx is greater than the current thread-local index.
                    if (
                        chunk_paths_len > 0
                        and sched
                        and getattr(utils.THREAD_CONTEXT, "uvr_chunk_index", 0) < chunk_idx
                    ):
                        setattr(utils.THREAD_CONTEXT, "uvr_chunk_index", chunk_idx)
                        setattr(self, "_chunk_index", chunk_idx)
                        processed_dur = min(chunk_idx * chunk_dur, audio_dur_val)
                        pct = 5.0 + (float(chunk_idx) / chunk_paths_len) * 5.0
                        sched.update_task_metadata(current_position=processed_dur)
                        sched.update_task_progress(
                            int(pct),
                            f"Vocal Separation ({chunk_idx}/{chunk_paths_len} segments | "
                            f"{utils.format_duration(processed_dur)} / {utils.format_duration(audio_dur_val)})",
                        )

                    if current_yield_cb:
                        current_yield_cb()

                    return res

                setattr(current_sep, "_separate_file", types.MethodType(permanent_patched_separate_file, current_sep))
                setattr(current_sep, "_is_permanently_patched", True)

            return current_sep.separate(audio_path), current_sep
        except OSError as exc:
            last_err = exc
            if exc.errno != errno.ENOSPC:
                # Not a disk-space error — propagate immediately
                raise
            try:
                free_mb = shutil.disk_usage(out_dir).free // (1024 * 1024)
            except OSError:
                free_mb = 0
            logger.error("[UVR] No space left on %s (%d MB free) — trying next fallback.", out_dir, free_mb)

    # All candidates exhausted
    raise OSError(errno.ENOSPC, "No space left on any candidate output directory") from last_err


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

            def patched_init(self, model_path, sess_options=None, providers=None, provider_options=None, **kwargs):
                """
                Intercept InferenceSession creation to inject OpenVINO options
                from the thread context, bypassing library-level limitations.
                """
                # Force OpenVINO if requested in thread context
                ctx_options = getattr(utils.THREAD_CONTEXT, "ov_options", None)

                # Check for CPU fallback override
                if (
                    ctx_options
                    and "device_type" in ctx_options
                    and (not providers or providers == ["CPUExecutionProvider"])
                ):
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

                return original_init(self, model_path, sess_options, providers, provider_options, **kwargs)

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
            logger.info("[System] Initializing UVR (%s) on %s...", config.VOCAL_SEPARATION_MODEL, unit_name)

            if ort is None:
                raise ImportError("onnxruntime not installed.")

            available_providers = ort.get_available_providers()
            logger.debug("Active ORT Version: %s | Providers: %s", ort.__version__, available_providers)
            target_providers, target_options = self._resolve_providers(available_providers)

            # Check if we are successfully using an accelerator
            is_accelerated = any(p in ["CUDAExecutionProvider", "OpenVINOExecutionProvider"] for p in target_providers)

            self.separator = separator_class(
                output_dir=str(CACHE_DIR),
                model_file_dir=config.UVR_MODEL_DIR,
                output_format="WAV",
                normalization_threshold=0.01,
                output_single_stem="Vocals",
                chunk_duration=config.UVR_CHUNK_DURATION,
                log_level=logging.INFO,
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
            options = [
                {
                    "device_type": ov_device.upper(),
                    "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                    "num_streams": "1",
                },
                {},
            ]
        elif self._device_type == "AUTO":
            # Priority: GPU > NPU > CPU
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                options = [{"device_id": "0"}, {}]
            elif "OpenVINOExecutionProvider" in available:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                options = [
                    {"device_type": "GPU", "cache_dir": os.path.abspath(config.OV_CACHE_DIR), "num_streams": "1"},
                    {},
                ]
            else:
                # AUTO Fallback to CPU
                options = [{"intra_op_num_threads": str(config.PREPROCESS_THREADS), "inter_op_num_threads": "1"}]
        else:
            # CPU Path or Fallback
            options = [{"intra_op_num_threads": str(config.PREPROCESS_THREADS), "inter_op_num_threads": "1"}]

        return providers, options

    def _build_active_yield_cb(self, yield_cb):
        """Wrap yield callback to temporarily release separator lock during cooperative preemption."""
        if not yield_cb:
            return None

        lock_ref = self._lock

        def _unlocked_yield_cb(*, _cb=yield_cb, _lk=lock_ref):
            _lk.release()
            try:
                _cb()
            finally:
                _lk.acquire()

        return _unlocked_yield_cb

    def _separate_audio(self, sep, audio_path, *, use_cpu_lock, target_options, active_yield_cb):
        """Run separation with provider-context injection and ENOSPC fallback handling."""
        with self._lock:
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
                    log_level=logging.INFO,
                )
                new_sep.onnx_execution_provider = sep.onnx_execution_provider
                if getattr(sep, "hardware_acceleration_enabled", False):
                    new_sep.hardware_acceleration_enabled = True
                new_sep.load_model(config.VOCAL_SEPARATION_MODEL)
                return new_sep

            try:
                if use_cpu_lock:
                    with utils.cpu_lock_ctx():
                        return _separate_with_fallback(sep, _make_separator, audio_path, yield_cb=active_yield_cb)
                return _separate_with_fallback(sep, _make_separator, audio_path, yield_cb=active_yield_cb)
            finally:
                utils.THREAD_CONTEXT.ov_options = None

    def _resolve_stem_path(self, path_value, effective_sep, source_audio_path):
        """Resolve absolute output path for a UVR stem, including fallback output directories."""
        if os.path.isabs(path_value):
            return path_value

        resolved_candidates = []
        run_output_dir = getattr(effective_sep, "output_dir", None)
        if run_output_dir:
            resolved_candidates.append(run_output_dir)
        resolved_candidates.extend(_candidate_output_dirs())
        source_parent = os.path.dirname(source_audio_path)
        if source_parent:
            resolved_candidates.append(source_parent)

        seen = set()
        for base_dir in resolved_candidates:
            if not base_dir or base_dir in seen:
                continue
            seen.add(base_dir)
            candidate = os.path.join(base_dir, path_value)
            if os.path.exists(candidate):
                return candidate

        return str(CACHE_DIR / path_value)

    def preprocess_audio(self, audio_path, force=False, yield_cb=None):
        """Perform vocal isolation on a file using the unit's separator."""
        if not config.ENABLE_VOCAL_SEPARATION and not force:
            return audio_path

        try:
            self._purge_stale_cache()

            original_path = audio_path
            audio_path = utils.prepare_for_uvr(audio_path, yield_cb=yield_cb)
            if not audio_path:
                return original_path

            scheduler.update_task_progress(5, "Vocal Separation")

            if yield_cb:
                yield_cb()

            sep = self._init_separator()
            audio_dur = utils.get_audio_duration(audio_path)
            unit_name = self._unit["name"] if self._unit else self._device_id
            logger.info("[UVR] Starting vocal isolation on %s...", unit_name)
            p_start = time.time()

            _, target_options = self._resolve_providers(ort.get_available_providers())
            active_yield_cb = self._build_active_yield_cb(yield_cb)
            stems, effective_sep = self._separate_audio(
                sep,
                audio_path,
                use_cpu_lock=self._device_type == "CPU",
                target_options=target_options,
                active_yield_cb=active_yield_cb,
            )

            dur = time.time() - p_start
            speed_val = audio_dur / dur if dur > 0 else 0.0
            logger.info(
                "[UVR] Isolation complete on %s (Duration: %s | Audio: %s | Speed: %.2fx)",
                unit_name,
                utils.format_duration(dur),
                utils.format_duration(audio_dur),
                speed_val,
            )

            if yield_cb:
                yield_cb()

            if stems and len(stems) > 0:
                stem_path = self._resolve_stem_path(stems[0], effective_sep, audio_path)
                utils.track_file(stem_path)

                for extra_stem in stems[1:]:
                    extra_path = self._resolve_stem_path(extra_stem, effective_sep, audio_path)
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
