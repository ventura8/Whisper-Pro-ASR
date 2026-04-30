
"""
Vocal Isolation and Signal Preprocessing

This module provides vocal isolation capabilities using the UVR (Ultimate Vocal Remover)
MDX-NET architecture. It implements hardware-specific optimizations for ONNX Runtime,
including OpenVINO and CUDA backends, and handles both file-level and
segment-level audio cleaning.
"""
import gc
import logging
import os
from pathlib import Path
import threading
import time
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

from modules import config
from modules import utils


logger = logging.getLogger(__name__)

# --- [ENGINE CONFIGURATION] ---
logging.getLogger("audio_separator").setLevel(logging.INFO)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# pylint: disable=global-statement  # Required for lazy loading pattern
def _lazy_import_separator():
    """Lazy import of audio-separator components."""
    return Separator


# pylint: disable=global-statement  # Required for lazy loading pattern
def apply_onnx_optimizations():
    """
    Monkeypatch ONNX Runtime and audio-separator for hardware acceleration.
    Must be called BEFORE any heavy AI imports or Separator instantiation.
    """
    global ort
    try:
        if ort is None:
            # pylint: disable=import-outside-toplevel,import-error  # Intentional lazy import
            import onnxruntime as loaded_ort
            ort = loaded_ort

        if getattr(ort.InferenceSession, "_is_patched", False) is not True:
            logger.debug("Optimization: Deep-patching ONNX InferenceSession...")

            # Save the original constructor
            original_init = ort.InferenceSession.__init__

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
            ort.InferenceSession.__init__ = patched_init
            ort.InferenceSession._is_patched = True  # pylint: disable=protected-access

        # Also patch audio-separator class if available to prevent internal CPU fallback flags
        try:
            from audio_separator.separator import Separator as S  # pylint: disable=import-outside-toplevel
            if getattr(S, "_is_patched", False) is not True:  # pylint: disable=protected-access
                logger.debug("Optimization: Patching Separator class detection logic...")
                S.check_onnxruntime = lambda self: None
                S._is_patched = True  # pylint: disable=protected-access
        except ImportError:
            pass

    except Exception as patch_err:  # pylint: disable=broad-exception-caught
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

            sep = self._init_separator()
            if yield_cb:
                yield_cb()

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

                try:
                    if use_cpu_lock:
                        with utils.cpu_lock_ctx():
                            stems = sep.separate(audio_path)
                    else:
                        stems = sep.separate(audio_path)
                finally:
                    # CRITICAL: Clear hardware options to prevent leaking into VAD/Whisper logic
                    utils.THREAD_CONTEXT.ov_options = None

            dur = time.time() - p_start
            logger.info("[UVR] Isolation complete on %s (Duration: %.2fs)", unit_name, dur)

            if yield_cb:
                yield_cb()

            # Return the path to the isolated vocals
            if stems and len(stems) > 0:
                stem_path = stems[0]
                if not os.path.isabs(stem_path):
                    stem_path = str(CACHE_DIR / stem_path)

                # Register for request-local cleanup
                utils.track_file(stem_path)

                # Cleanup unused stems to prevent disk/memory leakage
                for extra_stem in stems[1:]:
                    try:
                        extra_path = extra_stem if os.path.isabs(
                            extra_stem) else str(CACHE_DIR / extra_stem)

                        # Also track extra stems just in case they fail to remove here
                        utils.track_file(extra_path)

                        if os.path.exists(extra_path) and extra_path != stem_path:
                            os.remove(extra_path)
                    except (IOError, OSError) as cleanup_err:
                        logger.debug("[UVR] Failed to cleanup extra stem: %s", cleanup_err)

                return stem_path
            return audio_path

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[UVR] Processing failed on %s: %s", self._device_id, e)
            return audio_path

    def _purge_stale_cache(self):
        """Remove old preprocessing artifacts from the tmpfs cache."""
        try:
            # Only purge if cache directory is on a RAM-disk or designated temp area
            for item in CACHE_DIR.iterdir():
                if item.is_file() and (time.time() - item.stat().st_mtime) > 3600:
                    item.unlink()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def offload(self):
        """Purge model from VRAM/RAM when idle."""
        with self._lock:
            if self.separator:
                logger.info("[System] Offloading UVR engine from %s", self._device_id)
                self.separator = None
                gc.collect()
                utils.clear_gpu_cache()
