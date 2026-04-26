
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
try:
    import torch
except ImportError:
    torch = None

from . import config, utils


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
    try:
        if ort is None:
            return

        if getattr(ort.InferenceSession, "_is_patched", False) is not True:
            logger.debug("[System] Optimization: Patching ONNX...")
            # pylint: disable=protected-access
            ort.InferenceSession._is_patched = True
    except Exception as patch_err:  # pylint: disable=broad-exception-caught
        logger.warning("[System] Failed to apply ONNX optimizations: %s", patch_err)


class PreprocessingManager:
    """
    Orchestrates audio cleaning models (UVR/MDX-NET) for a specific hardware unit.
    """

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

            # Execution Provider Configuration
            apply_onnx_optimizations()
            if ort is None:
                raise ImportError("onnxruntime not installed.")
            available = ort.get_available_providers()

            target_providers, target_options = self._resolve_providers(available)

            self.separator = separator_class(
                output_dir=str(CACHE_DIR),
                model_file_dir=config.UVR_MODEL_DIR,
                output_format="WAV",
                normalization_threshold=0.9,
                log_level=logging.WARNING
            )

            # Injection of hardware pining
            self.separator.onnx_execution_provider = target_providers
            self.separator.onnx_provider_options = target_options

            self.separator.load_model(config.VOCAL_SEPARATION_MODEL)
            return self.separator

    def _resolve_providers(self, available):
        """Map the unit's hardware to ONNX execution providers."""
        providers = ["CPUExecutionProvider"]
        options = [{}]

        if self._device_type == "CUDA":
            if "CUDAExecutionProvider" in available:
                dev_idx = int(self._device_id.split(":")[-1]) if ":" in self._device_id else 0
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                options = [{"device_id": str(dev_idx)}, {}]
        elif self._device_type in ["GPU", "NPU", "OpenVINO"]:
            if "OpenVINOExecutionProvider" in available:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                options = [{
                    "device_type": self._device_id,
                    "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                    "num_streams": "1"
                }, {}]

        return providers, options

    def process_audio_file(self, audio_path, yield_cb=None):
        """Perform vocal isolation on a file using the unit's separator."""
        if not config.ENABLE_VOCAL_SEPARATION:
            return audio_path

        try:
            sep = self._init_separator()

            # Standard audio-separator output format:
            # Output is written to output_dir with a specific naming convention
            # We must find the resulting 'Vocal' or 'Instrument' stem.

            # MDX-NET usually produces stems with the model name in the filename
            if yield_cb:
                yield_cb()

            unit_name = self._unit["name"] if self._unit else self._device_id
            logger.info("[UVR] Starting vocal isolation on %s...", unit_name)
            p_start = time.time()

            use_cpu_lock = self._device_type == "CPU"
            with self._lock:
                if use_cpu_lock:
                    with utils.cpu_lock_ctx():
                        stems = sep.separate(audio_path)
                else:
                    stems = sep.separate(audio_path)

            dur = time.time() - p_start
            logger.info("[UVR] Isolation complete on %s (Duration: %.2fs)", unit_name, dur)

            if yield_cb:
                yield_cb()

            # Return the path to the isolated vocals
            if stems and len(stems) > 0:
                stem_path = stems[0]
                if not os.path.isabs(stem_path):
                    stem_path = str(CACHE_DIR / stem_path)
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
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
