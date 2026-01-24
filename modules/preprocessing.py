"""
Vocal Isolation and Signal Preprocessing

This module provides vocal isolation capabilities using the UVR (Ultimate Vocal Remover)
MDX-NET architecture. It implements hardware-specific optimizations for ONNX Runtime,
including OpenVINO and CUDA backends, and handles both file-level and chunk-level 
audio cleaning.
"""
import gc
import logging
import os
from pathlib import Path
import threading
import time
import uuid

import torch  # pylint: disable=import-error
import soundfile as sf  # pylint: disable=import-error
import numpy as np

from . import config, utils

# Lazy-loaded modules for hardware coordination
ort = None  # pylint: disable=invalid-name
Separator = None  # pylint: disable=invalid-name


logger = logging.getLogger(__name__)

# --- [ENGINE CONFIGURATION] ---
logging.getLogger("audio_separator").setLevel(logging.INFO)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Singleton manager instance
_INSTANCE = None


# Core engine: audio-separator provides the UVR/MDX-NET implementation
# Lazy import pattern allows hardware patching before the module is loaded
# pylint: disable=global-statement  # Required for lazy loading pattern
def _lazy_import_separator():
    """Lazy import of audio-separator components."""
    global Separator
    if Separator is None:
        try:
            # pylint: disable=import-outside-toplevel  # Intentional lazy import for hardware patching
            from audio_separator.separator import Separator as OriginalSeparator
            Separator = OriginalSeparator
        except ImportError:
            Separator = None
    return Separator


# pylint: disable=global-statement,too-many-statements  # Required for lazy loading pattern; complex hardware patching
def apply_onnx_optimizations():
    """
    Monkeypatch ONNX Runtime and audio-separator for hardware acceleration.
    Must be called BEFORE any heavy AI imports or Separator instantiation.
    """
    global ort
    try:
        if ort is None:
            # pylint: disable=import-outside-toplevel,import-error  # Intentional lazy import for hardware patching
            import onnxruntime as loaded_ort
            ort = loaded_ort

        if getattr(ort.InferenceSession, "_is_patched", False) is not True:
            logger.debug(
                "[System] Optimization: Patching ONNX Runtime for "
                "shared hardware priority..."
            )

            # --- [PATCH 1: ORT Provider Priority] ---
            original_get_providers = ort.get_available_providers

            def patched_get_providers():
                available = original_get_providers()
                if (config.PREPROCESS_DEVICE == "CUDA" and
                        "CUDAExecutionProvider" in available):
                    return ["CUDAExecutionProvider"] + [
                        p for p in available if p != "CUDAExecutionProvider"
                    ]

                if (config.PREPROCESS_DEVICE in ["GPU", "NPU"] and
                        "OpenVINOExecutionProvider" in available):
                    return ["OpenVINOExecutionProvider"] + [
                        p for p in available if p != "OpenVINOExecutionProvider"
                    ]

                return available

            ort.get_available_providers = patched_get_providers

            # 2. Configure global session options template
            try:
                thread_count = int(config.PREPROCESS_THREADS)
                os.environ["OMP_NUM_THREADS"] = str(thread_count)
                os.environ["MKL_NUM_THREADS"] = str(thread_count)
                os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(thread_count)
                os.environ["ORT_INTER_OP_NUM_THREADS"] = str(thread_count)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            # --- [PATCH 3: InferenceSession Threading] ---
            # Explicitly wrap InferenceSession to force thread limits via SessionOptions
            original_session = ort.InferenceSession

            # pylint: disable=too-few-public-methods  # Internal wrapper class with only __init__
            class ThreadLimitedSession(original_session):
                """Internal session class with CPU thread constraints."""
                _is_patched = True

                def __init__(self, path_or_bytes, sess_options=None,
                             providers=None, provider_options=None, **kwargs):

                    if sess_options is None:
                        sess_options = ort.SessionOptions()

                    try:
                        thread_count = int(config.PREPROCESS_THREADS)
                        sess_options.intra_op_num_threads = thread_count
                        sess_options.inter_op_num_threads = thread_count
                        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                        # Only log once to avoid clutter
                        if not getattr(PreprocessingManager, "_logged_threads", False):
                            logger.info(
                                "[System] Preprocessing Resource Limit: %d threads", thread_count)
                            PreprocessingManager._logged_threads = True
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass

                    if providers:
                        # Normalize providers to handle both string and tuple formats
                        # We inject options for OpenVINO if requested and missing
                        new_providers = []
                        for provider in providers:
                            is_ov = isinstance(
                                provider, str) and provider == "OpenVINOExecutionProvider"
                            if is_ov:
                                ov_opts = {
                                    "device_type": config.PREPROCESS_DEVICE,
                                    "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                                    "num_streams": "1"
                                }
                                new_providers.append((provider, ov_opts))
                            else:
                                new_providers.append(provider)
                        providers = new_providers

                    super().__init__(path_or_bytes, sess_options, providers,
                                     provider_options, **kwargs)

            ort.InferenceSession = ThreadLimitedSession
            # pylint: disable=protected-access
            ort.InferenceSession._is_patched = True

            # --- [PATCH 4: audio-separator Hardware Detection] ---
            # audio-separator 0.41.1 doesn't recognize OpenVINO as valid acceleration.
            # We patch it to recognize OpenVINO and force GPU usage if requested.
            separator_class = _lazy_import_separator()
            if separator_class:
                original_setup_torch = separator_class.setup_torch_device

                def patched_setup_torch(self, system_info_arg):
                    # Suppress original noisy fallback message
                    sep_logger = logging.getLogger("audio_separator")
                    original_level = sep_logger.level
                    sep_logger.setLevel(logging.ERROR)
                    try:
                        original_setup_torch(self, system_info_arg)
                    finally:
                        sep_logger.setLevel(original_level)

                    # Force to OpenVINO if requested and available
                    available_providers = ort.get_available_providers()
                    if (config.PREPROCESS_DEVICE in ["GPU", "NPU"] and
                            "OpenVINOExecutionProvider" in available_providers):
                        self.logger.info(
                            "[System] Overriding Separator backend: -> OpenVINO (%s)",
                            config.PREPROCESS_DEVICE
                        )
                        ov_opts = {
                            "device_type": config.PREPROCESS_DEVICE,
                            "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                            "num_streams": "1"
                        }
                        self.onnx_execution_provider = [
                            ("OpenVINOExecutionProvider", ov_opts)]
                        self.onnx_provider_options = [ov_opts]

                    # Force to CUDA if requested and available
                    elif (config.PREPROCESS_DEVICE == "CUDA" and
                          "CUDAExecutionProvider" in available_providers):
                        self.logger.info(
                            "[System] Overriding Separator backend: -> CUDA")
                        # Force CUDA provider
                        self.onnx_execution_provider = [
                            "CUDAExecutionProvider", "CPUExecutionProvider"]
                        self.onnx_provider_options = [{}, {}]

                separator_class.setup_torch_device = patched_setup_torch

            ort.InferenceSession._is_patched = True

    except Exception as patch_err:  # pylint: disable=broad-exception-caught
        logger.warning(
            "[System] Failed to apply ONNX Runtime optimizations: %s", patch_err
        )


def get_manager():
    """Access the global PreprocessingManager instance."""
    # pylint: disable=global-statement  # Required for singleton pattern
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PreprocessingManager()
    return _INSTANCE


class PreprocessingManager:
    """
    Orchestrates audio cleaning models (UVR/MDX-NET).

    Responsible for hardware acceleration patching, model lifecycle management,
    and concurrent processing locks.
    """
    _logged_threads = False
    _logged_providers = False

    def __init__(self):
        self.device = config.PREPROCESS_DEVICE
        self.separator = None
        self._lock = threading.Lock()

    def ensure_models_loaded(self):
        """
        Pre-loads and compiles UVR models to minimize runtime latency.

        This method triggers hardware-specific kernel compilation (OpenCL/CUDA)
        by performing a synthetic 'warmup' inference pass.
        """
        if config.ENABLE_VOCAL_SEPARATION or config.ENABLE_LD_PREPROCESSING:
            try:
                logger.debug(
                    "[System] Warmup: Initializing UVR isolation engine...")
                self._init_separator()
                self._perform_warmup()
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "[System] UVR Warmup failed (System will compile on first request): %s",
                    err
                )

    def _perform_warmup(self):
        """Execute a dummy inference pass to trigger engine compilation."""
        logger.debug("[System] Warmup: Compiling kernels for %s...",
                     config.PREPROCESS_DEVICE_NAME)

        # Create a 1-second synthetic white noise signal
        dummy_path = os.path.join("/tmp", f"warmup_{uuid.uuid4().hex}.wav")
        try:
            sample_rate = 44100
            duration = 1.0
            noise = np.random.normal(0, 0.1, int(sample_rate * duration))
            sf.write(dummy_path, noise, sample_rate)

            # Silence standard output during warmup
            original_level = logging.getLogger("audio_separator").level
            logging.getLogger("audio_separator").setLevel(logging.ERROR)

            try:
                with self._lock:
                    self.separator.separate(dummy_path)
            finally:
                logging.getLogger("audio_separator").setLevel(original_level)

            logger.debug("[System] UVR Warmup: Isolation engine ready.")

        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def _init_separator(self):
        """Initialize the underlying Separator instance with hardware optimizations."""
        if self.separator is not None:
            return

        # Enforce thread limits EARLY before any ONNX Runtime sessions are created
        try:
            threads = int(config.PREPROCESS_THREADS)
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)
            # Force OpenMP to respect thread limits
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(threads)
            os.environ["ORT_INTER_OP_NUM_THREADS"] = str(threads)
            # Limit inter-op parallelism
            # threading.set_interop_parallelism_limit(1) # pylint: disable=no-member
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Configure OpenVINO/OpenCL persistent caching
        abs_cache_dir = os.path.abspath(config.OV_CACHE_DIR)
        os.environ["OV_CACHE_DIR"] = abs_cache_dir
        os.environ["OV_GPU_CACHE_DIR"] = abs_cache_dir
        os.environ["cl_cache_dir"] = abs_cache_dir

        # Lazy import of components and hardware patching
        apply_onnx_optimizations()
        separator_class = _lazy_import_separator()
        if separator_class is None:
            logger.warning(
                "[Prep] audio-separator library missing. Vocal isolation disabled.")
            return

        # Providers: https://onnxruntime.ai/docs/execution-providers/
        available = ort.get_available_providers() if ort else []
        target_providers, target_options = self._configure_providers(available, [], [])

        # Log active backend for user visibility
        primary = target_providers[0] if target_providers else "CPUExecutionProvider"

        nice_name = "CPU"
        if "CUDA" in primary:
            nice_name = "NVIDIA (CUDA)"
        elif "OpenVINO" in primary:
            nice_name = "OpenVINO (NPU/GPU)"
        elif "Tensorrt" in primary:
            nice_name = "TensorRT (GPU)"

        logger.info("[System] Overriding Separator backend: -> %s", nice_name)
        logger.debug("[System] Provider details: %s | Options: %s",
                     primary, target_options[0] if target_options else "None")

        self.separator = separator_class(
            output_dir=str(CACHE_DIR),
            model_file_dir=config.UVR_MODEL_DIR,
            output_format="WAV",
            normalization_threshold=0.9
        )

        # Manually set provider configuration to ensure hardware acceleration
        # audio-separator 0.41.1 doesn't accept these in __init__
        self.separator.onnx_execution_provider = target_providers
        self.separator.onnx_provider_options = target_options

        logger.debug("[System] Loading MDX-NET weight profile: %s",
                     config.VOCAL_SEPARATION_MODEL)
        self.separator.load_model(config.VOCAL_SEPARATION_MODEL)

    @staticmethod
    def _log_available_providers():
        if not PreprocessingManager._logged_providers and ort:
            logger.debug("[System] ONNX Providers: %s",
                         ort.get_available_providers())
            PreprocessingManager._logged_providers = True

    @staticmethod
    def _configure_providers(available, providers, provider_options):
        """Map abstract device requests to concrete ONNX providers."""
        target_providers = list(providers)
        target_options = list(provider_options)

        # Case 1: NVIDIA Acceleration
        if config.PREPROCESS_DEVICE == "CUDA":
            if "CUDAExecutionProvider" in available:
                target_providers = ["CUDAExecutionProvider"]
            else:
                logger.warning(
                    "[System] CUDA requested for UVR, but provider is unavailable.")

        # Case 2: Intel Acceleration (GPU/NPU via OpenVINO)
        elif config.PREPROCESS_DEVICE in ["GPU", "NPU"]:
            if "OpenVINOExecutionProvider" in available:
                ov_opts = {
                    "device_type": config.PREPROCESS_DEVICE,
                    "cache_dir": os.path.abspath(config.OV_CACHE_DIR),
                    "num_streams": "1"
                }
                # Use the tuple format [('ProviderName', {options})] to ensure compatibility
                # with architecture classes that only pass 'providers' to ONNX Runtime.
                target_providers = [("OpenVINOExecutionProvider", ov_opts)]
                target_options = [ov_opts]
            else:
                logger.warning(
                    "[System] OpenVINO hardware requested but provider is missing.")

        # Always ensure CPU is available for fallback
        if "CPUExecutionProvider" not in target_providers:
            target_providers.append("CPUExecutionProvider")
            target_options.append({})

        return target_providers, target_options

    def process_audio_file(self, file_path, yield_cb=None): # pylint: disable=unused-argument
        """Perform vocal isolation on a media file from disk."""
        if not (config.ENABLE_VOCAL_SEPARATION or config.ENABLE_LD_PREPROCESSING):
            return str(file_path)

        logger.info("[Prep] Executing vocal isolation (UVR/MDX-NET)...")
        start = time.time()
        output_files = []
        vocal_path = None

        try:
            self._init_separator()
            input_path = str(file_path)

            with self._lock:
                output_files = self.separator.separate(input_path)

            vocal_path = self._identify_vocals(output_files)

            if vocal_path and os.path.exists(vocal_path):
                logger.info(
                    "[Prep] Isolated vocals finalized in %s.",
                    utils.format_duration(time.time() - start)
                )
                return vocal_path

            return str(file_path)

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("[Prep] Vocal isolation task failed: %s", e)
            return str(file_path)
        finally:
            self._cleanup_stems(output_files, vocal_path)

    def _identify_vocals(self, output_files):
        """Locate the (Vocals) stem in the collection of output files."""
        vocal_path = next((f for f in output_files if "(Vocals)" in f), None)

        # Reconciliation of relative paths from audio-separator
        if vocal_path and not os.path.exists(vocal_path):
            possible = os.path.join(str(CACHE_DIR), vocal_path)
            if os.path.exists(possible):
                vocal_path = possible
        return vocal_path

    def _cleanup_stems(self, output_files, keep_path=None):
        """Securely remove auxiliary stems while preserving the target track."""
        for out_f in output_files:
            if out_f and out_f != keep_path and os.path.exists(out_f):
                try:
                    os.remove(out_f)
                except Exception: # pylint: disable=broad-exception-caught
                    pass

    def process_audio_chunk(self, chunk, sr=16000, yield_cb=None):
        """Clean an in-memory audio chunk by oscillating to disk."""
        if not (config.ENABLE_VOCAL_SEPARATION or config.ENABLE_LD_PREPROCESSING):
            return chunk

        if np.max(np.abs(chunk)) < 0.005:
            return chunk

        try:
            self._init_separator()
            tmp_id = uuid.uuid4().hex
            in_path = os.path.join(str(CACHE_DIR), f"chunk_in_{tmp_id}.wav")
            sf.write(in_path, chunk, sr, subtype='PCM_16')

            if yield_cb:
                yield_cb()

            with self._lock:
                output_files = self.separator.separate(in_path)

            vocal_path = self._identify_vocals(output_files)
            final_chunk = chunk

            if vocal_path and os.path.exists(vocal_path):
                final_chunk = self._postprocess_chunk(vocal_path, sr, len(chunk))

            # Atomic cleanup
            if os.path.exists(in_path):
                os.remove(in_path)
            self._cleanup_stems(output_files)

            return final_chunk

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.warning("[Prep] Chunk isolation failed: %s", e)
            return chunk

    def _postprocess_chunk(self, vocal_path, target_sr, target_len):
        """Load and normalize cleaned audio to original buffer specifications."""
        data, read_sr = sf.read(vocal_path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Simple resample back to inference SR
        if read_sr != target_sr:
            indices = np.linspace(
                0, len(data) - 1, int(len(data) * (target_sr / read_sr))
            ).astype(int)
            data = data[indices]

        # Ensure buffer length remains identical to prevent batch desync
        if len(data) != target_len:
            if len(data) > target_len:
                data = data[:target_len]
            else:
                padded = np.zeros(target_len, dtype=data.dtype)
                padded[:len(data)] = data
                data = padded

        return data

    def offload(self):
        """Release isolation engine assets to free hardware resources."""
        if self.separator:
            logger.info("[System] Offloading UVR engine...")
            del self.separator
            self.separator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
