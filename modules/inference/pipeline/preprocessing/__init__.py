"""Vocal Isolation and Signal Preprocessing."""

import gc
import importlib
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None
import sys

from modules.core import config, utils
from modules.inference import scheduler
from modules.inference.pipeline import openvino_provider_dispatch, openvino_resolver

from . import execution as preprocessing_execution
from . import provider as preprocessing_provider
from .helpers import apply_onnx_optimizations as _apply_onnx_optimizations
from .helpers import existing_stem_candidates as _existing_stem_candidates_impl
from .helpers import log_isolation_complete as _log_isolation_complete_impl
from .helpers import run_optional_yield as _run_optional_yield_impl
from .helpers import separate_with_fallback as _separate_with_fallback_impl
from .helpers import stem_resolution_candidates as _stem_resolution_candidates_impl

logger = logging.getLogger(__name__)
_OPENVINO_INIT_LOCKS: dict[str, threading.Lock] = {}
_OPENVINO_INIT_LOCKS_GUARD = threading.Lock()


def _openvino_init_lock_key(device_id: str, device_type: str) -> str:
    """Return lock key for OpenVINO init serialization."""
    target = (device_id or device_type or "OPENVINO").upper()
    family = openvino_resolver.openvino_device_family(target) or target
    if family in {"GPU", "NPU"}:
        return family
    return target


def _get_or_create_openvino_init_lock(key: str) -> threading.Lock:
    lock = _OPENVINO_INIT_LOCKS.get(key)
    if lock is not None:
        return lock
    lock = threading.Lock()
    _OPENVINO_INIT_LOCKS[key] = lock
    return lock


def _openvino_init_lock_for(device_id: str, device_type: str) -> threading.Lock:
    """Return a stable lock scoped by accelerator family/slot for OpenVINO init."""
    key = _openvino_init_lock_key(device_id, device_type)
    with _OPENVINO_INIT_LOCKS_GUARD:
        return _get_or_create_openvino_init_lock(key)


class UVRAcceleratorUnavailableError(RuntimeError):
    """Raised when UVR cannot initialize the requested accelerator provider."""


# --- [ENGINE CONFIGURATION] ---
logging.getLogger("audio_separator").setLevel(logging.INFO)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None


def _lazy_import_separator():
    """Lazy import of audio-separator components."""
    return Separator


def apply_onnx_optimizations():
    """Wrapper to call apply_onnx_optimizations with the current module context."""
    _apply_onnx_optimizations(sys.modules[__name__])


def _get_available_openvino_devices() -> list[str]:
    """Compatibility wrapper for OpenVINO device discovery."""
    return openvino_resolver.get_available_openvino_devices()


def _find_matching_openvino_device(requested: str, available: list[str]) -> str:
    """Compatibility wrapper for OpenVINO device matching."""
    return openvino_resolver.find_matching_openvino_device(requested, available)


def _openvino_retry_candidates(requested: str) -> list[str]:
    """Build retry candidates using preprocessing-specific device discovery hooks."""
    return preprocessing_provider.build_openvino_retry_candidates(requested, _get_available_openvino_devices())


def _resolve_openvino_device_type(device_id: str) -> str:
    """Resolve OpenVINO device type while preserving preprocessing fallback behavior."""
    return preprocessing_provider.resolve_openvino_device_type_for_preprocessing(device_id, _get_available_openvino_devices())


def _reload_onnxruntime_from_intel_path() -> bool:
    """Reload ONNX Runtime from Intel path and update module-level reference."""
    if not openvino_resolver.reload_onnxruntime_from_intel_path():
        return False
    sys.modules[__name__].ort = importlib.import_module("onnxruntime")
    return True


def _get_or_import_ort_module():
    """Return current onnxruntime module, importing and caching it when absent."""
    current_ort = getattr(sys.modules[__name__], "ort", None)
    if current_ort is not None:
        return current_ort
    try:
        imported = importlib.import_module("onnxruntime")
    except ImportError:
        return None
    sys.modules[__name__].ort = imported
    return imported


def _ensure_openvino_onnxruntime(device_type: str) -> None:
    """Ensure OpenVINO-capable ONNX Runtime is loaded for Intel-target requests."""
    if not openvino_resolver.is_openvino_target(device_type):
        return
    current_ort = _get_or_import_ort_module()
    if current_ort is None:
        return
    if openvino_resolver.has_openvino_provider(current_ort):
        return
    logger.info("[UVR] OpenVINO provider missing in active ORT; hot-reloading from Intel runtime path.")
    if not _reload_onnxruntime_from_intel_path():
        logger.warning("[UVR] Failed to hot-reload ONNX Runtime from Intel runtime path.")


def _cpu_provider_config():
    """Return CPU provider configuration."""
    return ["CPUExecutionProvider"], [{}]


def _cuda_or_cpu_provider_config(device_id: str, available):
    """Return CUDA provider configuration or CPU fallback."""
    return openvino_provider_dispatch.cuda_or_cpu_provider_config(device_id, available)


def _openvino_provider_config(device_id: str):
    """Return OpenVINO provider configuration with concrete device aliasing."""
    return preprocessing_provider.openvino_provider_config_for_preprocessing(
        device_id,
        str(config.OV_CACHE_DIR),
        _get_available_openvino_devices(),
        config.PREPROCESS_THREADS,
    )


def _auto_provider_config(available):
    """Return AUTO provider configuration for preprocessing workloads."""
    return preprocessing_provider.resolve_provider_config_for_preprocessing(
        "AUTO",
        "AUTO",
        available,
        _get_available_openvino_devices(),
        str(config.OV_CACHE_DIR),
        preprocess_threads=config.PREPROCESS_THREADS,
    )


def _resolve_provider_config(device_type: str, device_id: str, available):
    """Resolve execution providers for preprocessing using split helper module."""
    return preprocessing_provider.resolve_provider_config_for_preprocessing(
        device_type,
        device_id,
        available,
        _get_available_openvino_devices(),
        str(config.OV_CACHE_DIR),
        preprocess_threads=config.PREPROCESS_THREADS,
    )


def _candidate_output_dirs() -> list[str]:
    """Return ordered candidate output directories for UVR stem files."""
    candidates = [
        str(CACHE_DIR),
        os.path.abspath(config.PERSISTENT_TEMP_DIR),
        tempfile.gettempdir(),
    ]

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def _separate_with_fallback(sep, sep_factory, audio_path, yield_cb=None):
    """Run UVR separation with ENOSPC fallback using module-local candidate rules."""
    return _separate_with_fallback_impl(
        sep,
        sep_factory,
        audio_path,
        yield_cb=yield_cb,
        hooks={"candidate_dirs_fn": _candidate_output_dirs, "disk_usage_fn": shutil.disk_usage},
    )


def _enable_separator_acceleration_flag(separator, target_providers, unit_name: str):
    preprocessing_execution.enable_separator_acceleration_flag(separator, target_providers, unit_name)


def _try_openvino_candidate_load(separator, device_id: str, candidate: str, first_error: Exception) -> tuple[bool, Exception]:
    return preprocessing_execution.try_openvino_candidate_load(separator, device_id, candidate, first_error)


def _stem_resolution_candidates(effective_sep, source_audio_path):
    """Compatibility wrapper retained for tests and legacy callers."""
    return _stem_resolution_candidates_impl(effective_sep, source_audio_path, _candidate_output_dirs)


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

        _ensure_openvino_onnxruntime(self._device_type)

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
        """Get the manager lock used to serialize separator access per device."""
        return self._lock

    def _init_separator(self):
        """Initialize the separator instance pinned to the assigned unit."""
        with self._lock:
            if self.separator is not None:
                return self.separator
            self.separator = self._create_separator_for_unit()
            self._load_separator_model(self.separator)
            return self.separator

    def _create_separator_for_unit(self):
        separator_class = _lazy_import_separator()
        if separator_class is None:
            raise ImportError("audio-separator not installed.")
        if ort is None:
            raise ImportError("onnxruntime not installed.")
        available_providers = ort.get_available_providers()
        logger.debug("Active ORT Version: %s | Providers: %s", ort.__version__, available_providers)
        target_providers, target_options = self._resolve_providers(available_providers)
        unit_name = self._unit["name"] if self._unit else self._device_id
        actual_provider = target_providers[0] if target_providers else "CPUExecutionProvider"
        logger.info(
            "[System] Initializing UVR (%s) on %s... [ONNX provider: %s]",
            config.VOCAL_SEPARATION_MODEL,
            unit_name,
            actual_provider,
        )
        separator = preprocessing_execution.create_separator(_lazy_import_separator, str(CACHE_DIR))
        separator.onnx_execution_provider = target_providers
        openvino_resolver.set_openvino_context_options(target_options)
        _enable_separator_acceleration_flag(separator, target_providers, unit_name)
        return separator

    def _load_separator_model_default(self, separator):
        try:
            separator.load_model(config.VOCAL_SEPARATION_MODEL)
        except (RuntimeError, ValueError, ImportError, OSError, TypeError, AttributeError, KeyError) as e:
            if getattr(separator, "onnx_execution_provider", None) != ["CPUExecutionProvider"]:
                logger.warning(
                    "[UVR] Accelerator '%s' (providers: %s) failed to load model: %s (type: %s); falling back to CPU mode",
                    self._device_id,
                    getattr(separator, "onnx_execution_provider", None),
                    e,
                    type(e).__name__,
                    exc_info=True,
                )
                separator.onnx_execution_provider = ["CPUExecutionProvider"]
                separator.load_model(config.VOCAL_SEPARATION_MODEL)
                return
            logger.error("[System] Failed to load UVR model: %s", e)
            self.separator = None
            raise
        finally:
            utils.THREAD_CONTEXT.ov_options = None

    def _load_separator_model_openvino_serialized(self, separator):
        with _openvino_init_lock_for(self._device_id, self._device_type):
            logger.info(
                "[UVR] Acquired OpenVINO init lock for %s (requested=%s)",
                self._device_id,
                self._device_type,
            )
            logger.info(
                "[UVR] OpenVINO runtime context providers=%s available_devices=%s",
                ort.get_available_providers() if ort is not None else [],
                openvino_resolver.get_available_openvino_devices(),
            )
            try:
                separator.load_model(config.VOCAL_SEPARATION_MODEL)
            except (RuntimeError, ValueError, ImportError, OSError, TypeError, AttributeError, KeyError) as e:
                if self._retry_openvino_separator_load(separator, e):
                    return
                logger.warning(
                    "[UVR] Accelerator '%s' unavailable after OpenVINO retries; falling back to CPU: %s",
                    self._device_id,
                    e,
                )
                separator.onnx_execution_provider = ["CPUExecutionProvider"]
                utils.THREAD_CONTEXT.ov_options = None
                separator.load_model(config.VOCAL_SEPARATION_MODEL)
                return
            finally:
                utils.THREAD_CONTEXT.ov_options = None

    def _load_separator_model(self, separator):
        try:
            if openvino_resolver.is_openvino_target(self._device_type):
                self._load_separator_model_openvino_serialized(separator)
                return
            self._load_separator_model_default(separator)
            return
        except Exception:
            self.separator = None
            raise
        finally:
            utils.THREAD_CONTEXT.ov_options = None

    def _attempt_openvino_retries(self, separator, retries: list[str], requested: str, first_error: Exception) -> bool:
        for candidate in openvino_resolver.alternate_openvino_candidates(retries, requested):
            loaded, first_error = _try_openvino_candidate_load(separator, self._device_id, candidate, first_error)
            if loaded:
                return True
            if openvino_resolver.is_openvino_runtime_loader_error(first_error):
                logger.warning(
                    "[UVR] OpenVINO runtime loader failure persisted while retrying %s; aborting OpenVINO retries",
                    candidate,
                )
                return False
        return False

    def _retry_openvino_separator_load(self, separator, first_error: Exception) -> bool:
        """Retry OpenVINO initialization on alternate Intel devices before CPU fallback."""
        if openvino_resolver.is_openvino_runtime_loader_error(first_error):
            openvino_resolver.disable_all_openvino_families_for_runtime()
            logger.warning("[UVR] OpenVINO runtime loader failure detected; disabling Intel OpenVINO families for this process")
            return False

        requested = (self._device_id or "").upper()
        retries = openvino_resolver.dedupe_openvino_retry_candidates(_openvino_retry_candidates(requested))
        if not retries:
            return False

        return self._attempt_openvino_retries(separator, retries, requested, first_error)

    def _resolve_providers(self, available):
        """Map the unit's hardware to ONNX execution providers."""
        return _resolve_provider_config(self._device_type, self._device_id, available)

    def _allocate_openvino_device(self, device_id: str) -> str:
        if openvino_resolver.is_openvino_family_disabled(device_id):
            return "CPU"
        return _resolve_openvino_device_type(device_id)

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
        return preprocessing_execution.separate_audio(
            self._lock,
            sep,
            audio_path,
            use_cpu_lock=use_cpu_lock,
            target_options=target_options,
            active_yield_cb=active_yield_cb,
            lazy_import_separator=_lazy_import_separator,
            separate_with_fallback=_separate_with_fallback,
        )

    def _resolve_stem_path(self, path_value, effective_sep, source_audio_path):
        """Resolve absolute output path for a UVR stem, including fallback output directories."""
        if os.path.isabs(path_value):
            return path_value

        resolved_candidates = _stem_resolution_candidates_impl(effective_sep, source_audio_path, _candidate_output_dirs)
        for candidate in _existing_stem_candidates_impl(path_value, resolved_candidates):
            return candidate
        return str(CACHE_DIR / path_value)

    def _run_preprocess_pipeline(self, audio_path, yield_cb=None):
        """Run the vocal-separation preprocessing pipeline and return resolved output path."""
        original_path = audio_path
        audio_path = utils.prepare_for_uvr(audio_path, yield_cb=yield_cb)
        if not audio_path:
            return original_path
        scheduler.update_task_progress(5, "Vocal Separation")
        _run_optional_yield_impl(yield_cb)
        sep = self._init_separator()
        stems, effective_sep = self._run_isolation_pipeline(sep, audio_path, yield_cb=yield_cb)
        _run_optional_yield_impl(yield_cb)
        return self._resolve_isolation_output_path(stems, effective_sep, audio_path)

    def preprocess_audio(self, audio_path, force=False, yield_cb=None):
        """Perform vocal isolation on a file using the unit's separator."""
        if not config.ENABLE_VOCAL_SEPARATION and not force:
            return audio_path

        try:
            self._purge_stale_cache()
            return self._run_preprocess_pipeline(audio_path, yield_cb=yield_cb)
        except (UVRAcceleratorUnavailableError, AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            return self._handle_preprocess_error(audio_path, yield_cb, e)

    def _handle_preprocess_error(self, audio_path, yield_cb, error: Exception):
        if isinstance(error, UVRAcceleratorUnavailableError) or self._should_cpu_fallback(error):
            return self._run_cpu_fallback(audio_path, yield_cb, error)
        logger.error("[UVR] Processing failed on %s: %s", self._device_id, error)
        return audio_path

    def _should_cpu_fallback(self, error: Exception) -> bool:
        return openvino_resolver.is_openvino_target(self._device_type) and openvino_resolver.is_openvino_session_fallback_error(error)

    def _run_cpu_fallback(self, audio_path, yield_cb, error: Exception):
        logger.warning("[UVR] Falling back to CPU preprocessing after %s failed: %s", self._device_id, error)
        original_device_id = self._device_id
        original_device_type = self._device_type
        self.separator = None
        self._device_id = "CPU"
        self._device_type = "CPU"
        try:
            return self._run_preprocess_pipeline(audio_path, yield_cb=yield_cb)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as cpu_error:
            logger.error("[UVR] CPU fallback preprocessing failed: %s", cpu_error)
            return audio_path
        finally:
            self._device_id = original_device_id
            self._device_type = original_device_type

    def _run_isolation_pipeline(self, sep, audio_path, yield_cb=None):
        audio_dur = utils.get_audio_duration(audio_path)
        unit_name = self._unit["name"] if self._unit else self._device_id
        actual_provider = getattr(sep, "onnx_execution_provider", ["CPUExecutionProvider"])
        logger.info(
            "[UVR] Starting vocal isolation on %s [ONNX: %s]...",
            unit_name,
            actual_provider[0] if actual_provider else "CPUExecutionProvider",
        )
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
        _log_isolation_complete_impl(unit_name, p_start, audio_dur)
        return stems, effective_sep

    def _resolve_isolation_output_path(self, stems, effective_sep, audio_path):
        return preprocessing_execution.resolve_isolation_output_path(self._resolve_stem_path, stems, effective_sep, audio_path)

    def _purge_stale_cache(self):
        """Remove old preprocessing artifacts from the tmpfs cache."""
        with suppress(Exception):
            # Only purge if cache directory is on a RAM-disk or designated temp area
            for item in CACHE_DIR.iterdir():
                if item.is_file() and (time.time() - item.stat().st_mtime) > 3600:
                    item.unlink()

    def offload(self):
        """Purge model from VRAM/RAM when idle."""
        with self._lock:
            if self.separator:
                logger.info("[System] Offloading UVR engine from %s", self._device_id)
                self.separator = None
                gc.collect()
                utils.clear_gpu_cache()
