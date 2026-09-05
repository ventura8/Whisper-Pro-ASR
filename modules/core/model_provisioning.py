"""Runtime model provisioning.

Models are no longer baked into the container image. They are downloaded on first
startup into the persistent cache volume and reused on every subsequent start.

The download bodies live here rather than in ``scripts/preload_model.py`` so that the
build-time preloader and the runtime provisioner share one implementation. Heavy
dependencies (faster-whisper, audio-separator, huggingface-hub) are resolved through
``importlib`` at call time so importing this module stays cheap.
"""

import hashlib
import importlib
import logging
import os
import shutil
import tempfile
import threading

from modules.core import model_integrity

logger = logging.getLogger(__name__)

# Failure modes a download can surface: network/transport, filesystem, and the assorted
# errors the hub/separator libraries raise. Startup must degrade, never die, on any of them.
_DOWNLOAD_ERRORS = (RuntimeError, ValueError, OSError, EOFError, ImportError, TypeError, KeyError)

WHISPER_ID = "Systran/faster-whisper-large-v3"
OV_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
# Re-exported, not restated. The digest lives in model_integrity, which is where every
# verifier reads it from; keeping a second literal here meant a model update had to be
# applied twice, and a half-applied one would have this module accept a file the verifier
# rejects.
UVR_MODEL_SHA256 = model_integrity.UVR_MDX_HQ3_SHA256
UVR_DIRECT_URL = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{UVR_MODEL}"

# Provisioning state, consumed by the scheduler queue gate and the telemetry status.
MODEL_READY = threading.Event()
# Set only once startup actually launches provisioning. The scheduler gate stays inert
# unless this is set, so unit tests and embedded usage keep their existing behaviour.
PROVISION_STARTED = threading.Event()
_PROVISION_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
PROGRESS = {"stage": "Idle", "percent": 0, "error": None}


def get_progress():
    """Return a snapshot of the current provisioning progress."""
    with _STATE_LOCK:
        return dict(PROGRESS)


def is_ready():
    """Return True once every required model is present and validated."""
    return MODEL_READY.is_set()


def should_gate_tasks():
    """Return True while tasks must wait for provisioning to finish."""
    return PROVISION_STARTED.is_set() and not MODEL_READY.is_set()


def _set_progress(stage, percent=None, error=None):
    with _STATE_LOCK:
        PROGRESS["stage"] = stage
        if percent is not None:
            PROGRESS["percent"] = percent
        PROGRESS["error"] = error


def _validate_uvr(path):
    return model_integrity.verify_onnx_model_file(path, min_bytes=10 * 1024 * 1024, expected_sha256=UVR_MODEL_SHA256)


def _stream_to_file(url, target_file, expected_sha256, timeout=120):
    """Stream a URL to ``target_file``, verifying sha256 before an atomic replace."""
    requests = importlib.import_module("requests")
    target_dir = os.path.dirname(os.path.abspath(target_file))
    os.makedirs(target_dir, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="model_dl_")
        digest = hashlib.sha256()
        try:
            with os.fdopen(temp_fd, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
            os.replace(temp_path, target_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def ensure_ct2_whisper(target_dir, model_id=WHISPER_ID):
    """Ensure the CTranslate2 faster-whisper model exists in ``target_dir``."""

    def _download():
        faster_whisper = importlib.import_module("faster_whisper")
        faster_whisper.download_model(model_id, output_dir=target_dir)

    return model_integrity.download_with_integrity_retry(
        download_fn=_download,
        validator_fn=model_integrity.verify_ct2_model_dir,
        target_path=target_dir,
        max_retries=2,
        description=f"Faster-Whisper Model ({model_id})",
    )


def ensure_openvino_whisper(target_dir, model_id=OV_MODEL_ID):
    """Ensure the pre-converted OpenVINO Whisper IR exists in ``target_dir``."""

    def _download():
        hub = importlib.import_module("huggingface_hub")
        # No local_dir_use_symlinks: huggingface_hub 1.x removed the parameter and its
        # signature takes no **kwargs, so passing it raises TypeError before a single byte
        # is fetched. Its behaviour is the modern default anyway -- a local_dir download
        # writes real files, not symlinks into the hub cache.
        hub.snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            max_workers=4,
        )

    return model_integrity.download_with_integrity_retry(
        download_fn=_download,
        validator_fn=model_integrity.verify_openvino_model_dir,
        target_path=target_dir,
        max_retries=2,
        description=f"OpenVINO Whisper IR ({model_id})",
    )


def _uvr_via_separator(target_dir, temp_root):
    """Load the UVR model through audio-separator, which fetches it if missing."""
    separator_mod = importlib.import_module("audio_separator.separator")
    uvr_temp_out = os.path.abspath(os.path.join(temp_root, "temp_uvr_preload"))
    os.makedirs(uvr_temp_out, exist_ok=True)
    try:
        sep = separator_mod.Separator(model_file_dir=target_dir, output_dir=uvr_temp_out)
        sep.load_model(UVR_MODEL)
    finally:
        shutil.rmtree(uvr_temp_out, ignore_errors=True)


def ensure_uvr_model(target_dir, temp_root):
    """Ensure the UVR vocal-separation model exists in ``target_dir``."""
    target_file = os.path.join(target_dir, UVR_MODEL)
    os.makedirs(target_dir, exist_ok=True)

    def _download():
        try:
            _uvr_via_separator(target_dir, temp_root)
        except (RuntimeError, ValueError, OSError, EOFError, ImportError, TypeError, KeyError) as exc:
            logger.warning("Separator load_model failed (%s), attempting direct download...", exc)
            _stream_to_file(UVR_DIRECT_URL, target_file, UVR_MODEL_SHA256)

    return model_integrity.download_with_integrity_retry(
        download_fn=_download,
        validator_fn=_validate_uvr,
        target_path=target_file,
        max_retries=2,
        description=f"UVR Model ({UVR_MODEL})",
    )


def _engines_to_provision(config) -> list:
    """Every engine this deployment will load.

    A hybrid host runs Faster-Whisper on its CUDA/AMD units and Intel-Whisper on its
    Intel units, so it needs both weight formats present -- provisioning only the
    "primary" engine would leave half the pool unable to initialise.
    """
    resolver = getattr(config, "engines_in_use", None)
    if callable(resolver):
        return list(resolver())
    return [str(getattr(config, "ASR_ENGINE", "")).upper()]


def _whisper_asset_for_engine(config, engine: str):
    """Return the (label, provision-callable) pair for one engine's weights, or None.

    openai-whisper downloads its own checkpoint on first load, so provisioning it here
    would fetch several gigabytes of CTranslate2 weights it can never read.
    """
    if engine == "OPENAI-WHISPER":
        return None
    if engine == "INTEL-WHISPER":
        return ("OpenVINO Whisper", lambda: ensure_openvino_whisper(config.OV_MODEL_PATH))
    return ("Whisper", _ct2_provisioner(config, engine))


def _ct2_provisioner(config, engine: str):
    """Return a callable that downloads the CT2 weights for ``engine`` into the cache dir.

    Two separate arguments. `ensure_ct2_whisper(target_dir, model_id)` was being called with
    the model value as its *first* parameter, so a custom ASR_MODEL such as
    "Systran/faster-whisper-medium" was used as the download directory -- creating a literal
    "Systran/..." path and downloading the default repo into it. It went unnoticed because
    the default MODEL_ID already resolves to the CT2 cache directory (config_model_paths),
    making the two arguments accidentally interchangeable.
    """
    # getattr with MODEL_ID as the fallback: for default weights the two are the same
    # resolved directory, and a config object that predates CT2_CACHE_DIR -- or a test
    # double standing in for one -- must keep working.
    target_dir = getattr(config, "CT2_CACHE_DIR", None) or config.MODEL_ID
    repo_id = _ct2_repo_id(config, engine)
    # A resolved cache directory is a destination, not a repository id; only a genuine
    # repo id (or local path) may be forwarded as the thing to download.
    if repo_id == target_dir:
        return lambda: ensure_ct2_whisper(target_dir)
    return lambda: ensure_ct2_whisper(target_dir, repo_id)


def _ct2_repo_id(config, engine: str) -> str:
    """The repository (or local path) this engine's CT2 weights come from."""
    by_engine = getattr(config, "MODEL_ID_BY_ENGINE", None)
    if getattr(config, "HYBRID_ENGINES", False) and by_engine:
        return by_engine.get(engine, config.MODEL_ID)
    return config.MODEL_ID


def _required_assets(config):
    """Return the (label, callable) provisioning steps required by the active config."""
    assets = [asset for asset in (_whisper_asset_for_engine(config, engine) for engine in _engines_to_provision(config)) if asset]
    if getattr(config, "ENABLE_VOCAL_SEPARATION", False):
        assets.append(("UVR", lambda: ensure_uvr_model(config.UVR_MODEL_DIR, config.PERSISTENT_TEMP_DIR)))
    return assets


def _fail(label, reason):
    """Record a provisioning failure and release the scheduler gate.

    The gate is opened deliberately: leaving it closed would queue every task forever
    behind a download that is never going to succeed. Letting them through surfaces the
    real engine error instead of an indefinite wait.
    """
    _set_progress(f"Failed: {label}", error=reason)
    logger.error("Model provisioning failed for %s: %s", label, reason)
    MODEL_READY.set()


def provision_models(config):
    """Download every model the active configuration needs.

    Safe to call repeatedly: ``download_with_integrity_retry`` skips assets that are
    already present and valid, so warm starts cost only a validation pass.
    """
    with _PROVISION_LOCK:
        if MODEL_READY.is_set():
            return True
        assets = _required_assets(config)
        total = len(assets)
        for index, (label, provision) in enumerate(assets):
            _set_progress(f"Downloading {label}", int(index * 100 / total))
            try:
                if not provision():
                    _fail(label, f"Could not provision {label}")
                    return False
            except _DOWNLOAD_ERRORS as exc:
                _fail(label, str(exc))
                return False
        _set_progress("Ready", 100)
        MODEL_READY.set()
        logger.info("[Provisioning] All required models are ready.")
        return True


def start_background_provisioning(config):
    """Kick off provisioning in a daemon thread so startup never blocks on downloads."""
    if MODEL_READY.is_set():
        return None
    PROVISION_STARTED.set()
    thread = threading.Thread(target=provision_models, args=(config,), name="model-provisioning", daemon=True)
    thread.start()
    return thread
