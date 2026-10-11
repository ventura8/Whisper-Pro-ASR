"""Regression checks for restart-safe Compose runtime settings."""

from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).parents[2] / "docker-compose.yml"


def _service():
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))["services"]["whisper-pro-asr"]


def _env():
    raw = _service()["environment"]
    if isinstance(raw, dict):
        return raw
    out = {}
    for item in raw:
        key, _, value = str(item).partition("=")
        out[key] = value
    return out


def test_tmpfs_mounts_remain_writable_after_compose_restart():
    """Compose must remount transient directories writable by the service user."""
    assert "/tmp/whisper:size=2G,mode=1777" in _service()["tmpfs"]


def test_no_jit_cache_is_left_on_tmpfs():
    """A compiled-kernel cache on a RAM disk is recompiled on every restart.

    /tmp/numba-cache used to be a tmpfs mount, filed under SSD write protection. That
    trade is backwards for a JIT cache: it is written once and read forever, so putting
    it on a RAM disk pays the compile cost again on every restart to avoid a few MB of
    writes. Measured on an Intel Arc iGPU, where the equivalent SYCL cache cost 517s cold
    against 39.7s once it survived a container recreate.
    """
    for mount in _service()["tmpfs"]:
        assert "cache" not in str(mount).lower(), f"{mount} puts a cache on tmpfs"


def test_every_kernel_cache_lands_in_the_persistent_volume():
    """Kernel/JIT caches must sit under the bind-mounted model_cache, not in the container.

    Anything else is silently rebuilt whenever the container is recreated, with no error
    and no log line -- only a first request that is an order of magnitude slower.
    """
    env = _env()
    cache_vars = [
        "XDG_CACHE_HOME",  # Intel neo_compiler_cache, ze_intel_npu_cache, torch
        "SYCL_CACHE_DIR",
        "NEO_CACHE_DIR",
        "cl_cache_dir",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "NUMBA_CACHE_DIR",
    ]
    for var in cache_vars:
        assert var in env, f"{var} is not set; its cache would land in the container"
        assert env[var].startswith("/app/model_cache/"), f"{var}={env[var]} is not persistent"


def test_sycl_persistent_caching_is_enabled():
    """SYCL_CACHE_DIR alone does nothing: SYCL only writes to disk when told to."""
    assert _env().get("SYCL_CACHE_PERSISTENT") == "1"


def test_build_cache_export_is_not_max():
    """mode=max exports every intermediate layer of every stage.

    On a validation host that reached 95GB and filled the root filesystem to zero bytes,
    after which the build failed with "no space left on device" while still leaving a
    plausible-looking image behind.
    """
    cache_to = " ".join(str(entry) for entry in _service()["build"]["cache_to"])
    assert "mode=max" not in cache_to
    assert "BUILDX_CACHE_MODE:-min" in cache_to


def test_model_settings_reach_the_container_from_the_environment():
    """remote_validate.sh --env ASR_MODEL=... used to be silently ignored: the value was hardcoded."""
    env = _env()
    assert env["ASR_MODEL"] == "${ASR_MODEL:-Systran/faster-whisper-large-v3}"
    assert env["ASR_COMPUTE_TYPE"] == "${ASR_COMPUTE_TYPE:-AUTO}"
    assert env["MAX_NPU_UNITS"] == "${MAX_NPU_UNITS:-1}"


def test_nemo_settings_are_forwarded_with_the_code_defaults():
    """Each NEMO_* setting reaches the container, defaulting to what config_nemo defaults to."""
    env = _env()
    assert env["NEMO_MODEL"] == "${NEMO_MODEL:-}"
    assert env["NEMO_QUANTIZATION"] == "${NEMO_QUANTIZATION:-AUTO}"
    assert env["NEMO_MAX_CHUNK_SEC"] == "${NEMO_MAX_CHUNK_SEC:-20}"
    assert env["NEMO_BATCH_SIZE"] == "${NEMO_BATCH_SIZE:-4}"
    assert env["NEMO_COMPANION_DEVICE"] == "${NEMO_COMPANION_DEVICE:-AUTO}"
    assert env["NEMO_GPU_MEM_LIMIT_MB"] == "${NEMO_GPU_MEM_LIMIT_MB:-0}"


def test_the_cpu_target_honours_the_requested_engine():
    """docker-compose.cpu.yml hardcoded AUTO, so --engine PARAKEET never reached a CPU run."""
    override = yaml.safe_load((COMPOSE_PATH.parent / "docker-compose.cpu.yml").read_text(encoding="utf-8"))
    assert "ASR_ENGINE=${ASR_ENGINE:-AUTO}" in override["services"]["whisper-pro-asr"]["environment"]
