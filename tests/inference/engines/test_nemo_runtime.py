"""Loading NeMo models on the right provider (modules/inference/engines/nemo_runtime.py)."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.core import config, nemo_catalog
from modules.inference.engines import nemo_runtime
from tests.inference.engines.nemo_fixtures import FakeModel, fake_onnx_asr, fake_onnxruntime

PARAKEET = nemo_catalog.CATALOG["nemo-parakeet-tdt-0.6b-v3"]
CANARY = nemo_catalog.CATALOG["nemo-canary-1b-v2"]
CUDA_UNIT = {"type": "CUDA", "id": "cuda:1", "name": "RTX"}
GPU_UNIT = {"type": "GPU", "id": "GPU.0", "name": "Arc"}
CPU_UNIT = {"type": "CPU", "id": "CPU", "name": "Host CPU"}


def _names(providers):
    """Provider names, without their option dicts."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def test_cuda_unit_gets_the_cuda_provider_on_its_own_card():
    """cuda:1 runs on device 1, with requested-size arena growth."""
    providers = nemo_runtime.providers_for(CUDA_UNIT, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert _names(providers) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    options = providers[0][1]
    assert options["device_id"] == 1
    assert options["arena_extend_strategy"] == "kSameAsRequested"


def test_intel_gpu_gets_openvino_without_a_kernel_cache():
    """A kernel cache_dir segfaults a second same-process session."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.get_available_openvino_devices", return_value=["CPU", "GPU.0"]):
        providers = nemo_runtime.providers_for(GPU_UNIT, ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert _names(providers)[0] == "OpenVINOExecutionProvider"
    options = providers[0][1]
    assert options["device_type"] == "GPU.0"
    assert options["num_streams"] == "1"
    assert "cache_dir" not in options


@pytest.mark.parametrize(
    ("unit", "available"),
    [
        # A CUDA unit in a worker that imported the CPU build must not claim CUDA.
        (CUDA_UNIT, ["CPUExecutionProvider"]),
        (GPU_UNIT, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        # NPU and AMD run NeMo on the CPU by design.
        ({"type": "NPU", "id": "NPU"}, ["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
        ({"type": "AMD", "id": "amd:0"}, ["ROCMExecutionProvider", "CPUExecutionProvider"]),
        (CPU_UNIT, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ],
)
def test_everything_else_is_the_cpu(unit, available):
    """No provider is claimed that the unit or the imported build cannot supply."""
    assert nemo_runtime.providers_for(unit, available) == ["CPUExecutionProvider"]


def test_session_options_follow_asr_threads_and_quiet_the_runtime(monkeypatch):
    """ASR_THREADS bounds ONNX Runtime; its warnings are silenced."""
    monkeypatch.setattr(config, "ASR_THREADS", 6)
    options = nemo_runtime.session_options(fake_onnxruntime([]))
    assert (options.intra_op_num_threads, options.inter_op_num_threads, options.log_severity_level) == (6, 1, 3)


def test_vendor_runtime_reload_is_for_intel_gpu_units_only():
    """Only an Intel GPU unit needs the OpenVINO build hot-loaded."""
    with mock.patch("modules.inference.pipeline.openvino_resolver.ensure_openvino_onnxruntime") as ensure:
        nemo_runtime.ensure_vendor_runtime("CUDA")
        nemo_runtime.ensure_vendor_runtime("NPU")
        ensure.assert_not_called()
        nemo_runtime.ensure_vendor_runtime("GPU")
    ensure.assert_called_once_with("GPU")


@pytest.fixture(name="runtime")
def _runtime(monkeypatch):
    """Install fake onnx_asr / onnxruntime modules offering CUDA."""
    asr = fake_onnx_asr(FakeModel("CUDAExecutionProvider"))
    monkeypatch.setitem(sys.modules, "onnx_asr", asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    return asr


def test_load_passes_explicit_providers_and_warms_up(runtime, caplog):
    """Providers are explicit, and one warm-up recognise runs at load."""
    model, device = nemo_runtime.load(PARAKEET, "/cache/nemo/p", None, CUDA_UNIT)
    assert device == "cuda"
    args, kwargs = runtime.load_model.call_args
    assert args == (PARAKEET.name, "/cache/nemo/p")
    assert kwargs["quantization"] is None
    assert _names(kwargs["providers"]) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert len(model.calls) == 1, "one warm-up recognise at load"
    assert "ONNX Runtime 1.29.0" in caplog.text


def test_canary_warm_up_names_a_language(runtime):
    """Canary's warm-up must name a language like every other call."""
    model, _ = nemo_runtime.load(CANARY, "/cache/nemo/c", None, CUDA_UNIT)
    assert runtime.load_model.call_count == 1
    assert model.calls[0]["language"] == "en"


def test_load_silences_the_runtimes_default_logger(runtime, monkeypatch):
    """ORT 1.29 warns through its default logger before any session option applies."""
    ort = sys.modules["onnxruntime"]
    calls = []
    monkeypatch.setattr(ort, "set_default_logger_severity", calls.append, raising=False)
    nemo_runtime.load(PARAKEET, "/p", None, CUDA_UNIT)
    assert calls == [3]
    assert runtime.load_model.call_count == 1


def test_an_accelerator_that_cannot_run_the_graph_falls_back_to_the_cpu(monkeypatch, caplog):
    """A failed warm-up reloads on the CPU and reports it."""
    failing = FakeModel("CUDAExecutionProvider", fail_on=1)
    cpu = FakeModel("CPUExecutionProvider")
    asr = SimpleNamespace(load_model=mock.MagicMock(side_effect=[failing, cpu]))
    monkeypatch.setitem(sys.modules, "onnx_asr", asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    model, device = nemo_runtime.load(PARAKEET, "/p", None, CUDA_UNIT)
    assert (model, device) == (cpu, "cpu")
    assert _names(asr.load_model.call_args.kwargs["providers"]) == ["CPUExecutionProvider"]
    assert "reloading on the CPU" in caplog.text


def test_a_cpu_load_failure_is_not_retried(monkeypatch):
    """A CPU load that fails has nothing to fall back to."""
    asr = SimpleNamespace(load_model=mock.MagicMock(side_effect=RuntimeError("missing file")))
    monkeypatch.setitem(sys.modules, "onnx_asr", asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="missing file"):
        nemo_runtime.load(PARAKEET, "/p", "int8", CPU_UNIT)
    assert asr.load_model.call_count == 1


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("CUDAExecutionProvider", "cuda"), ("OpenVINOExecutionProvider", "GPU.0"), ("CPUExecutionProvider", "cpu")],
)
def test_actual_device_reads_the_encoder_session(provider, expected):
    """The device reported is the encoder session's, in execution_status spelling."""
    assert nemo_runtime.actual_device(FakeModel(provider), GPU_UNIT) == expected


def test_actual_device_tolerates_an_onnx_asr_without_the_private_session():
    """A moved private name reads as the CPU, never as a crash."""
    assert nemo_runtime.actual_device(SimpleNamespace(), CUDA_UNIT) == "cpu"
