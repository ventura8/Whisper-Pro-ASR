"""Which ONNX Runtime variant gets loaded, and who decides.

ONNX Runtime is used by UVR, VAD and hardware detection -- not by any ASR engine.
FASTER-WHISPER is CTranslate2, INTEL-WHISPER is OpenVINO GenAI, OPENAI-WHISPER and
WHISPERX are torch. So an explicitly named preprocess device is the one that decides
which variant must be importable, and it has to outrank an ASR device left on AUTO.

The bug these pin: the NVIDIA check ran first and was the only vendor check that never
received preprocess_device, so on a hybrid NVIDIA+Intel host with ASR_DEVICE=AUTO it
matched unconditionally. That loaded onnxruntime-gpu, which has no OpenVINO provider, and
ASR_PREPROCESS_DEVICE=GPU ran UVR through CPUExecutionProvider while the logs named the
iGPU -- 0.48x against 2.40x on CUDA, five times slower while claiming acceleration.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

from unittest import mock

from modules.core import bootstrap


def _all_libs_present(path):
    return str(path).startswith("/app/libs/")


def _resolve(device, preprocess_device, *, nvidia=False, intel=False, amd=False):
    with mock.patch("modules.core.bootstrap.os.path.exists", side_effect=_all_libs_present):
        return bootstrap._resolve_target_library(device, preprocess_device, nvidia, intel, amd)


class TestExplicitPreprocessDeviceWins:
    """An explicitly named preprocess device decides the ONNX variant, ahead of the ASR device."""

    def test_intel_preprocess_beats_auto_asr_on_a_hybrid_host(self):
        """The reported bug: NVIDIA present, ASR on AUTO, Intel explicitly requested."""
        lib, reason = _resolve("auto", "GPU", nvidia=True, intel=True)
        assert lib == "/app/libs/intel"
        assert reason == "Intel OpenVINO"

    def test_npu_preprocess_also_selects_the_intel_runtime(self):
        """The NPU is an Intel target too, so it selects the Intel ONNX runtime."""
        lib, _ = _resolve("auto", "NPU", nvidia=True, intel=True)
        assert lib == "/app/libs/intel"

    def test_amd_preprocess_beats_auto_asr(self):
        """An explicit AMD preprocess device wins over an ASR device left on AUTO."""
        lib, reason = _resolve("auto", "AMD", nvidia=True, amd=True)
        assert lib == "/app/libs/amd"
        assert reason == "AMD ROCm"

    def test_explicit_cuda_preprocess_still_selects_nvidia(self):
        """Naming CUDA for preprocessing selects the NVIDIA runtime, as before."""
        lib, _ = _resolve("auto", "CUDA", nvidia=True, intel=True)
        assert lib == "/app/libs/nvidia"


class TestAutoBehaviourIsUnchanged:
    """AUTO must fall through to the original ASR-device-led order."""

    def test_auto_pair_on_nvidia_still_picks_nvidia(self):
        """With both devices on AUTO, an NVIDIA host resolves exactly as it always did."""
        lib, _ = _resolve("auto", "auto", nvidia=True)
        assert lib == "/app/libs/nvidia"

    def test_auto_pair_on_intel_only_still_picks_intel(self):
        """With both devices on AUTO, an Intel-only host resolves to the Intel runtime."""
        lib, _ = _resolve("auto", "auto", intel=True)
        assert lib == "/app/libs/intel"

    def test_explicit_asr_device_still_leads_when_preprocess_is_auto(self):
        """AUTO preprocessing defers to the ASR device, leaving the original order intact."""
        lib, _ = _resolve("cuda", "auto", nvidia=True, intel=True)
        assert lib == "/app/libs/nvidia"

    def test_empty_preprocess_device_is_treated_as_auto(self):
        """An empty value is not a request; it must behave exactly like AUTO."""
        lib, _ = _resolve("auto", "", nvidia=True)
        assert lib == "/app/libs/nvidia"


class TestRequestedRuntimeMustExistInTheImage:
    """A request for a runtime the image does not ship must not claim it."""

    def test_intel_request_falls_through_when_the_image_has_no_intel_libs(self):
        """A runtime the image does not ship cannot be selected, however explicitly it is asked for."""

        def only_nvidia(path):
            return str(path) == "/app/libs/nvidia"

        with mock.patch("modules.core.bootstrap.os.path.exists", side_effect=only_nvidia):
            lib, _ = bootstrap._resolve_target_library("auto", "GPU", True, True, False)

        assert lib == "/app/libs/nvidia", "must not select an Intel runtime this image lacks"

    def test_intel_request_without_intel_hardware_is_still_honoured_if_explicit(self):
        """`GPU` is an explicit request; _should_use_intel_path treats it as such."""
        lib, _ = _resolve("auto", "GPU", nvidia=True, intel=False)
        assert lib == "/app/libs/intel"


class TestExplicitCpuAndAbsentRuntimes:
    """The two ends of the range: CPU asked for outright, and nothing available at all."""

    def test_an_explicit_cpu_preprocess_device_beats_every_accelerator_present(self):
        """CPU is a request like any other, and it is checked before the vendor table.

        Worth pinning separately from the vendor cases: UVR on the CPU is a legitimate
        choice on a host whose accelerator is busy with ASR, and resolving it to a vendor
        runtime because the silicon happens to be there would override the operator on the
        one setting they stated explicitly.
        """
        lib, reason = _resolve("auto", "cpu", nvidia=True, intel=True, amd=True)
        assert lib == "/app/libs/cpu"
        assert reason == "CPU"

    def test_a_cpu_request_is_matched_regardless_of_casing_or_padding(self):
        """Operators write .env by hand, so the match is on the normalized value."""
        assert _resolve("auto", "  CPU  ", nvidia=True)[0] == "/app/libs/cpu"

    def test_a_vendor_request_the_image_cannot_serve_falls_back_to_the_cpu_runtime(self):
        """Falling through to CPU is what keeps an ONNX Runtime on sys.path at all.

        The image build uninstalls the global onnxruntime, so a resolution that returned the
        absent vendor path would leave the process with no ONNX Runtime whatsoever -- UVR
        and VAD both failing at import rather than degrading.
        """

        def only_cpu(path):
            return str(path) == "/app/libs/cpu"

        with mock.patch("modules.core.bootstrap.os.path.exists", side_effect=only_cpu):
            lib, reason = bootstrap._resolve_target_library("cuda", "auto", True, False, False)

        assert (lib, reason) == ("/app/libs/cpu", "CPU Runtime")

    def test_with_no_runtime_directories_at_all_resolution_reports_no_path(self):
        """Nothing to select is a real state, and it must be reported rather than guessed."""
        with mock.patch("modules.core.bootstrap.os.path.exists", return_value=False):
            assert bootstrap._resolve_target_library("auto", "auto", False, False, False) == (None, "Default")

    def test_a_dual_nvidia_amd_host_on_auto_takes_the_amd_runtime(self):
        """Decided before either vendor's own check, so the order is not enumeration luck."""
        lib, reason = _resolve("auto", "auto", nvidia=True, amd=True)
        assert (lib, reason) == ("/app/libs/amd", "AMD ROCm")
