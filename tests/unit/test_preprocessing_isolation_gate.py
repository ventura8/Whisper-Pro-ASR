"""Which units may run vocal isolation out-of-process.

Isolating UVR is worth real acceleration -- on an Arc 140T the worker separates at
4.6-5.0x through OpenVINOExecutionProvider, against 0.25x when the in-process manager
falls back to the CPU provider. It is not safe on every Intel GPU: on UHD Graphics the
worker dies with SIGSEGV, intermittently (the first request succeeded and a concurrent
pair killed it), so the decision is made from OpenVINO's reported architecture rather
than from the device type alone.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

from unittest import mock

import pytest

from modules.inference.pipeline import preprocessing
from modules.inference.pipeline.preprocessing import isolation_policy


class TestParseArchitecture:
    """Reading the GPU generation out of OpenVINO's DEVICE_ARCHITECTURE string."""

    def test_reads_major_and_minor(self):
        """An Arc-class architecture string yields its major and minor version."""
        assert preprocessing._parse_intel_gpu_arch("GPU: vendor=0x8086 arch=v12.74.0") == (12, 74)

    def test_reads_the_uhd_form(self):
        """The older UHD form parses the same way, and is what the gate rejects."""
        assert preprocessing._parse_intel_gpu_arch("GPU: vendor=0x8086 arch=v12.0.0") == (12, 0)

    @pytest.mark.parametrize("value", ["", "nonsense", "GPU: vendor=0x8086", None])
    def test_unparseable_is_none(self, value):
        """Anything without a version reads as unknown rather than as a number."""
        assert preprocessing._parse_intel_gpu_arch(value) is None


class TestIsolationDecision:
    """Which units may run vocal isolation out of process, by GPU generation."""

    def _with_arch(self, arch):
        # Patched where the decision reads it: preprocessing re-exports the name for
        # callers, but isolation_supported resolves it inside isolation_policy.
        return mock.patch.object(isolation_policy, "intel_gpu_arch_for", return_value=arch)

    def test_arc_class_gpu_is_isolated(self):
        """Measured working on Intel(R) Graphics [0x7d51], arch v12.74."""
        with self._with_arch((12, 74)):
            assert preprocessing._isolation_supported({"type": "GPU", "id": "GPU.0"})

    def test_alchemist_boundary_is_included(self):
        """The boundary itself is supported: Alchemist is the oldest generation that survives a worker."""
        with self._with_arch((12, 55)):
            assert preprocessing._isolation_supported({"type": "GPU", "id": "GPU.0"})

    def test_uhd_class_gpu_stays_in_process(self):
        """Measured crashing on Intel(R) UHD Graphics, arch v12.0."""
        with self._with_arch((12, 0)):
            assert not preprocessing._isolation_supported({"type": "GPU", "id": "GPU.0"})

    def test_just_below_the_boundary_stays_in_process(self):
        """One minor version below the boundary keeps the in-process manager."""
        with self._with_arch((12, 54)):
            assert not preprocessing._isolation_supported({"type": "GPU", "id": "GPU.0"})

    def test_unknown_generation_stays_in_process(self):
        """Guessing wrong costs a native crash, so an unreadable device is not isolated."""
        with self._with_arch(None):
            assert not preprocessing._isolation_supported({"type": "GPU", "id": "GPU.0"})

    def test_npu_is_never_isolated(self):
        """Never exercised on an NPU; an untested guess here costs a segfault."""
        assert not preprocessing._isolation_supported({"type": "NPU", "id": "NPU.0"})

    @pytest.mark.parametrize("device_type", ["CUDA", "AMD", "CPU"])
    def test_other_vendors_are_always_isolated(self, device_type):
        """The generation gate is an Intel question; other vendors are isolated unconditionally."""
        assert preprocessing._isolation_supported({"type": device_type, "id": device_type.lower()})

    def test_gpu_architecture_is_not_consulted_for_other_vendors(self):
        """Reading OpenVINO properties on a CUDA host would be pointless work.

        Patched on isolation_policy, where the decision resolves it: patching the name
        preprocessing re-exports left the assertion vacuous, because nothing ever calls
        that binding.
        """
        with mock.patch.object(isolation_policy, "intel_gpu_arch_for") as probe:
            preprocessing._isolation_supported({"type": "CUDA", "id": "cuda:0"})
        probe.assert_not_called()


class TestForeignGpuArchitectureIsNotTrusted:
    """OpenVINO's "GPU" is not necessarily the Intel one.

    On a hybrid host with an NVIDIA OpenCL ICD installed, OpenVINO enumerates the NVIDIA
    card as a plain "GPU". Observed on an RTX 3080 + Intel UHD laptop:

        GPU: vendor=0x10de arch=v8.6.0

    That is the NVIDIA compute capability. Comparing it against an Intel generation
    boundary is meaningless -- it happened to fall below the boundary there, so the answer
    was accidentally safe, but a higher number would have enabled a worker on a part known
    to segfault in one.
    """

    def _with_architecture(self, architecture):
        core = mock.MagicMock()
        core.get_property.return_value = architecture
        ov = mock.MagicMock()
        ov.Core.return_value = core
        return mock.patch("importlib.import_module", return_value=ov)

    def test_a_non_intel_vendor_reads_as_unknown(self):
        with self._with_architecture("GPU: vendor=0x10de arch=v8.6.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") is None

    def test_a_high_version_from_another_vendor_does_not_enable_isolation(self):
        """The failure that matters: a foreign arch above the Intel boundary."""
        with self._with_architecture("GPU: vendor=0x10de arch=v99.0.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") is None
            assert isolation_policy.isolation_supported({"type": "GPU", "id": "GPU"}) is False

    def test_intel_silicon_is_still_read(self):
        with self._with_architecture("GPU: vendor=0x8086 arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") == (12, 74)

    def test_an_architecture_without_a_vendor_field_is_still_read(self):
        """Older plugins omit the vendor id; the name check upstream already covers those."""
        with self._with_architecture("GPU: arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") == (12, 74)
