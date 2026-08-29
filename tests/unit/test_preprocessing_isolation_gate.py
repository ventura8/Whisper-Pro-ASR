"""Which units may run vocal isolation out-of-process.

Isolating UVR is worth real acceleration -- on an Arc 140T the worker separates at
4.6-5.0x through OpenVINOExecutionProvider, against 0.25x when the in-process manager
falls back to the CPU provider. It is not safe on every Intel GPU: on UHD Graphics the
worker dies with SIGSEGV, intermittently (the first request succeeded and a concurrent
pair killed it), so the decision is made from OpenVINO's reported architecture rather
than from the device type alone.
"""

# The unit under test is the module's internals; reaching them by name is the point.

import importlib
from unittest import mock

import pytest

from modules.inference.pipeline import preprocessing
from modules.inference.pipeline.preprocessing import isolation_policy


@pytest.fixture(autouse=True)
def _clear_arch_cache():
    """Reset the per-device architecture cache around every test.

    ``intel_gpu_arch_for`` memoises successful reads because a GPU's generation cannot
    change while the process runs and the probe costs a full OpenVINO plugin enumeration.
    These tests do change it -- several ask about "GPU" with a different mocked architecture
    each time -- so without this the second one is answered from the first one's cache.
    """
    isolation_policy._ARCH_CACHE.clear()
    yield
    isolation_policy._ARCH_CACHE.clear()


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
        """Patch only the OpenVINO import, delegating every other module to the real one.

        ``return_value=ov`` answered *any* importlib.import_module call with the OpenVINO
        mock, so an unrelated lazy import inside the code under test silently received a
        MagicMock instead of the module it asked for -- a trap that hides real import
        breakage and makes a failure here point at the wrong place.
        """
        core = mock.MagicMock()
        core.get_property.return_value = architecture
        ov = mock.MagicMock()
        ov.Core.return_value = core
        real_import_module = importlib.import_module

        def _import(name, *args, **kwargs):
            return ov if name == "openvino" else real_import_module(name, *args, **kwargs)

        return mock.patch("importlib.import_module", side_effect=_import)

    def test_a_non_intel_vendor_reads_as_unknown(self):
        """OpenVINO's "GPU" may be another vendor's card; its arch says nothing about Intel."""
        with self._with_architecture("GPU: vendor=0x10de arch=v8.6.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") is None

    def test_a_high_version_from_another_vendor_does_not_enable_isolation(self):
        """The failure that matters: a foreign arch above the Intel boundary."""
        with self._with_architecture("GPU: vendor=0x10de arch=v99.0.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") is None
            assert isolation_policy.isolation_supported({"type": "GPU", "id": "GPU"}) is False

    def test_intel_silicon_is_still_read(self):
        """A confirmed Intel vendor id means the generation is the one being asked about."""
        with self._with_architecture("GPU: vendor=0x8086 arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") == (12, 74)

    def test_an_architecture_without_a_vendor_field_is_not_trusted(self):
        """There is no upstream name check, so a vendor-less string is unverified silicon.

        This case used to be read as Intel, on the stated grounds that "the name check
        upstream already covers those". No such check exists: isolation_supported only tests
        that the unit's type is "GPU", which on a hybrid host is exactly when OpenVINO's
        "GPU" may be another vendor's card. Failing open there risks a worker on a part known
        to segfault; failing closed costs the in-process path, which is slower and safe --
        the trade this module's docstring already commits to for every other unknown.
        """
        with self._with_architecture("GPU: arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU") is None

    def test_a_vendor_less_architecture_does_not_enable_isolation(self):
        """The consequence that matters: no worker is spawned for it."""
        with self._with_architecture("GPU: arch=v99.0.0"):
            assert isolation_policy.isolation_supported({"type": "GPU", "id": "GPU"}) is False
