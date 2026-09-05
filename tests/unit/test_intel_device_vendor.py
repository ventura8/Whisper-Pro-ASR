"""An OpenVINO "GPU" is not necessarily an Intel GPU.

OpenVINO enumerates any GPU its installed plugins can see. On a host whose image also
carries NVIDIA's OpenCL ICD -- the `full` image does -- it reports the NVIDIA card as
plain "GPU":

    devices: ['CPU', 'GPU']
      GPU: NVIDIA GeForce RTX 5090 (dGPU)
         arch: GPU: vendor=0x10de arch=v12.0.0

Registering that as an Intel unit is not harmless: with per-unit engines it is handed
INTEL-WHISPER, which cannot load on an NVIDIA card, and every request routed to it
returns 500. That was 31 failures across the real-audio suite on the `full` image.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

from unittest import mock

import pytest

from modules.core import config_helpers


def _core(architecture=None, name="Some Device", arch_raises=False):
    core = mock.MagicMock()

    def get_property(dev, prop):  # pylint: disable=unused-argument
        if prop == "DEVICE_ARCHITECTURE":
            if arch_raises:
                raise RuntimeError("property not supported by this plugin")
            return architecture
        return name

    core.get_property.side_effect = get_property
    return core


class TestVendorIdIsAuthoritative:
    """DEVICE_ARCHITECTURE carries the PCI vendor id, which settles the question outright."""

    def test_intel_vendor_id_is_accepted(self):
        """Intel's own vendor id (0x8086) identifies the device as Intel silicon."""
        core = _core("GPU: vendor=0x8086 arch=v12.74.0", "Intel(R) Graphics [0x7d51] (iGPU)")
        assert config_helpers._is_intel_ov_device(core, "GPU.0")

    def test_nvidia_vendor_id_is_rejected(self):
        """The exact device that broke the full image."""
        core = _core("GPU: vendor=0x10de arch=v12.0.0", "NVIDIA GeForce RTX 5090 (dGPU)")
        assert not config_helpers._is_intel_ov_device(core, "GPU")

    def test_vendor_id_beats_a_misleading_name(self):
        """A non-Intel vendor id rejects the device however Intel-sounding its name is."""
        core = _core("GPU: vendor=0x10de arch=v12.0.0", "Intel-ish Marketing Name")
        assert not config_helpers._is_intel_ov_device(core, "GPU")


class TestNameFallback:
    """Used only when no vendor id is available."""

    @pytest.mark.parametrize(
        "name",
        [
            "NVIDIA GeForce RTX 5090 (dGPU)",
            "AMD Radeon RX 7900",
            "Tesla T4",
            "Quadro P2000",
        ],
    )
    def test_named_other_vendors_are_rejected(self, name):
        """Without a vendor id, a name that says NVIDIA or AMD is enough to reject it."""
        assert not config_helpers._is_intel_ov_device(_core(arch_raises=True, name=name), "GPU")

    def test_intel_name_is_accepted(self):
        """An Intel-sounding name is accepted when no vendor id is available."""
        core = _core(arch_raises=True, name="Intel(R) Arc(TM) Graphics")
        assert config_helpers._is_intel_ov_device(core, "GPU")

    def test_unidentifiable_device_is_assumed_intel(self):
        """Pre-existing behaviour for plugins that expose neither: do not drop a real unit.

        The bug this module guards against reported its vendor id, so the authoritative
        path covers it. Rejecting unknowns instead would silently remove working Intel
        accelerators on any plugin that does not implement the property.
        """
        core = _core(arch_raises=True, name="Some Accelerator")
        assert config_helpers._is_intel_ov_device(core, "GPU")

    def test_architecture_without_a_vendor_field_falls_back_to_the_name(self):
        """An architecture string carrying no vendor id falls through to the name check."""
        core = _core("GPU: v12.0.0", "NVIDIA GeForce RTX 5090")
        assert not config_helpers._is_intel_ov_device(core, "GPU")
