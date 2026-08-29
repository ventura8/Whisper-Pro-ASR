"""An isolated worker must not re-probe accelerator hardware.

The parent already decided which unit the worker serves, so probing again is redundant --
and not free: ctranslate2.get_cuda_device_count() maps the NVIDIA driver libraries into
the process even when CUDA_VISIBLE_DEVICES hides every device, so an Intel-only UVR
worker still appeared in nvidia-smi. Skipping the probe keeps a worker's address space
confined to the vendor it serves.
"""

import os
from unittest import mock

from modules.core import config_helpers


def test_worker_context_is_detected_from_the_environment():
    """A worker interpreter is identified by the variable its parent sets, nothing else."""
    with mock.patch.dict(os.environ, {"WHISPER_WORKER_CONTEXT": "1"}):
        assert config_helpers.is_worker_context() is True

    with mock.patch.dict(os.environ, {}, clear=True):
        assert config_helpers.is_worker_context() is False


def test_worker_context_skips_every_vendor_probe():
    """No CUDA, AMD or OpenVINO enumeration may happen inside a worker."""
    units: list[dict] = []
    with mock.patch.dict(os.environ, {"WHISPER_WORKER_CONTEXT": "1"}):
        with (
            mock.patch.object(config_helpers, "_detect_cuda_hardware") as cuda,
            mock.patch.object(config_helpers, "_detect_amd_hardware") as amd,
            mock.patch.object(config_helpers, "_detect_intel_hardware") as intel,
        ):
            device, prep, compute = config_helpers.detect_hardware(1, 1, 1, 1, units)

    cuda.assert_not_called()
    amd.assert_not_called()
    intel.assert_not_called()
    assert (device, prep, compute) == ("CPU", "CPU", "int8")


def test_worker_context_still_yields_a_usable_unit_list():
    """config asserts a non-empty pool; the worker gets a CPU placeholder it never uses."""
    units: list[dict] = []
    with mock.patch.dict(os.environ, {"WHISPER_WORKER_CONTEXT": "1"}):
        config_helpers.detect_hardware(1, 1, 1, 1, units)

    assert units == [{"type": "CPU", "id": "CPU", "name": "Host CPU"}]


def test_the_api_process_still_probes():
    """The short-circuit must apply only to workers, never to the parent."""
    units: list[dict] = []
    with mock.patch.dict(os.environ, {}, clear=True):
        with (
            mock.patch.object(config_helpers, "_detect_cuda_hardware") as cuda,
            mock.patch.object(config_helpers, "_detect_amd_hardware"),
            # No create=True: this test exists to prove the probe is reached, so the
            # symbol must actually exist. create=True would invent it after a rename and
            # keep passing against a function nobody calls.
            mock.patch.object(config_helpers, "_detect_intel_hardware"),
        ):
            config_helpers.detect_hardware(1, 1, 1, 1, units)

    cuda.assert_called_once()
