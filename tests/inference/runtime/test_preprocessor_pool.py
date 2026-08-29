"""Which device a task's vocal isolation lands on, and how many managers get built.

Every branch here decides where UVR runs, and getting it wrong is silent: the request
succeeds, the transcript is correct, and only a `nvidia-smi`/`intel_gpu_top` sample taken
during separation shows the work went to the wrong device. That is the failure recorded in
docs/REMOTE_VALIDATION.md, where ASR_PREPROCESS_DEVICE=GPU ran UVR on CUDA at five times
the speed of the "Intel" path the banner was naming.

Building a real PreprocessingManager loads the UVR model onto a device, so
``preprocessing.create_manager`` is stubbed throughout: the assertions are about which unit
it is asked for and how many times it is called, which is exactly what the leak and the
misrouting are made of.
"""

from __future__ import annotations

import threading
from unittest import mock

import pytest

from modules.inference.runtime import preprocessor_pool


def _unit(unit_type: str, unit_id: str | None = None) -> dict:
    return {"type": unit_type, "id": unit_id or unit_type, "name": f"{unit_type} unit"}


def _manager(device_type: str) -> mock.MagicMock:
    manager = mock.MagicMock(name=f"manager-{device_type}")
    manager.device_type = device_type
    return manager


@pytest.fixture(name="created")
def _created():
    """Stub create_manager, recording the unit each call was given."""
    calls: list[dict | None] = []

    def create(unit=None):
        """Build a fake manager, recording the unit it was given."""
        calls.append(unit)
        return _manager(unit["type"] if unit else "CUDA")

    with mock.patch.object(preprocessor_pool.preprocessing, "create_manager", side_effect=create):
        yield calls


class TestCreatingTheSharedPreprocessor:
    """Which unit the shared per-type preprocessor is built against."""

    def test_a_scheduler_pool_unit_of_that_type_is_preferred(self, created):
        """A scheduler pool unit of that type is preferred."""
        with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("GPU", "GPU.0")]):
            with mock.patch.object(preprocessor_pool.config, "DETECTED_UNITS", [_unit("GPU", "GPU.9")]):
                preprocessor = preprocessor_pool._create_shared_preprocessor("GPU")

        assert created == [_unit("GPU", "GPU.0")], "a unit still in the scheduler pool is the better assignment"
        assert preprocessor.device_type == "GPU"

    def test_a_unit_only_the_pre_filter_snapshot_still_holds_is_used(self, created):
        """DETECTED_UNITS is itself ASR-filtered; DETECTED_HARDWARE_UNITS is not.

        DETECTED_UNITS is snapshotted after the Intel restriction and the requested-device
        narrowing have already run, so an ASR_DEVICE=NPU request prunes the iGPU from it --
        while UVR, on ONNX Runtime, can still use that iGPU. Only the wider snapshot has it.
        """
        with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("NPU", "NPU.0")]):
            with mock.patch.object(preprocessor_pool.config, "DETECTED_UNITS", [_unit("NPU", "NPU.0")]):
                with mock.patch.object(preprocessor_pool.config, "DETECTED_HARDWARE_UNITS", [_unit("NPU", "NPU.0"), _unit("GPU", "GPU.0")]):
                    preprocessor = preprocessor_pool._create_shared_preprocessor("GPU")

        assert created == [_unit("GPU", "GPU.0")]
        assert preprocessor.device_type == "GPU"

    def test_a_unit_detection_found_but_the_engine_filter_removed_is_still_used(self, created):
        """HARDWARE_UNITS is filtered to what the ASR engine can drive; UVR reaches further.

        This is the hybrid NVIDIA+Intel case: CTranslate2 cannot touch the Intel iGPU, so it
        is pruned from the scheduler pool -- but UVR is ONNX Runtime and runs on it happily.
        Looking only at the filtered pool fell through to an unassigned create_manager(),
        which resolves to CUDA, so ASR_PREPROCESS_DEVICE=GPU silently ran on the NVIDIA card.
        """
        with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("CUDA", "cuda:0")]):
            with mock.patch.object(preprocessor_pool.config, "DETECTED_UNITS", [_unit("GPU", "GPU.0")]):
                preprocessor = preprocessor_pool._create_shared_preprocessor("GPU")

        assert created == [_unit("GPU", "GPU.0")]
        assert preprocessor.device_type == "GPU"

    def test_a_requested_type_that_does_not_exist_falls_back_and_warns(self, created):
        """A requested type that does not exist falls back and warns."""
        with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("CUDA", "cuda:0")]):
            with mock.patch.object(preprocessor_pool.config, "DETECTED_UNITS", []):
                with mock.patch.object(preprocessor_pool.logger, "warning") as warned:
                    preprocessor_pool._create_shared_preprocessor("NPU")

        assert created == [None], "an unassigned manager is the documented fallback"
        warned.assert_called_once()

    def test_a_missing_detected_units_attribute_is_tolerated(self, created, monkeypatch):
        """config is reloaded in places; the lookup must not depend on the attribute existing.

        The attribute is DELETED, not patched to []. Patching it to an empty list leaves
        `getattr(config, "DETECTED_UNITS", [])` reading a present attribute, so the
        missing-attribute branch this test is named for never ran.
        """
        monkeypatch.setattr(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("NPU", "NPU.0")])
        monkeypatch.delattr(preprocessor_pool.config, "DETECTED_UNITS", raising=False)
        assert not hasattr(preprocessor_pool.config, "DETECTED_UNITS"), "the attribute must be absent for this test to mean anything"

        preprocessor_pool._create_shared_preprocessor("NPU")

        assert created == [_unit("NPU", "NPU.0")]


class TestTheSharedCacheIsBuiltExactlyOnce:
    """Two tasks arriving together must not each load the UVR model."""

    def test_a_cache_hit_does_not_build_anything(self, created):
        """A cache hit does not build anything."""
        existing = _manager("NPU")
        pool = {"PREPROCESS::NPU": existing}

        assert preprocessor_pool._shared_preprocessor_for_type(pool, "NPU") is existing
        assert created == []

    def test_concurrent_misses_produce_one_manager_and_one_pool_entry(self):
        """The double-checked lock is the whole point of this path.

        Without the re-check under the lock, the loser's manager was overwritten in the pool
        while still holding its ONNX session and its share of device memory -- leaked for the
        life of the process, and on an NPU enough to make the second load fail outright.
        """
        pool: dict = {}
        built: list[mock.MagicMock] = []
        first_is_building = threading.Event()
        second_has_missed = threading.Event()
        roles = iter([True, False])
        roles_lock = threading.Lock()

        def slow_create(unit=None):
            """Build one manager, releasing only once the other thread has missed the cache.

            The barrier this replaces was waited on *before* entering
            _shared_preprocessor_for_type, which proves only that both threads started --
            not that both got past the unlocked cache check. Whether the second one ever
            reached the contended window was down to scheduling, so the test could pass
            without exercising the re-check under the lock at all. Here the first build
            cannot finish until the second thread has provably missed the cache.
            """
            first_is_building.set()
            assert second_has_missed.wait(timeout=5.0), "the second caller never reached the cache-miss path"
            manager = _manager("NPU")
            built.append(manager)
            return manager

        def arrive():
            """Enter the contended path, in one of two explicitly assigned roles."""
            with roles_lock:
                is_first = next(roles)
            if is_first:
                return preprocessor_pool._shared_preprocessor_for_type(pool, "NPU")
            # The second caller waits until the first is inside the builder, then confirms
            # the cache is still empty -- which is exactly the window the double-checked
            # lock exists to close -- before contending for the lock.
            assert first_is_building.wait(timeout=5.0), "the first caller never started building"
            assert pool.get("PREPROCESS::NPU") is None, "the pool was populated before the build finished"
            second_has_missed.set()
            return preprocessor_pool._shared_preprocessor_for_type(pool, "NPU")

        results: list = []
        with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", [_unit("NPU", "NPU.0")]):
            with mock.patch.object(preprocessor_pool.config, "DETECTED_UNITS", []):
                with mock.patch.object(preprocessor_pool.preprocessing, "create_manager", side_effect=slow_create):
                    threads = [threading.Thread(target=lambda: results.append(arrive()), daemon=True) for _ in range(2)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=5.0)
                        assert not thread.is_alive(), "a shared-preprocessor build deadlocked"

        assert len(built) == 1, f"the UVR model was loaded {len(built)} times for one key"
        assert results[0] is results[1] is pool["PREPROCESS::NPU"]


class TestColocatingWithTheTasksOwnUnit:
    """Whether UVR runs on the task's unit or on the one the operator configured.

    The count that decides this comes from the detected hardware, not from how many
    preprocessors happen to have been built. Those differ constantly: /status is polled long
    before any task binds one, and the first task of a run sees only its own -- so a
    pool-derived count made a two-accelerator host behave like a single-accelerator one and
    routed the NVIDIA unit's isolation to the Intel iGPU, on the dashboard and in the run.
    """

    def _colocate(self, units, unit_preprocessor, preprocess_device):
        with mock.patch.object(preprocessor_pool.config, "PREPROCESS_DEVICE", preprocess_device):
            with mock.patch.object(preprocessor_pool.config, "HARDWARE_UNITS", units):
                return preprocessor_pool._should_colocate_with_unit(unit_preprocessor)

    def test_a_unit_with_no_preprocessor_cannot_be_colocated_with(self):
        """A unit with no preprocessor cannot be colocated with."""
        assert self._colocate([_unit("GPU", "GPU.0")], None, "GPU") is False

    def test_a_cpu_unit_is_never_colocated_with(self):
        """Only accelerators are worth spreading work across."""
        assert self._colocate([_unit("CPU")], _manager("CPU"), "GPU") is False

    def test_the_units_own_type_matching_the_request_colocates(self):
        """The unit the operator asked for runs its own isolation, whatever else exists."""
        assert self._colocate([_unit("GPU", "GPU.0")], _manager("GPU"), "GPU") is True

    def test_a_single_accelerator_does_not_override_the_configured_device(self):
        """The hybrid NVIDIA+Intel regression: one CUDA unit passed this check and took UVR.

        With a single accelerator there is no parallelism to protect, so co-locating only
        overrides what the operator asked for -- while the startup banner went on naming the
        Intel iGPU.
        """
        assert self._colocate([_unit("CUDA", "cuda:0")], _manager("CUDA"), "GPU") is False

    def test_two_accelerator_units_colocate_so_they_run_in_parallel(self):
        """Each accelerator runs its own isolation, so two tasks proceed at once."""
        units = [_unit("CUDA", "cuda:0"), _unit("GPU", "GPU.0")]
        assert self._colocate(units, _manager("CUDA"), "GPU") is True

    def test_the_decision_does_not_depend_on_what_has_been_built_yet(self):
        """An empty pool must not make a two-accelerator host look single-accelerator.

        This is the defect the hardware-derived count fixes: at /status time nothing is
        bound, and the NVIDIA card reported the Intel GPU for its vocal isolation.
        """
        units = [_unit("CUDA", "cuda:0"), _unit("GPU", "GPU.0")]
        assert self._colocate(units, _manager("CUDA"), "GPU") is True, "no preprocessor is built yet"


class TestTheUnitKeyedLookupThatBacksTheDashboard:
    """`observed_preprocessor_for_unit` and the helpers only it reaches.

    These paths answer "where would this unit's isolation run", with no side effects, so the
    dashboard can report routing without building a manager. They went in with the hybrid-host
    fix and no coverage of their own, which is how a per-file gate that had not run to
    completion since could stay silent about them.
    """

    def _config(self, units, preprocess_device):
        return mock.patch.multiple(
            preprocessor_pool.config,
            HARDWARE_UNITS=units,
            DETECTED_UNITS=[],
            DETECTED_HARDWARE_UNITS=[],
            PREPROCESS_DEVICE=preprocess_device,
        )

    def test_a_unit_of_the_wanted_type_is_found_by_its_id(self):
        """The pool is keyed by unit id, so the type has to be resolved through the unit list."""
        pool = {"GPU.0": _manager("GPU")}
        with self._config([_unit("GPU", "GPU.0")], "GPU"):
            found = preprocessor_pool._unit_preprocessor_by_type(pool, "gpu")
        assert found is pool["GPU.0"], "lower-case request must still match the GPU unit"

    def test_a_matching_unit_with_nothing_built_yet_is_not_a_hit(self):
        """A unit can exist while its preprocessor does not; that is a miss, not a None manager."""
        with self._config([_unit("GPU", "GPU.0")], "GPU"):
            assert preprocessor_pool._unit_preprocessor_by_type({}, "GPU") is None

    def test_an_unaccelerated_preprocess_device_reports_the_units_own_preprocessor(self):
        """With UVR on the CPU there is no shared device to route to, so the unit answers."""
        pool = {"cuda:0": _manager("CUDA")}
        with self._config([_unit("CUDA", "cuda:0")], "CPU"):
            observed = preprocessor_pool.observed_preprocessor_for_unit(pool, "cuda:0")
        assert observed is pool["cuda:0"]

    def test_a_colocating_unit_reports_its_own_preprocessor(self):
        """The unit the operator named runs its own isolation, so that is what is reported."""
        pool = {"GPU.0": _manager("GPU")}
        with self._config([_unit("GPU", "GPU.0")], "GPU"):
            observed = preprocessor_pool.observed_preprocessor_for_unit(pool, "GPU.0")
        assert observed is pool["GPU.0"]

    def test_a_routing_unit_reports_the_shared_device_not_its_own(self):
        """The recorded defect: the CUDA card reported UVR on CUDA while it ran on the iGPU.

        A single accelerator, so there is no parallelism to protect and the CUDA unit does not
        colocate: its isolation goes to the Intel manager, and that is what must be reported.
        """
        pool = {"cuda:0": _manager("CUDA"), "GPU.0": _manager("GPU")}
        with self._config([_unit("CUDA", "cuda:0")], "GPU"):
            observed = preprocessor_pool.observed_preprocessor_for_unit(pool, "cuda:0")
        assert observed is pool["GPU.0"], "the CUDA unit routes to the Intel GPU manager"

    def test_a_routing_unit_falls_back_to_the_shared_key_when_no_unit_manager_exists(self):
        """Nothing is built per-unit yet, so only the shared PREPROCESS:: entry can answer."""
        pool = {"PREPROCESS::GPU": _manager("GPU")}
        with self._config([_unit("CUDA", "cuda:0")], "GPU"):
            observed = preprocessor_pool.observed_preprocessor_for_unit(pool, "cuda:0")
        assert observed is pool["PREPROCESS::GPU"]

    def test_an_unknown_unit_id_does_not_colocate(self):
        """A unit id absent from the hardware list must not raise on the way to False."""
        with self._config([_unit("GPU", "GPU.0")], "GPU"):
            assert preprocessor_pool._unit_would_colocate("nonexistent") is False

    def test_a_cpu_unit_never_colocates(self):
        """Only accelerators run their own isolation."""
        with self._config([_unit("CPU", "cpu")], "GPU"):
            assert preprocessor_pool._type_would_colocate("CPU") is False

    def test_the_preferred_lookup_falls_through_to_the_unit_list(self):
        """A manager whose own device_type does not match is still the right unit's manager.

        The by-type scan over the pool reads each manager's device_type, so a manager that
        reports something else is a miss there and can only be found through the unit list.
        """
        pool = {"GPU.0": _manager("CPU")}
        with self._config([_unit("GPU", "GPU.0")], "GPU"):
            assert preprocessor_pool.preferred_preprocessor(pool) is pool["GPU.0"]
