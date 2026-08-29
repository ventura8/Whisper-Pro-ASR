"""When per-unit engines are used, and what it takes to turn them on.

Hybrid mode lets a host with both a CUDA/AMD GPU and an Intel GPU/NPU run each unit's
native engine in its own worker, so both accelerators stay busy. It is off unless asked
for: the engine a request runs on would otherwise depend on which unit the scheduler
happened to pick, and AUTO deliberately resolves to one engine on every host.

These pin the resolution rules rather than the config module's import-time value, which
depends on the machine the tests run on.
"""

import pytest

from modules.core.config_resolution import resolve_hybrid_engines


def _resolve(*, isolate=True, intel=True, cuda=True, env=""):
    """Call the shipped decision directly.

    This used to be a hand-written mirror of the logic in config.py, which meant the tests
    could keep passing while the real rule drifted away from them.
    """
    return resolve_hybrid_engines(
        isolate_engines=isolate,
        has_intel_unit=intel,
        has_cuda_or_amd_unit=cuda,
        env_value=env,
    )


class TestOffUnlessRequested:
    """The default is off, even on a host that could run it."""

    def test_off_when_nothing_is_set(self):
        """The default is off, on a host that could otherwise run it."""
        assert not _resolve(env="")

    @pytest.mark.parametrize("env", ["maybe", "yes please", "1.0", "enabled"])
    def test_an_unrecognised_value_does_not_enable_it(self, env):
        """Only an explicit yes counts; noise must not silently change the engine per unit."""
        assert not _resolve(env=env)

    @pytest.mark.parametrize("env", ["false", "0", "no", "off", "FALSE", "Off", " Off "])
    def test_explicit_disable_stays_off(self, env):
        """An explicit no is honoured, in any spelling or case."""
        assert not _resolve(env=env)


class TestExplicitOptIn:  # pylint: disable=too-few-public-methods
    """The only way to turn hybrid mode on."""

    @pytest.mark.parametrize("env", ["true", "1", "yes", "on", "TRUE", "On", " on "])
    def test_enabled_by_an_explicit_yes(self, env):
        """Passed verbatim: config normalises case and whitespace, so the test must too."""
        assert _resolve(env=env)


class TestHardRequirements:
    """No override can satisfy these: the contexts genuinely cannot share a process."""

    @pytest.mark.parametrize("env", ["", "true", "1"])
    def test_off_without_isolation(self, env):
        """Without process isolation the two runtimes would share an interpreter, which they cannot."""
        assert not _resolve(isolate=False, env=env)

    @pytest.mark.parametrize("env", ["", "true"])
    def test_off_without_an_intel_unit(self, env):
        """Hybrid mode needs one unit of each vendor family; asking cannot conjure one."""
        assert not _resolve(intel=False, env=env)

    @pytest.mark.parametrize("env", ["", "true"])
    def test_off_without_a_cuda_or_amd_unit(self, env):
        """Hybrid mode needs one unit of each vendor family; asking cannot conjure one."""
        assert not _resolve(cuda=False, env=env)
