"""Bootstrap configuration for test suite monkeypatches."""

import os
import threading

# Default subtitle promo to false for tests to prevent test pollution
os.environ.setdefault("SUBTITLE_PROMO_ENABLED", "false")

# Global monkeypatch to ensure all threads spawned during tests are daemon threads
# to prevent pytest from hanging at exit if background threads are left alive.
_original_thread_init = threading.Thread.__init__


def _patched_thread_init(self, *args, **kwargs):
    kwargs["daemon"] = True
    _original_thread_init(self, *args, **kwargs)


threading.Thread.__init__ = _patched_thread_init
