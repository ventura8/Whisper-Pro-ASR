"""Tests for the priority model access queue mechanism (Scheduler)."""
import threading
import time
from unittest import mock
import pytest
import modules.model_manager as mm

# pylint: disable=protected-access


@pytest.fixture(autouse=True)
def reset_mm_state():
    """Reset global state and threading primitives before each test."""
    mm._TRANSCRIBING_SESSIONS = 0
    mm._PRIORITY_REQUESTS = 0
    mm._PAUSE_REQUESTED.clear()
    mm._PAUSE_CONFIRMED.clear()
    mm._RESUME_EVENT.set()
    # Fresh locks to avoid any state pollution
    mm._MODEL_LOCK = threading.Lock()
    mm._TRANSCRIBING_LOCK = threading.Lock()
    mm._PRIORITY_LOCK = threading.Lock()
    mm._PRIORITY_SEQUENTIAL_LOCK = threading.RLock()
    mm.WHISPER = None
    yield


def test_request_priority_sets_flags():
    """Test that request_priority sets the correct flags."""
    mm.request_priority()

    assert mm._PRIORITY_REQUESTS == 1
    assert mm._PAUSE_REQUESTED.is_set()
    assert not mm._RESUME_EVENT.is_set()

    # Cleanup
    mm.release_priority()


def test_release_priority_clears_flags():
    """Test that release_priority clears flags when counter reaches 0."""
    mm._PRIORITY_REQUESTS = 1
    mm._PAUSE_REQUESTED.set()
    mm._RESUME_EVENT.clear()

    mm.release_priority()

    assert mm._PRIORITY_REQUESTS == 0
    assert not mm._PAUSE_REQUESTED.is_set()
    assert mm._RESUME_EVENT.is_set()


def test_multiple_priority_requests_tracked():
    """Test that multiple priority requests are tracked correctly."""
    mm.request_priority()
    mm.request_priority()

    assert mm._PRIORITY_REQUESTS == 2

    mm.release_priority()
    assert mm._PRIORITY_REQUESTS == 1
    assert mm._PAUSE_REQUESTED.is_set()

    mm.release_priority()
    assert mm._PRIORITY_REQUESTS == 0
    assert not mm._PAUSE_REQUESTED.is_set()


def test_release_priority_doesnt_go_negative():
    """Test that release_priority doesn't make counter negative."""
    mm._PRIORITY_REQUESTS = 0
    mm.release_priority()

    assert mm._PRIORITY_REQUESTS == 0


def test_model_lock_ctx_acquires_lock():
    """Test that model_lock_ctx context manager works."""
    assert not mm._MODEL_LOCK.locked()

    with mm.model_lock_ctx():
        assert mm._MODEL_LOCK.locked()

    assert not mm._MODEL_LOCK.locked()


def test_transcription_increments_session_count():
    """Test that run_transcription properly manages session count."""
    mock_whisper = mock.MagicMock()
    mock_info = mock.MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 10.0
    mock_info.all_language_probs = None
    mock_whisper.transcribe.return_value = (iter([]), mock_info)

    mm.WHISPER = mock_whisper

    with mock.patch('modules.config.ENABLE_VOCAL_SEPARATION', False):
        with mock.patch('modules.model_manager._init_transcription_stats'):
            with mock.patch(
                'modules.model_manager._post_process_vad',
                side_effect=lambda r, p: r
            ):
                # Run transcription
                mm.run_transcription("/fake.wav", language="en")

    # After completion, session count should be 0
    assert mm._TRANSCRIBING_SESSIONS == 0


def test_wait_for_priority_blocks():
    """Test that _wait_for_priority blocks when pause is requested."""
    # Set up a scenario where _PAUSE_REQUESTED is set
    mm._PAUSE_REQUESTED.set()
    mm._RESUME_EVENT.clear()

    blocked = []

    def blocking_task():
        mm.wait_for_priority()
        blocked.append(True)

    t = threading.Thread(target=blocking_task)
    t.start()

    # Give the thread time to block
    time.sleep(0.2)
    assert len(blocked) == 0  # Should still be blocked

    # Release
    mm._RESUME_EVENT.set()
    t.join(timeout=2.0)

    assert len(blocked) == 1  # Should have unblocked


def test_wait_for_priority_yields_lock():
    """Test that wait_for_priority releases and re-acquires the model lock."""
    # 1. Acquire the lock
    # pylint: disable=consider-using-with
    mm._MODEL_LOCK.acquire()
    assert mm._MODEL_LOCK.locked()

    # 2. Request priority
    mm.request_priority()

    # 3. Call wait_for_priority with the lock in a separate thread
    # so we can check the lock status from this thread
    def yielding_task():
        mm.wait_for_priority(model_lock=mm._MODEL_LOCK)

    t = threading.Thread(target=yielding_task)
    t.start()

    # Wait for the task to report confirmation (indicates it reached the wait point)
    start_time = time.time()
    while not mm._PAUSE_CONFIRMED.is_set() and time.time() - start_time < 2.0:
        time.sleep(0.01)

    assert mm._PAUSE_CONFIRMED.is_set()
    # 4. Check that the lock was released!
    assert not mm._MODEL_LOCK.locked()

    # 5. Resume
    mm.release_priority()
    t.join(timeout=2.0)

    # 6. Check that the lock was re-acquired (and subsequently released by our join/task?
    # Actually wait_for_priority re-acquires it. Since we acquired it manually at Step 1,
    # we should check if it's locked now.)
    assert mm._MODEL_LOCK.locked()

    # Cleanup: manually release since we acquired manually
    mm._MODEL_LOCK.release()


def test_priority_tasks_are_sequential():
    """Test that multiple priority tasks run one-by-one."""
    # 1. Start continuous priority task 1
    mm.request_priority()
    # 2. Try to start priority task 2 in a thread
    started_2 = []

    def task_2():
        mm.request_priority()
        started_2.append(True)
        mm.release_priority()

    t2 = threading.Thread(target=task_2)
    t2.start()

    # 3. Task 2 should be blocked on the sequential lock
    time.sleep(0.2)
    assert len(started_2) == 0

    # 4. Release Task 1
    mm.release_priority()

    # 5. Task 2 should now proceed
    t2.join(timeout=2.0)
    assert len(started_2) == 1
