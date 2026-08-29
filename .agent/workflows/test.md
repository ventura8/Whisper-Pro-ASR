---
description: Test transcription endpoints
---

# Test Endpoints

// turbo-all

1. Health check:

```bash
curl http://localhost:9000/
```

1. Status:

```bash
curl http://localhost:9000/status
```

1. Transcribe local file:

```bash
curl -X POST "http://localhost:9000/asr?local_path=/movies/test.mp4&language=en"
```

1. Transcribe uploaded file:

```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/asr
```

1. Verify uploads remain available after a restart:

```bash
# Fail fast. Without this, a failed upload or a failed assertion below is just a non-zero
# exit nothing reads, and the restart and history request run anyway -- so the block ends
# with a successful-looking curl that says nothing about the transcription that failed.
set -euo pipefail

# Every curl below is bounded. Without --connect-timeout/--max-time a single hung socket
# blocks indefinitely, and the 120-iteration bound on the poll loop stops meaning anything:
# the loop cannot time out if one iteration never returns.
CURL_LIMITS=(--connect-timeout 5 --max-time 30)

# Defaulted before the loop: under `set -u` the case below reads $status even when the loop
# exits without ever reaching the 2xx branch (every poll transient, or the 120 iterations
# exhausted), and an unbound variable there aborts with a shell error rather than the
# "never reached a terminal state" diagnosis the block is written to give.
status="unknown"

task_id=$(curl -sS "${CURL_LIMITS[@]}" -X POST -F "audio_file=@test.mp3" http://localhost:9000/asr | jq -r '.task_id // .id // .history_id')
test -n "$task_id" && test "$task_id" != "null"

# Wait for the task to reach a terminal state before restarting. Restarting mid-flight
# tests whether an in-progress task survives a kill -- which it does not, by design -- so
# the history lookup below failed for a reason that has nothing to do with persistence.
for _ in $(seq 1 120); do
  # `|| echo unknown` and the -f/--fail: under `set -euo pipefail` above, a single dropped
  # connection or a 502 while the service restarts a worker aborted the whole block on a
  # non-zero curl -- so a transient blip looked identical to a task that never finished.
  # A failed poll now falls through to the "unknown" branch and simply tries again.
  # Three outcomes, not two. A dropped connection or a 5xx while the service restarts a
  # worker is transient and worth retrying; a 404 (wrong task id) or a body that is not
  # JSON at all never becomes valid by waiting, and collapsing both into "unknown" spent
  # ten minutes re-asking a question that was already answered.
  body=$(curl -sS "${CURL_LIMITS[@]}" -o /dev/null -w '%{http_code}' "http://localhost:9000/history/$task_id" || echo 000)
  case "$body" in
    000|5??) sleep 5; continue ;;                       # transient: retry
    2??) ;;                                             # fall through and read it
    *) echo "history lookup failed permanently (HTTP $body)" >&2; exit 1 ;;
  esac
  if ! status=$(curl -fsS "${CURL_LIMITS[@]}" "http://localhost:9000/history/$task_id" | jq -e -r '.status'); then
    echo "history response was not valid JSON with a .status field" >&2; exit 1
  fi
  case "$status" in completed|failed) break ;; esac
  sleep 5
done
# Both are terminal, and both are worth checking persistence for -- a failed task's history
# entry must survive a restart exactly as a completed one's does. Asserting completed here
# threw the run away before the restart, so the one thing this block exists to test never
# ran. What is NOT acceptable is neither: that means the loop timed out mid-flight, and a
# restart would then be testing whether an in-progress task survives a kill (it does not,
# by design), which is the misdiagnosis the comment above warns about.
case "$status" in
  completed|failed) echo "task reached terminal state: $status" ;;
  *) echo "task never reached a terminal state (last status: $status)" >&2; exit 1 ;;
esac

docker compose restart whisper-pro-asr

# And wait for the service to come back: the restart returns as soon as the container is
# up, well before the app is serving, so the history request raced it and got a connection
# refused that looked like lost history.
for _ in $(seq 1 60); do
  curl -sf "${CURL_LIMITS[@]}" http://localhost:9000/status >/dev/null && break
  sleep 2
done

# The point of the whole block: the entry must still be there, and still hold the same
# terminal status it had before the restart. `curl -f` alone asserted only that the endpoint
# answered 200 -- an empty object, or an entry rebuilt in a different state, passed it.
after=$(curl -fsS "${CURL_LIMITS[@]}" "http://localhost:9000/history/$task_id" | jq -e -r '.status')
test "$after" = "$status" || { echo "history did not survive the restart intact: $status -> $after" >&2; exit 1; }
echo "history survived the restart with status $after"
```

1. Transcribe with custom parameters:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?initial_prompt=Technical%20discussion&vad_filter=true&word_timestamps=true&max_line_width=42&max_line_count=2"
```

1. Transcribe with speaker diarization:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?diarize=true&min_speakers=2&max_speakers=5"
```

1. Detect language:

```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/detect-language
```

1. Get VTT output:

```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?output=vtt"
```
