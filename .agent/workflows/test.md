---
description: Test transcription endpoints
---

# Test Endpoints

// turbo-all

1. Health check:
```bash
curl http://localhost:9000/
```

2. Status:
```bash
curl http://localhost:9000/status
```

3. Transcribe local file:
```bash
curl -X POST "http://localhost:9000/asr?local_path=/movies/test.mp4&language=en"
```

4. Transcribe uploaded file:
```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/asr
```

5. Transcribe with custom parameters:
```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?initial_prompt=Technical%20discussion&vad_filter=true&word_timestamps=true&max_line_width=42&max_line_count=2"
```

6. Transcribe with speaker diarization:
```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?diarize=true&min_speakers=2&max_speakers=5"
```

7. Detect language:
```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/detect-language
```

8. Get VTT output:
```bash
curl -X POST -F "audio_file=@test.mp3" "http://localhost:9000/asr?output=vtt"
```
