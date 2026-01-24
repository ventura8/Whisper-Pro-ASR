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

5. Detect language:
```bash
curl -X POST -F "audio_file=@test.mp3" http://localhost:9000/detect-language
```
