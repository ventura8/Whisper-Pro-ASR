# Release v1.0.3 — Dynamic Language Identification

This update introduces a major overhaul of the language detection system, replacing the multi-zone voting mechanism with a more robust and efficient dynamic chunking approach designed specifically for long-form media like movies.

## 🚀 Key Improvements

### 🎯 Dynamic Language Detection (v2)
- **Automated Chunk Scaling**: The identification engine now calculates an optimal sample size based on the total media duration.
- **Minimum 5-Minute Sample**: Ensures that even short clips have enough context for high-confidence identification.
- **Movie Optimization**: For a standard 4-hour movie, the system now extracts a 12-minute representative sample (5% of duration), significantly improving accuracy over 30-second snippets.
- **Single-Pass Efficiency**: Replaced the iterative 15-scan voting system with a single, high-fidelity inference pass. This reduces FFmpeg overhead and simplifies the processing pipeline.

### ⚡ Performance & Stability
- **Reduced Latency**: By eliminating multiple parallel extraction tasks and redundant inference cycles, `/detect-language` responses are now significantly faster.
- **Lower Memory Pressure**: The single-pass approach ensures more predictable VRAM/RAM utilization during the identification phase.
- **FFmpeg 8.1.0 Refinement**: Optimized the extraction command to use direct seek and duration flags for near-instantaneous chunk creation from any media container.

## 🛠️ Internal Changes
- **Version**: Bumped to `1.0.3`.
- **Logic**: Refactored `modules/language_detection.py` to remove the multi-zone voting engine.
- **Tests**: Updated unit tests to verify dynamic scaling calculations across various file lengths.

## 📦 Deployment
Update your `docker-compose.yml` to use the `1.0.3` tag (or `latest`) and restart the service:
```bash
docker compose pull
docker compose up -d
```

---
*Optimized for high-precision identification in complex media libraries.*
