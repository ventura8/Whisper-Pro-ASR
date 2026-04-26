# Whisper Pro ASR v1.0.4 Release Notes

## 🚀 Production Stability & Observability Finalization

This release focuses on industrial-grade stability, long-term data persistence, and comprehensive system observability. 

### 📉 Advanced Memory Hygiene
*   **Nuclear RAM Purge**: Implemented proactive memory reclamation using `malloc_trim(0)` and `ctranslate2.clear_caches()` after every task.
*   **Sub-500MB Idle Footprint**: Verified memory returns to baseline immediately after inference, even in high-throughput environments.

### 💾 Persistent Storage Tiering
*   **Dual-Tier History**: 
    *   **RAM Cache**: Strictly capped at the last 20 tasks for peak dashboard performance.
    *   **SSD Persistence**: Atomic merge-on-save preserves up to 1000 tasks in `/app/data/task_history.json`.
*   **Host-Volume Persistence**: Updated `docker-compose.yml` to support host-mounted volumes for history and logs, ensuring data survives container updates.
*   **Audit Logging**: System logs (`whisper_pro.log`) now persist in the state volume.

### 🖥️ Enhanced Observability
*   **Live Lifecycle Logs**: The dashboard now streams real-time logs for *all* processing stages, including **Initializing** (UVR/Montage) and **Active** (ASR).
*   **Translation Progress**: Added dynamic UI feedback for translation tasks, with live segment updates and accurate status labeling.
*   **Stylized Identity**: Custom Whisper soundwave favicon for professional browser tab identification.

### 🛡️ System Hardening
*   **Safe Startup Cleanup**: Refined temporary asset cleanup to target only specific processing directories, protecting operational files and test artifacts.
*   **CI Pipeline Stabilization**: Resolved all test suite regressions and achieved a 100% pass rate with 92% code coverage.

---
**Deployment Note**: Please ensure you update your `docker-compose.yml` to include the `./state:/app/data` volume mount to take advantage of the new persistence features.
