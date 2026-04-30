# Release v1.0.5 - Industrial Resource Hygiene & Storage Stability

This update hardens the Whisper Pro ASR engine against long-term resource leaks and optimizes the storage lifecycle for industrial-grade reliability.

## 🚀 Key Enhancements

### 🛡️ Centralized Storage Hygiene
Implemented a multi-layered defense against storage accumulation during high-volume processing.
- **Request-Local File Tracking**: Introduced a thread-local `tracked_files` registry that automatically captures every temporary asset created during a request (raw uploads, 16kHz standardized WAVs, isolated stems, and HQ prep files).
- **Guaranteed Atomic Cleanup**: Every API route now features a robust `cleanup_files()` call in the `finally` block, ensuring a **100% reclamation rate** of transient storage space even after fatal task errors.

### 📜 Hardened Diagnostic Logging
Engineered a persistent logging architecture that ensures operational records are never lost.
- **Persistence Across Restarts**: Fixed a critical regression to ensure `whisper_pro.log` survives app restarts and container updates via a hardened `TimedRotatingFileHandler` initialization.
- **Real-Time Log Synchronization**: Implemented a mandatory flush-to-disk sequence for the `/logs/download` endpoint, guaranteeing that downloaded logs always contain the absolute latest system entries.
- **Aggressive Cache Busting**: Configured the log download stream with zero-caching headers to prevent stale diagnostic data from being served by browsers or proxies.

### ⚡ Performance & Stability
- **Refined API Lifecycle**: Aligned all inference routes with the new centralized cleanup architecture to prevent memory and storage fragmentation.
- **Hardware Stability**: Validated re-entrant locking and memory reclamation logic to maintain a ~1GB idle footprint under heavy load.

---
*For deployment instructions, refer to the [README.md](README.md) or [ARCHITECTURE.md](docs/ARCHITECTURE.md).*
