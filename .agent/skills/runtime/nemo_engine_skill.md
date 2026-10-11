# NeMo Engines (PARAKEET / CANARY) Skill

Use when touching `ASR_ENGINE=PARAKEET|CANARY`, `NEMO_*` settings, or any module named
`nemo_*` / `config_nemo` / `nemo_catalog`.

## Shape

- One `NemoOnnxEngine` per hardware unit (`modules/inference/engines/nemo_onnx_engine.py`),
  so the scheduler, model pool and pipeline need nothing NeMo-specific.
- Inside it: the NeMo model on onnx-asr, plus a **Whisper companion**
  (`FasterWhisperEngine` on `ASR_MODEL`). The companion answers every `detect_language`
  call and transcribes whatever `nemo_routing.choose()` sends to Whisper. `ASR_MODEL` keeps
  naming Whisper weights; for these engines they are the companion's.
- `nemo_routing.py` decides per call: no language, a language outside `nemo_catalog.EU25`,
  or translation on Parakeet go to Whisper; Canary translates X->en natively. Never route
  an unknown language to NeMo: the models silently transcribe it into the nearest language
  they know.

## Invariants

- **Supply chain**: `NEMO_MODEL` may only name an entry in `modules/core/nemo_catalog.py`,
  each pinned to one Hugging Face revision and downloaded by `model_provisioning`, never by
  onnx-asr itself.
- **Providers are explicit** (`nemo_runtime.providers_for`): onnx-asr's default is every
  provider the imported ONNX Runtime build offers, which on the OpenVINO build silently
  moves a CPU unit onto OpenVINO. `actual_device()` reports what the encoder really runs on.
- **Chunking** (`nemo_chunking.py`): spans come from `clip_timestamps` first (speech
  regions and resumable_decode's resume point), then Silero VAD, then fixed windows; any
  span over `NEMO_MAX_CHUNK_SEC` is cut at its quietest point. Honouring `clip_timestamps`
  is what keeps pause/resume and region decoding working -- do not bypass it.
- **Lazy segments**: transcribe yields a batch (`NEMO_BATCH_SIZE` chunks) at a time, so
  closing the stream stops at the next batch boundary (cooperative preemption).
- **Timing** (`nemo_segments.py`): Parakeet words are rebuilt from per-token start times;
  Canary has no times, so lines (one per sentence, wrapped at 84 characters) are spread
  across their chunk by text length (documented approximation, bounded by the chunk length).
- **Detection load**: on a NeMo engine, language detection is the CTranslate2 companion's
  work; `metrics_discovery` counts it as accelerator load only on a CUDA unit whose
  companion is not forced to the CPU.
- **Bad `NEMO_MODEL`** is refused (default loaded) and logged once per value, not per
  `/status` poll.
- **Canary obeys the language it is given**: it decodes with that language as source *and*
  target, so a region mislabelled by detection comes back translated, not transcribed.
  Parakeet takes no language prompt. Weigh this before routing more work to Canary.
- **GPU memory**: CUDA out-of-memory on load falls back to the CPU (`_is_out_of_memory`).
  On small cards pair `NEMO_GPU_MEM_LIMIT_MB` with `NEMO_COMPANION_DEVICE=CPU`.

## Validation

- Mocked tests: `tests/inference/engines/test_nemo_*.py`, `tests/unit/test_nemo_engine_config.py`.
- Real hardware is mandatory (see `remote_hardware_validation_skill.md`): run the accuracy
  test and real-audio smoke set with `ASR_ENGINE=PARAKEET` and `CANARY` on an NVIDIA host
  and on the CPU, and confirm the banner's `ASR Runtime` row names the provider actually used.
- Engine comparisons (WER/speed per language): `tests/real_audio/test_engine_benchmark.py`,
  results recorded in `docs/ASR_ENGINE_EVALUATION.md`.
