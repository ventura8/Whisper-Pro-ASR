# ASR Engine Evaluation: NVIDIA Parakeet and Canary vs Whisper

Should NVIDIA's open ASR models replace Whisper large-v3 as this service's default? This
document records the answer, the published evidence, what was measured here, and what a
future default switch would have to show.

## Decision

**No, not as a wholesale replacement. `AUTO` stays `FASTER-WHISPER`.** `PARAKEET` and
`CANARY` ship as opt-in engines, each with a Whisper companion that keeps every
Whisper-only capability intact. Four facts decide it:

1. **Coverage.** Parakeet-TDT-0.6B-v3 and Canary-1B-v2 cover 25 European languages. Whisper
   covers 99. Bazarr sends whatever the library holds: Japanese, Korean, Chinese, Hindi,
   Arabic, Turkish. These models do not fail on such audio. They transcribe it into the
   nearest language they know, which no caller can tell from a real transcript.
2. **No language identification.** `/detect-language`, gap filling and per-region run
   detection all need a language-ID head. Parakeet decides internally and does not report
   the result. Canary must be *told* the language, and without one it translates to English
   (measured: German audio sent with no language came back as an English translation). A
   Whisper model has to stay loaded either way.
3. **Translation.** Whisper translates 99 languages to English. Canary translates its 25.
   Parakeet translates none.
4. **Policy.** `AUTO_DEFAULT_ENGINE` exists so a deployment decodes identically across the
   fleet. The real-audio matrix, the long-form defect history and the per-engine manifest
   expectations are all calibrated against Faster-Whisper.

Where they win is speed, English accuracy, and a transducer decoder (Parakeet) that cannot
fall into Whisper's repetition loops. A future default would therefore be "Parakeet for its
25 languages, Whisper for the rest", which is exactly what `ASR_ENGINE=PARAKEET` already
does. Switching `AUTO` to it requires all of the following, measured with `--suite bench`
on every host class:

- On covered languages, mean WER at or below Faster-Whisper's, on tier A and tier B.
- On languages routed to Whisper, results equal to the Faster-Whisper baseline. This also
  proves the routing.
- On the long-form fixture, no more missed windows and no more text in silent passages.
- `/detect-language` latency no worse on any host, including where the companion runs on
  the CPU.
- No silent CPU fallback in `/status` on any accelerator host.

### What the measurements so far say (RTX 3080 8 GB, Intel NUC Arc)

- **Speed where they apply.** On covered languages with the language given, Parakeet is
  2.5x faster than Faster-Whisper on CUDA (RTF 0.063 vs 0.159) and 1.5x faster than
  INTEL-WHISPER on Arc. Canary sits between the two.
- **Accuracy on the synthetic matrix.** Whisper is more accurate. Covered-language WER is
  0.077 (Faster-Whisper) and 0.076 (INTEL-WHISPER), against 0.104 (Parakeet) and
  0.089–0.094 (Canary). On *real* Romanian speech an independent run found the opposite
  for Canary (6.5–6.9% vs 8.9–10.1%), so the synthetic numbers cannot decide this alone.
- **Everything that needs Whisper got slower** on both hosts: `/detect-language`,
  auto-detected requests, other languages and Parakeet translations. The companion had to
  sit on the CPU (8 GB card; no CTranslate2 backend on Intel), so detection took 2–3x as
  long, and long-form with per-region detection ran at about 0.5x real time.
- **Routing works.** Results for the languages sent to Whisper match the baseline (WER
  0.34 on both). Every NeMo row ran on its accelerator per `/status`; nothing fell back
  silently once the 8 GB settings were applied.

**Follow-ups worth doing before any default discussion:**

- An fp16 ONNX export of Canary. Under transformers bf16 it needs ~2 GB, against 6.6 GB for
  the onnx-asr fp32 export, and that alone decides whether the companion fits beside it on
  8 GB cards.
- A real-speech benchmark set per language (FLEURS or VoxPopuli) next to the TTS matrix.

## How the engines work

The design is detailed in `docs/ARCHITECTURE.md` under "NVIDIA NeMo Engines". In brief:

- **Runtime.** The NeMo models run through
  [onnx-asr](https://github.com/istupakov/onnx-asr) 0.12.0 on the ONNX Runtime build each
  image already ships under `/app/libs/<vendor>`: 1.29 CUDA, 1.24 OpenVINO, 1.27 CPU.
  There is no NeMo/PyTorch stack and no new image target.
- **Weights.** The istupakov ONNX exports, pinned by revision in
  `modules/core/nemo_catalog.py`.
  - `nemo-parakeet-tdt-0.6b-v3` @ `8f23f0c0`
  - `nemo-parakeet-tdt-0.6b-v2` @ `0bbb45a3` (English only)
  - `nemo-canary-1b-v2` @ `5ebc1520`
- **Routing.** This happens per decode call (`nemo_routing`). A call with no language, a
  language outside the 25, or a translation Parakeet cannot do goes to the Whisper
  companion. Everything else goes to the NeMo model.

## Published evidence (not measured here)

| | Whisper large-v3 | Parakeet-TDT-0.6B-v3 | Canary-1B-v2 |
| :--- | :--- | :--- | :--- |
| Parameters | 1.55B | 0.6B | 1B |
| Languages | 99 | 25 European | 25 European |
| Language ID output | yes | no | no (language must be given) |
| Translation | X→en (99) | none | X↔en (25) |
| Open ASR Leaderboard, English avg WER | ~7.4 | 6.32 (RTFx 3330) | 7.15 (RTFx 749) |
| Open ASR Leaderboard, de/fr/it/es/pt avg WER | — | 4.81 (RTFx 1720) | 4.60 (RTFx 634) |
| FLEURS 25-language avg WER (model cards) | — | 11.97 | 8.40 |
| Long audio | 30 s windows | up to 24 min full attention (NeMo) | chunked |
| Timestamps via onnx-asr | — | token-level | none |
| Decoder | autoregressive | transducer (TDT) | autoregressive |

**Sources:**

- NVIDIA model cards: [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  and [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2).
- *Canary-1B-v2 & Parakeet-TDT-0.6B-v3*, [arXiv 2509.14128](https://arxiv.org/abs/2509.14128).
- *Open ASR Leaderboard*, [arXiv 2510.06961](https://arxiv.org/abs/2510.06961).
- [NeMo "Choosing a Model"](https://docs.nvidia.com/nemo/speech/latest/starthere/choosing_a_model.html).
- [onnx-asr documentation](https://istupakov.github.io/onnx-asr/).

Treat leaderboard numbers as clean read speech. They say little about film audio, music
beds or code-switching.

**Out of scope, noted for later:**

- Canary-Qwen-2.5B: English only, and needs the NeMo SALM stack.
- Nemotron 3.5 ASR (June 2026): 40 locales, but streaming-first, GPU-only and NeMo-only,
  with no documented timestamps.
- Parakeet-unified-en (April 2026): English only.

## Measured here

### Model spike: onnx-asr directly, RTX 3080 laptop (8 GB) + Intel UHD, 2026-10-09

All runs used 20 s German chunks, three timed passes after a warm-up. RTFx means seconds of
audio per second of compute.

| Model | Provider | Quantization | Batch | RTFx | Peak VRAM |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Parakeet v3 | CUDA | fp32 | 1 | 82 | 3.2 GB |
| Parakeet v3 | CUDA | fp32 | 4 | 123 | 4.0 GB |
| Parakeet v3 | CUDA | int8 | 4 | 7.6 | 1.8 GB |
| Parakeet v3 | OpenVINO (UHD iGPU) | fp32 | 4 | 9.1 | — |
| Parakeet v3 | OpenVINO (UHD iGPU) | int8 | 4 | fails to execute | — |
| Parakeet v3 | CPU, 4 threads (i9-11900H) | fp32 / int8 | 4 | 4.3 / 4.0 | — |
| Canary-1B-v2 | CUDA | fp32 | 1 / 2 / 4 | 12.7 / 17.5 / 23.5 | 6.6 / 7.8 / 7.8 GB |
| Canary-1B-v2 | CUDA | int8 | 1 / 4 | 1.5 / 1.2 | 2.0 / 6.2 GB |
| Canary-1B-v2 | CPU, 4 threads | int8 | 4 | 2.3 | — |

**Behaviour observed:**

- Both models transcribed the English accuracy fixture word-perfectly.
- Parakeet rendered "faulen Hund" as "Faulenhund" when given a single chunk.
- Canary with `target_language="en"` translated the German clip correctly. With no
  language it translated anyway.
- Canary produced a full English sentence from 50 ms of silence. The engine drops chunks
  under 0.1 s, and its clips come from VAD.
- Tokens carry a leading space at word starts, and timestamps are token starts in 0.08 s
  steps. That is what `nemo_segments` rebuilds words from.

These results set the defaults:

- `NEMO_QUANTIZATION=AUTO` is fp32 on CUDA and Intel GPU and int8 on the CPU.
- `NEMO_BATCH_SIZE=4`.
- The CUDA arena uses `kSameAsRequested`, which measured 4.3 GB peak against 5.0 GB for
  power-of-two growth.

### Service runs

| Host | Engine / settings | Suite | Result | Device evidence |
| :--- | :--- | :--- | :--- | :--- |
| RTX 3080 laptop (8 GB), overlay image | PARAKEET, companion large-v3 fp16 on GPU | accuracy | 1/9: every request OOM | NeMo worker 6.4 GB + LD vocal isolation 1.3 GB filled the card; CTranslate2 then reported `invalid device ordinal` |
| same | PARAKEET, companion `int8_float16` on GPU | accuracy | 9/9 **but `ASR=CPU measured`** | NeMo model fell back to the CPU under memory pressure; a foreign process from another session also held VRAM |
| same | PARAKEET, `NEMO_COMPANION_DEVICE=CPU` | accuracy | 9/9; NeMo on CUDA until a mid-decode arena OOM moved it to the CPU | led to the chunk-by-chunk OOM retry and `NEMO_GPU_MEM_LIMIT_MB` |

### RTX 3080 laptop (8 GB), full `nvidia-intel` build, 2026-10-10

These rows ran with the 8 GB settings: `NEMO_COMPANION_DEVICE=CPU`,
`NEMO_GPU_MEM_LIMIT_MB=3072` and `MAX_GPU_UNITS=0`. Before the chunk-by-chunk OOM retry
and the memory cap, the same card failed every request or silently fell back to the CPU
(see the overlay rows above).

| Row | Result | Device evidence |
| :--- | :--- | :--- |
| PARAKEET accuracy | 9/9 | `ASR=CUDA measured=True`; peak 5.5 GB VRAM; GPU utilisation 100% during decode; `route=nemo` logged |
| PARAKEET code-switch (6 clips, 30 checks) | 30/30 | CUDA, measured |
| PARAKEET bench | see table | CUDA, measured |
| PARAKEET long-form (6 variants, 20 min each) | 30 passed, 14 failed: every throughput budget, and the episode variant, which hit the 3-hour run limit | CUDA, measured |

PARAKEET bench results:

| Mode (bucket) | Mean WER | Median RTF | Median wall |
| :--- | ---: | ---: | ---: |
| Transcribe, language given: Parakeet's 25 (82 clips) | 0.104 | **0.063** | 0.24 s |
| Transcribe, language given: other languages (33 clips; Whisper on CPU) | 0.348 | 2.16 | 13.5 s |
| Transcribe, language auto-detected: Parakeet's 25 (10 clips) | 0.085 | 2.08 | 23.0 s |
| Translate (10 clips; Whisper on CPU) | 0.500 | 4.54 | 38.5 s |
| `/detect-language` (11 clips, all correct) | – | – | 10.3 s |

The same card, same suites, with the other two engines and two more device shapes:

| Mode (bucket) | FASTER-WHISPER (CUDA, fp16) | PARAKEET | CANARY (`NEMO_BATCH_SIZE=1`) |
| :--- | :--- | :--- | :--- |
| Transcribe, language given: Parakeet's 25 (82 clips), WER / RTF / wall | **0.077** / 0.159 / 0.70 s | 0.104 / **0.063** / 0.24 s | 0.089 / 0.084 / 0.44 s |
| Transcribe, language given: other languages (33 clips) | **0.340** / 0.153 / 1.0 s | 0.348 / 2.16 / 13.5 s | 0.348 / 1.96 / 11.3 s |
| Transcribe, language auto-detected: Parakeet's 25 (10 clips) | **0.055** / 0.29 / 5.5 s | 0.085 / 2.08 / 23.0 s | 0.098 / 2.15 / 23.5 s |
| Translate (10 clips) | 0.494 / 0.28 / 5.3 s | 0.500 / 4.54 / 38.5 s | **0.483** / 2.12 / 22.9 s |
| `/detect-language` median wall | **3.5 s** | 10.3 s | 9.5 s |
| Accuracy suite | 9/9 | 9/9 | 8/9 ("whisperpro", as on the NUC) |

Two device shapes, both 9/9:

- **PARAKEET on the Intel UHD iGPU alone** (`--device GPU`, `MAX_CUDA_UNITS=0`): `ready on
  Intel(R) UHD Graphics (iGPU) (device=GPU.0, quantization=fp32)`, `/status`
  `ASR=Intel GPU measured=True`.
- **PARAKEET on the hybrid pool** (cuda:0 + UHD iGPU, no limits): every task landed on
  cuda:0, so only the `PARAKEET-CUDA` worker ever started. The iGPU unit reported
  `measured=False` because it was never needed. That is the scheduler's
  fastest-idle-unit choice working as designed.

**Long-form throughput, and why.** Each 20-minute clip ran at about 0.5x real time: 31–42
minutes of wall time, against RTF 0.15–0.2 for Faster-Whisper on the same card.

- **Not the decode.** Parakeet's own decoding is fast (RTF 0.06 on clips).
- **The detection.** Decoding by speech region (`ASR_SEGMENT_FIRST`) detects the language
  of every speech region before decoding, and with the companion forced onto the CPU,
  hundreds of large-v3 detections on the CPU dominate. The English variant, where there is
  little to switch, finished at 0.98x.
- **Consequence.** On an 8 GB card the NeMo engines are fast only for requests that name
  their language and stay inside the 25. With the companion on a GPU large enough to hold
  it, this cost should drop to Faster-Whisper's. The RTX 5090 rows will say.

### Intel NUC (Core Ultra 255H: Arc 140T iGPU + AI Boost NPU), full `intel` build, 2026-10-10

Each engine ran the same suites from a fresh stack (`--device GPU`, vocal separation off),
with clips from the generated matrix: 146 requests, 13 skipped for clips the matrix lacks.
Device evidence for every row is `/status` `ASR=Intel GPU measured=True`. For the NeMo rows
the log also shows `ONNX Runtime 1.24.1 from /app/libs/intel ... requested
['OpenVINOExecutionProvider', ...]` and `ready on Intel(R) Graphics [0x7d51] (iGPU)
(device=GPU, quantization=fp32)`, with no fallback. The NeMo companion ran on the CPU (the
NPU unit's path), because Faster-Whisper has no Intel GPU backend.

| Mode (bucket) | INTEL-WHISPER (OpenVINO, iGPU) | PARAKEET | CANARY |
| :--- | :--- | :--- | :--- |
| Transcribe, language given: Parakeet's 25 (82 clips), mean WER / median RTF / median wall | **0.076** / 0.437 / 1.63 s | 0.104 / 0.288 / 0.90 s | 0.094 / **0.190** / 0.78 s |
| Transcribe, language given: other languages (33 clips, sent to Whisper) | **0.278** / 0.500 | 0.339 / 1.23 (CPU companion) | 0.339 / 1.30 (CPU companion) |
| Transcribe, language auto-detected: Parakeet's 25 (10 clips) | **0.057** / 0.40 / 7.7 s | 0.085 / 1.98 / 22.9 s | 0.098 / 1.94 / 22.2 s |
| Translate to English (10 clips) | **0.472** / 0.28 / 6.6 s | 0.461 / 3.58 / 33.6 s (Whisper on CPU) | 0.483 / 1.89 / 21.8 s (native) |
| `/detect-language`, median wall (11 clips, all correct) | **4.8 s** | 10.0 s | 9.9 s |
| Accuracy suite | 9/9 | 9/9 | 8/9* |
| Code-switch suite | — | 30/30 | — |

\* Canary wrote the product name in the accuracy fixture as "whisperpro asr" (two words
off where the test allows one). The decode was otherwise word-perfect. This is the model's
spelling of a proper noun, recorded rather than worked around.

Also on the NUC:

- **PARAKEET on the CPU** (GPU and NPU units removed): accuracy 9/9,
  `ASR=CPU measured`.
- **INTEL-WHISPER on GPU + NPU** after the scheduler's idle-unit change: accuracy 9/9.

**What the NUC shows:**

- **Accuracy.** On this hardware, INTEL-WHISPER on the iGPU is *more accurate* than both
  NeMo models on their own languages (0.076 against 0.104 and 0.094). It is also more
  accurate everywhere else.
- **Speed on covered languages.** The NeMo models are only 1.5x (Parakeet) to 2.3x
  (Canary) faster there, which is far from the 30x CUDA gap.
- **Everything that needs Whisper is slower.** That covers detection, auto-detected
  requests, other languages and Parakeet's translations. The companion has to run on the
  CPU, while INTEL-WHISPER uses the iGPU. Detection, the request Bazarr makes most often,
  takes twice as long (10 s against 4.8 s).
- **Verdict.** For an Intel host serving Bazarr, INTEL-WHISPER remains the better engine.

### RTX 5090 (32 GB), `nvidia` build via Windows 11 / WSL2, 2026-10-11

Same branch, same audio-matrix clips, every engine on `cuda:0` (companion on CUDA too).
Benchmark: 82 NeMo-covered clips with the language given, 33 routed to Whisper, 10 each for
auto-detection and translation. WER/CER is the mean; RTF the median.

| engine | WER, language given | RTF, language given | WER, auto-detect | translate WER | long-form (44 checks) | 20-min suite wall |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| FASTER-WHISPER | 0.077 | 0.171 | 0.055 | 0.494 | 44/44 | 16.6 min |
| PARAKEET | 0.104 | 0.026 | 0.085 | 0.494 (Whisper) | 44/44 | 7.4 min |
| CANARY | 0.088 | 0.057 | 0.098 | 0.483 (native) | 42/44 | 5.2 min |

- PARAKEET decodes about 6.5x faster than Whisper large-v3 on the same card, at a WER about
  three points higher on these synthetic clips; CANARY sits between them on both.
- Whisper-routed clips score identically (0.336) under every engine: they *are* Whisper.
- CANARY's two long-form misses (`longform_film`, Russian `longform_film_mono`) are **not
  timing**: every segment lands on its utterance. They are two recognition behaviours,
  established by re-running both clips and comparing each failed window with the transcript:
  - **Russian:** 42 of its 48 misses are one TTS sentence, "Скажи мне правду.", which CANARY
    hears as "Кажем неправду." every time it recurs. A model error on that voice and line;
    the other engines get it right.
  - **Short foreign lines are translated into the run's language.** Per-region detection
    labels a one-second Dutch or Italian line inside a German stretch as `de`. CANARY takes
    the language it is given as both source and target, so it *translates*: Dutch "Ik weet
    niet wat je bedoelt." comes back as German "Ich weiß nicht, was du meinst.", Italian
    "Dimmi la verità." as "Dimme die Wahrheit.". FASTER-WHISPER and PARAKEET receive the same
    wrong label but write what they hear, which is why they pass. CANARY amplifies a
    detection mistake into a wrong-language subtitle, so it is the weaker choice for
    code-switched material.
- Accuracy test: PARAKEET and FASTER-WHISPER 9/9, CANARY 8/9 (writes "whisperpro").

### Real speech, Romanian (independent run, 2026-10-10)

A separate evaluation ran on the RTX 5090 with real speech rather than this repo's synthetic
TTS matrix. It used `transformers` bf16, not onnx-asr, and diacritics were kept.

| | Whisper large-v3 | Canary-1B-v2 | Parakeet-TDT-0.6B-v3 |
| :--- | ---: | ---: | ---: |
| FLEURS-ro WER (883 utterances) | 8.91% | **6.90%** | 12.60% |
| VoxPopuli-ro WER (1,330 utterances) | 10.14% | **6.45%** | 9.15% |
| Long-form, 35 min synthetic | 17.2% | **7.4%** | 11.9% |
| Invented text on reversed-speech probes (chars/min, after VAD and filters) | 841 | 590 | **451** |

On the same evaluation's NUC CPU runs (8 threads), real-time factors were Whisper
large-v3 int8 0.60, Canary 0.071 and Parakeet 0.055.

This points the other way from the synthetic matrix above. On real Romanian speech Canary
beats Whisper clearly, and both NeMo models hallucinate less on non-speech. The two are
consistent with each other: the matrix's Piper TTS clips are short and clean, which suits
Whisper, and say little about natural speech, long-form audio or non-speech. Any default
decision must rest on real-speech data per language. A synthetic WER gap alone is not
enough in either direction.

The RTX 5090 comparison, with both models and the companion on the GPU, is recorded above. Every
host's runs are tracked in `docs/RELEASE_VALIDATION_PLAN.md`, rows P1–P6.

**What the 3080 runs already establish:**

- **Fit.** On 8 GB the three tenants (NeMo model, companion, CUDA LD vocal isolation) do
  not fit with their defaults.
- **Fallback.** The fallback design works as intended: requests kept succeeding.
- **Validation.** For these engines `/status` must be read every run, because a passing
  suite does not show where the work ran.

## Behaviour differences to expect

- **Options with no NeMo counterpart.** `initial_prompt` and `beam_size` have no effect on
  NeMo-served calls; the prompt is logged at debug level. They still apply to calls routed
  to the companion.
- **Canary timing.** Canary returns no token times through onnx-asr. Its subtitle lines
  are timed from their chunk (at most 20 s), split proportionally across sentences, and a
  sentence longer than 84 characters is wrapped at word boundaries so an unpunctuated
  stretch never becomes one 20-second line.
  Parakeet lines come from word timings, at most 7 s and 84 characters, split at sentence
  ends and pauses over 0.8 s.
- **Punctuation and casing** come from the model (both emit them), not from Whisper.
- **With `ASR_SEGMENT_FIRST=false`,** an auto-detected file is one call in the file's
  language. Parakeet copes because it detects internally. Canary is prompted with the
  file's language throughout.
- **Detection runs on the companion's device.** On an Intel host where the companion runs
  on the CPU, `/detect-language` is slower than under INTEL-WHISPER on the iGPU. Measure it
  before recommending these engines for a Bazarr deployment there.
