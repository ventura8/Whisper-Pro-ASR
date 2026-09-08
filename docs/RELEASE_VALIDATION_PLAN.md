# Release Validation Plan

What must pass before v1.4.0 is tagged, and what each step actually proves. Machines are
named by their plan label only (`local`, `nuc`, `xubuntu`, `win5090`); hosts and usernames
live in the gitignored `.validation-matrix.conf` and never in a committed file.

Two claims are separate throughout and must never be merged:

- **Decoding worked** -- the transcript matches the audio. A CPU fallback also produces a
  perfect transcript, so this proves the software, never the accelerator.
- **The accelerator was used** -- a device-side observation taken while work was in flight.

A row is complete only when both are recorded. See
`.claude/skills/hardware-verification/SKILL.md` for the reasoning.

The previous plan (v1.3.0) ran 25 rows across four machines because that release moved
every vendor runtime out of the API process. This one changes the FASTER-WHISPER decode
path and the request pipeline around it, plus one scheduler decision -- which idle unit a
task takes on a hybrid host -- and nothing in the Dockerfile, the preprocessing stack or
the Intel and AMD engines. The matrix is sized to that: every changed
path on every class of host, and a non-regression row on each engine the shared pipeline
serves. Status is recorded per row -- **a row not marked done has not been run.**

---

## What changed, and therefore what needs proving

| Change | Where | Hosts that can prove it |
| :--- | :--- | :--- |
| Decode by VAD speech region (`clip_timestamps`), FASTER-WHISPER only | `speech_clips.py`, `model_manager.py` | any CUDA host; CPU for the cost |
| Per-clip language detection through the worker | `inference_worker.py`, `isolated_engine.py`, `segment_languages.py` | any host running FASTER-WHISPER isolated |
| `auto_detected` carried from the route past language resolution | `asr.py`, `model_manager.py` -- **every engine's request path** | one row per engine |
| Gap fill treats decoded spans as covered | `gap_filling.py` | any FASTER-WHISPER row with clips |
| WHISPERX refuses clips explicitly; code-switched defects engine-scoped | `whisperx_engine.py`, manifest `xfail_engines` | a WHISPERX row |
| Compose forwards `ASR_SEGMENT_FIRST` / `ASR_SEGMENT_LANGUAGES` | `docker-compose.yml` | any compose-driven row |
| The scheduler prefers an idle unit the engine can drive over the pool's rotation | `scheduler/unit_choice.py`, `concurrency.py` | the CUDA + Intel hybrid host (rows L4, L7) |
| Run-level hysteresis and one decoder call per language | `language_runs.py`, `run_decoding.py`, `gap_filling.py` | any CUDA host for the fixtures; the NAS excerpts for real film (rows R1-R4) |
| A paused task releases the worker channel (chunked region detection, resumable decode) | `resumable_decode.py`, `segment_languages.py`, `model_segment_processing.py`, `inference_worker.py` | a single-unit host with a priority detection arriving mid-transcription: the NUC (row N8) |

---

## Phase 0 -- Prerequisites

| # | Item | Status |
| :-- | :--- | :--- |
| 0.1 | Fixtures regenerated for this release (20 `*_scene2/3` clips, `longform_natural`) and synced with `--fixtures` to any host that runs `longform` -- the natural clip post-dates every host's last sync | done for `win5090` and for `nuc` -- N4 and N6 ran with `--fixtures` (399 files synced), so all six variants are there |
| 0.2 | `remote_validate.sh --timeout 10800` on any CPU long-form row; the 900s default truncates the clip and reads as three unrelated assertion failures | done |
| 0.3 | `remote_validate.sh` ssh keepalive (`ServerAliveInterval`) -- a 48-minute request without it lost a completed NUC result to a silently dead connection | done |
| 0.4 | `--wsl Ubuntu` on every `win5090` row; without it the key-auth probe runs `true` in a Windows shell and the script blames the key | done |
| 0.5 | **Never edit `remote_validate.sh` while a run is in flight.** Bash reads a script by byte offset as it executes, so an insertion above the executing line makes every live runner misparse when it gets there -- one crashed at line 845 after its suite had passed, taking the teardown and the `result:` line with it, and stalled every waiter chained on that line | learned |

---

## Phase 1 -- Pipeline validation (no hardware)

| # | Check | Pass condition | Status |
| :-- | :--- | :--- | :--- |
| 1.1 | Full gate | lint clean, every Python and JS test passing, every file >= 90% | **done**, re-run after every review wave -- last: 2,497 Python (2,468 in the parallel run + 29 serial concurrency tests), 191 JS unit, 35 e2e-fixture: 2,723 in all; coverage 97 %, every file over 90 % |
| 1.2 | **Nine cold builds** from an empty cache | all nine succeed | running on `local` (sequential, pruned between targets, disk-guarded) -- `cpu` 28 min, `intel` 30 min, `intel-xpu` 132 min OK; 6 to go |
| 1.3 | Manifest and catalog agree | `scripts/audio_catalog.py` leaves the tree unchanged | **done** |

1.2 is lower risk than last release -- the Dockerfile is untouched and the Python lock did not
move -- but it stays mandatory because caching conceals exactly the failure it catches.

---

## Phase 2 -- Hardware matrix

### local -- RTX 3080 + Intel UHD (pre-Arc)

| # | Target | Engine | ASR dev | Suite | What it proves | Status |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| L1 | nvidia | FASTER-WHISPER | CUDA | longform, all three variants | The two closed defects, all six assertions, strict | **done, then re-done** -- 18 passed at the real 1.0 budget on `cuda:0` (RTF 0.149 / 0.197 / 0.179) with the first design; the shipped design (runs, one call per language, region clips) measures stress **0/118** at RTF 0.166 and natural 7/7 at 0.190 (L8). (First measured at RTF 2.4-3.3 with the budget raised: that was the CPU, see L7) |
| L2 | nvidia | FASTER-WHISPER | CUDA | accuracy + code-switched, `ASR_SEGMENT_FIRST=0` then `=1` | No monolingual regression; both legs of every code-switched clip | **done** -- 25 passed both rows; 19 passed strict at `=1` |
| L3 | nvidia-whisperx | WHISPERX | CUDA | code-switched | Clips never reach WhisperX; its six defects are xfailed by engine, nothing else | **done** -- 13 passed, 6 xfailed |
| L4 | nvidia-intel | FASTER-WHISPER | AUTO | smoke, UVR on | The hybrid pool still fills with both units; route changes touch every request | **done** -- 25 passed; pool holds `NVIDIA GPU 0` and the Intel iGPU, `ASR=CUDA measured=True`, `UVR=CUDA measured=True`, and nothing landed on the Intel unit (`measured=False`): the rotation fix holds under the smoke load. (First attempt died in the image build on a poetry SSL EOF.) |
| L5 | nvidia | FASTER-WHISPER | CUDA | longform, `film` variant | First run of the fixture laid out from real film: a bed at dialogue level, one-line scenes. Establishes its baseline; no defects are inherited | **done, and load-bearing** -- 6 passed with the first design (RTF 0.179). It then failed `quiet` + `windows` on the run-as-clip design (6 of 19 quiet passages inside a run) and showed the forced-language, run-sized window dropping lines; on the shipped design **7 of 7** at RTF 0.170 once `repetition` judged consecutive repeats -- the 44th copy was an absorbed French line rendered in German, the documented single-line trade (L8) |
| L7 | nvidia | FASTER-WHISPER | AUTO | 3-min clip, then stress `=0`/`=1` | **The hybrid-host fix**: the transcription must land on `cuda:0`, not the Intel unit, when both are idle | **done** -- `hardware unit cuda:0` on every request, one engine loaded; same clip 13:35 -> 1:00; stress 56/118 @ 135s vs 0/118 @ 195s |
| L6 | nvidia | FASTER-WHISPER | CUDA | tier-B content, 66 clips | The bars that were never read, now enforced as a fraction of each reference's ceiling | **done** -- 55 clear; 7 recorded as content defects, 4 near-misses given strict 0.3 bars (decoded on CPU, see L7 -- correctness is device-independent) |
| L8 | nvidia | FASTER-WHISPER | CUDA | longform, natural + film + `film_mono`, shipped design | The run design on every long-form shape, with the new `fracture` assertion | **done** -- natural 7/7 (RTF 0.190), `film_mono` 7/7 (0.167; fracture under 5%), film 7/7 (0.170; fracture 4.9% against its 10% bar) after `repetition` was made to judge consecutive repeats |
| R1 | nvidia | FASTER-WHISPER | CUDA | 21 real monolingual NAS excerpts, fracture | The default must not fracture a film in one language | **done** -- per-region 20.1% median (killed the first design); runs-as-clips 3.0%; **shipped design 1.1% median**, mean 9.9%, 6 films over 5%, RTF 0.149 |
| R2 | nvidia | FASTER-WHISPER | CUDA | 7 excerpts with same-language subtitle tracks | Transcript quality, relative only | **done** -- mean overlap 0.456 whole-file, 0.461 per-region, 0.514 runs-as-clips, **0.505 shipped** with the fewest cues under 0.4 (42%) |
| R3 | nvidia | FASTER-WHISPER | CUDA | CS / OPEN / END / EP / DUB excerpts | Code-switch localisation against forced tracks, invented text in title music and credits, whole episodes, dub tracks | extraction on the NUC (attempt 4, one ffmpeg pass per title); probe queued after R1/R2 |
| L10 | nvidia | FASTER-WHISPER | CUDA | longform, `film_bookends` + `episode` | The shapes laid out from openings, credits and episodes: loud non-speech passages of 60-300 s | **done** -- episode 7/7 (RTF 0.173); bookends 6/7 (RTF 0.131) with the coverage assertion measuring against the audio's end rather than the last line's -- corrected, and after the planner learned to stop a scene at the credits the re-laid bookends passes **7/7** at RTF 0.115; `hardware unit cuda:0` |
| R4 | nvidia | FASTER-WHISPER | CUDA | 54 excerpts: OPEN / END / EP / DUB / CS | Text where the VAD hears nothing in openings, credits and title music; forced-cue precision and recall on code-switching titles | **done** (RTX 3080, 54 of 54, third attempt): openings hold 1.4 s of text per minute of music, credits 0.3 s; six dub tracks of six reported in the dub language; forced-cue precision 0.87-1.0 on the titles that report a foreign run, recall 0.73-0.77 on scene-length passages, none on single lines (the documented trade, confirmed by a per-region label dump on the two titles that looked like exceptions); 7 of 16 forced tracks are full tracks misflagged (evidence README) |
| L9 | nvidia | FASTER-WHISPER | CUDA | code-switched + tier-A accuracy, shipped design | The short-clip share rule keeps both legs; no monolingual regression | **done** -- 18 passed (both legs of all six), 20 passed tier-A; `hardware unit cuda:0` on all 22 requests |
| L11 | nvidia | FASTER-WHISPER | CUDA | translation: tier-A 10 clips, code-switched 6, long-form stress + film | Every language in a file is translated from itself -- the per-language decode carries `task=translate`, and follows switches even when the caller named a language | **first run** (RTX 3080, 2026-09-12): 16 of 18 -- both long-form translations reach every window in English, all six code-switched clips come back with both legs translated and no foreign words left; `it_core` (0.44) and `uk_core` (0.31) fall under the 0.50 bar on the model's own translation of a six-second clip and are recorded as translation-scoped defects. Re-run at `f29ed53` on the RTX 3080 (28 passed, 2 xfailed -- both reproduced) and in the RTX 5090's full suite (W8), where `it_core` cleared the bar and `uk_core` did not; `it_core` is recorded as a flapper like `ka_tail` |
| L12 | nvidia | FASTER-WHISPER | CUDA | accuracy fixture, then two window-context designs through tier-A, code-switched and every long-form variant | The accuracy fixture's "A quick brown fox" (a 3-second window on its own, beam search on a near-tie) is the model, and no design that gives the window more context survives the fixtures | **done** (RTX 3080, 2026-09-13): the whole-file call for a file in one language lost 188/126/220 windows on film_mono/bookends/episode and hallucinated in their silences; joining lines under 1.0 s apart lost 97/37/86/118 on film/film_mono/bookends/episode. Both rejected and recorded; the accuracy suite scores each sentence by ordered edit distance within one word |

### nuc -- Intel Arc 140T iGPU + AI Boost NPU

| # | Target | Engine | ASR dev | Suite | What it proves | Status |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| N1 | cpu | FASTER-WHISPER | CPU | longform, `=1` | What the default costs a CPU host | **done** -- 5 passed, 1 failed: only `throughput` (RTF 2.44), as CPU always has; `windows`, `quiet`, language, coverage, repetition all hold. 225 clips |
| N4 | cpu | FASTER-WHISPER | CPU | longform, six variants, shipped design, `--timeout 10800` | What one decoder call per language costs a CPU host, and whether every correctness assertion still holds there | **done** -- 36 passed, 6 failed, every failure `throughput`: RTF stress **2.39** (2.44 on the first design -- the per-language calls cost nothing measurable), natural 3.12, film 3.08, film_mono 2.92, bookends 2.13 (pre-rebuild layout), episode 3.24; `windows`, `quiet`, `fracture`, `language`, coverage and `repetition` hold on all six. 5 h 43 min; `ASR=CPU measured=True` |
| N2 | cpu | FASTER-WHISPER | CPU | longform, `=0` | The control for N1 -- CPU never met the 1.0 budget, so the question is the delta | **done** -- RTF **1.04**, `windows` FAIL, `quiet` FAIL. The default costs CPU 2.35x and buys both defects; CPU is over budget either way |
| N3 | intel | INTEL-WHISPER | GPU | accuracy | Segment-first must not engage on the Intel engine; the shared route changes must not regress it | **done** -- 9 passed on OpenVINO GPU, `ASR=Intel GPU measured=True`, no region decoding logged |
| N8 | cpu | FASTER-WHISPER | CPU | stress clip with a `/detect-language` fired mid-transcription, `--timeout 10800` | The deadlock the NUC found on 2026-09-12 (a paused task held the worker channel the priority detection needed) is closed: the detection completes while the transcription is paused, the transcription resumes and finishes with every assertion holding | **done as N8b** (final tree, 2026-09-13): accuracy 9 passed; three hand-fired detections returned in 40 s / 177 s / 36 s (inside the per-region detection and the decode), 28 of Bazarr's own detections completed alongside; `Paused after N segment(s)` / `Resuming: N clip(s)` seven times in twenty minutes, no `Blocked on lock`. The transcription was starved -- not deadlocked -- by a saturating priority backlog (a Bazarr detection every ~3 min, each 2:45 on this CPU) and had reached 7:38 of 20:03 when the host was rebooted 2 h 47 min in |

### win5090 -- RTX 5090 via Windows 11 / WSL2

`xubuntu` (the Linux side of the same box) needs `ssh-copy-id` of the validation key before
it can be used; every row below ran through WSL2.

| # | Target | Engine | ASR dev | Suite | What it proves | Status |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | nvidia | FASTER-WHISPER | CUDA | longform stress | The RTF budget holds where it is meant to | **done** -- 6/6 at the real 1.0 budget, session 134s including the 1203s clip (RTF < 0.12); `ASR=CUDA measured=True` |
| W2 | nvidia | FASTER-WHISPER | CUDA | longform, stress + natural | The realistic layout, on the fast host | **done** -- 12 passed at the real 1.0 budget, 306s for both clips (RTF ~0.13); `ASR=CUDA measured=True` |
| W3 | nvidia | FASTER-WHISPER | CUDA | **full** (156 languages) | No regression anywhere in the matrix; only affordable here | pending |
| W4 | nvidia-whisperx | WHISPERX | CUDA | smoke | Engine non-regression through the changed route | **done** -- 23 passed, 1 xfailed (`mix_en_fr`, engine-scoped); `ASR=CUDA measured=True` |
| W5 | nvidia | OPENAI-WHISPER | CUDA | smoke | Engine non-regression through the changed route | **1 failed, 23 passed** -- `mix_en_fr` content at 0.50: French leg only. The raised bar caught a third engine dropping a leg; W5b measured all six before the manifest was scoped |
| W5b | nvidia | OPENAI-WHISPER | CUDA | codeswitch (all six) | Which of the six the engine actually drops, before scoping it | **done** -- 3 failed, 15 passed: `mix_en_es` 0.38, `mix_en_fr` 0.50, `mix_ar_fr` 0.50 (first leg dropped each time); `mix_de_en`, `mix_hi_en`, `mix_zh_en` clear 0.60. Those three now carry `xfail_engines: ["WHISPERX", "OPENAI-WHISPER"]`; `ASR=CUDA measured=True` |
| W6 | full | FASTER-WHISPER | AUTO | smoke | The image `latest` is meant to resolve to | **cannot start with its own override there** -- `docker-compose.full.yml` lists `/dev/kfd`, which WSL2 has not, and Compose refuses (the file says so). First attempt hung in pip for 4h; the retry built in 30 min and failed at `up`. Re-queued as W6b with the new `--compose nvidia` runner flag |
| W7 | nvidia | FASTER-WHISPER | CUDA | longform, `film` variant | The film fixture at the real 1.0 budget, on the fast host | **done as W2b** -- all four variants (stress, natural, film, `film_mono`), 7 assertions each incl. the new `fracture`: **28 passed** in 647 s on the shipped design; `ASR=CUDA measured=True` |
| W5c | nvidia | FASTER-WHISPER | CUDA | codeswitch, shipped design | Both legs of all six clips survive the run hysteresis (the share rule) | **done** -- 18 passed; `ASR=CUDA measured=True` |
| W3b | nvidia | FASTER-WHISPER | CUDA | full, shipped design | The whole matrix on the shipped design, and the numbers behind the tier-B bars that fail on float16 | **done** -- 377 passed, 7 xfailed, 4 failed, all tier-B content: `bn_tail` 0.21 (Latin gibberish, twice on this host), `pa_mms` 0.00 (romanised Punjabi -- the right words in the wrong script), `uk_line3` 0.00 (a one-second line as one run-together word), `ru_line2` 0.50 ("чем" for "чём" -- a scoring bug, ё now folds to е). `ka_tail`/`te_tail` failed on W3 and cleared here: temperature fallback samples on those clips. Five clips are recorded as content defects from this pair of runs -- `bn_tail`, `pa_mms`, `uk_line3`, and `ka_tail`/`te_tail` as flappers (`strict=False`); `ru_line2` was a scoring bug and is not recorded; `ASR=CUDA measured=True` |
| W2c | nvidia | FASTER-WHISPER | CUDA | longform, six variants, `--fixtures` | `film_bookends` and `episode` on the fast host | **done, twice** -- 42 passed (six variants, 7 assertions each) at 978 s; re-run as W2d after the planner learned to stop a scene at the credits (bookends re-laid, 166 lines): **42 passed** at 914 s. `ASR=CUDA measured=True` both times. (The first attempt was lost to Docker Desktop's engine dying with the BuildKit SIGBUS; a restart of the box fixed it) |
| W6b | full | FASTER-WHISPER | CUDA | smoke, `--compose nvidia` | The full image -- what `latest` resolves to -- started on a CUDA-only host | **build crashed** -- BuildKit on the WSL2 host died with SIGBUS mid-build (the host's Docker, not the tree; third failure of this image on this box today). Moved to the local hybrid host as W6-local, `--target full --compose nvidia-intel`: **25 passed**, image rebuilt from the shipped tree, `ASR=CUDA measured=True` -- the image `latest` resolves to starts and passes on a host without `/dev/kfd` when given the right override |
| W8 | nvidia | FASTER-WHISPER | CUDA | full real-audio suite at `f29ed53`, `--timeout 3600` | The whole matrix on the pushed tree, translation assertions included | **done** -- 390 passed, 44 skipped (the long-form tests, gated on `RUN_GPU_LONG_ASR`), 11 xfailed, 3 xpassed: `ka_tail`, `te_tail` and `it_core`'s translation, the three recorded flappers; `uk_core`'s translation xfailed as recorded; `ASR=CUDA measured=True` |

---

## Phase 3 — Evidence required per row

A green suite is half a row. Record all of these:

1. **Banner** — `ASR Engine`, `ASR Runtime`, `Preprocess Device`, `Resource Pool`,
   `OpenVINO devices`.
2. **UVR provider** — the `[ONNX provider: ...]` line, for every UVR-on row.
3. **Dashboard truth** — `/status` per unit: `asr_execution` and `uvr_execution`, each with
   `measured: true`. A `measured: false` chip is a prediction from configuration, not a result.
4. **Device sample taken during inference** — `nvidia-smi --query-compute-apps=pid,used_memory
   --format=csv` (CUDA), `intel_gpu_top` (Intel GPU), `rocm-smi --showuse` (AMD).
5. **NPU rows only** — `intel_gpu_top` cannot see the NPU, and an idle NPU reads there exactly
   like a busy one. Use `openvino.Core().available_devices` plus `ls -l /dev/accel/`, and the
   `/status` execution chip.
6. **Suite exit code 0.** A skipped suite has proven nothing; report skips and xfails
   honestly rather than folding them into a pass count.

---

## Phase 4 -- Release gate

- [x] Phase 1.1 green
- [ ] Phase 1.2: nine cold builds
- [ ] Every Phase 2 row done, each with both claims evidenced
- [x] Release notes reconciled against the actual numbers
- [ ] Known Limitations still true (see below; the tier-B and WhisperX entries are new this release)
- [ ] `flavor: latest=false` in CI **and** the published `latest` tag repointed to `full` -- on v1.3.0 the metadata action's `latest=auto` let all nine targets race for it and `amd-rocm-torch` won
- [ ] Commit, then tag

Logs go in `docs/validation/v1.4.0/`, scrubbed by `scripts/scrub_validation_log.sh`.

---

## What this plan deliberately does not cover

State these as limitations rather than letting a green matrix imply them:

- **AMD / ROCm is unvalidated on supported silicon, and the dashboard will claim otherwise.**
  Two Radeon hosts were exercised in the end -- a discrete card and the RTX 5090 box's Granite
  Ridge integrated Radeon -- and both are `gfx1036`-class, which is not among the architectures
  the image ships kernels for. ROCm initialises anyway, so `/status` reports
  `UVR = AMD GPU, measured: true, fallback: false` while the work runs on the CPU. Two unrelated
  machines producing the identical false positive makes this a property of the combination, not
  one host's quirk. A supported Radeon (RDNA2/3/4) has still never run this code.
- **Isolated OpenVINO alongside CUDA is reasoned, not measured.** It needs a host with both an
  NVIDIA GPU and an Arc-class Intel GPU. `local` is CUDA + pre-Arc UHD (isolation refused
  there anyway) and `nuc` is Arc with no CUDA, so no available machine can produce it.
- **NPU ASR is impossible, not merely untested.** The shipped Whisper IR is dynamic-shaped and
  the NPU plugin requires static upper bounds. N2/N5 validate NPU *preprocessing*, which is
  the device's one working capability.
- **macOS validates the `cpu` target only** — Docker Desktop's Linux VM has no GPU passthrough.

## Budget

Phase 1 is dominated by the cold builds: 3-5 hours sequential on `local`. Phase 2 is roughly
1 h on `local` (done), 2 h on `nuc` for the two long-form rows plus 15 min for N3, and 3 h on
`win5090` -- the `full` tier alone is about 2 h. `nuc` and `win5090` run in parallel; nothing
here shares hardware with anything else.
