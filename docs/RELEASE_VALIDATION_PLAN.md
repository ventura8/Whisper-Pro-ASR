# Release Validation Plan

What must pass before v1.3.0 is tagged, and what each step actually proves. Machines are
named by their plan label only (`local`, `nuc`, `xubuntu`, `win5090`); hosts and usernames
live in the gitignored `.validation-matrix.conf` and never in a committed file.

Two claims are separate throughout and must never be merged:

- **Decoding worked** — the transcript matches the audio. A CPU fallback also produces a
  perfect transcript, so this proves the software, never the accelerator.
- **The accelerator was used** — a device-side observation taken while work was in flight.

A row is complete only when both are recorded. See
`.claude/skills/hardware-verification/SKILL.md` for the reasoning.

---

## Phase 0 — Prerequisites (blocking, do these first)

| # | Item | Why it blocks |
| :-- | :--- | :--- |
| 0.1 | **Decide the piper-tts 1.7.0 -> 1.8.0 fixture question** | The toolchain lock now resolves piper 1.8.0 while the committed fixtures were rendered with 1.7.0. Running the matrix against a fixture set the toolchain no longer reproduces measures nothing durable. Either regenerate and re-commit the matrix, or pin piper back to 1.7.0 in `requirements.lock`. |
| 0.2 | **Teach `validation_matrix.sh` the two missing axes** | Its runner passed only `--target/--engine/--preprocess/--suite`, so it could express neither **UVR on/off** nor an explicit **ASR device**, which 11 of the 25 rows below need. Done: `device` and `separation` columns were added, and `remote_validate.sh` gained `--env` for the isolation rows it still cannot express. |
| 0.3 | **Sync fixtures to each machine** (`--fixtures`, ~3.2 GB) | Only the ~10-language core tier is committed; `smoke` and deeper need the generated matrix. |
| 0.4 | **Confirm `validation_matrix.sh` runs remote-only plans** | A defect here was reported and never reproduced. If it misbehaves, fall back to per-row `remote_validate.sh` invocations rather than debugging it during a release. |

---

## Trigger: moving torch

Any change to `torch`, `torchvision` or `torchaudio` **requires re-running the CUDA rows on
hardware before it can be merged**, not just a green gate.

torch supplies its own cuDNN through the `nvidia-cudnn-cu13` wheel while the image also
installs one from apt, and the two must agree. A torch 2.14 bump moved the wheel from
9.20.0.48 to 9.24.0.43 and every torch-engine request began returning HTTP 500
(`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`) -- with the full suite green, all nine images
building, and FASTER-WHISPER unaffected because CTranslate2 never touches cuDNN. The
mismatch was a minor version difference against an apt package pinned as `=*`, so no static
check can see it. Rows W3/W4 (WHISPERX and OPENAI-WHISPER on CUDA) are what caught it and
are what must be re-run.

## Phase 1 — Pipeline validation (no hardware)

Runs anywhere; do it before booking machine time.

| # | Check | Command | Pass condition |
| :-- | :--- | :--- | :--- |
| 1.1 | Full gate | `scripts/ci/build-and-test.sh` | Lint clean, ~2200 Python + 191 JS tests pass, every file >=90% coverage, badge regenerated |
| 1.2 | **Nine cold builds** | `docker buildx build --target <t> --build-arg RUNTIME_BASE=<base> --output type=cacheonly .` | All nine succeed from an **empty** cache |
| 1.3 | Fixture determinism | `scripts/generate_fixtures_docker.sh all` then `git status` | Working tree clean (depends on 0.1) |
| 1.4 | Fixture coverage | `scripts/generate_fixtures_docker.sh verify` | Gaps are only the known no-voice languages |
| 1.5 | Toolchain lock integrity | Re-run 1.3 on a wiped `.fixture-tooling/site` | Installs under `--require-hashes` with no skip warnings |

**1.2 is not optional.** v1.3.0 as committed could not build any standalone target from a
cold cache, and the build cache — plus CI's `cache-from` — hid it completely. This is the one
check whose entire purpose is that caching conceals the failure it catches.

---

## Phase 2 — Hardware matrix

Machines run in parallel; rows within a machine run in sequence (each rebuilds the stack and
binds port 9000). `UVR` is `ENABLE_VOCAL_SEPARATION`. Suite tiers: `accuracy` ~4 min,
`smoke` ~20 min, `full` ~2 h, `longform` ~9 min.

### local — RTX 3080 + Intel UHD (pre-Arc)

The only CUDA+Intel hybrid available, so it is the only machine that can exercise the
cross-vendor guards at all.

| # | Target | Engine | ASR dev | Prep dev | UVR | Suite | What it proves |
| :-- | :--- | :--- | :--- | :--- | :-- | :--- | :--- |
| L1 | nvidia-intel | FASTER-WHISPER | AUTO | AUTO | on | smoke | Hybrid default; both units in the pool; both stages measured CUDA |
| L2 | nvidia-intel | FASTER-WHISPER | CUDA | GPU | on | accuracy | **Bootstrap reachability fix** — must report `CUDAExecutionProvider`, not a CPU fallback |
| L3 | nvidia-intel | FASTER-WHISPER | CUDA | GPU | on | accuracy | **Provider guard fails closed** with `ASR_ISOLATE_PREPROCESSING=1`: UHD is pre-Arc, isolation is refused, OpenVINO must stay blocked |
| L4 | nvidia-intel | FASTER-WHISPER | AUTO | AUTO | **off** | accuracy | UVR-off path; no separator is built |
| L5 | nvidia-whisperx | WHISPERX | CUDA | AUTO | on | smoke | Diarization image |
| L6 | nvidia | OPENAI-WHISPER | CUDA | AUTO | on | smoke | torch CUDA path |
| L7 | cpu | FASTER-WHISPER | CPU | CPU | on | accuracy | CPU baseline |
| L8 | nvidia-intel | FASTER-WHISPER | AUTO | AUTO | on | accuracy | `ASR_ISOLATE_ENGINES=0` — the in-process engine path |

### nuc — Intel Arc 140T iGPU + AI Boost NPU

| # | Target | Engine | ASR dev | Prep dev | UVR | Suite | What it proves |
| :-- | :--- | :--- | :--- | :--- | :-- | :--- | :--- |
| N1 | intel | INTEL-WHISPER | GPU | GPU | on | smoke | Primary Intel path; OpenVINO EP on the iGPU |
| N2 | intel | INTEL-WHISPER | GPU | NPU | on | smoke | NPU preprocessing — the NPU's only working capability |
| N3 | intel | INTEL-WHISPER | GPU | GPU | **off** | accuracy | UVR-off on Intel |
| N4 | intel | FASTER-WHISPER | AUTO | GPU | on | smoke | **Either-stage pool rule**: ASR falls to CPU, UVR runs on the iGPU, and the unit stays in the pool |
| N5 | intel | FASTER-WHISPER | AUTO | NPU | on | accuracy | Same rule with the NPU doing isolation |
| N6 | intel | INTEL-WHISPER | **NPU** | GPU | on | accuracy | `device_probe` refuses the NPU for ASR and says so; ASR runs on CPU |
| N7 | intel-xpu | OPENAI-WHISPER | GPU | GPU | on | smoke | XPU torch path (Arc-class only) |
| N8 | intel | INTEL-WHISPER | GPU | GPU | on | accuracy | `ASR_ISOLATE_PREPROCESSING=0` — in-process OpenVINO on Arc |
| N9 | cpu | FASTER-WHISPER | CPU | CPU | on | accuracy | CPU baseline |

### xubuntu — RTX 5090 (Linux)

The fastest CUDA host, so it carries the expensive tiers.

| # | Target | Engine | ASR dev | Prep dev | UVR | Suite | What it proves |
| :-- | :--- | :--- | :--- | :--- | :-- | :--- | :--- |
| X1 | nvidia | FASTER-WHISPER | CUDA | CUDA | on | **full** | All 156 language entries |
| X2 | nvidia | WHISPERX | CUDA | CUDA | on | smoke | Engine comparison point |
| X3 | nvidia | OPENAI-WHISPER | CUDA | CUDA | on | smoke | Engine comparison point |
| X4 | nvidia | FASTER-WHISPER | CUDA | CUDA | on | **longform** | Chunk boundaries, VAD across long pauses, decoder loops |
| X5 | full | FASTER-WHISPER | AUTO | AUTO | on | smoke | The default published image |
| X6 | nvidia | FASTER-WHISPER | CUDA | CPU | **off** | accuracy | UVR off with CPU preprocessing |

### win5090 — Windows 11 + WSL2 (same physical box as xubuntu)

Dual boot: **never schedule these alongside the xubuntu rows.**

| # | Target | Engine | ASR dev | Prep dev | UVR | Suite | What it proves |
| :-- | :--- | :--- | :--- | :--- | :-- | :--- | :--- |
| W1 | nvidia | FASTER-WHISPER | CUDA | AUTO | on | smoke | CUDA reaches a Linux container through WSL2 — **and capture the accelerator evidence**, which is currently a stated gap |
| W2 | — | — | — | — | — | audit only | `audit_hardware_windows.ps1` correctly reports Intel GPU/NPU as unreachable and AMD as `/dev/dxg` -> CPU |
| W3 | nvidia-whisperx | WHISPERX | CUDA | AUTO | on | smoke | Diarization image on WSL2 |
| W4 | nvidia | OPENAI-WHISPER | CUDA | AUTO | on | smoke | torch CUDA path |
| W5 | nvidia | FASTER-WHISPER | CUDA | AUTO | off | accuracy | UVR-off decode path |
| W7 | nvidia | FASTER-WHISPER | CUDA | CUDA | on | longform | 20-minute stress on the 5090 |

> **The `full` target cannot run here, and is deliberately absent.** Its override passes
> `/dev/kfd` through for the AMD half of the image, and Compose refuses to start when a
> listed device does not exist -- WSL2 exposes an AMD adapter as `/dev/dxg`, never
> `/dev/kfd`. A `full` row fails with `error gathering device information while adding
> custom device "/dev/kfd"` before a single test runs. It is the same reason the built-in
> `local` rows use `nvidia-intel`. `full` needs a native Linux host that really has an AMD
> GPU; on `xubuntu`, confirm `/dev/kfd` exists before scheduling X5.

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

## Phase 4 — Release gate

- [x] Phase 1 fully green, including the nine cold builds
- [x] All 25 rows run, each with both claims evidenced
- [x] Release notes reconciled against the actual numbers (test count, coverage, per-machine evidence)
- [x] Known Limitations still true — see the non-claims below
- [ ] Commit the staged release-prep files, then tag

Rows added after the plan was written, once it became clear the matrix proved decoding but not
device: the isolation toggles (`L2`, `L3`, `L8`, `N8`) that the runner could not express until
`remote_validate.sh` gained `--env`; `X1`-`X9`, which re-ran every preprocessing claim with
measured evidence and executed all 177 real-engine tests against the shipped tree; and
`W1`-`W4` on Windows/WSL2 for the same reason. Logs are in `docs/validation/v1.3.0/`, scrubbed
of host addresses by `scripts/scrub_validation_log.sh`.

Phase 1's nine cold builds were re-run last, after the final Dockerfile change rather than
before it: 9 of 9. That ordering matters here more than most places, since the earlier sweep
had been invalidated twice by later edits, and a stale cold-build result is indistinguishable
from a fresh one in the notes.

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

Phase 1 is about an hour, dominated by the cold builds. Phase 2 is roughly 2 h on `local`,
2 h on `nuc`, and 4 h on `xubuntu` (the `full` tier alone is ~2 h); run in parallel that is
about **4 hours wall clock**, plus the `win5090` rows which must wait for `xubuntu` to finish
because they share hardware.
