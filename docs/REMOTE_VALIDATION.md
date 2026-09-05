# Remote Hardware Validation

Every test in this repository except the real-audio suite mocks the ASR engine, so a
broken accelerator path passes them all. Proving a path therefore requires the hardware —
and no single machine has every accelerator. This document describes validating a build
target on a second machine over SSH.

See [`docs/SETUP.md`](SETUP.md) for the image table and
`.claude/skills/hardware-verification/SKILL.md` for what "validated" means.

## One command

```bash
scripts/remote_validate.sh <user>@<host>                     # preflight + hardware audit
scripts/remote_validate.sh <user>@<host> --device NPU --full # + build, run, accuracy suite
```

The script handles key creation, access checks, the hardware audit, sync, build, startup
and the test run. It stops with an exact command for the two steps that need your password
(authorising the key, and the `docker` group). Everything below explains what it does.

## What you will be asked for

To drive a remote host I need the **IP/hostname**, the **username**, and confirmation that
the login user can run `docker` without sudo. I will also ask what hardware you expect the
machine to have, so the audit can be checked against it.

I will not ask for a password, and cannot use one. Authorise a key in your own terminal
instead:

```bash
ssh-copy-id -i ~/.ssh/whisper_remote_validation.pub <user>@<host>
```

## Bootstrapping a host: which script to run

Two axes matter — the shell on the machine *driving* the validation (the operator), and the
OS of the machine *being* validated (the target). Every combination is covered:

| Target \ Operator shell | bash (Linux / macOS) | PowerShell (Windows) |
| :--- | :--- | :--- |
| **Linux** | `scripts/setup_linux_remote.sh <user>@<host>` | `scripts\setup_linux_remote.ps1 -RemoteHost <user>@<host>` |
| **macOS** | `scripts/setup_macos_remote.sh <user>@<host>` | `scripts\setup_macos_remote.ps1 -RemoteHost <user>@<host>` |
| **Windows** | `scripts/setup_windows_remote.sh <user>@<host>` | `bash scripts/setup_windows_remote.sh <user>@<host>` * |

\* There is no PowerShell port of the Windows bootstrapper, so run the shell script through
Git Bash or WSL (`bash scripts/setup_windows_remote.sh ...`) -- PowerShell cannot execute a
`.sh` directly. The command it prints, which the footnote below describes, is what runs on
the target and is unchanged.

The Windows bootstrapper's whole job is to emit one self-contained PowerShell command
to paste on the target. That command is plain text, so it can be produced from either
operator shell; only the target needs PowerShell, and
`scripts/setup_windows_remote.ps1` is what actually runs *there*.

Each bootstrapper generates the SSH key if absent, prints exactly one block to paste,
waits for the host to come up, verifies Docker is reachable, and finishes by printing the
`remote_validate.sh` command to run next. The only manual steps are the two that need the
remote user's own password — authorising the key, and (on Linux) joining the `docker`
group.

After bootstrapping, validation itself always runs through `scripts/remote_validate.sh`.

For repeated validation, `scripts/validation_matrix.sh` accepts a seventh `transport`
column: `linux` (default), `wsl`, or `wsl:<distro>`. Its first column identifies the
physical machine, so Linux and Windows dual-boot rows must share that label; the matrix
then serializes them instead of starting two conflicting GPU runs.

## Intel NPU: not usable for this model

Verified on an Intel Core Ultra 255H (NPU arch 3720, `Intel(R) AI Boost`) on 2026-09-03,
with a current driver stack (`intel-level-zero-npu` / `intel-fw-npu` 1.32.1, OpenVINO
2026.3.1). **The NPU cannot run the Whisper OpenVINO IR this project ships.**

What happens without the guard: the NPU *builds* a `WhisperPipeline` in about 4 seconds,
the service reports healthy, the banner prints `ASR Runtime: OpenVINO (NPU)` — and every
request returns HTTP 500:

```text
L0 pfnAppendGraphExecute result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe
```

The cause is not this codebase. A stock `openvino_genai.WhisperPipeline` outside the
project fails identically on the same model directory, while the GPU returns text from it.
Compiling the encoder directly on the NPU names it outright:

```text
Upper bounds are not specified for node ... bounds are '[9223372036854775807, 1500, 20, 64]'
```

The exported IR is dynamic-shaped (`[?,?]`, `[?,?,1280]`); the NPU plugin requires static
upper bounds. Supporting it would need a statically-reshaped re-export, which is a real
piece of work and questionable value: `large-v3` is a poor fit for this class of NPU.

**What the service does now.** A probe runs before the banner prints (subprocess, so no
OpenVINO context is created in the main process). If the NPU cannot execute, `DEVICE` and
the scheduler pool are downgraded to the Intel GPU *before* anything is reported, and the
log says plainly that the request was not honoured:

```text
[System] ASR_DEVICE=NPU cannot execute this model: ...
[System] Falling back to the Intel GPU. ASR_DEVICE=NPU was NOT honoured.
```

Set `VERIFY_RUNTIME=false` to skip the probe — but then an `ASR_DEVICE=NPU` host serves
500s and the banner claims a device nothing ran on.

**Intel GPU is validated and works.** On the same machine the iGPU transcribes correctly
with UVR preprocessing, 9/9 accuracy tests passing. `--device NPU` is only worth passing
to confirm this limitation still holds on a newer driver.

## Remote platforms: what each can actually validate

The container stack is Linux. Windows and macOS hosts run it inside a VM, and that VM's
accelerator passthrough — not the host's hardware — decides what can be proven.

| Remote OS | Reachable accelerators | Notes |
| :--- | :--- | :--- |
| **Linux** | NVIDIA, AMD ROCm, Intel GPU, Intel NPU | Full support; devices map directly |
| **Windows** (Docker Desktop / WSL2) | NVIDIA CUDA only | Intel GPU/NPU are **not** exposed to Linux containers — no `/dev/dri`, no `/dev/accel`. AMD gets `/dev/dxg`, which detects the card but falls back to CPU |
| **macOS** (Docker Desktop) | none | The Linux VM has no GPU passthrough; Apple Silicon GPU and ANE are unreachable. Only the `cpu` target is meaningful |

Consequences worth stating before anyone spends a build on it:

- A **Windows** host is worth using for CUDA, and for confirming the images boot and fall
  back cleanly. It cannot validate Intel NPU or Intel GPU, whatever the machine contains —
  those need a Linux host.
- A **macOS** host validates the `cpu` target and nothing else. A green suite there says
  the software works; it says nothing about any accelerator.

### Shell differences

- **Linux / macOS** — SSH lands in a POSIX shell; commands run unchanged.
- **Windows** — SSH lands in PowerShell or `cmd`, where `uname`, `df` and the container
  invocations do not work. Run everything through WSL instead, either by SSHing directly
  into the WSL distro (simplest) or by wrapping each command:

  ```bash
  ssh <user>@<host> 'wsl -e bash -lc "<command>"'
  ```

  `scripts/remote_validate.sh --wsl` applies that wrapper automatically. Docker must be
  reachable *inside* WSL, which is the default with Docker Desktop's WSL2 integration
  enabled for that distro.

- **macOS** — `stat -c` does not exist (BSD `stat` uses `-f`), and there are no render
  nodes to read a GID from, so the Intel-specific steps are skipped entirely.

## Step 1: audit the remote, vendor-agnostically

Do not assume which accelerator the machine has — a host offered for its Intel NPU may
also carry an NVIDIA or AMD GPU, and the most valuable target may not be the one that
prompted the offer. The repo's auditor already detects every vendor and recommends a
`BUILD_TARGET`, and it can run on a host with nothing installed by piping it over:

```bash
ssh -o BatchMode=yes <user>@<host> 'bash -s -- --json' < scripts/audit_hardware.sh
```

It reports NVIDIA (plus whether the container toolkit works), AMD `/dev/kfd`, Intel render
nodes and their GID, Intel NPU `/dev/accel`, and free build space. Choose targets from
that output, not from what the owner mentioned.

## Step 2: confirm each present vendor can actually execute

Availability flags lie. Run only the checks for vendors the audit found.

**NVIDIA** — the toolkit must work inside a container, not just on the host:

```bash
# IMAGE must name the exact build under test -- the digest `docker images --digests` shows
# for it, or its commit-tagged form. `whisper-pro-asr:nvidia` is a moving tag: it is
# reassigned by every build, so probing it can report a pass for an image that is not the
# one you validated, which is the whole failure mode this document exists to prevent.
IMAGE='whisper-pro-asr@sha256:<digest of the build under test>'
ssh <user>@<host> "docker run --rm --gpus all $IMAGE python3 -c \"import ctranslate2;print('ct2 cuda devices:', ctranslate2.get_cuda_device_count())\""
```

**AMD** — a card whose gfx architecture is not in the shipped ROCm kernels still
transcribes, silently on the CPU:

```bash
# Same rule as the NVIDIA probe above: name the exact build, never the moving tag.
IMAGE='whisper-pro-asr@sha256:<digest of the build under test>'
ssh <user>@<host> "docker run --rm --device /dev/kfd --device /dev/dri $IMAGE rocm-smi --showhw"
```

**Intel GPU (XPU)** — `torch.xpu.is_available()` returns True on iGPUs that then fail real
inference, so a load-and-transcribe is the only meaningful check. Write the probe to a
file and mount it, rather than fighting nested shell quoting:

```bash
cat > /tmp/xpu_probe.py <<'EOF'
import logging, warnings, numpy as np, whisper
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
whisper.load_model("tiny", device="xpu").transcribe(np.zeros(32000, dtype="float32"))
logging.getLogger("xpu_probe").info("XPU INFERENCE OK")
EOF
ssh <user>@<host> 'cat > /tmp/xpu_probe.py' < /tmp/xpu_probe.py
IMAGE='whisper-pro-asr@sha256:<digest of the build under test>'
ssh <user>@<host> "docker run --rm --device /dev/dri:/dev/dri \
  --group-add \"\$(stat -c %g /dev/dri/renderD128)\" -u root \
  -v /tmp/xpu_probe.py:/probe.py $IMAGE python3 /probe.py"
```

**Intel NPU** — presence of `/dev/accel` is not enough; OpenVINO enumeration is
authoritative (see the NPU section below).

## Requirements for driving it remotely

| Requirement | Why |
| :--- | :--- |
| Key-based SSH working non-interactively | A password prompt never returns in an automated session |
| Login user in the `docker` group | A `sudo` prompt cannot be answered non-interactively |
| 5–25 GB free, plus ~3 GB per model format | Targets range from `cpu` (~4.9 GB) to `amd-rocm-torch` (~21.8 GB) |

Verify the whole chain in one command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 <user>@<host> 'id -nG | tr " " "\n" | grep -qx docker && echo ACCESS OK'
```

`BatchMode=yes` matters: it turns a would-be password prompt into an immediate error
rather than a hang.

Credentials are never typed on your behalf. If the host wants a password, configure key
authentication first:

```bash
# -i, so the dedicated validation key is authorised rather than whatever identity happens
# to be first in the agent. remote_validate.sh only ever offers this one key, so copying a
# different one leaves it still refusing to connect.
ssh-copy-id -i ~/.ssh/whisper_remote_validation.pub <user>@<host>
```

## What the validation actually checks

1. **Provenance** — the image is built on the remote host from the current tree, or copied
   with its tag recorded. An image from another commit proves nothing about the change.
2. **What loaded** — read from the startup banner (`ASR Engine`, `Resource Pool`,
   `OpenVINO devices`, image edition), not from an HTTP 200. A successful response is
   entirely compatible with a silent CPU fallback.
3. **That the accelerator ran** — sampled *during* inference with `intel_gpu_top`,
   `nvidia-smi --query-compute-apps` or `rocm-smi`. A correct transcript alone does not
   prove it; CPU fallback transcribes correctly too.
4. **Which process held the device** — engines run in isolated workers, so the device
   holder is normally a spawned child, not the API process.

## Intel NPU

- Pass **both** `/dev/dri` and `/dev/accel`, plus the host render GID via `group_add`.
- Set `ASR_DEVICE=NPU` explicitly. `AUTO` ranks CUDA > AMD > GPU > NPU, so a machine with
  an iGPU will never select the NPU on its own.
- OpenVINO enumeration is authoritative: if the banner's `OpenVINO devices` lists no
  `NPU`, the device node alone does not make it usable and no NPU claim can be made.

## Reporting

A remote result carries its provenance: host, image tag, source commit. "Decoding worked"
and "the accelerator was used" are reported separately, each with evidence. If the
hardware turns out not to support the path, that is the finding — not a run to repeat
until it looks green.

## Long-form defect status after the repetition fix (RTX 5090, 2026-09-03)

`condition_on_previous_text=False` + `no_repeat_ngram_size=3` were added to
`FasterWhisperEngine` to break the recorded decoder-loop defect. **Neither setting is the
fix it was taken for**, and the numbers below were measured on a truncated transcript --
see the two corrections that follow. `no_repeat_ngram_size` blocks a token run repeating
*inside* one 30-second chunk, while these repeats are one sentence recurring *across*
chunks, which n-gram blocking cannot see; the later sweep found the count unmoved at 11-12
under every decode parameter tried. Measured on the 20-minute clip on an RTX 5090:

| property | before | after |
| --- | --- | --- |
| worst sentence repetition | 18x | 7x |
| coverage (last segment end / duration) | stops short | 765s / 1203s |
| throughput | — | 93s wall for 1203s audio (RTF 0.077) |

> **Both numbers in that table are wrong, and the "coverage" defect was never real.**
> See the section below. They are kept here only so the correction has something to point
> at; do not quote them.

### Correction: the coverage defect was a response-truncation bug (2026-09-04)

Instrumenting `_fill_language_gaps` and the pre-post-processing point on the 5090 showed
the transcript was complete inside the pipeline and lost on the way out:

| stage | segments | max segment end |
| --- | --- | --- |
| after gap-fill | 169 | 1202.28s |
| pre-post-processing | 169 | 1202.28s |
| final HTTP response | **100** | **765.57s** |

The cause is `history_manager._truncate_large_segments`, which keeps the *history file*
small by clipping a result to 100 segments. It mutated `result["segments"]` in place, and
`log_completed_task` shallow-copies `task_data` only afterwards — so it clipped the very
dict the request handler returns. **Every response over 100 segments was silently
truncated for the client.** Fixed by replacing the result dict instead of mutating it
(`tests/monitoring/test_history_manager.py::TestTruncateLargeSegments`).

Measured on the same clip with the fix, via a `modules/` bind-mount over the image:

| property | before fix | after fix |
| --- | --- | --- |
| segments returned | 100 | **169** |
| last segment end | 765.57s | **1202.28s** of 1203.05s |
| coverage ratio | 0.636 | **0.9994** (gate is 0.90) |
| worst sentence repetition | 7x | **11x** |

Two consequences worth stating plainly:

- **`coverage` is not a model defect.** Gap-fill was working the whole time — it found all
  72 uncovered gaps out to 1202.9s. The entry should leave the manifest's `known_defects`
  once a test run confirms it XPASSes.
- **`repetition` is about twice as bad as recorded, not better.** The truncation was
  discarding 69 tail segments, and repeated sentences sat among them, so every long-audio
  repetition count this project has taken — including the `18x -> 7x` improvement claimed
  above — was measured on a clipped transcript. The honest figure against a threshold of 5
  is **11x**. `condition_on_previous_text=False` + `no_repeat_ngram_size=3` may still have
  helped, but the size of that win is unmeasured; the 18x baseline was clipped too. The
  sweep below then showed no decode parameter moves it at all, so treat the pair as
  unproven rather than as the mitigation.

Note that `repetition_penalty` -- which *is* effective, and is what `intel_engine.py` sets
for the same class of runaway decoding -- is an OpenVINO GenAI setting on the Intel path,
not a CTranslate2 one. It is not interchangeable with the two settings above, and OpenVINO
GenAI rejects it outright under beam search.

An earlier hypothesis recorded here — that VAD saw no speech past 765s — was also measured
and disproven (173 regions, 295s of speech past the cutoff, last region ending 1202.87s),
as was a follow-up guess that the 5090 image predated gap-fill (it did not; the image
contained all three fixes).

## Decoder mitigations that do not work (RTX 5090, 2026-09-04)

Six mitigations swept against the long-form clip in one process, so the baseline and
the candidates share a machine, a model load and a warm cache. Scored with the suite's
own metrics. Run directly against the engine (no UVR), which is why the baseline's
quiet-window count is worse than the full pipeline's: **UVR preprocessing is already
removing about half the quiet-window hallucination on its own.**

| config | worst repeats | noisy quiet windows (of 25) | last segment end | wall |
| --- | --- | --- | --- | --- |
| current settings | 12 | 14 | 1202.2s | 39.4s |
| `vad_parameters.threshold=0.6` | 11 | 9 | 1202.3s | 15.4s |
| `log_prob_threshold=-0.6` | 12 | 9 | 1202.2s | 34.9s |
| `no_speech_threshold=0.4` | 12 | 14 | 1202.2s | 14.8s |
| `compression_ratio_threshold=2.0` | 12 | 15 | 1201.9s | 19.6s |
| `hallucination_silence_threshold=2.0` | 12 | **8** | 1185.4s | 15.0s |

**Do not re-run this sweep expecting a different answer.** No decode parameter moves
the repetition count; the spread is 11-12 everywhere. `hallucination_silence_threshold`
is the only one that helps quiet windows materially (14 -> 8) and it costs coverage
(1202s -> 1185s) and needs `word_timestamps=True`, so it is not a free win.

The reason nothing moved: `no_repeat_ngram_size` blocks a token run repeating *inside*
one 30-second chunk, but these repeats are one sentence appearing across chunks spread
over the whole clip. Cross-chunk repetition is invisible to n-gram blocking. And the
repeats are not a loop at all -- of 11, zero land in real Spanish windows, 3 in silence
and 8 in other languages' windows.

So the remaining long-form defect is one behaviour, not three: **the model locks onto a
single detected language and emits that language's text over audio that is not in it.**
It surfaces as `windows` (78 of 118 windows below threshold, with the detected language
the only clean one) and `quiet` (8 of 25 windows). A real fix needs per-segment language
verification -- detecting that a segment's audio does not match the forced language and
re-decoding it -- not another decode-parameter tweak. Gap-fill cannot reach it either:
gap-fill re-transcribes speech no segment covers, and these regions are covered, just
wrongly.

## Intel NUC long-form matrix (20-minute clip, 2026-09-04)

Both device arrangements, every engine the `intel` image ships. Pool was `GPU, NPU`
throughout, confirming the NPU stays schedulable while ASR falls back appropriately.

| engine | prep | ASR ran on | UVR ran on | result |
| --- | --- | --- | --- | --- |
| FASTER-WHISPER | NPU | CPU (int8) | `Intel(R) AI Boost` | 2 failed, 4 xfailed (15:00) |
| OPENAI-WHISPER | NPU | CPU (int8) | `Intel(R) AI Boost` | 2 failed, 4 xfailed (15:00) |
| INTEL-WHISPER | NPU | OpenVINO (GPU) | `Intel(R) AI Boost` | **1 passed**, 4 xfailed (08:42) |
| INTEL-WHISPER | GPU | OpenVINO (GPU) | `Intel(R) Graphics` (iGPU) | **1 passed**, 4 xfailed (08:36) |
| OPENAI-WHISPER | GPU | CPU (int8) | `Intel(R) Graphics` (iGPU) | 2 failed, 4 xfailed (15:00) |
| FASTER-WHISPER | GPU | CPU (int8) | `Intel(R) Graphics` (iGPU) | 2 failed, 4 xfailed (15:01) |

**The two failures on every CPU-ASR row are the expected consequence of CPU decoding, not
new defects.** `LONG_ASR_MAX_RTF` is 1.0 -- 20 minutes of audio must decode in under 20
minutes -- and CPU decoding of `large-v3` cannot meet it. Every CPU row stopped at the
900s ceiling exactly, i.e. it was cut off rather than finishing slowly.
`test_multiple_languages_are_recognized_across_the_clip` then fails as a knock-on: a
truncated run never reaches the later languages.

Only INTEL-WHISPER clears the budget (~8:40 for 20 minutes of audio), on either
preprocessing device. That is the clean demonstration that the Intel GPU path is
genuinely accelerated rather than silently falling back: same clip, same machine, roughly
half the wall clock of the CPU rows and inside the budget instead of hitting the ceiling.

UVR on the NPU and UVR on the iGPU produce identical pass/fail outcomes, so both
preprocessing devices are validated.

## Hybrid NVIDIA + Intel: the Intel iGPU is never used (2026-09-04)

On a laptop with an RTX 3080 and an `Intel(R) UHD Graphics` iGPU, running the
`nvidia-intel` image, the Intel device is detected for display but never enters the
resource pool:

```text
Preprocess Device : Intel(R) UHD Graphics (iGPU)     <- startup banner
Resource Pool     : cuda:0                           <- same banner, next line
PREPROCESS_DEVICE = GPU
HARDWARE_UNITS    = [('CUDA', 'cuda:0', 'NVIDIA GPU 0')]
```

With `ASR_PREPROCESS_DEVICE=GPU` set explicitly, UVR still runs on CUDA:

```text
[UVR] Starting vocal isolation on NVIDIA GPU 0 [ONNX: CUDAExecutionProvider]
```

Reproduced with `ASR_DEVICE=CUDA` and with `ASR_DEVICE=AUTO`; the unit list is the same
either way. The container can see the hardware -- OpenVINO inside it reports
`['CPU', 'GPU.0', 'GPU.1']` and `/dev/dri` is mapped -- so this is unit registration, not
device access.

The mechanism: `_shared_preprocessor_for_type("GPU")` looks for a unit of that type in
`config.HARDWARE_UNITS`, finds none, and falls back to `preprocessing.create_manager()`
with no assigned unit, which resolves to CUDA. Nothing raises and nothing warns, so the
request looks entirely successful.

**What this means in practice.** On a hybrid host the documented split (UVR on the Intel
iGPU, ASR on the NVIDIA GPU) does not happen -- both stages land on CUDA. That is not a
disaster on its own, since CUDA UVR is fast, but the banner claims otherwise, so any
measurement attributing preprocessing to the iGPU on such a host is wrong.

Not yet diagnosed: whether the non-registration is deliberate (this is an older UHD part,
not Arc) or a gap in detection when an NVIDIA GPU is present. The NUC, which has no
NVIDIA GPU, registers `GPU, NPU` correctly, so the ranking logic around a present CUDA
device is the place to look first.

### Follow-on: the ONNX Runtime variant follows the ASR device, not the preprocess device

With the unit-registration fix above, `ASR_PREPROCESS_DEVICE=GPU` now reaches the Intel
iGPU on a hybrid host -- the log names it and the request is honoured:

```text
[System] Initializing UVR ... on Intel(R) UHD Graphics (iGPU)... [ONNX provider: CPUExecutionProvider]
[UVR] Isolation complete on Intel(R) UHD Graphics (iGPU) (Duration: 00:00:13 | Speed: 0.48x)
```

Note the provider: **CPUExecutionProvider**, not OpenVINO. `/app/libs/intel` does ship
`OpenVINOExecutionProvider` (verified inside the container), but it is never loaded here.
`bootstrap._check_nvidia_library` runs before the Intel check and
`_should_use_nvidia_path` matches on `device == "auto"` alone whenever NVIDIA hardware is
present, so a hybrid host with `ASR_DEVICE=AUTO` always gets `onnxruntime-gpu` -- which has
no OpenVINO provider, so the Intel unit degrades to CPU.

Measured cost on an RTX 3080 + UHD Graphics laptop: UVR at 2.40x on CUDA against 0.48x on
the "Intel" path, i.e. **five times slower while naming the accelerator** -- the failure
shape this project has been bitten by before.

So the routing is now correct and the acceleration is not. Worth noting that the ASR
engines do not use ONNX Runtime at all (CTranslate2 and OpenVINO GenAI do their own
thing); only UVR and VAD do. That suggests the variant should follow
`ASR_PREPROCESS_DEVICE` rather than `ASR_DEVICE`, but changing that ordering affects every
image and host and was not attempted here.

Until then, on a hybrid NVIDIA+Intel host leave `ASR_PREPROCESS_DEVICE` unset: CUDA UVR is
much faster than a CPU-bound Intel path, and `AUTO` co-locates it with the ASR unit.

## Engine comparison, controlled (RTX 5090, 2026-09-04)

One machine, the `full` image, the same corpus for every engine: 24 single-language clips
scored on word overlap, plus the 20-minute long-form clip. Only `ASR_ENGINE` changed, with
a container recreate between runs. This is the first side-by-side this project has had --
every earlier engine claim was a by-product of defect hunting on different machines.

| metric | FASTER-WHISPER | WHISPERX | OPENAI-WHISPER |
| --- | ---: | ---: | ---: |
| mean word overlap (24 clips) | 0.8675 | **0.8719** | 0.8716 |
| language detected correctly | 24/24 | 24/24 | 24/24 |
| long-form RTF | 0.0299 | **0.0134** | 0.1179 |
| long-form wall clock | 36.0s | **16.2s** | 141.8s |
| 24 short clips, wall clock | **26.1s** | 34.5s | 405.5s |
| hallucinated quiet windows (of 25) | 8 | **0** | 11 |
| window misses (of 118) | **56** | 109 | 89 |
| worst repetition | 8 | **6** | 20 |
| coverage ratio | 0.9993 | 0.9996 | 0.9993 |
| segments emitted | 229 | 56 | 259 |

**Accuracy is a tie.** 0.8675 / 0.8719 / 0.8716 is a 0.5% spread, and all three identify
the language of all 24 clips. Any claim that one Whisper engine is "more accurate" than
another on clean speech is not measuring anything real. What actually separates them is
speed and failure mode.

**WHISPERX is the fast one, not the heavy one.** 2.2x faster than FASTER-WHISPER on the
20-minute clip and the only engine with *zero* hallucinated quiet windows -- its VAD-batched
chunking never hands silence to the decoder. The same batching is why it misses 109 of 118
multilingual windows and emits 56 coarse segments against 229: it commits hard to one
language and to long spans.

**FASTER-WHISPER is the multilingual one.** 56 window misses against 89 and 109, with fine
segment granularity. That is why it stays the default here -- multilingual audio is this
project's purpose -- not because it wins on accuracy or speed.

**OPENAI-WHISPER loses on every axis**: 15x slower on short clips (405s against 26s), the
most hallucination (11/25) and the worst looping (20 repeats). Its only reason to exist is
the XPU path on Intel GPUs, and even there OpenVINO beats it (see the Intel table below).

None of the three populates per-segment languages on its own; `segment_languages` comes
from the pipeline's gap-fill, not from the engine.

### Choosing

- **Multilingual, or unknown content** -> FASTER-WHISPER (the default).
- **Monolingual, and speed or clean silence matters** -> WHISPERX. Subtitles benefit twice:
  no invented text in pauses, and diarization is available.
- **OPENAI-WHISPER** -> only for XPU torch on an Intel GPU.
