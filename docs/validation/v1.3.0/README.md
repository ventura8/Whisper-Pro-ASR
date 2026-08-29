# v1.3.0 isolation evidence

Raw container logs behind the isolation claims in the release notes and in the
"Correction to the correction" section of `docs/REMOTE_VALIDATION.md`. Kept because those
claims are about which device *actually* served a request, and a transcript alone can never
show that -- only the runtime's own device lines and `/status` can.

Rows prefixed `L` ran on the local RTX 3080 + Intel UHD laptop, the only host available with
both a CUDA GPU and an Intel iGPU. `N` rows ran on the Intel NUC, `X` rows on the RTX 5090.

Remote transcripts are passed through `scripts/scrub_validation_log.sh` before landing here:
it keeps the evidence sections and replaces the SSH target, remote hostname and address with
`<user>@<host>`. Host addresses must never reach a commit, and "remember to scrub it" is not
a control.

| log | what it varies | why it exists |
| --- | --- | --- |
| `L3_isolate_preprocessing_on.log` | `ASR_ISOLATE_PREPROCESSING=1` | First run that exposed the guard never firing: UVR at 0.90x on the CPU while every log line named the iGPU. |
| `L2_recheck_default_isolation.log` | shipped defaults, before the fix | Confirms the same CPU fallback with no environment overrides at all. |
| `L2_after_fix.log` | shipped defaults, after the fix | Source rebuilt; UVR moves to CUDA. |
| `L2_after_fix_rebuilt.log` | shipped defaults, image rebuilt | The measurement quoted in the docs: `Speed: 8.34x`, `UVR=CUDA measured=True`. |
| `L8_isolate_engines_off.log` | `ASR_ISOLATE_ENGINES=0` | In-process engines. Exposed language detection being handed a path (`'str' object has no attribute 'dtype'`), a 500 on every gap-fill that the isolated default hides. |

| `N8_isolate_preprocessing_off.log` | `ASR_ISOLATE_PREPROCESSING=0`, Intel NUC | In-process preprocessing on Intel: `ASR=Intel GPU measured=True | UVR=Intel GPU measured=True`, 4.7x. |
| `X1_whisperx_uvr.log` | RTX 5090, `nvidia-whisperx`, WHISPERX | `ASR=CUDA / UVR=CUDA`, both measured; UVR 12-16x. |
| `X2_openai_uvr.log` | RTX 5090, `nvidia`, OPENAI-WHISPER | `ASR=CUDA / UVR=CUDA`, both measured; UVR ~15.8x. |
| `X3_faster_uvr.log` | RTX 5090, `nvidia`, FASTER-WHISPER | `ASR=CUDA / UVR=CUDA`, both measured; UVR ~15.9x. |
| `X4_full_auto_uvr.log` | RTX 5090 + Radeon iGPU, `full`, AUTO | Two units. See the caveat below -- this row is why the completion log now names the provider. |

The plan's xubuntu WHISPERX row names the `nvidia` target; X1 ran against `nvidia-whisperx`,
which is the target that carries WhisperX and was already built on that host. Recorded rather
than substituted silently.

## Full real-engine coverage (X5-X8)

The 177 `real_asr`-marked tests cannot run in the gate container -- no weights, no GPU, no
live service -- so they run here. No single tier covers them: `full` leaves the six long-form
tests skipped because they are gated behind `RUN_GPU_LONG_ASR=1`, which only the `longform`
tier sets. All four rows below ran against the same commit, each reporting
`ASR=CUDA / UVR=CUDA` with both measured.

| log | tier | result |
| --- | --- | --- |
| `X5_full_matrix.log` | `full` | 158 passed, 1 xfailed, 3 xpassed, 6 skipped (the long-form six) |
| `X6_accuracy.log` | `accuracy` | 9 passed |
| `X7_longform.log` | `longform`, FASTER-WHISPER | 3 passed, 2 xfailed, 1 skipped -- the skip is what prompted X8. |
| `X8_longform_whisperx.log` | `longform`, WHISPERX | Skipped the same test, disproving the assumption that only FASTER-WHISPER lacked the capability. |
| `X9_longform_language_gap.log` | `longform`, FASTER-WHISPER | After the fix: **3 passed, 3 xfailed, 0 skipped**. |

The last three rows are one investigation. `test_multiple_languages_are_recognized_across_the_clip`
called `pytest.skip` whenever a segment carried no language, so it had never executed on any
engine while still being counted in the matrix. X8 ruled out an engine-specific cause, and the
installed faster-whisper `Segment` was confirmed to have no `language` field at all. It is now a
manifest-recorded known defect: visible as an XFAIL with its evidence attached, and it becomes an
XPASS by itself if per-segment language is ever exposed.

The two `xfailed` are the known long-form defects -- window misses and hallucinated speech in
quiet passages -- tracked in the suite rather than hidden. The three `xpassed` are
engine-dependent entries whose manifest notes record that removing them once already had to be
reverted.

## Windows / WSL2 (W1-W4)

Same physical box as the `X` rows, booted into Windows 11 and driven through WSL2. Re-run
because the earlier Windows pass recorded only the startup banner; these carry the measured
figure, and they are also the first rows to show the completion log's new provider field --
`Isolation complete on NVIDIA GPU 0 [ONNX: CUDAExecutionProvider]`, so a CPU fallback could no
longer read as a GPU run.

| log | target / engine | tier | result |
| --- | --- | --- | --- |
| `W1_whisperx_smoke.log` | `nvidia-whisperx` / WHISPERX | smoke | passed, UVR ~14.7x |
| `W2_openai_smoke.log` | `nvidia` / OPENAI-WHISPER | smoke | passed, UVR ~14.6x |
| `W3_faster_accuracy.log` | `nvidia` / FASTER-WHISPER | accuracy, separation off | passed, UVR ~33x |
| `W4_faster_longform.log` | `nvidia` / FASTER-WHISPER | longform | 3 passed, 3 xfailed, 0 skipped; UVR 40.5x on the 20-minute clip |

All four report `ASR=CUDA measured=True | UVR=CUDA measured=True`.

A first attempt failed all four in preflight: after the reboot the Windows clock was three
hours behind UTC, WSL inherited it, and apt refused every repository with "Release file is not
valid yet". Recorded because it costs an hour to diagnose from inside a failed build, and the
preflight check that named it is the reason it cost minutes instead.

## The X4 caveat

That host has an AMD Granite Ridge integrated Radeon (`1002:13c0`) alongside the RTX 5090, so
the `full` image built a second unit and the row exercised both. Two things in it must not be
read as acceleration:

- The NVIDIA unit's `/status` says `uvr_execution: {device: CPU, accelerated: false,
  fallback: true, measured: true}` -- correct -- while the log line for the same unit read
  `Isolation complete on NVIDIA GPU 0`, because that line named the *unit* the task held
  rather than the device that ran it. The line now carries the ONNX provider too.
- The AMD unit reports `UVR = AMD GPU, measured: true, fallback: false` at 3.06x on a
  `gfx1036`-class part the image ships no kernels for. This is the second host to produce that
  exact false positive; see Known Limitations in the release notes.

Each file starts with the row name, the image target, the engine and device requested, the
environment overrides, and the commit the container was built from.

## Why the `measured=` column exists

The startup banner names the device that was *requested*. `/status` names the device an
inference actually touched, and the two disagree exactly when something has silently fallen
back -- which is the failure this project keeps hitting. Until 2026-09-08 `remote_validate.sh`
recorded only the banner, so every remote row proved decoding and suite pass but proved
nothing about which device served preprocessing. A row whose accelerator carries
`measured=False` has not validated that accelerator.
