# Remote Hardware Validation Skill

This skill documents how to validate accelerator paths on a host other than the working
machine, over SSH. It complements `runtime/intel_hardware_inference_skill.md` and
`runtime/amd_hardware_inference_skill.md`, which describe the runtime behaviour itself;
this one covers executing that validation elsewhere without producing a false green.

## When To Use

- The target accelerator does not exist locally (Intel NPU, Intel Arc, AMD Radeon).
- A defect reproduces only on other silicon.
- Before marking any image or device hardware-validated when its accelerator is absent
  from the working machine.

## Access Preconditions

- Key-based SSH must already work non-interactively. Verify with
  `ssh -o BatchMode=yes -o ConnectTimeout=10 <host> true`. `BatchMode=yes` converts a
  password prompt into an immediate failure rather than an indefinite hang, which is the
  difference between a clear error and a stalled session.
- The login user must be in the `docker` group; a `sudo` password prompt cannot be
  answered non-interactively.
- Never transmit or type credentials on the user's behalf. If a host needs a password,
  stop and ask the owner to configure key auth.
- Disk: shipped targets range from ~4.9 GB (`cpu`) to ~21.8 GB (`amd-rocm-torch`), plus
  roughly 3 GB of `model_cache` per weight format the selected engine provisions.

## Validation Rules

1. **Provenance beats convenience.** Build the target on the remote host from the current
   tree, or copy a known image and record its tag. A green run against an image built from
   another commit proves nothing about the change under test.
2. **Read the banner, not the HTTP status.** `ASR Engine`, `ASR Engine Source`,
   `Resource Pool`, `OpenVINO devices` and the image edition establish what actually
   loaded. A 200 response is compatible with a full CPU fallback.
3. **Sample the accelerator during inference.** `intel_gpu_top -J`, `nvidia-smi
   --query-compute-apps`, `rocm-smi --showuse`. An idle device during a passing test means
   the accelerator was not exercised.
4. **Attribute work to the right process.** Engines run in isolated worker processes, so
   the device holder is normally a `multiprocessing` spawn child. Map container PID to host
   PID through the `NSpid` field of `/proc/<host-pid>/status`.

## Intel NPU Notes

- Pass `/dev/dri` **and** `/dev/accel`, with the host render GID via `group_add`.
- OpenVINO enumeration is authoritative: a present `/dev/accel` node with no `NPU` entry in
  `OpenVINO devices` is not a schedulable unit, and no NPU claim may be made from it.
- `ASR_DEVICE=NPU` must be explicit. AUTO ranks CUDA > AMD > GPU > NPU, so on a mixed host
  AUTO will never select the NPU.

## Known Platform Limits Worth Checking First

- **`torch.xpu.is_available()` is not proof.** It reports True on iGPUs that then fail
  real inference with `level_zero backend failed with error: 45
  (UR_RESULT_ERROR_INVALID_ARGUMENT)`. Verify with an actual `whisper.load_model(...,
  device="xpu")` plus a transcribe call, not with the availability flag.
- **OpenVINO GPU/NPU cannot run inside a `multiprocessing` spawn child.** Its ONNX Runtime
  session creation segfaults there, reliably. OpenVINO preprocessing therefore runs
  in-process while CUDA/AMD preprocessing is isolated; do not "fix" a remote crash by
  re-enabling isolation for OpenVINO devices.
- **Never set `ONEAPI_DEVICE_SELECTOR` to an empty string.** Intel's SYCL runtime rejects
  an empty value with an uncaught C++ exception and aborts the process; pass the variable
  through only when the host actually sets it.

## Reporting Standard

State the host, image tag, and source commit. Separate "decoding worked" from "the
accelerator was used", each with its own evidence. If the remote hardware does not support
the path, report that as the finding rather than as an inconclusive run.
