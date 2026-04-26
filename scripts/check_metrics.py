import os
import glob


def find_metrics():
    print("Searching for Intel/NPU metrics in /sys...")

    # Intel GPU
    gpu_paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
    print(f"GPU Busy Percent Paths: {gpu_paths}")
    for p in gpu_paths:
        try:
            with open(p, 'r') as f:
                print(f"{p}: {f.read().strip()}%")
        except:
            pass

    # NPU (Newer kernels use /sys/class/accel)
    npu_paths = glob.glob("/sys/class/accel/accel*/device/utilization")
    print(f"NPU Utilization Paths: {npu_paths}")

    # Try alternate NPU paths (sometimes in drm)
    npu_alt = glob.glob("/sys/class/drm/accel*/device/utilization")
    print(f"NPU Alt Paths: {npu_alt}")


if __name__ == "__main__":
    find_metrics()
