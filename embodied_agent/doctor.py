"""Environment doctor: check the machine is ready and report the device.

    python -m embodied_agent.doctor

Prints a friendly, plain-language report: Python, the required packages, and
whether your NVIDIA GPU (e.g. an RTX 4060 Ti) is visible to PyTorch -- with a
tiny GPU test so you know training will actually use it. Exit code 0 means
"ready to run".
"""
from __future__ import annotations

import importlib
import platform
import sys

OK, WARN, BAD = "✅", "⚠️", "❌"


def _check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 9)
    note = ""
    if v >= (3, 14):
        note = "  (very new — GPU/CUDA PyTorch wheels may not exist yet; " \
               "Python 3.12 is the safe choice)"
    elif not ok:
        note = "  (need 3.9+)"
    print(f"{OK if ok else BAD} Python {v.major}.{v.minor}.{v.micro}{note}")
    return ok


def _check_packages() -> bool:
    ok = True
    for name in ("numpy", "torch", "matplotlib", "yaml", "imageio"):
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "?")
            print(f"{OK} {name} {ver}")
        except Exception:
            print(f"{BAD} {name} is NOT installed  (run the installer)")
            ok = False
    return ok


def _physical_nvidia_gpus() -> list:
    """GPU names reported by the driver (nvidia-smi), independent of PyTorch."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _check_gpu() -> str:
    try:
        import torch
    except Exception:
        print(f"{BAD} PyTorch missing -- cannot check the GPU")
        return "cpu"
    print(f"   (PyTorch build: {torch.__version__})")
    build = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        cpu_build = build is None or "+cpu" in torch.__version__
        gpus = _physical_nvidia_gpus()
        if gpus:
            # a real GPU is present but PyTorch can't use it -- the usual cause
            print(f"{BAD} Found an NVIDIA GPU ({', '.join(gpus)}) but PyTorch "
                  f"cannot use it.")
            if cpu_build:
                print(f"     The installed PyTorch is the CPU-only build "
                      f"({torch.__version__}). You need the CUDA build.")
            if sys.version_info >= (3, 14):
                print(f"     Likely cause: Python {sys.version_info.major}."
                      f"{sys.version_info.minor} has no CUDA PyTorch wheels yet."
                      f"  Use Python 3.12 (see the fix below).")
            print("     Fix: install Python 3.12 from python.org, delete the "
                  ".venv folder, and re-run the installer.")
        else:
            print(f"{WARN} No CUDA GPU visible to PyTorch -- training will use "
                  f"the CPU (slower but works).")
            print("     If you do have an NVIDIA GPU, install its drivers and "
                  "the CUDA build of PyTorch (the installer does this).")
        return "cpu"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / (1024 ** 3)
    print(f"{OK} GPU: {name}  ({vram:.1f} GB VRAM, CUDA {torch.version.cuda})")
    try:  # a tiny real op to prove the GPU works
        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print(f"{OK} GPU compute test passed -- training will run on the GPU.")
    except Exception as e:
        print(f"{WARN} GPU present but a test op failed ({e}); using CPU.")
        return "cpu"
    return "cuda"


def main() -> int:
    print("=" * 60)
    print(" Embodied Agent -- environment check")
    print(f" {platform.platform()}")
    print("=" * 60)
    py = _check_python()
    pkgs = _check_packages()
    device = _check_gpu()
    print("-" * 60)
    if py and pkgs:
        where = "your GPU \U0001f680" if device == "cuda" else "the CPU"
        print(f"{OK} Ready to run on {where}.")
        print("   Next:  python -m embodied_agent.gui      (open the dashboard)")
        return 0
    print(f"{BAD} Not ready -- run the installer (install.sh / install.bat) "
          f"and try again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
