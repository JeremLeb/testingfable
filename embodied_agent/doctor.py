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
    print(f"{OK if ok else BAD} Python {v.major}.{v.minor}.{v.micro}"
          f"{'' if ok else '  (need 3.9+)'}")
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


def _check_gpu() -> str:
    try:
        import torch
    except Exception:
        print(f"{BAD} PyTorch missing -- cannot check the GPU")
        return "cpu"
    if not torch.cuda.is_available():
        print(f"{WARN} No CUDA GPU visible to PyTorch -- training will use the "
              f"CPU (slower but works).")
        print("     If you have an NVIDIA GPU, install the CUDA build of "
              "PyTorch (the installer does this for you).")
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
