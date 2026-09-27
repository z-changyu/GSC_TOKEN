#!/usr/bin/env python
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def package(name):
    try: return version(name)
    except PackageNotFoundError: return None


def main():
    try: smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.strip()
    except OSError: smi = "unavailable"
    import torch
    report = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "transformers": package("transformers"), "datasets": package("datasets"),
              "nvidia_smi": smi or "failed/no output"}
    print(json.dumps(report, indent=2))
    if not torch.cuda.is_available(): print("WARN: CUDA is unavailable; GPU model checks cannot pass.")


if __name__ == "__main__": main()

