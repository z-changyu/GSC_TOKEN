# Environment record

Recorded 2026-08-22 in Conda environment `GSC_Token`.

| Item | Observed status |
|---|---|
| OS | Linux 5.15, x86_64 |
| Python | 3.10.20 |
| Conda | 23.9.0 |
| `nvcc` | CUDA toolkit 12.1.105 |
| NVIDIA driver | CUDA driver API 12.2 (`12020`) inferred from PyTorch error; `nvidia-smi` cannot communicate with the driver |
| GPU | Unavailable/not identifiable while driver is inaccessible |
| Initial PyTorch | 2.13.0+cu130 (incompatible with driver 12.2) |
| Target PyTorch | stable 2.5.1+cu121 with torchvision 0.20.1+cu121 |
| Transformers | Not installed: official PyPI connection stalled/interrupted |
| Datasets | 5.0.1 |
| CUDA available | No |

Compatibility decision: CUDA 12.1 wheels are compatible with the locally installed toolkit and a CUDA 12.2-capable driver. Do not change the system toolkit or driver from this environment. The incompatible cu130 packages must not be used for experiments.

Commands:

```bash
conda create -n GSC_Token python=3.10 -y
conda activate GSC_Token
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
python -m pip install --index-url https://pypi.org/simple \
  -r requirements_GSC_Token.txt
python scripts/verify_environment.py
```

Installation blocker: the official torch wheel was cached, but `nvidia-cuda-nvrtc-cu12` repeatedly suffered incomplete downloads; the official PyPI Transformers request also stalled. Retry when network connectivity is stable. Do not treat this YAML as proof that every target package is currently installed; `verify_environment.py` is authoritative.

