# GSC Token

Research scaffold for **Pragmatics-Aware Token Selection and Scheduling for Generative Semantic Communication over Time-Varying Channels**.

The current repository is a first-stage, configuration-driven pilot scaffold. It preserves the paper's token-level definitions in separate modules and explicitly gates model experiments when genuine attention, embedding gradients, or hardware support cannot be verified. It does not claim to reproduce the old BLIP-2/Flan-T5 results.

## Quick start

```bash
conda activate GSC_Token
export PYTHONPATH="$PWD"
python scripts/verify_environment.py
bash scripts/download_datasets.sh
python scripts/build_manifests.py --dataset scienceqa --split validation \
  --num-samples 150 --seed 2026 --output data/manifests/scienceqa_val.jsonl
python scripts/inspect_datasets.py --dataset scienceqa --split validation
python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python scripts/test_model_loading.py --gpu 0 --attention-implementation eager
CUDA_VISIBLE_DEVICES=0 python scripts/run_pilot.py --config configs/experiments/pilot_scienceqa.yaml
```

Read [the experiment protocol](docs/experiment_protocol.md), [metric definitions](docs/metrics_spec.md), and [method-critical open questions](docs/open_questions.md) before running models. Dataset downloads and model weights are intentionally not committed.

## Current gate

The host NVIDIA driver is inaccessible to `nvidia-smi`, and the pre-existing cu130 PyTorch build is incompatible with the detected CUDA 12.2 driver. Official dependency downloads were interrupted. Consequently dataset/model/pilot outputs are correctly marked blocked rather than synthesized.
