from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def environment_snapshot() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            from importlib.metadata import version as get_version
            return get_version(name)
        except Exception:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    gpu = None
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        pass
    return {"git_commit": commit, "python": sys.version, "platform": platform.platform(),
            "torch": version("torch"), "transformers": version("transformers"),
            "datasets": version("datasets"), "cuda_available": cuda_available,
            "gpu_name": gpu, "hostname": socket.gethostname()}


def create_run_directory(root: str | Path, label: str, config: dict[str, Any]) -> Path:
    run_dir = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2) + "\n", encoding="utf-8")
    return run_dir


def configure_logging(path: str | Path) -> logging.Logger:
    logger = logging.getLogger("gsc_token")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger

