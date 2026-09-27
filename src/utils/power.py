"""Best-effort GPU energy accounting, with two independent backends.

NVML is preferred because it is an in-process query. When it is unavailable the
sampler falls back to ``nvidia-smi``, which is slower but keeps the energy axis
of Section VI populated instead of null. Neither backend is available on every
machine, so every caller must tolerate ``None``.
"""
from __future__ import annotations

import os
import shutil
import subprocess


class PowerSampler:
    def __init__(self) -> None:
        self._nvml, self._handle = None, None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml, self._handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml, self._handle = None, None
        self._backend = "nvml" if self._handle is not None else None
        if self._backend is None and shutil.which("nvidia-smi"):
            self._backend = "nvidia-smi"

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend(self):
        return self._backend

    def watts(self):
        if self._backend == "nvml":
            try:
                return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except Exception:
                return None
        if self._backend == "nvidia-smi":
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                     "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
                    capture_output=True, text=True, timeout=10, check=True).stdout.strip()
                return float(out.splitlines()[0])
            except Exception:
                return None
        return None

    @staticmethod
    def joules(start_watts, end_watts, seconds):
        if start_watts is None or end_watts is None:
            return None
        return (start_watts + end_watts) / 2.0 * seconds
