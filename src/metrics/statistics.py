from __future__ import annotations

import numpy as np


def mean_std_ci95(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0: raise ValueError("values cannot be empty")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "ci95": 1.96 * std / np.sqrt(arr.size)}


def aubc(budgets: list[float], accuracies: list[float]) -> float:
    if len(budgets) != len(accuracies) or len(budgets) < 2:
        raise ValueError("Need equal-length curves with at least two points")
    x, y = np.asarray(budgets, float), np.asarray(accuracies, float)
    order = np.argsort(x); x, y = x[order], y[order]
    span = x[-1] - x[0]
    if span <= 0: raise ValueError("Budget range must be positive")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x) / span)


def target_accuracy_saving(target: float, proposed_budget: float, baseline_budget: float) -> float:
    del target  # Target is retained in the API for auditable callers.
    if baseline_budget <= 0: raise ValueError("baseline_budget must be positive")
    return 1.0 - proposed_budget / baseline_budget
