#!/usr/bin/env python
"""Section VI-B-6: complexity and optimality.

Two independent parts.

**Optimality.** Proposition 1 (Eq. 64) claims ``F(y_alg) >= F(y*) / (2(1+kappa))``
with ``kappa = beta d_M / 2``. The claim is checked against the *exact* optimum
obtained by enumerating every ``(|R|+1)^N`` assignment, on instances whose
coupling matrix is rescaled to hit a prescribed ``kappa`` so that the bound can
be probed as the weak-coupling regime tightens.

**Complexity.** Eq. (68) predicts the scheduler cost as
``O(L_c H N_k^2 + C_B + N_k |R|^2 log(N_k |R|) + T_max N_k^2 |R|)``. The last
three terms belong to this implementation and are timed here; the attention and
backward terms are reported separately by the pilot because they live in the
model forward pass, not in the scheduler.

Everything runs on the CPU, so the GPU stays free for the accuracy experiments.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pragmatic.utility import coupling_regime
from src.scheduling import certified_rounding, refine, solve_base_lp
from src.scheduling.exhaustive import exhaustive_optimum
from src.scheduling.problem import SchedulingProblem
from src.utils.io import atomic_json
from src.utils.logging import configure_logging, create_run_directory
from src.utils.seed import set_global_seed

LEVEL_POOL = (48.0, 32.0, 16.0)


def make_instance(num_tokens, num_levels, beta, kappa, seed):
    """Random instance whose coupling is rescaled to the requested kappa."""
    generator = torch.Generator().manual_seed(seed)
    utility = torch.rand(num_tokens, generator=generator)
    coupling = torch.rand(num_tokens, num_tokens, generator=generator).triu(1)
    if kappa > 0 and beta > 0:
        regime = coupling_regime(coupling, beta)
        if regime["d_M"] <= 0:
            raise ValueError("degenerate coupling matrix")
        coupling = coupling * ((2.0 * kappa / beta) / regime["d_M"])
    elif kappa == 0 or beta == 0:
        coupling = torch.zeros_like(coupling)
    q = torch.linspace(0.9, 0.5, num_levels)
    rates = torch.tensor(LEVEL_POOL[:num_levels])
    # A budget that makes the choice non-trivial: roughly a third of the tokens
    # at the middle protection level.
    budget = float(rates[len(rates) // 2]) * num_tokens / 3.0
    return SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta,
                             q=q, rates=rates, budget=budget)


def algorithm1_value(problem, t_max):
    lp = solve_base_lp(problem)
    rounded = certified_rounding(problem, lp)
    if t_max <= 0:
        return rounded.utility, rounded.level, lp.lp_value
    refined = refine(problem, rounded.level, t_max=t_max)
    return refined.utility, refined.level, lp.lp_value


def optimality_sweep(args, logger):
    rows = []
    for num_tokens in args.optimality_tokens:
        for num_levels in args.optimality_levels:
            for beta in args.beta_values:
                for kappa in args.kappa_values:
                    if kappa > 0 and beta <= 0:
                        continue
                    ratios, bounds = [], []
                    for instance_index in range(args.instances):
                        seed = args.seed + instance_index
                        problem = make_instance(num_tokens, num_levels, beta, kappa, seed)
                        value, _, lp_value = algorithm1_value(problem, args.t_max)
                        optimum = exhaustive_optimum(problem)["value"]
                        if optimum <= 0:
                            continue
                        ratios.append(value / optimum)
                        bounds.append(1.0 / (2.0 * (1.0 + kappa)))
                        rows.append({
                            "num_tokens": num_tokens, "num_levels": num_levels,
                            "beta": beta, "kappa": kappa, "instance": instance_index,
                            "algorithm_value": value, "lp_value": lp_value,
                            "exhaustive_optimum": optimum,
                            "optimality_ratio": value / optimum,
                            "proposition_bound": 1.0 / (2.0 * (1.0 + kappa))})
                    if ratios:
                        logger.info("optimality N=%d |R|=%d beta=%.2f kappa=%.2f "
                                    "min_ratio=%.4f mean_ratio=%.4f bound=%.4f",
                                    num_tokens, num_levels, beta, kappa,
                                    min(ratios), statistics.fmean(ratios), bounds[0])
    return rows


def complexity_sweep(args, logger):
    rows = []
    for num_tokens in args.complexity_tokens:
        for num_levels in args.complexity_levels:
            for t_max in args.tmax_values:
                problem = make_instance(num_tokens, num_levels, args.complexity_beta,
                                        args.complexity_kappa, args.seed)
                samples = []
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    lp = solve_base_lp(problem)
                    lp_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    rounded = certified_rounding(problem, lp)
                    round_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    if t_max > 0:
                        refined = refine(problem, rounded.level, t_max=t_max)
                        moves = refined.num_moves
                    else:
                        moves = 0
                    refine_seconds = time.perf_counter() - start
                    samples.append((lp_seconds + round_seconds, refine_seconds, moves))
                total = [first + second for first, second, _ in samples]
                rows.append({
                    "num_tokens": num_tokens, "num_levels": num_levels, "t_max": t_max,
                    "lp_seconds": statistics.fmean([first for first, _, _ in samples]),
                    "refine_seconds": statistics.fmean([second for _, second, _ in samples]),
                    "total_seconds": statistics.fmean(total),
                    "moves": statistics.fmean([float(moves) for _, _, moves in samples])})
                logger.info("complexity N=%d |R|=%d T_max=%d total=%.4fs "
                            "(lp+round=%.4fs refine=%.4fs moves=%.1f)",
                            num_tokens, num_levels, t_max, rows[-1]["total_seconds"],
                            rows[-1]["lp_seconds"], rows[-1]["refine_seconds"],
                            rows[-1]["moves"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--optimality-tokens", type=int, nargs="*", default=[6, 8, 10])
    parser.add_argument("--optimality-levels", type=int, nargs="*", default=[1, 3])
    parser.add_argument("--beta-values", type=float, nargs="*", default=[0.0, 0.25, 0.5])
    parser.add_argument("--kappa-values", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.9])
    parser.add_argument("--complexity-tokens", type=int, nargs="*",
                        default=[20, 40, 80, 160, 320])
    parser.add_argument("--complexity-levels", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--tmax-values", type=int, nargs="*", default=[0, 4, 8])
    parser.add_argument("--complexity-beta", type=float, default=0.25)
    parser.add_argument("--complexity-kappa", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/complexity")
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "scheduling_complexity_optimality",
                               vars(args))
    logger = configure_logging(run / "run.log")

    optimality_rows = optimality_sweep(args, logger)
    complexity_rows = complexity_sweep(args, logger)

    summary = {
        "seed": args.seed, "instances_per_cell": args.instances, "t_max": args.t_max,
        "optimality": {
            "worst_ratio": min((row["optimality_ratio"] for row in optimality_rows),
                               default=None),
            "violations": [row for row in optimality_rows
                           if row["optimality_ratio"] < row["proposition_bound"] - 1e-9],
            "rows": optimality_rows,
        },
        "complexity": complexity_rows,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)

    print(json.dumps({
        "optimality_cells": len(optimality_rows),
        "worst_optimality_ratio": summary["optimality"]["worst_ratio"],
        "proposition_violations": len(summary["optimality"]["violations"]),
        "complexity_cells": len(complexity_rows),
        "output": str(output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
