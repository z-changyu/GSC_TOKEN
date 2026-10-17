"""Coupling-aware local refinement, Eq. (56)-(59) and Algorithm 1, steps 10-17.

Every candidate move is scored by its *exact* change in Eq. (34), not by a
re-ranking heuristic, and only a strictly positive gain is accepted. Writing
``w_i = u'_i z_i`` and ``c_i = 1 + beta * sum_{k != i} M^{<->}_{ki} w_k`` (the
incident coefficient of Eq. 18 evaluated on the current reception degrees), the
three gains of Section IV-E collapse to

    insertion   Delta^+_{j,ell}  = q_{j,ell} u'_j c_j
    removal     Delta^-_{m,s}    = q_{m,s}   u'_m c_m
    swap        Gamma            = Delta^+_{j,ell} - Delta^-_{m,s}
                                   - beta M^{<->}_{jm} u'_j u'_m q_{j,ell} q_{m,s}
    mode change Gamma_{m:s->ell} = (q_{m,ell} - q_{m,s}) u'_m c_m

The subtracted term in the swap is the coupling that both formulas count but
that no longer exists once ``m`` leaves the set, which is exactly Eq. (58).
Pure removal is never profitable because ``c_m >= 1``, so it is only ever
considered as one half of a swap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..pragmatic.coupling import incident_coupling
from .problem import SchedulingProblem


@dataclass
class RefinementResult:
    level: torch.Tensor
    utility: float
    num_moves: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _state(problem: SchedulingProblem, level: torch.Tensor):
    selected = level > 0
    mode = (level - 1).clamp(min=0)
    z = torch.zeros(problem.num_tokens, dtype=torch.float32)
    if bool(selected.any()):
        z[selected] = problem.q[selected, mode[selected]]
    used = torch.zeros(problem.num_tokens, dtype=torch.float32)
    if bool(selected.any()):
        used[selected] = problem.rates[mode[selected]]
    return selected, mode, z, used


def _level_to_schedule(problem: SchedulingProblem, level: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(problem.mode_value)
    selected = level > 0
    if bool(selected.any()):
        y[selected, (level[selected] - 1)] = 1.0
    return y


def refine(problem: SchedulingProblem, level: torch.Tensor, t_max: int = 8,
           tolerance: float = 1e-9) -> RefinementResult:
    """Improve a feasible integral schedule by positive-gain local moves."""
    level = level.clone()
    incident = incident_coupling(problem.coupling)
    utility = float(problem.expected_utility(_level_to_schedule(problem, level)))
    moves: list[dict[str, Any]] = []

    for _ in range(max(0, int(t_max))):
        selected, mode, _z, used = _state(problem, level)
        # Exact marginal decomposition of Eq. (34) under joint reception:
        #   add_i(ell)  = u'_i q_{i,ell}
        #   pair_i(ell) = beta u'_i sum_{j in T} M^<->_ij u'_j Q_{ell, lvl_j}
        #   insertion_i(ell) = add + pair   (gain of adding i at ell)
        #   removal_i        = add + pair evaluated at i's current level
        # With Q = q q^T this is algebraically the product model the gains used
        # before, so schedules are unchanged whenever joint_q is not supplied.
        add = problem.base_utility.reshape(-1, 1) * problem.q
        pair = problem.coupling_affinity(level, incident)
        insertion = add + pair
        idx_current = (mode - 1).clamp(min=0)
        gathered = (add.gather(1, idx_current.reshape(-1, 1))
                    + pair.gather(1, idx_current.reshape(-1, 1))).reshape(-1)
        removal = torch.where(selected, gathered, torch.zeros_like(gathered))
        budget = float(problem.budget)
        spent = float(used.sum())
        negative_inf = torch.full((), -math.inf, dtype=torch.float32)

        best_gain, best_move = float(tolerance), None

        # (a) insertion of an unselected token
        fits = (spent + problem.rates.reshape(1, -1)) <= budget + 1e-6
        candidate = torch.where(fits, insertion, negative_inf)
        candidate = torch.where(selected.reshape(-1, 1), negative_inf, candidate)
        flat = int(candidate.argmax())
        if math.isfinite(float(candidate.reshape(-1)[flat])):
            token, lvl = divmod(flat, problem.num_levels)
            if float(candidate[token, lvl]) > best_gain:
                best_gain, best_move = float(candidate[token, lvl]), ("insert", int(token), int(lvl))

        if bool(selected.any()):
            headroom = budget - spent + used.reshape(-1, 1)      # spent - r_s + r_ell <= C

            # (b) protection-mode change on a selected token, Eq. (59):
            #     Delta = insertion(i, ell_new) - removal(i) at the current level.
            change = insertion - removal.reshape(-1, 1)
            change = torch.where(problem.rates.reshape(1, -1) <= headroom + 1e-6,
                                 change, negative_inf)
            change = torch.where(selected.reshape(-1, 1), change, negative_inf)
            flat = int(change.argmax())
            if math.isfinite(float(change.reshape(-1)[flat])):
                token, lvl = divmod(flat, problem.num_levels)
                if float(change[token, lvl]) > best_gain:
                    best_gain, best_move = float(change[token, lvl]), ("mode", int(token), int(lvl))

            # (c) swap an unselected token in for a selected one, Eq. (58):
            #     the pending pair (incoming, outgoing) must be netted out, with the
            #     joint probability Q(ell_in, lvl_out) rather than q_in * q_out.
            shared = problem.beta * incident * torch.outer(problem.base_utility,
                                                           problem.base_utility)
            pair_out = problem.joint_q[:, idx_current].t()       # [o, ell] = Q(ell, lvl_o)
            for lvl in range(problem.num_levels):
                rate = float(problem.rates[lvl])
                gains = (insertion[:, lvl].reshape(-1, 1) - removal.reshape(1, -1)
                         - shared * pair_out[:, lvl].reshape(1, -1))
                feasible = (rate <= headroom + 1e-6) & (~selected).reshape(-1, 1) & selected.reshape(1, -1)
                gains = torch.where(feasible, gains, negative_inf)
                flat = int(gains.argmax())
                if math.isfinite(float(gains.reshape(-1)[flat])):
                    token, outgoing = divmod(flat, problem.num_tokens)
                    if float(gains[token, outgoing]) > best_gain:
                        best_gain = float(gains[token, outgoing])
                        best_move = ("swap", int(token), int(lvl), int(outgoing))

        if best_move is None:
            break

        if best_move[0] == "insert":
            level[best_move[1]] = best_move[2] + 1
        elif best_move[0] == "mode":
            level[best_move[1]] = best_move[2] + 1
        else:
            _, incoming, lvl, outgoing = best_move
            level[outgoing] = 0
            if incoming != outgoing:
                level[incoming] = lvl + 1
        moves.append({"move": best_move, "gain": best_gain})

        updated = float(problem.expected_utility(_level_to_schedule(problem, level)))
        # Every accepted move must strictly increase the exact objective; if it
        # does not, the gain algebra and the objective have diverged.
        if updated <= utility - 1e-6:
            raise RuntimeError(f"local move {best_move} did not increase Eq. (34): "
                               f"{updated} <= {utility}")
        utility = updated

    return RefinementResult(level=level, utility=utility, num_moves=len(moves),
                            metadata={"moves": moves, "t_max": int(t_max)})
