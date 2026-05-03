"""Federated Averaging (FedAvg) aggregation algorithm.

Reference
---------
McMahan et al. "Communication-Efficient Learning of Deep Networks from
Decentralized Data", AISTATS 2017.  https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def federated_average(
    updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Compute the weighted average of client model updates (FedAvg).

    Each element of *updates* is ``(state_dict, num_samples)``.  The
    contribution of each client is proportional to its dataset size,
    so clients with more data have a stronger influence on the global
    model — matching the original FedAvg paper.

    Parameters
    ----------
    updates:
        List of ``(client_state_dict, num_samples)`` tuples.

    Returns
    -------
    averaged_state_dict:
        Weighted-average model weights ready to be loaded into the
        global model.

    Raises
    ------
    ValueError:
        If *updates* is empty.
    """
    if not updates:
        raise ValueError("No client updates provided to federated_average.")

    total_samples = sum(n for _, n in updates)
    if total_samples == 0:
        raise ValueError("Total sample count across clients is zero.")

    # Initialise accumulator with zeros matching the first client's layout
    reference_state = updates[0][0]
    averaged: Dict[str, torch.Tensor] = {
        key: torch.zeros_like(tensor, dtype=torch.float32)
        for key, tensor in reference_state.items()
    }

    for state_dict, num_samples in updates:
        weight = num_samples / total_samples
        for key, tensor in state_dict.items():
            averaged[key] += tensor.float() * weight

    return averaged


def simple_average(
    updates: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Unweighted (uniform) average of client state dicts.

    Useful for ablation studies where all clients have identical dataset
    sizes or when sample counts are unavailable.
    """
    if not updates:
        raise ValueError("No updates provided.")

    averaged: Dict[str, torch.Tensor] = {
        key: torch.zeros_like(tensor, dtype=torch.float32)
        for key, tensor in updates[0].items()
    }
    n = len(updates)
    for state_dict in updates:
        for key, tensor in state_dict.items():
            averaged[key] += tensor.float() / n

    return averaged


def adaptive_federated_average(
    updates: List[Tuple[Dict[str, torch.Tensor], int, float]],
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Adaptive FedAvg: weight each client by num_samples * (1 + beta * metric).

    Each element of *updates* should be ``(state_dict, num_samples, metric)``
    where *metric* is a per-client quality signal (e.g. local validation accuracy)
    in range [0, 1]. The contribution of each client is proportional to
    num_samples * (1 + beta * metric).
    """
    if not updates:
        raise ValueError("No client updates provided to adaptive_federated_average.")

    # compute weighted total
    weights = []
    total_weight = 0.0
    for _sd, n, metric in updates:
        w = n * (1.0 + beta * float(metric))
        weights.append(w)
        total_weight += w

    if total_weight == 0:
        raise ValueError("Total adaptive weight across clients is zero.")

    # init accumulator from first state
    reference_state = updates[0][0]
    averaged: Dict[str, torch.Tensor] = {
        key: torch.zeros_like(tensor, dtype=torch.float32)
        for key, tensor in reference_state.items()
    }

    for (state_dict, _n, _metric), w in zip(updates, weights):
        for key, tensor in state_dict.items():
            averaged[key] += tensor.float() * (w / total_weight)

    return averaged
