"""Evaluation metrics and training visualisation helpers."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


def safe_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity, guarding against overflow."""
    if loss != loss:
        return float("nan")
    if loss > 50:
        return float("inf")
    return math.exp(loss)


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy for a batch of logits and targets."""
    if logits.numel() == 0:
        return float("nan")
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(correct)


def word_error_rate(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Next-token WER (single-token sequence): 1 - top-1 accuracy."""
    if predictions.numel() == 0:
        return float("nan")
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    return float((predictions != targets).float().mean().item())


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean categorical cross-entropy loss for logits/targets."""
    if logits.numel() == 0:
        return float("nan")
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    return float(loss.item())


def aggregate_batch_metrics(
    logits_batches: Sequence[torch.Tensor],
    targets_batches: Sequence[torch.Tensor],
    top_k: int = 5,
) -> dict:
    """Aggregate evaluation metrics from batches of logits and targets."""
    if not logits_batches:
        return {
            "loss": float("nan"),
            "perplexity": float("nan"),
            "top1_accuracy": float("nan"),
            f"top{top_k}_accuracy": float("nan"),
            "wer": float("nan"),
        }

    logits = torch.cat(logits_batches, dim=0)
    targets = torch.cat(targets_batches, dim=0)
    loss = cross_entropy_from_logits(logits, targets)
    top1 = top_k_accuracy(logits, targets, k=1)
    topk = top_k_accuracy(logits, targets, k=top_k)
    preds = logits.argmax(dim=-1)
    wer = word_error_rate(preds, targets)
    return {
        "loss": loss,
        "perplexity": safe_perplexity(loss),
        "top1_accuracy": top1,
        f"top{top_k}_accuracy": topk,
        "wer": wer,
    }


def plot_training_history(
    history,  # List[RoundMetrics]
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot train/test loss and test accuracy across communication rounds.

    Parameters
    ----------
    history : List of ``RoundMetrics`` objects.
    save_path : If given, save the figure to this path (PNG).
    show : If True, display the figure interactively.
    """
    rounds = [m.round_num for m in history]
    train_loss = [m.avg_train_loss for m in history]
    test_loss = [m.avg_test_loss for m in history]
    test_acc = [m.avg_test_accuracy for m in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Federated Training Progress", fontsize=14, fontweight="bold")

    # Loss curves
    axes[0].plot(rounds, train_loss, "o-", label="Train Loss", color="#2196F3")
    axes[0].plot(rounds, test_loss, "s--", label="Test Loss", color="#F44336")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss vs. Round")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(rounds, test_acc, "^-", color="#4CAF50")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].set_title("Test Accuracy vs. Round")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def print_metrics_table(history) -> None:
    """Pretty-print a summary table of all round metrics."""
    header = f"{'Round':>6}  {'Clients':>7}  {'Train Loss':>10}  {'Test Loss':>9}  {'Test Acc':>8}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for m in history:
        print(
            f"{m.round_num:>6}  {m.num_clients:>7}  "
            f"{m.avg_train_loss:>10.4f}  {m.avg_test_loss:>9.4f}  "
            f"{m.avg_test_accuracy:>8.4f}"
        )
    print(separator)
