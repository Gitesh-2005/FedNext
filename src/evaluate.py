"""Inference and evaluation script for a trained global model.

Usage
-----
  python -m src.evaluate --checkpoint models/global_model.pt \\
                                                 --prompt "to be or not to be"

By default, the script also evaluates the checkpoint on the saved
federated client test sets and reports:
    - categorical cross-entropy loss
    - perplexity
    - top-N accuracy
    - word error rate (WER)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import Vocabulary, load_client_datasets
from src.models.lstm import LSTMLanguageModel
from src.utils.metrics import (
    aggregate_batch_metrics,
    safe_perplexity,
    top_k_accuracy,
    word_error_rate,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-word prediction inference")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint")
    parser.add_argument("--prompt", default=None, help="Optional context string for prediction")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to show")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top-N accuracy threshold for dataset evaluation",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Only run prompt prediction; skip dataset-level metrics",
    )
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "vocab" not in ckpt:
        logger.error(
            "Checkpoint does not contain vocabulary. "
            "Re-run training with the updated src/train.py."
        )
        sys.exit(1)

    vocab: Vocabulary = ckpt["vocab"]
    seq_len = ckpt.get("seq_len", 20)
    model_cfg = ckpt["model_cfg"]
    model = LSTMLanguageModel(**model_cfg).to(device)
    model.load_state_dict(ckpt["global_state_dict"])
    model.eval()

    if not args.skip_full_eval:
        federation_dir = ckpt.get("federation_dir")
        if not federation_dir:
            logger.error("Checkpoint is missing federation_dir for full evaluation.")
            sys.exit(1)

        client_data = load_client_datasets(
            federation_dir=federation_dir,
            vocab=vocab,
            seq_len=seq_len,
            test_fraction=float(ckpt.get("test_fraction", 0.1)),
            min_samples=int(ckpt.get("min_samples", 10)),
            non_iid=bool(ckpt.get("non_iid", False)),
            num_clients=ckpt.get("num_clients", None),
            dirichlet_alpha=float(ckpt.get("dirichlet_alpha", 0.5)),
            seed=int(ckpt.get("seed", 42)),
        )

        logits_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []

        for client_id, (_train_ds, test_ds) in client_data.items():
            if test_ds is None or len(test_ds) == 0:
                continue
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits, _ = model(x)
                logits_batches.append(logits.detach().cpu())
                target_batches.append(y.detach().cpu())

        metrics = aggregate_batch_metrics(logits_batches, target_batches, top_k=args.top_n)
        print("\nDataset evaluation")
        print("-" * 72)
        print(f"Categorical Cross-Entropy Loss : {metrics['loss']:.4f}")
        print(f"Perplexity                    : {metrics['perplexity']:.4f}")
        print(f"Top-1 Accuracy                : {metrics['top1_accuracy']:.4f}")
        print(f"Top-{args.top_n} Accuracy            : {metrics[f'top{args.top_n}_accuracy']:.4f}")
        print(f"Word Error Rate (WER)         : {metrics['wer']:.4f}")
        print("-" * 72)

    if args.prompt:
        tokens = vocab.encode(args.prompt)
        if len(tokens) < seq_len:
            tokens = [vocab.token2idx["<PAD>"]] * (seq_len - len(tokens)) + tokens
        else:
            tokens = tokens[-seq_len:]

        context = torch.tensor(tokens, dtype=torch.long, device=device)
        top_ids, top_probs = model.predict_next(context, top_k=args.top_k)

        print(f"\nPrompt : {args.prompt!r}")
        print(f"Top-{args.top_k} next-token predictions:")
        print("-" * 40)
        for rank, (idx, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
            token = vocab.idx2token.get(idx, "<UNK>")
            print(f"  {rank}. {token:<20}  p={prob:.4f}")


if __name__ == "__main__":
    main()
