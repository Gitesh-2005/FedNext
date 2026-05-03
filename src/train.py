"""Main training script for Federated Learning text generation.

Usage
-----
  python -m src.train                      # uses config/default.yaml
  python -m src.train --config my.yaml     # custom config
  python -m src.train --rounds 5           # CLI overrides

The script:
1. Builds a shared vocabulary from all client text files.
2. Creates per-client DataLoaders.
3. Instantiates FederatedServer and runs N communication rounds.
4. Saves the final global model and training plots.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_global_vocab, load_client_datasets
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.utils.metrics import plot_training_history, print_metrics_table


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_model_cfg(cfg: dict) -> dict:
    m = cfg["model"]
    return {
        "vocab_size": m["vocab_size"],
        "embedding_dim": m["embedding_dim"],
        "hidden_dim": m["hidden_dim"],
        "num_layers": m["num_layers"],
        "dropout": m["dropout"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning – Text Generation")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument("--rounds", type=int, help="Override federation.rounds")
    parser.add_argument("--clients-per-round", type=int, help="Override clients_per_round")
    parser.add_argument("--lr", type=float, help="Override training.lr")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # CLI overrides
    if args.rounds:
        cfg["federation"]["rounds"] = args.rounds
    if args.clients_per_round:
        cfg["federation"]["clients_per_round"] = args.clients_per_round
    if args.lr:
        cfg["training"]["lr"] = args.lr

    setup_logging(cfg["logging"]["level"])
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------ #
    # 1. Vocabulary
    # ------------------------------------------------------------------ #
    federation_dir = cfg["data"]["federation_dir"]
    if not os.path.isdir(federation_dir):
        logger.error(
            "Federation directory not found: %s\n"
            "Run `bash src/download_dataset.sh` and then `python src/preprocess_shakespeare.py` first.",
            federation_dir,
        )
        sys.exit(1)

    logger.info("Building vocabulary from all client files …")
    vocab = build_global_vocab(
        federation_dir=federation_dir,
        vocab_size=cfg["model"]["vocab_size"],
    )

    # ------------------------------------------------------------------ #
    # 2. Client datasets
    # ------------------------------------------------------------------ #
    logger.info("Loading per-client datasets …")
    client_data = load_client_datasets(
        federation_dir=federation_dir,
        vocab=vocab,
        seq_len=cfg["model"]["seq_len"],
        test_fraction=cfg["data"]["test_fraction"],
        min_samples=cfg["data"]["min_samples"],
        non_iid=cfg["federation"].get("non_iid", False),
        num_clients=cfg["federation"].get("num_clients", None),
        dirichlet_alpha=cfg["federation"].get("dirichlet_alpha", 0.5),
        seed=cfg["training"]["seed"],
    )

    batch_size = cfg["federation"]["local_batch_size"]
    clients: dict[str, FederatedClient] = {}
    for client_id, (train_ds, test_ds) in client_data.items():
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = (
            DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            if test_ds and len(test_ds) > 0
            else None
        )
        clients[client_id] = FederatedClient(client_id, train_loader, test_loader, device)

    logger.info("Registered %d federated clients", len(clients))

    # ------------------------------------------------------------------ #
    # 3. Update vocab_size to actual vocabulary size
    # ------------------------------------------------------------------ #
    model_cfg = build_model_cfg(cfg)
    model_cfg["vocab_size"] = vocab.size   # actual size ≤ configured max

    # ------------------------------------------------------------------ #
    # 4. Federated training
    # ------------------------------------------------------------------ #
    server = FederatedServer(
        model_cfg=model_cfg,
        clients=clients,
        device=device,
        clients_per_round=cfg["federation"]["clients_per_round"],
        seed=cfg["training"]["seed"],
        aggregation_method=cfg["federation"].get("aggregation_method", "fedavg"),
        aggregation_beta=cfg["federation"].get("aggregation_beta", 1.0),
    )

    history = server.run(
        rounds=cfg["federation"]["rounds"],
        local_epochs=cfg["federation"]["local_epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
    )

    # ------------------------------------------------------------------ #
    # 5. Results
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FEDERATED TRAINING SUMMARY")
    print("=" * 60)
    print_metrics_table(history)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    server.save(
        "models/global_model.pt",
        extra_metadata={
            "vocab": vocab,
            "seq_len": cfg["model"]["seq_len"],
            "federation_dir": federation_dir,
            "non_iid": cfg["federation"].get("non_iid", False),
            "dirichlet_alpha": cfg["federation"].get("dirichlet_alpha", 0.5),
            "num_clients": cfg["federation"].get("num_clients", None),
            "test_fraction": cfg["data"].get("test_fraction", 0.1),
            "min_samples": cfg["data"].get("min_samples", 10),
            "seed": cfg["training"].get("seed", 42),
        },
    )
    plot_training_history(history, save_path="reports/training_history.png", show=False)
    logger.info("Training complete. Checkpoint saved to models/global_model.pt")


if __name__ == "__main__":
    main()
