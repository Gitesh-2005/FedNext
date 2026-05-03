"""Federated Learning server – coordinates global training rounds."""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.federated.aggregation import federated_average, adaptive_federated_average
from src.federated.client import FederatedClient
from src.models.lstm import LSTMLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics collected at the end of one communication round."""

    round_num: int
    participating_clients: List[str]
    avg_train_loss: float
    avg_test_loss: float
    avg_test_accuracy: float
    num_clients: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_clients = len(self.participating_clients)

    def __str__(self) -> str:
        return (
            f"Round {self.round_num:>3} | "
            f"clients={self.num_clients} | "
            f"train_loss={self.avg_train_loss:.4f} | "
            f"test_loss={self.avg_test_loss:.4f} | "
            f"test_acc={self.avg_test_accuracy:.4f}"
        )


class FederatedServer:
    """
    Central coordinator for Federated Learning.

    Responsibilities
    ----------------
    * Maintain the global model.
    * Sample a subset of clients each round (partial participation).
    * Broadcast global weights → collect local updates → aggregate.
    * Track per-round metrics for downstream analysis.
    """

    def __init__(
        self,
        model_cfg: Dict,
        clients: Dict[str, FederatedClient],
        device: torch.device,
        clients_per_round: int = 5,
        seed: int = 42,
        aggregation_method: str = "fedavg",
        aggregation_beta: float = 1.0,
    ) -> None:
        self.model_cfg = model_cfg
        self.clients = clients
        self.device = device
        self.clients_per_round = min(clients_per_round, len(clients))
        self._rng = random.Random(seed)
        self.aggregation_method = aggregation_method.lower()
        self.aggregation_beta = aggregation_beta

        # Initialise the global model
        self.global_model = LSTMLanguageModel(**model_cfg).to(device)
        logger.info(
            "Global model initialised  |  parameters: %d",
            self.global_model.count_parameters(),
        )

        self.history: List[RoundMetrics] = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(
        self,
        rounds: int,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
    ) -> List[RoundMetrics]:
        """
        Execute *rounds* federated communication rounds.

        Returns
        -------
        history : List of :class:`RoundMetrics`, one per round.
        """
        for r in range(1, rounds + 1):
            logger.info("=== Round %d / %d ===", r, rounds)
            metrics = self._run_round(r, local_epochs, lr, weight_decay, grad_clip)
            self.history.append(metrics)
            logger.info(str(metrics))

        logger.info("Federated training complete.")
        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_round(
        self,
        round_num: int,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        grad_clip: float,
    ) -> RoundMetrics:
        """Execute a single communication round."""
        selected = self._sample_clients()
        global_state = copy.deepcopy(self.global_model.state_dict())

        # updates format: for fedavg/simple -> (state_dict, n)
        # for adaptive -> (state_dict, n, metric)
        updates: List = []
        train_losses: List[float] = []

        for client_id in selected:
            client = self.clients[client_id]
            updated_sd, n_samples, train_loss = client.local_train(
                global_state_dict=global_state,
                model_cfg=self.model_cfg,
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )
            if self.aggregation_method == "adaptive":
                # evaluate client's updated model locally to get a quality metric
                tl_local, ta_local = client.evaluate(updated_sd, self.model_cfg)
                metric = 0.0 if ta_local != ta_local else float(ta_local)
                updates.append((updated_sd, n_samples, metric))
            else:
                updates.append((updated_sd, n_samples))
            train_losses.append(train_loss)
        # Aggregate updates into the new global model
        if self.aggregation_method == "adaptive":
            new_global_state = adaptive_federated_average(updates, beta=self.aggregation_beta)
        else:
            new_global_state = federated_average(updates)
        self.global_model.load_state_dict(new_global_state)

        # Evaluate on all participating clients' test sets
        test_losses, test_accs = [], []
        for client_id in selected:
            tl, ta = self.clients[client_id].evaluate(new_global_state, self.model_cfg)
            if tl == tl:  # skip NaN
                test_losses.append(tl)
                test_accs.append(ta)

        return RoundMetrics(
            round_num=round_num,
            participating_clients=selected,
            avg_train_loss=sum(train_losses) / max(len(train_losses), 1),
            avg_test_loss=sum(test_losses) / max(len(test_losses), 1),
            avg_test_accuracy=sum(test_accs) / max(len(test_accs), 1),
        )

    def _sample_clients(self) -> List[str]:
        """Randomly sample *clients_per_round* client IDs."""
        all_ids = list(self.clients.keys())
        return self._rng.sample(all_ids, self.clients_per_round)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str, extra_metadata: Optional[Dict] = None) -> None:
        payload = {
            "global_state_dict": self.global_model.state_dict(),
            "model_cfg": self.model_cfg,
            "history": self.history,
            "aggregation_method": self.aggregation_method,
            "aggregation_beta": self.aggregation_beta,
        }
        if extra_metadata:
            payload.update(extra_metadata)

        torch.save(payload, path)
        logger.info("Model checkpoint saved to %s", path)

    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> LSTMLanguageModel:
        device = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = LSTMLanguageModel(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["global_state_dict"])
        return model
