"""Federated Learning client – handles local training on a single device."""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.lstm import LSTMLanguageModel

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Simulates a single participant in a Federated Learning system.

    Each client:
    1. Receives the current global model weights from the server.
    2. Trains locally for *local_epochs* using its private dataset.
    3. Returns the updated model weights (and dataset size) to the server.

    No raw data ever leaves the client.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def local_train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        model_cfg: Dict,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Train locally starting from *global_state_dict*.

        Parameters
        ----------
        global_state_dict : State dict broadcast by the server.
        model_cfg : Kwargs forwarded to ``LSTMLanguageModel.__init__``.
        local_epochs : Number of passes over the local dataset.
        lr : Learning rate for local SGD.
        weight_decay : L2 regularisation coefficient.
        grad_clip : Max gradient norm (0 to disable).

        Returns
        -------
        updated_state_dict : Model weights after local training.
        num_samples : Size of the local training set (used for weighted avg).
        avg_loss : Mean cross-entropy over the final local epoch.
        """
        # Clone global weights into a local model (no in-place mutation)
        model = LSTMLanguageModel(**model_cfg).to(self.device)
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        final_loss = 0.0
        num_samples = len(self.train_loader.dataset)  # type: ignore[arg-type]

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            final_loss = epoch_loss / max(num_samples, 1)

        logger.debug(
            "[%s] local_epochs=%d  loss=%.4f  samples=%d",
            self.client_id,
            local_epochs,
            final_loss,
            num_samples,
        )
        return model.state_dict(), num_samples, final_loss

    def evaluate(
        self,
        state_dict: Dict[str, torch.Tensor],
        model_cfg: Dict,
    ) -> Tuple[float, float]:
        """
        Evaluate *state_dict* on this client's held-out test set.

        Returns
        -------
        loss : Average cross-entropy loss.
        accuracy : Top-1 next-word accuracy.
        """
        if self.test_loader is None:
            return float("nan"), float("nan")

        model = LSTMLanguageModel(**model_cfg).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = model(x)
                total_loss += criterion(logits, y).item() * x.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
