"""Integration test: full federated training loop with synthetic data."""

import copy
import random

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.lstm import LSTMLanguageModel


VOCAB_SIZE = 100
SEQ_LEN = 8
BATCH_SIZE = 4
NUM_SAMPLES = 32

MODEL_CFG = {
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 16,
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout": 0.0,
}


def _make_synthetic_loader(n_samples=NUM_SAMPLES, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(1, VOCAB_SIZE, (n_samples, SEQ_LEN))
    y = torch.randint(0, VOCAB_SIZE, (n_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def _make_clients(n=3, device=torch.device("cpu")):
    clients = {}
    for i in range(n):
        loader = _make_synthetic_loader(seed=i)
        clients[f"client_{i}"] = FederatedClient(
            client_id=f"client_{i}",
            train_loader=loader,
            test_loader=loader,  # reuse train as test for simplicity
            device=device,
        )
    return clients


class TestFederatedClient:
    def test_local_train_returns_state_dict(self):
        device = torch.device("cpu")
        client = FederatedClient(
            "c0", _make_synthetic_loader(), _make_synthetic_loader(), device
        )
        global_sd = LSTMLanguageModel(**MODEL_CFG).state_dict()
        updated_sd, n_samples, loss = client.local_train(
            global_state_dict=global_sd,
            model_cfg=MODEL_CFG,
            local_epochs=1,
            lr=0.01,
        )
        assert set(updated_sd.keys()) == set(global_sd.keys())
        assert n_samples == NUM_SAMPLES
        assert isinstance(loss, float)

    def test_local_train_does_not_mutate_global_state(self):
        device = torch.device("cpu")
        client = FederatedClient(
            "c0", _make_synthetic_loader(), None, device
        )
        global_sd = LSTMLanguageModel(**MODEL_CFG).state_dict()
        original_sd = copy.deepcopy(global_sd)
        client.local_train(global_sd, MODEL_CFG, local_epochs=2, lr=0.01)
        for key in global_sd:
            assert torch.equal(global_sd[key], original_sd[key]), (
                f"Global state was mutated at key: {key}"
            )

    def test_evaluate_returns_finite_metrics(self):
        device = torch.device("cpu")
        loader = _make_synthetic_loader()
        client = FederatedClient("c0", loader, loader, device)
        sd = LSTMLanguageModel(**MODEL_CFG).state_dict()
        loss, acc = client.evaluate(sd, MODEL_CFG)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_evaluate_without_test_loader_returns_nan(self):
        device = torch.device("cpu")
        client = FederatedClient("c0", _make_synthetic_loader(), None, device)
        sd = LSTMLanguageModel(**MODEL_CFG).state_dict()
        loss, acc = client.evaluate(sd, MODEL_CFG)
        import math
        assert math.isnan(loss) and math.isnan(acc)


class TestFederatedServer:
    def test_run_produces_history(self):
        device = torch.device("cpu")
        clients = _make_clients(n=3, device=device)
        server = FederatedServer(
            model_cfg=MODEL_CFG,
            clients=clients,
            device=device,
            clients_per_round=2,
            seed=0,
        )
        history = server.run(rounds=2, local_epochs=1, lr=0.01)
        assert len(history) == 2
        for m in history:
            assert m.num_clients == 2

    def test_global_model_weights_change_after_training(self):
        device = torch.device("cpu")
        clients = _make_clients(n=3, device=device)
        server = FederatedServer(
            model_cfg=MODEL_CFG,
            clients=clients,
            device=device,
            clients_per_round=2,
            seed=0,
        )
        initial_sd = copy.deepcopy(server.global_model.state_dict())
        server.run(rounds=1, local_epochs=2, lr=0.05)
        final_sd = server.global_model.state_dict()

        changed = any(
            not torch.equal(initial_sd[k], final_sd[k])
            for k in initial_sd
        )
        assert changed, "Global model weights did not change after training"

    def test_save_and_load(self, tmp_path):
        device = torch.device("cpu")
        clients = _make_clients(n=2, device=device)
        server = FederatedServer(
            model_cfg=MODEL_CFG, clients=clients, device=device,
            clients_per_round=1, seed=0,
        )
        server.run(rounds=1, local_epochs=1, lr=0.01)
        ckpt = str(tmp_path / "model.pt")
        server.save(ckpt)

        loaded_model = FederatedServer.load_model(ckpt, device=device)
        for k in server.global_model.state_dict():
            assert torch.allclose(
                loaded_model.state_dict()[k].float(),
                server.global_model.state_dict()[k].float(),
                atol=1e-6,
            )
