"""Unit tests for the FedAvg aggregation algorithm."""

import copy

import pytest
import torch

from src.federated.aggregation import federated_average, simple_average
from src.models.lstm import LSTMLanguageModel


def make_state(vocab_size=50, hidden_dim=32):
    model = LSTMLanguageModel(
        vocab_size=vocab_size, embedding_dim=16, hidden_dim=hidden_dim, num_layers=1, dropout=0.0
    )
    return model.state_dict()


class TestFederatedAverage:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="No client updates"):
            federated_average([])

    def test_single_client_returns_identical_weights(self):
        sd = make_state()
        result = federated_average([(sd, 100)])
        for key in sd:
            assert torch.allclose(result[key].float(), sd[key].float(), atol=1e-6)

    def test_equal_weights_uniform_average(self):
        """With equal sample counts, FedAvg == simple mean."""
        sd1 = make_state()
        sd2 = make_state()
        # Set known values
        for key in sd1:
            sd1[key].fill_(1.0)
            sd2[key].fill_(3.0)

        result = federated_average([(sd1, 10), (sd2, 10)])
        for key in result:
            expected = 2.0
            assert torch.allclose(
                result[key].float(), torch.full_like(result[key], expected), atol=1e-5
            ), f"Key {key}: expected {expected}, got {result[key].float().mean()}"

    def test_weighted_average_correctness(self):
        """Client with 3× more data should have 3× more influence."""
        sd1 = make_state()
        sd2 = make_state()
        for key in sd1:
            sd1[key].fill_(0.0)
            sd2[key].fill_(4.0)

        # 25 vs 75 samples → expected avg = 0.25*0 + 0.75*4 = 3.0
        result = federated_average([(sd1, 25), (sd2, 75)])
        for key in result:
            assert torch.allclose(
                result[key].float(), torch.full_like(result[key], 3.0), atol=1e-5
            )

    def test_output_keys_match_input(self):
        sd = make_state()
        result = federated_average([(sd, 1)])
        assert set(result.keys()) == set(sd.keys())

    def test_raises_on_zero_total_samples(self):
        sd = make_state()
        with pytest.raises(ValueError, match="zero"):
            federated_average([(sd, 0)])


class TestSimpleAverage:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            simple_average([])

    def test_two_clients_midpoint(self):
        sd1 = make_state()
        sd2 = make_state()
        for key in sd1:
            sd1[key].fill_(0.0)
            sd2[key].fill_(2.0)

        result = simple_average([sd1, sd2])
        for key in result:
            assert torch.allclose(
                result[key].float(), torch.ones_like(result[key]), atol=1e-5
            )
