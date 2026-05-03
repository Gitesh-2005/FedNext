"""Unit tests for LSTMLanguageModel."""

import pytest
import torch

from src.models.lstm import LSTMLanguageModel


VOCAB_SIZE = 200
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2
BATCH_SIZE = 4
SEQ_LEN = 10


@pytest.fixture
def model() -> LSTMLanguageModel:
    return LSTMLanguageModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )


class TestLSTMLanguageModel:
    def test_output_shape(self, model):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, (h, c) = model(x)
        assert logits.shape == (BATCH_SIZE, VOCAB_SIZE), "Wrong logit shape"
        assert h.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
        assert c.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)

    def test_hidden_state_passthrough(self, model):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        h0 = model.init_hidden(BATCH_SIZE, torch.device("cpu"))
        logits, (h, c) = model(x, h0)
        assert h.shape == (NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)

    def test_parameter_count_positive(self, model):
        assert model.count_parameters() > 0

    def test_no_nan_in_logits(self, model):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, _ = model(x)
        assert not torch.isnan(logits).any(), "NaN detected in logits"

    def test_predict_next_topk_shape(self, model):
        context = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        ids, probs = model.predict_next(context, top_k=5)
        assert ids.shape == (5,)
        assert probs.shape == (5,)

    def test_predict_next_probabilities_sum_leq_one(self, model):
        context = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        _, probs = model.predict_next(context, top_k=VOCAB_SIZE)
        assert abs(probs.sum().item() - 1.0) < 1e-3

    def test_predict_next_probabilities_non_negative(self, model):
        context = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        _, probs = model.predict_next(context, top_k=5)
        assert (probs >= 0).all()

    def test_padding_idx_gradient_zeroed(self, model):
        x = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)  # all PAD
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        pad_grad = model.embedding.weight.grad[0]
        assert pad_grad.abs().sum().item() == 0.0, "PAD embedding should have zero grad"

    def test_gradient_flow(self, model):
        model.train()
        x = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,))
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
