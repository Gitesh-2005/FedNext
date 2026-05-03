# FedNext

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](#validation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FedNext is a federated text-generation project. It trains a shared LSTM language model across many client partitions without centralizing raw text. The current pipeline supports **Non-IID client splits**, **FedAvg**, **adaptive aggregation**, and a standalone evaluator that reports **categorical cross-entropy loss**, **perplexity**, **top-N accuracy**, and **word error rate (WER)**.

## What’s Included

- Character-level corpus preprocessing
- Non-IID client generation using a Dirichlet split
- Federated training with partial participation
- Multiple aggregation modes: `fedavg`, `adaptive`, `simple`
- Evaluation on saved client test sets plus prompt-based next-token prediction
- Unit and integration tests for aggregation, clients, server, and model behavior

## Repository Layout

```text
.
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── federation/
├── models/
├── reports/
├── src/
│   ├── data/
│   ├── federated/
│   ├── models/
│   ├── utils/
│   ├── evaluate.py
│   └── train.py
├── tests/
├── notebooks/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Data Flow

1. Download the source text corpus.
2. Preprocess it into per-character client files.
3. Optionally split clients with a Dirichlet-based Non-IID sampler.
4. Build a shared vocabulary from all client text.
5. Train the global model with federated rounds.
6. Save the checkpoint with vocabulary and split metadata.
7. Evaluate the model on the saved federated test sets.

## Model

The project currently uses a compact character-level LSTM language model:

- Embedding: 64
- Hidden size: 128
- LSTM layers: 1
- Dropout: 0.1
- Observed vocabulary size in the current dataset run: 55 tokens

This setup was chosen because character-level modeling fits the source corpus better than the earlier word-level setup and converged much faster.

## Training Configuration

The active defaults live in `config/default.yaml`:

| Section | Setting | Value |
| --- | --- | --- |
| `model` | `vocab_size` | 256 |
| `model` | `embedding_dim` | 64 |
| `model` | `hidden_dim` | 128 |
| `model` | `num_layers` | 1 |
| `model` | `dropout` | 0.1 |
| `model` | `seq_len` | 40 |
| `federation` | `rounds` | 20 |
| `federation` | `clients_per_round` | 10 |
| `federation` | `local_epochs` | 5 |
| `federation` | `aggregation_method` | `adaptive` |
| `federation` | `non_iid` | `true` |
| `federation` | `dirichlet_alpha` | `0.5` |
| `training` | `lr` | `0.1` |
| `training` | `weight_decay` | `0.0` |
| `training` | `grad_clip` | `1.0` |

## Dataset Details

Source: Project Gutenberg eBook 100, a public-domain literary corpus.

After preprocessing, the current workspace produced:

| Item | Value |
| --- | --- |
| Raw corpus | `data/raw/shakespeare.txt` |
| Character client files | 91 |
| Synthetic Non-IID clients loaded for training | 20 |
| Observed vocabulary size | 55 tokens |
| Federated test split | 10% |

The preprocessing and non-IID split are deterministic with the configured training seed, so the same split can be reproduced from the saved checkpoint.

## Results

Recent training run with Non-IID adaptive aggregation:

| Round | Clients | Train Loss | Test Loss | Test Acc |
| --- | --- | ---: | ---: | ---: |
| 1 | 10 | 1.3782 | 1.2864 | 0.6329 |
| 2 | 10 | 0.9713 | 1.1717 | 0.6667 |
| 3 | 10 | 0.8465 | 1.0734 | 0.6980 |
| 4 | 10 | 0.6880 | 1.0421 | 0.7092 |
| 5 | 10 | 0.6091 | 1.1068 | 0.6951 |
| 6 | 10 | 0.6075 | 1.1046 | 0.6958 |
| 7 | 10 | 0.5428 | 1.1008 | 0.6881 |
| 8 | 10 | 0.5092 | 1.0948 | 0.7093 |
| 9 | 10 | 0.4893 | 1.0542 | 0.7020 |
| 10 | 10 | 0.4880 | 1.0809 | 0.7074 |
| 11 | 10 | 0.4621 | 1.0493 | 0.7151 |
| 12 | 10 | 0.4562 | 1.1096 | 0.6993 |
| 13 | 10 | 0.4950 | 1.0670 | 0.7207 |
| 14 | 10 | 0.4219 | 1.1048 | 0.7083 |
| 15 | 10 | 0.4477 | 1.1744 | 0.6958 |
| 16 | 10 | 0.4055 | 1.1761 | 0.6930 |
| 17 | 10 | 0.4667 | 1.0485 | 0.7126 |
| 18 | 10 | 0.3720 | 1.0535 | 0.7158 |
| 19 | 10 | 0.4188 | 1.0218 | 0.7247 |
| 20 | 10 | 0.3742 | 1.0740 | 0.7159 |

Evaluation on the saved checkpoint:

| Metric | Value |
| --- | ---: |
| Categorical Cross-Entropy Loss | 1.2191 |
| Perplexity | 3.3842 |
| Top-1 Accuracy | 0.6431 |
| Top-5 Accuracy | 0.8350 |
| Word Error Rate | 0.3569 |

## Quick Start

### Install

```bash
git clone https://github.com/Gitesh-2005/FedNext.git
cd FedNext
pip install -e ".[dev]"
```

### Prepare data

```bash
curl -L -o data/raw/shakespeare.txt "https://www.gutenberg.org/cache/epub/100/pg100.txt"
python src/preprocess_shakespeare.py data/raw/shakespeare.txt data/federation/
```

### Train

```bash
python -m src.train
```

### Evaluate

```bash
python -m src.evaluate --checkpoint models/global_model.pt --top-n 5
```

### Predict next tokens

```bash
python -m src.evaluate --checkpoint models/global_model.pt --prompt "to be or not to" --top-k 5 --skip-full-eval
```

## Validation

```bash
python -m pytest tests/ -q
```

Current result:

```text
24 passed
```

## Notes

- `data/raw/`, `data/federation/`, `models/`, `reports/`, and `logs/` are generated locally and should not be committed.
- The evaluator expects the checkpoint produced by `src.train`, because it stores the vocabulary and split metadata needed to reconstruct the evaluation dataset.

## License

MIT License. See [LICENSE](LICENSE).
