# FedNext - Project Status & Architecture

**Date Generated**: May 11, 2026

## Complete Current Status

FedNext is a functional, character-level federated learning system for text generation. The current project status demonstrates a complete pipeline capable of preprocessing raw text (specifically Shakespeare corpus), partitioning it into smaller Non-IID client datasets, executing a multi-round federated training loop using PyTorch, and evaluating the resulting global model.

**State of Completion:**
- **Data Pipeline**: Fully implemented. Handles text cleansing, sequence generation, and Non-IID client splitting using Dirichlet distributions.
- **Modeling**: Complete. Implements a multi-layer (currently 1-layer default) LSTM architecture for next-character prediction.
- **Federated Engine**: Complete. Supports Server-Client abstraction with partial client participation and multiple aggregation strategies (FedAvg, Adaptive).
- **Evaluation**: Integrated. Tracks Categorical Cross-Entropy, Perplexity, Top-N Accuracy, and Word Error Rate (WER).
- **Tooling**: Includes a comprehensive testing suite and Jupyter notebooks for manual exploration.

---

## High-Level Architecture

The system mimics a Federated Learning environment where training data never leaves the "local" client node. 

### Data Flow & Preprocessing
1. **Raw Ingestion**: A raw text corpus (`data/raw/shakespeare.txt`) is ingested.
2. **Tokenization**: Handled at the *character level*, avoiding large, sparse word vocabularies and improving convergence properties in local small-data environments.
3. **Partitioning**: 
   - Generates distinct text files for "clients" (e.g., individual Shakespearean characters).
   - Generates Non-IID splits via a Dirichlet sampler (`alpha = 0.5`) to mimic realistic, unevenly distributed federated data.
4. **Context Window**: Characters are chunked into sequences of `seq_len = 40` to predict the 41st character.

### Federated Training Loop
1. **Server Initialization**: The central server initializes a global `LSTMLanguageModel` and a shared vocabulary.
2. **Client Selection**: Each global round, the server samples a subset of clients (`clients_per_round = 10` out of `num_clients = 20`) to participate.
3. **Local Training**: Selected clients copy the global weights and train on their local partitioned subsets for `local_epochs = 5` using an SGD-based optimizer.
4. **Aggregation**: The server gathers the updated weight states from the participating clients and merges them. The current configuration uses the **adaptive** aggregation method.
5. **Iteration**: This process loops for `rounds = 20`.

---

## Technical Details

### Model Configuration (`LSTMLanguageModel`)
Located in `src/models/lstm.py`, the model is a straightforward sequential architecture tailored for sequence processing.

| Component | Specification | Description |
| --------- | ------------- | ----------- |
| **Input / Embedding** | `vocab_size: 256`, `embedding_dim: 64` | Embeds ASCII characters and punctuation into dense vectors. |
| **Recurrent Core** | `hidden_dim: 128`, `num_layers: 1` | Processes the sequence context to learn temporal dependencies. |
| **Regularization** | `dropout: 0.1` | Mitigates overfitting on small or specific client datasets. |
| **Output Head** | `Linear(128, 256)` | Projects the last LSTM hidden state to log-softmax over the vocabulary size. |

### Federation Configuration (`config/default.yaml`)
- **Number of Total Clients**: 20
- **Clients sampled per round (Fraction)**: 10 (50% Participation Rate)
- **Local Epochs (E)**: 5
- **Local Batch Size (B)**: 32
- **Aggregation Methods**: 
  - `fedavg`: Classical Federated Averaging, weighting clients by data size.
  - `adaptive`: Dynamically adjusts weights based on model updates or historical loss profiles.
  - `simple`: Straight mean averaging.
- **Heterogeneity**: Non-IID enabled (`dirichlet_alpha: 0.5`).

### Training Hyperparameters
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: `0.1`
- **Gradient Clipping**: `1.0` (Prevents exploding gradients common in recurrent architectures)
- **Weight Decay**: `0.0`
- **Seed**: `42` (For reproducibility of client selections and weight initializations)

---

## Project Structure & Dependencies

```text
federated-learning-project/
├── config/
│   └── default.yaml         # Active configuration constants
├── data/
│   ├── raw/                 # Original text corpus
│   └── federation/          # Preprocessed client datasets and splits
├── models/
│   └── global_model.pt      # Aggregated server model outcome
├── src/
│   ├── data/                # Dataset iterators and preprocessors
│   ├── federated/           # Aggregation algorithms, Client/Server classes
│   ├── models/              # Network architectures (LSTM)
│   ├── train.py             # Orchestrates the FL training loop
│   └── evaluate.py          # Centralized evaluation reporting
└── tests/                   # Aggregation, modeling, and client tests
```

**Environment:**
- **Language**: Python 3.9+
- **Core Library**: PyTorch 2.0+ 
