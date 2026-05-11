"""Shakespeare dataset loading and vocabulary construction."""

import logging
import os
import re
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from .base_dataset import BaseTokenizer, BaseFederatedDataset

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

class CharTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        return list(text.lower())

class WordTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

class Vocabulary:
    """Maps tokens <-> integer indices, with special token support."""

    def __init__(self, vocab_size: int = 10_000, tokenizer: str = "char"):
        self.vocab_size = vocab_size
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._built = False
        
        if tokenizer == "word":
            self.tokenizer = WordTokenizer()
        else:
            self.tokenizer = CharTokenizer()

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from a list of raw text strings."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.tokenizer.tokenize(text))

        self.token2idx = dict(SPECIAL_TOKENS)
        for token, _ in counter.most_common(self.vocab_size - len(SPECIAL_TOKENS)):
            idx = len(self.token2idx)
            self.token2idx[token] = idx

        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self._built = True
        logger.info(
            "Vocabulary built: %d tokens (from %d unique)",
            len(self.token2idx),
            len(counter),
        )

    def save(self, path: str) -> None:
        """Save the vocabulary to a JSON file."""
        if not self._built:
            logger.warning("Attempting to save an unbuilt vocabulary.")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token2idx": self.token2idx,
                "tokenizer": "word" if isinstance(self.tokenizer, WordTokenizer) else "char"
            }, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load vocabulary from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.token2idx = data.get("token2idx", {})
        self.idx2token = {int(v): k for k, v in self.token2idx.items()}
        
        tokenizer_type = data.get("tokenizer", "char")
        if tokenizer_type == "word":
            self.tokenizer = WordTokenizer()
        else:
            self.tokenizer = CharTokenizer()
            
        self._built = True

    def encode(self, text: str) -> List[int]:
        unk = self.token2idx["<UNK>"]
        return [self.token2idx.get(t, unk) for t in self.tokenizer.tokenize(text)]

    def decode(self, indices: List[int]) -> str:
        return " ".join(self.idx2token.get(i, "<UNK>") for i in indices)

    @property
    def size(self) -> int:
        return len(self.token2idx)

    def __len__(self) -> int:
        return self.size


class NextWordDataset(BaseFederatedDataset):
    """
    Sliding-window next-word prediction dataset.

    Each sample is (context_ids, target_id) where context is a fixed-length
    sequence of token indices and target is the immediately following token.
    """

    def __init__(
        self,
        text: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
        seq_len: int = 20,
        tokens: Optional[List[int]] = None,
    ) -> None:
        if tokens is None:
            if not vocab or not vocab._built:
                raise ValueError("Vocabulary must be built before creating a dataset.")
            tokens = vocab.encode(text)

        self.seq_len = seq_len

        self.inputs: List[List[int]] = []
        self.targets: List[int] = []

        for i in range(len(tokens) - seq_len):
            self.inputs.append(tokens[i : i + seq_len])
            self.targets.append(tokens[i + seq_len])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


def _get_cache_path(cache_dir: str, config_hash: str) -> str:
    return os.path.join(cache_dir, f"federated_clients_cache_{config_hash}.pt")

def load_client_datasets(
    federation_dir: str,
    vocab: Vocabulary,
    seq_len: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_samples: int = 10,
    non_iid: bool = False,
    num_clients: Optional[int] = None,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dict[str, Tuple[NextWordDataset, Optional[NextWordDataset], Optional[NextWordDataset]]]:
    """
    Load per-character text files from *federation_dir* and return a
    mapping ``{client_id: (train_dataset, val_dataset, test_dataset)}``.

    Parameters
    ----------
    federation_dir:
        Directory produced by ``preprocess_shakespeare.py``.
    vocab:
        Pre-built ``Vocabulary`` object (built on **all** client texts first).
    seq_len:
        Sliding-window context length.
    val_fraction:
        Fraction of each client's tokens held out for validation.
    test_fraction:
        Fraction of each client's tokens held out for evaluation.
    min_samples:
        Clients with fewer than this many samples are skipped.
    cache_dir:
        If set, caches tokenized splits to speed up consecutive runs.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        config_hash = hash(f"{federation_dir}_{seq_len}_{val_fraction}_{test_fraction}_{non_iid}_{num_clients}_{dirichlet_alpha}_{seed}")
        cache_path = _get_cache_path(cache_dir, str(config_hash))
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached client datasets from {cache_path}")
            try:
                cached_data = torch.load(cache_path)
                client_datasets = {}
                for cid, splits in cached_data.items():
                    tr_ds = NextWordDataset(seq_len=seq_len, tokens=splits['train'])
                    va_ds = NextWordDataset(seq_len=seq_len, tokens=splits['val']) if splits['val'] else None
                    te_ds = NextWordDataset(seq_len=seq_len, tokens=splits['test']) if splits['test'] else None
                    client_datasets[cid] = (tr_ds, va_ds, te_ds)
                return client_datasets
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")

    client_datasets: Dict[str, Tuple[NextWordDataset, Optional[NextWordDataset], Optional[NextWordDataset]]] = {}
    cache_payload = {}

    txt_files = [f for f in os.listdir(federation_dir) if f.endswith(".txt")]
    logger.info("Found %d client text files in %s", len(txt_files), federation_dir)

    # Non-IID splitting via Dirichlet over character files (dialogue lines)
    if non_iid:
        # Determine number of desired clients
        K = num_clients or len(txt_files)
        rng = np.random.default_rng(seed)
        # Collect lines per character file
        per_char_lines = []
        for fname in txt_files:
            path = os.path.join(federation_dir, fname)
            with open(path, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
            if lines:
                per_char_lines.append(lines)

        # Initialize client buckets
        client_buckets: List[List[str]] = [[] for _ in range(K)]

        for lines in per_char_lines:
            # draw proportions for this character across K clients
            props = rng.dirichlet([dirichlet_alpha] * K)
            counts = (props * len(lines)).astype(int)
            # adjust counts to sum exactly to len(lines)
            diff = len(lines) - counts.sum()
            while diff > 0:
                for i in range(K):
                    if diff <= 0:
                        break
                    counts[i] += 1
                    diff -= 1
            while diff < 0:
                for i in range(K):
                    if diff >= 0:
                        break
                    if counts[i] > 0:
                        counts[i] -= 1
                        diff += 1

            # shuffle lines and distribute
            perm = rng.permutation(len(lines))
            idx = 0
            for k in range(K):
                c = counts[k]
                if c > 0:
                    sel = [lines[i] for i in perm[idx: idx + c]]
                    client_buckets[k].extend(sel)
                    idx += c

        # Build datasets per generated client
        for i in range(K):
            client_lines = client_buckets[i]
            if not client_lines:
                continue
            text = " ".join(client_lines)
            tokens = vocab.encode(text)
            if len(tokens) < seq_len + min_samples:
                logger.debug("Skipping synthetic client_%d (only %d tokens)", i, len(tokens))
                continue
                
            test_split = max(seq_len + 1, int(len(tokens) * (1 - test_fraction)))
            val_split = max(seq_len + 1, int(test_split * (1 - val_fraction / (1 - test_fraction))))
            
            train_tokens = tokens[:val_split]
            val_tokens = tokens[val_split:test_split]
            test_tokens = tokens[test_split:]
            
            train_ds = NextWordDataset(seq_len=seq_len, tokens=train_tokens)
            val_ds = NextWordDataset(seq_len=seq_len, tokens=val_tokens) if val_tokens else None
            test_ds = NextWordDataset(seq_len=seq_len, tokens=test_tokens) if test_tokens else None
            
            if len(train_ds) < min_samples:
                continue
                
            client_datasets[f"client_{i}"] = (train_ds, val_ds, test_ds)
            cache_payload[f"client_{i}"] = {'train': train_tokens, 'val': val_tokens, 'test': test_tokens}

        logger.info("Loaded %d synthetic non-IID client datasets", len(client_datasets))
        if cache_dir:
            torch.save(cache_payload, _get_cache_path(cache_dir, str(config_hash)))
        return client_datasets

    # IID/default behaviour: one file == one client
    for fname in txt_files:
        client_id = fname[:-4]  # strip .txt
        path = os.path.join(federation_dir, fname)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()

        tokens = vocab.encode(text)
        if len(tokens) < seq_len + min_samples:
            logger.debug("Skipping %s (only %d tokens)", client_id, len(tokens))
            continue
            
        test_split = max(seq_len + 1, int(len(tokens) * (1 - test_fraction)))
        val_split = max(seq_len + 1, int(test_split * (1 - val_fraction / (1 - test_fraction))))
        
        train_tokens = tokens[:val_split]
        val_tokens = tokens[val_split:test_split]
        test_tokens = tokens[test_split:]
        
        train_ds = NextWordDataset(seq_len=seq_len, tokens=train_tokens)
        val_ds = NextWordDataset(seq_len=seq_len, tokens=val_tokens) if val_tokens else None
        test_ds = NextWordDataset(seq_len=seq_len, tokens=test_tokens) if test_tokens else None
        
        if len(train_ds) < min_samples:
            continue
            
        client_datasets[client_id] = (train_ds, val_ds, test_ds)
        cache_payload[client_id] = {'train': train_tokens, 'val': val_tokens, 'test': test_tokens}

    logger.info("Loaded %d client datasets", len(client_datasets))
    if cache_dir:
        torch.save(cache_payload, _get_cache_path(cache_dir, str(config_hash)))
    return client_datasets


def build_global_vocab(federation_dir: str, vocab_size: int) -> Vocabulary:
    """Build a single vocabulary from all client text files."""
    all_texts: List[str] = []
    for fname in os.listdir(federation_dir):
        if fname.endswith(".txt"):
            path = os.path.join(federation_dir, fname)
            with open(path, encoding="utf-8") as fh:
                all_texts.append(fh.read())

    vocab = Vocabulary(vocab_size=vocab_size)
    vocab.build(all_texts)
    return vocab
