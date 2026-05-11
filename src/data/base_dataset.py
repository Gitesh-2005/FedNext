import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a string into a list of string tokens."""
        pass

class BaseFederatedDataset(Dataset, ABC):
    """Abstract base class for a dataset used in federated learning tasks."""
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

