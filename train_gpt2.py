from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 8192
    vocab_size: int = 65
    embedding_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
