from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 8192
    vocab_size: int = 65
    n_embd: int = 768
    n_layers: int = 12
    n_heads: int = 12


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                # encode tokens into embeddings
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # add positional info into embeddings
                "wpe": nn.Embedding(config.block_size, config.embedding_dim),
                # transformer blocks
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                # final layer norm
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        # output of lm will have dimension of vocab_size (for softmax)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
