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


class CausalSelfAttention(nn.Module):
    # think of self attention as where the tokens communicate with each other
    def __init__(self, config: GPTConfig):
        super().__init__()


class MLP(nn.Module):
    # think of MLP as where the model thinks about the gathered information from attn
    def __init__(self, config: GPTConfig):
        super().__init__()
        # one linear layer to size 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # gelu with tanh approximation
        self.gelu = nn.GELU(approximate="tanh")
        # one linear layer back to size n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # layer norm -> self attention -> layer norm -> mlp
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
