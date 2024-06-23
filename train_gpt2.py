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
        # dimension of Q, K, V is n_embd / n_heads
        assert config.n_embd % config.n_heads == 0
        # K, Q, V projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection (not 100% sure why this is necessary bc we're doing this in the MLP)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        # this is NOT a 'bias', but a mask, but we are following the OpenAI/HF naming convention
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
        ).view(
            1, 1, config.block_size, config.block_size
        )  # view is so that we can apply to batch and head dimensions too

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # B:batch size, T:sequence length, C:embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(
            self.n_embd, dim=2
        )  # split on third dimension, [B, T, 3 *C]
        # nh = n_heads, hs = head size which is C // n_heads
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # [B, nh, T, hs]
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # [B, nh, T, hs]
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # [B, nh, T, hs]
        # attention (materializes the large (T, T) matrix for all queries and keys)
        # when transposed, k's shape becomes [B, nh, hs, T]
        # so the dot product is between [B, nh, T, hs] and [B, nh, hs, T] = [B, nh, T, T]
        # k.size(-1) is hs. we scale by 1/sqrt(hs) to prevent the dot product from blowing up (to make softmax more stable)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # we slice to sequence length because the mask is for block_size (set 0s to -inf)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # softmax on the last dimension [B, nh, T, T]
        # last dimension represents the attention scores for each token (with respect to all other tokens)
        # the second to last dimension is also T, but it represents different tokens in the sequence
        # we want to normalize across attention scores for single token, NOT across the sequence
        att = F.softmax(att, dim=-1)
        # att @ V will result in [B, nh, T, T] * [B, nh, T, hs] -> [B, nh, T, hs]
        # you can interpret this as B batches of n head outputs, and each head output
        # is a sequence (T long) of weighted sums of the original tokens
        # each weighted sums corresponds to a token in the sequence (for token i, softmax(qi @ kj) * vj), where j goes up to T
        y = att @ v
        # transpose rearranges dimensions so that outputs from diff heads are next to each other
        # specifically, transpose results in [B, T, nh, hs] (where it was originally [B, nh, T, hs])
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)  # C is nh * hs (n_embd)
        )  # reassmble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


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
