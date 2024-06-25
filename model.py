from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50k BPE merges + 256 bytes tokens + 1 <|end_of_sequence|> token
    )
    n_embd: int = 768  # embedding dimensionality
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                # encode tokens into embeddings
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # add positional info into embeddings
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                # transformer blocks
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # final layer norm
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # output of lm will have dimension of vocab_size (for softmax)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)  # runs it on all modules

    def _init_weights(self, module):
        # initialize weights according to gpt-3 paper
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # account for growing stdev inside the residual stream by scaling c_proj by 1 * n_layer**-0.5
                std *= (
                    2 * self.config.n_layer
                ) ** -0.5  # x2 because we have 2 layers per block
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # encode with wte and wpe before passing it in through h (and finally, ln_f)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape(T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # pos_emb will be broadcasted to (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # logits.view(-1, logits.size(-1)) reshapes logits to (B*T, vocab_size)
            # targets.view(-1) reshapes targets to (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class CausalSelfAttention(nn.Module):
    # think of self attention as where the tokens communicate with each other
    def __init__(self, config: GPTConfig):
        super().__init__()
        # dimension of Q, K, V is n_embd / n_head
        assert config.n_embd % config.n_head == 0
        # K, Q, V projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection (allowing different subspaces captured by different heads to share information)
        # MLP will be token independent while this projection layer uses information across different tokens
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # this is NOT a 'bias', but a mask, but we are following the OpenAI/HF naming convention
        self.register_buffer(  # register_buffer is a keyword in pytorch
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),  # view is so that we can apply to batch and head dimensions too
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # B:batch size, T:sequence length, C:embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(
            self.n_embd, dim=2
        )  # split on third dimension, [B, T, 3 *C]
        # nh = n_head, hs = head size which is C // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
