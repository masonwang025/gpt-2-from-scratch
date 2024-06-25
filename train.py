import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig
from data import DataLoaderLite
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# get a data batch (see play.ipynb)
import tiktoken

text = open("shakespeare.txt", "r").read()
text = text[:10000]
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

B, T = 4, 32
buf = torch.tensor(tokens[: B * T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# get logits
model = GPT(GPTConfig(vocab_size=50304))  # divisible by 128, or 2 ** 7
model.to(device)
model = torch.compile(model)

train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision("high")  # drop from higheest to tf32 for matmul

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16
    ):  # this uses bfloat16 for same scale but less precision
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(
        f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.2f}"
    )


# -------------------------------------------

import sys

sys.exit(0)

# -------------------------------------------

### FROM SECTION 1
num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

# prefix tokens
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(dim=0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) which is 5, 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    logits = model(x)  # (B, T, vocab_size)
    # inefficient sampling: throw away everything but logits from last token
    logits = logits[:, -1, :]  # (B, vocab_size)
    probs = F.softmax(logits, dim=-1)  # get the probabilities
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (num_return_sequences, 50)
    # select a token from the top-k probabilities
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    idx = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
    # gather the corresponding token indices by using the sampled indices
    xcol = torch.gather(topk_indices, 1, idx)
    # append the sampled token to the sequence
    x = torch.cat([x, xcol], dim=1)

# print the generated text
for i in range(num_return_sequences):
    # get the tokens for this sequence
    tokens = x[i, :max_length].tolist()
    # decode the sequence
    decoded = enc.decode(tokens)
    print(">", decoded)
