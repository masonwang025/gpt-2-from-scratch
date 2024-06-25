import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig
from data import DataLoaderLite
import time
import math

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# get logits
model = GPT(GPTConfig(vocab_size=50304))  # divisible by 128, or 2 ** 7
model.to(device)
model = torch.compile(model)

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1 # 10% of max_lr
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in betweeen, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
    
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B, T = 16, 1024 # B is micro batch size
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision("high")  # drop from higheest to tf32 for matmul

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16
        ):  # this uses bfloat16 for same scale but less precision
                logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss (instead of purely accumulating)
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the gradients
    # determine and set lr for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups: # only one param group here
        param_group["lr"] = lr 
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


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
