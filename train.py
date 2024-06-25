import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig
from data import DataLoaderLite
import time
import math
import os

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py

# for DDP: https://github.com/karpathy/build-nanogpt/commit/ba2554acce54487f07c13563247f2cce683e0d5e

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# create model
model = GPT(GPTConfig(vocab_size=50304), master_process=master_process)  # divisible by 128, or 2 ** 7
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1 # 10% of max_lr
warmup_steps = 715
max_steps = 19073
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
B, T = 32, 1024 # B is micro batch size
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision("high")  # drop from higheest to tf32 for matmul

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    t0 = time.time()
    
    # once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # average the loss across all processes
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # training loop
    model.train()
    
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):  # this uses bfloat16 for same scale but less precision
                logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss (instead of purely accumulating)
        loss_accum += loss.detach()
        if ddp:
            # if ddp, we need to sync the gradients across all processes
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the loss across all processes
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
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()


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
