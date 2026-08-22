import time
import math
import sys
import os
import urllib.request
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

# -------------------------------------------------------------
# 1. NanoGPT Architecture
# -------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=256, n_head=4, block_size=64):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd=256):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd=256, n_head=4, block_size=64):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=65, n_layer=4, n_head=4, n_embd=256, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                h=nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------------------------------------------------------
# 2. Synthetic & Cached Dataset Loader
# -------------------------------------------------------------
def get_dataset():
    data_path = "examples/shakespeare.txt"
    if not os.path.exists(data_path):
        # Generate clean synthetic text tokens if network unavailable
        text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them. " * 500
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size

def get_batch(data, batch_size=16, block_size=64):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y

# -------------------------------------------------------------
# 3. Training Function
# -------------------------------------------------------------
def train_experiment(optimizer_type: str, data, vocab_size, num_steps: int = 50):
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = GPT(vocab_size=vocab_size, n_layer=4, n_head=4, n_embd=256, block_size=64).to(device)

    # Separate 2D weight matrices from 1D biases/layernorms/embeddings for Muon
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

    if optimizer_type == "torch_adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, foreach=True)
    elif optimizer_type == "tikun_muon":
        opt = tikun.Muon(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=0.02, # Keller Jordan optimal Muon learning rate
            momentum=0.95,
            adamw_lr=1e-3,
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_type}")

    step_times = []
    losses = []

    # Warmup
    torch.manual_seed(0)
    x, y = get_batch(data, 16, 64)
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    start_train = time.perf_counter()
    for step in range(1, num_steps + 1):
        torch.manual_seed(step)
        x, y = get_batch(data, 16, 64)

        for p in model.parameters():
            p.grad = None

        t0 = time.perf_counter()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(loss.item())

        if step % 10 == 0 or step == num_steps:
            print(f"  Step {step:2d}/{num_steps} | Loss: {loss.item():.4f} | Time/Step: {np.mean(step_times[-10:]):.2f} ms")

    total_time = time.perf_counter() - start_train
    return {
        "total_time": total_time,
        "avg_step_time": np.mean(step_times),
        "final_loss": losses[-1],
        "losses": losses,
    }

def main():
    print("==========================================================================")
    print("🚀 THE KELLER JORDAN MUON EXPERIMENT (NanoGPT Transformer)")
    print("   Comparing Newton-Schulz Orthogonalized Momentum (Muon) vs. AdamW")
    print("==========================================================================")

    data, vocab_size = get_dataset()
    print(f"📊 Dataset: {len(data):,} tokens | Vocab Size: {vocab_size}")

    num_steps = 50

    # 1. Train with Standard AdamW
    print(f"\n⏳ [1/2] Training NanoGPT with Standard PyTorch AdamW (50 Steps)...")
    res_adamw = train_experiment("torch_adamw", data, vocab_size, num_steps)

    # 2. Train with Keller Jordan Muon
    print(f"\n⏳ [2/2] Training NanoGPT with Keller Jordan Muon (50 Steps)...")
    res_muon = train_experiment("tikun_muon", data, vocab_size, num_steps)

    # 3. Compare Convergence & Efficiency
    print("\n==========================================================================")
    print("🏆 KELLER JORDAN MUON EXPERIMENT RESULTS:")
    print("==========================================================================")
    print(f"  • Standard AdamW Final Loss: {res_adamw['final_loss']:.4f} (Total: {res_adamw['total_time']:.2f}s)")
    print(f"  • Tikun Muon Final Loss:     {res_muon['final_loss']:.4f} (Total: {res_muon['total_time']:.2f}s)")
    
    loss_improvement = res_adamw['final_loss'] - res_muon['final_loss']
    if loss_improvement > 0:
        print(f"  🎉 MUON WINS ON CONVERGENCE: Reached lower loss by {loss_improvement:.4f} in the exact same 50 steps!")
    else:
        print(f"  • Loss difference: {loss_improvement:.4f}")

if __name__ == "__main__":
    main()
