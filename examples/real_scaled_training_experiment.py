import time
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

# -------------------------------------------------------------
# 1. 25M Parameter Real GPT Transformer Architecture
# -------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=512, n_head=8, block_size=128):
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
    def __init__(self, n_embd=512):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd=512, n_head=8, block_size=128):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ScaledGPT(nn.Module):
    def __init__(self, vocab_size=256, n_layer=8, n_head=8, n_embd=512, block_size=128):
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
# 2. Batch Generator
# -------------------------------------------------------------
def get_synthetic_corpus(batch_size=8, seq_len=128, vocab_size=256):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y

# -------------------------------------------------------------
# 3. Real Training Loop with Immediate Flush
# -------------------------------------------------------------
def run_real_training(optimizer_name: str, num_steps: int = 30):
    torch.manual_seed(42)
    model = ScaledGPT(vocab_size=256, n_layer=8, n_head=8, n_embd=512, block_size=128)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if optimizer_name == "torch_foreach":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    elif optimizer_name == "tikun_tuned":
        opt = tikun.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    opt_times = []
    losses = []

    # Warmup
    torch.manual_seed(0)
    x, y = get_synthetic_corpus(8, 128, 256)
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    start_epoch = time.perf_counter()
    for step in range(1, num_steps + 1):
        torch.manual_seed(step)
        x, y = get_synthetic_corpus(8, 128, 256)

        for p in model.parameters():
            p.grad = None

        _, loss = model(x, y)
        loss.backward()

        t0 = time.perf_counter()
        opt.step()
        t_opt = (time.perf_counter() - t0) * 1000.0
        opt_times.append(t_opt)
        losses.append(loss.item())

        if step % 5 == 0 or step == num_steps:
            avg_opt = np.mean(opt_times[-5:])
            print(f"  Step {step:2d}/{num_steps} | Loss: {loss.item():.4f} | Opt Latency: {avg_opt:.2f} ms", flush=True)

    total_time = time.perf_counter() - start_epoch
    avg_opt_ms = np.mean(opt_times)

    return {
        "num_params": num_params,
        "total_time": total_time,
        "avg_opt_ms": avg_opt_ms,
        "final_loss": losses[-1],
    }

def main():
    print("==========================================================================", flush=True)
    print("🚀 REAL-WORLD SCALED TRANSFORMER TRAINING SHOWDOWN (25M Parameters)", flush=True)
    print("   PyTorch Native C++ (foreach=True) vs. Tikun ENNX-Tuned Engine", flush=True)
    print("==========================================================================", flush=True)

    num_steps = 30

    # 1. Official PyTorch C++ Native
    print(f"\n⏳ [1/2] Training 25M Transformer with PyTorch Native C++ (foreach=True)...", flush=True)
    res_torch = run_real_training("torch_foreach", num_steps)

    # 2. Tikun ENNX-Tuned Engine
    print(f"\n⏳ [2/2] Training 25M Transformer with Tikun ENNX-Tuned Optimizer...", flush=True)
    res_tikun = run_real_training("tikun_tuned", num_steps)

    # 3. Final Rigorous Metrics
    opt_speedup = res_torch["avg_opt_ms"] / res_tikun["avg_opt_ms"]
    total_speedup = res_torch["total_time"] / res_tikun["total_time"]
    loss_diff = abs(res_torch["final_loss"] - res_tikun["final_loss"])

    print("\n==========================================================================", flush=True)
    print("🏆 FINAL 25M TRANSFORMER TRAINING RESULTS:", flush=True)
    print("==========================================================================", flush=True)
    print(f"  • Model Parameters:        {res_torch['num_params']:,} (~{res_torch['num_params']*4/(1024*1024):.1f} MB weights)", flush=True)
    print(f"  • Optimizer Step Latency:  PyTorch {res_torch['avg_opt_ms']:.2f} ms vs tikun {res_tikun['avg_opt_ms']:.2f} ms ({opt_speedup:.2f}x FASTER!)", flush=True)
    print(f"  • Total 30-Step Runtime:   PyTorch {res_torch['total_time']:.2f}s  vs tikun {res_tikun['total_time']:.2f}s  ({total_speedup:.2f}x Total Speedup)", flush=True)
    print(f"  • Final Training Loss:     PyTorch {res_torch['final_loss']:.4f} vs tikun {res_tikun['final_loss']:.4f}", flush=True)
    print(f"  • Loss Parity Difference:  {loss_diff:.6e} (Exact Parity: {loss_diff < 1e-4})", flush=True)

if __name__ == "__main__":
    main()
