import time
import math
import sys
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

# -------------------------------------------------------------
# 1. Real GPT Transformer Architecture (NanoGPT)
# -------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=384, n_head=6, block_size=128):
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
    def __init__(self, n_embd=384):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd=384, n_head=6, block_size=128):
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
    def __init__(self, vocab_size=65, n_layer=6, n_head=6, n_embd=384, block_size=128):
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
# 2. Real Dataset Loader (Tiny Shakespeare)
# -------------------------------------------------------------
def get_dataset():
    data_path = "examples/shakespeare.txt"
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"📥 Downloading Tiny Shakespeare dataset from {url}...")
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size

def get_batch(data, batch_size=32, block_size=128):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y

# -------------------------------------------------------------
# 3. Side-by-Side Real Training Loop
# -------------------------------------------------------------
def train_model(optimizer_type: str, data, vocab_size, num_steps: int = 100):
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = GPT(vocab_size=vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=128).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if optimizer_type == "torch_foreach":
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    elif optimizer_type == "tikun":
        opt = tikun.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_type}")

    step_opt_times = []
    losses = []
    
    # Warmup
    torch.manual_seed(0)
    x, y = get_batch(data, 16, 128)
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    if hasattr(opt, "zero_grad"):
        opt.zero_grad()

    start_train = time.perf_counter()
    for step in range(1, num_steps + 1):
        torch.manual_seed(step) # Deterministic identical data sequence
        x, y = get_batch(data, 32, 128)

        # Zero grad
        for p in model.parameters():
            p.grad = None

        _, loss = model(x, y)
        loss.backward()

        t0 = time.perf_counter()
        opt.step()
        t_opt = (time.perf_counter() - t0) * 1000.0
        step_opt_times.append(t_opt)
        losses.append(loss.item())

        if step % 25 == 0 or step == num_steps:
            avg_opt = np.mean(step_opt_times[-25:])
            print(f"  Step {step:3d}/{num_steps} | Loss: {loss.item():.4f} | Opt Step: {avg_opt:.2f} ms")

    total_time = time.perf_counter() - start_train
    avg_opt_time = np.mean(step_opt_times)

    return {
        "total_time": total_time,
        "avg_opt_time": avg_opt_time,
        "final_loss": losses[-1],
        "losses": losses,
        "num_params": num_params,
    }

def main():
    print("==========================================================================")
    print("🧠 REAL NANOGPT TRANSFORMER TRAINING BENCHMARK (Tiny Shakespeare)")
    print("   Architecture: 6 Layers, 6 Heads, 384 Embedding Dim (~10.8M Params)")
    print("==========================================================================")

    data, vocab_size = get_dataset()
    print(f"📊 Dataset Loaded: {len(data):,} tokens | Vocab Size: {vocab_size}")

    num_steps = 100

    # 1. Train with PyTorch foreach=True
    print(f"\n⏳ [1/2] Training NanoGPT with PyTorch C++ Native (foreach=True)...")
    res_torch = train_model("torch_foreach", data, vocab_size, num_steps)

    # 2. Train with tikun.AdamW Drop-in
    print(f"\n⏳ [2/2] Training NanoGPT with tikun.AdamW Drop-in Optimizer...")
    res_tikun = train_model("tikun", data, vocab_size, num_steps)

    # 3. Summary & Parity
    opt_speedup = res_torch["avg_opt_time"] / res_tikun["avg_opt_time"]
    total_speedup = res_torch["total_time"] / res_tikun["total_time"]
    loss_diff = abs(res_torch["final_loss"] - res_tikun["final_loss"])

    print("\n==========================================================================")
    print("🏆 FINAL NANOGPT REAL-WORLD TRAINING RESULTS:")
    print("==========================================================================")
    print(f"  • Model Parameter Count:   {res_torch['num_params']:,} parameters")
    print(f"  • Optimizer Step Time:     PyTorch {res_torch['avg_opt_time']:.2f} ms vs tikun {res_tikun['avg_opt_time']:.2f} ms ({opt_speedup:.2f}x FASTER!)")
    print(f"  • End-to-End Total Time:   PyTorch {res_torch['total_time']:.2f}s  vs tikun {res_tikun['total_time']:.2f}s  ({total_speedup:.2f}x Total Speedup)")
    print(f"  • Final Cross-Entropy Loss:PyTorch {res_torch['final_loss']:.4f} vs tikun {res_tikun['final_loss']:.4f}")
    print(f"  • Mathematical Convergence Equivalence: {loss_diff < 1e-4} (Diff: {loss_diff:.6e})")

if __name__ == "__main__":
    main()
