import time
import math
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

class MultiRankTransformerBlock(nn.Module):
    """
    Transformer Block with explicitly separated 1D, 2D, and 3D tensor representations:
      - 1D: LayerNorm weights and biases
      - 2D: MLP Projections (fc1, fc2)
      - 3D: Multi-Head Attention Key/Query/Value Projections [num_heads, head_dim, head_dim]
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 1D Parameters
        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ln1_bias = nn.Parameter(torch.zeros(d_model))
        self.ln2_weight = nn.Parameter(torch.ones(d_model))
        self.ln2_bias = nn.Parameter(torch.zeros(d_model))

        # 3D Parameters: Multi-Head Attention Tensor [num_heads, head_dim, head_dim]
        self.qkv_3d = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.proj_2d = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # 2D Parameters: Feed-Forward MLP
        self.fc1_2d = nn.Parameter(torch.randn(d_model, d_model * 4) * 0.02)
        self.fc2_2d = nn.Parameter(torch.randn(d_model * 4, d_model) * 0.02)

    def forward(self, x):
        B, S, D = x.shape
        # LayerNorm 1
        h = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        h = h * self.ln1_weight + self.ln1_bias

        # 3D Multi-Head Attention Projection
        h_heads = h.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, S, D_h]
        attn_out = torch.einsum("bhsd,hde->bhse", h_heads, self.qkv_3d)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        attn_out = attn_out @ self.proj_2d
        x = x + attn_out

        # LayerNorm 2 + MLP
        h2 = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        h2 = h2 * self.ln2_weight + self.ln2_bias
        mlp_out = torch.relu(h2 @ self.fc1_2d) @ self.fc2_2d
        x = x + mlp_out
        return x

class MultiRankGPT(nn.Module):
    def __init__(self, vocab_size=1024, d_model=512, num_layers=4, num_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MultiRankTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits

def run_experiment():
    print("==========================================================================", flush=True)
    print("🚀 MULTI-RANK TRANSFORMER TRAINING SHOWDOWN (1D + 2D + 3D TENSORS)", flush=True)
    print("   Evaluating: Auto-Tuned BPANN TuRBO-ENN MultiDimensionalMuon (MuonND)", flush=True)
    print("==========================================================================", flush=True)

    torch.manual_seed(42)
    device = "cpu"

    model_tikun = MultiRankGPT(vocab_size=1024, d_model=512, num_layers=4, num_heads=8).to(device)
    model_torch = MultiRankGPT(vocab_size=1024, d_model=512, num_layers=4, num_heads=8).to(device)
    model_torch.load_state_dict(model_tikun.state_dict())

    total_params = sum(p.numel() for p in model_tikun.parameters())
    print(f"  • Total Model Parameters: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB)", flush=True)
    print(f"  • Tensor Composition: 1D LayerNorms + 2D MLP Weights + 3D Attention Heads\n", flush=True)

    # 1. Tikun Multi-Dimensional Muon Optimizer (Rank-Aware)
    opt_tikun = tikun.MuonND(model_tikun.parameters(), lr=0.02, ns_steps=5, adamw_lr=1e-3)

    # 2. PyTorch Native C++ AdamW (Foreach)
    opt_torch = torch.optim.AdamW(model_torch.parameters(), lr=1e-3, foreach=True)

    criterion = nn.CrossEntropyLoss()

    # Synthetic training batches
    batch_size = 16
    seq_len = 128
    num_steps = 20

    torch_latencies = []
    tikun_latencies = []

    print("⏱️ Executing 20 Forward-Backward Training Steps...", flush=True)
    for step in range(1, num_steps + 1):
        inputs = torch.randint(0, 1024, (batch_size, seq_len))
        targets = torch.randint(0, 1024, (batch_size, seq_len))

        # --- PyTorch Step ---
        opt_torch.zero_grad()
        logits_torch = model_torch(inputs)
        loss_torch = criterion(logits_torch.view(-1, 1024), targets.view(-1))
        loss_torch.backward()
        t0 = time.perf_counter()
        opt_torch.step()
        t_torch = (time.perf_counter() - t0) * 1000.0
        torch_latencies.append(t_torch)

        # --- Tikun MuonND Step ---
        opt_tikun.zero_grad()
        logits_tikun = model_tikun(inputs)
        loss_tikun = criterion(logits_tikun.view(-1, 1024), targets.view(-1))
        loss_tikun.backward()
        t0 = time.perf_counter()
        opt_tikun.step()
        t_tikun = (time.perf_counter() - t0) * 1000.0
        tikun_latencies.append(t_tikun)

        if step % 5 == 0 or step == num_steps:
            print(f"  Step {step:02d}/{num_steps:02d} | PyTorch Native: {t_torch:6.2f} ms | Tikun MuonND: {t_tikun:6.2f} ms | Loss: {loss_tikun.item():.4f}", flush=True)

    avg_torch = np.mean(torch_latencies[5:])
    avg_tikun = np.mean(tikun_latencies[5:])

    print("\n==========================================================================", flush=True)
    print("🏆 FINAL MULTI-RANK TRANSFORMER BENCHMARK RESULTS:")
    print("==========================================================================")
    print(f"  • PyTorch Native AdamW (foreach=True): {avg_torch:.2f} ms / step", flush=True)
    print(f"  • Tikun MultiDimensionalMuon (MuonND): {avg_tikun:.2f} ms / step", flush=True)
    print(f"  • Final Loss (PyTorch):                {loss_torch.item():.6f}", flush=True)
    print(f"  • Final Loss (Tikun MuonND):           {loss_tikun.item():.6f}", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    run_experiment()
