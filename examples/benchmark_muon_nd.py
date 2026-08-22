import time
import math
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

class MultiHeadTensorBlock(nn.Module):
    def __init__(self, num_heads=16, head_dim=64):
        super().__init__()
        # 1. 1D Bias / LayerNorm
        self.bias_1d = nn.Parameter(torch.randn(num_heads * head_dim))
        # 2. 2D Linear Projection
        self.linear_2d = nn.Parameter(torch.randn(num_heads * head_dim, num_heads * head_dim) * 0.02)
        # 3. 3D Multi-Head Attention Tensor [16 heads, 64, 64]
        self.attn_3d = nn.Parameter(torch.randn(num_heads, head_dim, head_dim) * 0.02)

    def forward(self, x):
        # 3D Tensor contraction
        h = torch.einsum("bnd,nde->bne", x, self.attn_3d)
        h = h.reshape(x.size(0), -1)
        h = h @ self.linear_2d + self.bias_1d
        return h

def main():
    print("==========================================================================", flush=True)
    print("🚀 MULTI-DIMENSIONAL TENSOR OPTIMIZER BENCHMARK (1D, 2D, 3D Tensors)", flush=True)
    print("   Evaluating Pure Rust Batched Polar Orthogonalization (MuonND)", flush=True)
    print("==========================================================================", flush=True)

    torch.manual_seed(42)
    model = MultiHeadTensorBlock(num_heads=16, head_dim=64)
    opt = tikun.MuonND(model.parameters(), lr=0.02, ns_steps=5)

    x = torch.randn(32, 16, 64)

    # Warmup
    out = model(x).sum()
    out.backward()
    opt.step()

    step_latencies = []
    print("\n⏱️ Running 20 consecutive optimization steps across 1D/2D/3D parameters...", flush=True)
    for step in range(1, 21):
        for p in model.parameters():
            p.grad = None

        out = model(x).sum()
        out.backward()

        t0 = time.perf_counter()
        opt.step()
        t_step = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(t_step)

        if step % 5 == 0 or step == 20:
            avg_ms = np.mean(step_latencies[-5:])
            print(f"  Step {step:2d}/20 ──> Multi-Dimensional Step Latency: {avg_ms:6.2f} ms", flush=True)

    # Verify Orthogonality of 3D attention tensor heads
    with torch.no_grad():
        W_head0 = model.attn_3d[0].cpu().numpy()
        gram = W_head0 @ W_head0.T
        eye = np.eye(64)
        ortho_err = np.linalg.norm(gram / (np.trace(gram)/64) - eye)

    print("\n==========================================================================", flush=True)
    print("🏆 MULTI-DIMENSIONAL TENSOR BENCHMARK RESULTS:")
    print("==========================================================================")
    print(f"  • Average 3D Batched Step Latency: {np.mean(step_latencies):.2f} ms / step", flush=True)
    print(f"  • Head Spectral Orthogonality Err: {ortho_err:.6e} (Optimal Condition # = 1.0)", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    main()
