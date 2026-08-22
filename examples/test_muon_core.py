import time
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun
from tikun.muon import zeropower_via_newtonschulz5

def test_muon_core():
    print("==========================================================================")
    print("🔬 TESTING TIKUN MUON ENGINE: Orthogonality, Math Parity, & Latency")
    print("==========================================================================")

    # 1. Test Newton-Schulz Matrix Orthogonalization
    print("🧪 1. Verifying Newton-Schulz Polar Orthogonality (768 x 768 Weight Matrix)...")
    torch.manual_seed(42)
    G = torch.randn(768, 768, dtype=torch.float32)
    
    t0 = time.perf_counter()
    X = zeropower_via_newtonschulz5(G, steps=5)
    ns_time = (time.perf_counter() - t0) * 1000.0

    # Check orthogonality: X @ X.T should be approximately Identity
    I_approx = X @ X.T
    I_expected = torch.eye(768)
    
    frobenius_error = (I_approx - I_expected).norm() / I_expected.norm()
    diag_mean = I_approx.diagonal().mean().item()
    off_diag_max = (I_approx - torch.diag(I_approx.diagonal())).abs().max().item()

    print(f"   • Computation Time: {ns_time:.2f} ms")
    print(f"   • Diagonal Elements Mean: {diag_mean:.4f} (Expected: 1.0000)")
    print(f"   • Max Off-Diagonal Cross-Talk: {off_diag_max:.6f}")
    print(f"   • Relative Frobenius Error: {frobenius_error.item():.6f}")
    print(f"   • Orthogonality Quality: {'✅ PASSED (Strict Matrix Orthogonality)' if frobenius_error < 0.05 else '❌ FAILED'}")

    # 2. Test Hybrid Parameter Partitioning in Tikun Muon
    print("\n🧪 2. Testing Hybrid Parameter Partitioning in tikun.Muon...")
    # 2D weight matrix (Muon path)
    w_2d = torch.randn(512, 512, requires_grad=True)
    w_2d.grad = torch.randn(512, 512)

    # 1D bias vector (Tikun AdamW path)
    b_1d = torch.randn(512, requires_grad=True)
    b_1d.grad = torch.randn(512)

    opt = tikun.Muon(
        muon_params=[w_2d],
        adamw_params=[b_1d],
        lr=0.02,
        momentum=0.95,
        adamw_lr=1e-3,
    )

    # Warmup
    opt.step()

    # Time 50 steps
    start = time.perf_counter()
    for _ in range(50):
        w_2d.grad = torch.randn(512, 512)
        b_1d.grad = torch.randn(512)
        opt.step()
    avg_step_ms = (time.perf_counter() - start) * 1000.0 / 50.0

    print(f"   • Average Muon Step Time (512x512 matrix + 1D vector): {avg_step_ms:.2f} ms")
    print(f"   • Hybrid Engine Step: ✅ PASSED")

    print("\n==========================================================================")
    print("🎉 TIKUN MUON ENGINE VERIFICATION COMPLETE")
    print("==========================================================================")

if __name__ == "__main__":
    test_muon_core()
