import time
import sys
import torch
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def benchmark_pytorch_comparison(num_params: int, name: str):
    print(f"\n==========================================================================")
    print(f"🥊 Low-Level Showdown: PyTorch C++ Engine vs. tikun ({name}: {num_params:,} Params)")
    print(f"==========================================================================")

    # 1. Setup PyTorch Tensors
    p_torch = torch.randn(num_params, dtype=torch.float32, requires_grad=True)
    g_torch = torch.randn(num_params, dtype=torch.float32)
    p_torch.grad = g_torch.clone()

    # PyTorch AdamW with foreach=True (C++ multi-tensor vectorized kernel)
    opt_torch = torch.optim.AdamW([p_torch], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=True)

    # Warmup PyTorch
    opt_torch.step()

    # Time PyTorch C++ Engine
    start = time.perf_counter()
    for _ in range(10):
        opt_torch.step()
    t_torch = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ PyTorch C++ Native (foreach=True): {t_torch:.2f} ms / step")

    # 2. Setup tikun Engine (Zero-Copy Resident Memory)
    p_np = p_torch.detach().numpy()
    g_np = g_torch.numpy()
    m1_np = np.zeros(num_params, dtype=np.float32)
    m2_np = np.zeros(num_params, dtype=np.float32)

    p_ptr = p_np.ctypes.data
    g_ptr = g_np.ctypes.data
    m1_ptr = m1_np.ctypes.data
    m2_ptr = m2_np.ctypes.data

    # Warmup tikun
    tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)

    # Time tikun Engine
    start = time.perf_counter()
    for _ in range(10):
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)
    t_tikun = (time.perf_counter() - start) * 1000 / 10
    print(f"⚡ tikun Rust Native Engine:           {t_tikun:.2f} ms / step")

    speedup = t_torch / t_tikun
    print(f"🏆 Result: tikun is {speedup:.2f}x faster than PyTorch C++ Native!")

def main():
    benchmark_pytorch_comparison(10_000_000, "10M")
    benchmark_pytorch_comparison(50_000_000, "50M")
    benchmark_pytorch_comparison(100_000_000, "100M")

if __name__ == "__main__":
    main()
