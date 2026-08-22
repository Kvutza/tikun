import time
import sys
import numpy as np
import torch

sys.path.insert(0, "target/release")
import tikun

def run_long_horizon_experiment(num_steps: int = 1000, num_params: int = 25_000_000):
    print("==========================================================================")
    print(f"🏃 1,000-STEP ENDURANCE TEST: {num_steps} Continuous Steps ({num_params:,} Params / {num_params * 16 / (1024*1024):.1f} MB RAM)")
    print("==========================================================================")

    np.random.seed(42)
    p_init = np.random.randn(num_params).astype(np.float32) * 0.02
    g_sim = np.random.randn(num_params).astype(np.float32) * 0.01

    lr = 0.001
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    decay = 0.01

    # -------------------------------------------------------------
    # 1. PyTorch C++ Native (foreach=True) Endurance
    # -------------------------------------------------------------
    print("\n⏳ 1. Running 1,000 continuous steps on PyTorch C++ (foreach=True)...")
    p_torch = torch.tensor(p_init.copy(), requires_grad=True)
    g_torch = torch.tensor(g_sim.copy())
    p_torch.grad = g_torch
    opt_torch = torch.optim.AdamW([p_torch], lr=lr, betas=(b1, b2), eps=eps, weight_decay=decay, foreach=True)

    # Warmup
    opt_torch.step()

    start_torch = time.perf_counter()
    for _ in range(num_steps):
        opt_torch.step()
    total_torch_sec = time.perf_counter() - start_torch
    avg_torch_ms = (total_torch_sec * 1000.0) / num_steps
    print(f"✅ PyTorch Completed: {total_torch_sec:.2f}s total | {avg_torch_ms:.2f} ms/step | {1000.0/avg_torch_ms:.1f} steps/sec")

    # -------------------------------------------------------------
    # 2. tikun Rust Engine Endurance
    # -------------------------------------------------------------
    print("\n⏳ 2. Running 1,000 continuous steps on tikun Rust Engine...")
    p_tikun = p_init.copy()
    g_tikun = g_sim.copy()
    m1_tikun = np.zeros(num_params, dtype=np.float32)
    m2_tikun = np.zeros(num_params, dtype=np.float32)

    p_ptr = p_tikun.ctypes.data
    g_ptr = g_tikun.ctypes.data
    m1_ptr = m1_tikun.ctypes.data
    m2_ptr = m2_tikun.ctypes.data

    # Warmup
    tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 1, lr, b1, b2, eps, decay)

    start_tikun = time.perf_counter()
    for step in range(1, num_steps + 1):
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", step, lr, b1, b2, eps, decay)
    total_tikun_sec = time.perf_counter() - start_tikun
    avg_tikun_ms = (total_tikun_sec * 1000.0) / num_steps
    print(f"✅ tikun Completed:   {total_tikun_sec:.2f}s total | {avg_tikun_ms:.2f} ms/step | {1000.0/avg_tikun_ms:.1f} steps/sec")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    speedup = total_torch_sec / total_tikun_sec
    print("\n==========================================================================")
    print("🏆 1,000-STEP ENDURANCE RESULTS SUMMARY:")
    print("==========================================================================")
    print(f"  • PyTorch C++: {total_torch_sec:.2f}s ({avg_torch_ms:.2f} ms / step)")
    print(f"  • tikun Rust:  {total_tikun_sec:.2f}s ({avg_tikun_ms:.2f} ms / step)")
    print(f"  • Sustained Speedup: tikun is {speedup:.2f}x faster across 1,000 steps!")

if __name__ == "__main__":
    run_long_horizon_experiment(1000, 25_000_000)
