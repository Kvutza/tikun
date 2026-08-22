import time
import sys
import torch
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def run_grand_showdown(num_params: int, name: str):
    print(f"\n==========================================================================")
    print(f"🥊 GRAND CPU SHOWDOWN: {name} ({num_params:,} Parameters / {num_params * 16 / (1024*1024):.1f} MB RAM)")
    print(f"==========================================================================")

    # -------------------------------------------------------------
    # 1. PyTorch C++ Vectorized AdamW (foreach=True)
    # -------------------------------------------------------------
    p_torch = torch.randn(num_params, dtype=torch.float32, requires_grad=True)
    g_torch = torch.randn(num_params, dtype=torch.float32)
    p_torch.grad = g_torch.clone()

    opt_torch_foreach = torch.optim.AdamW([p_torch], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=True)

    # Warmup
    opt_torch_foreach.step()

    start = time.perf_counter()
    for _ in range(10):
        opt_torch_foreach.step()
    t_torch_foreach = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ 1. PyTorch C++ (foreach=True):  {t_torch_foreach:.2f} ms / step")

    # -------------------------------------------------------------
    # 2. PyTorch C++ Standard AdamW (foreach=False)
    # -------------------------------------------------------------
    p_torch_std = torch.randn(num_params, dtype=torch.float32, requires_grad=True)
    p_torch_std.grad = g_torch.clone()
    opt_torch_std = torch.optim.AdamW([p_torch_std], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)

    opt_torch_std.step()

    start = time.perf_counter()
    for _ in range(10):
        opt_torch_std.step()
    t_torch_std = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ 2. PyTorch C++ (foreach=False): {t_torch_std:.2f} ms / step")

    # -------------------------------------------------------------
    # 3. Google Optax + JAX JIT (CPU)
    # -------------------------------------------------------------
    key = jax.random.PRNGKey(42)
    params_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)
    grads_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)

    optimizer_jax = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    opt_state_jax = optimizer_jax.init(params_jax)

    @jax.jit
    def step_optax(p, s, g):
        u, s = optimizer_jax.update(g, s, p)
        return optax.apply_updates(p, u), s

    p_j, s_j = step_optax(params_jax, opt_state_jax, grads_jax)
    jax.block_until_ready(p_j)

    start = time.perf_counter()
    for _ in range(10):
        p_j, s_j = step_optax(p_j, s_j, grads_jax)
    jax.block_until_ready(p_j)
    t_optax = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ 3. Google Optax + JAX JIT:     {t_optax:.2f} ms / step")

    # -------------------------------------------------------------
    # 4. tikun Rust Native Engine (ARM NEON SIMD)
    # -------------------------------------------------------------
    p_np = p_torch.detach().numpy()
    g_np = g_torch.numpy()
    m1_np = np.zeros(num_params, dtype=np.float32)
    m2_np = np.zeros(num_params, dtype=np.float32)

    p_ptr = p_np.ctypes.data
    g_ptr = g_np.ctypes.data
    m1_ptr = m1_np.ctypes.data
    m2_ptr = m2_np.ctypes.data

    tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)

    start = time.perf_counter()
    for _ in range(10):
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)
    t_tikun = (time.perf_counter() - start) * 1000 / 10
    print(f"⚡ 4. tikun Rust Engine:          {t_tikun:.2f} ms / step")

    # -------------------------------------------------------------
    # Summary Analysis
    # -------------------------------------------------------------
    data_gb = (num_params * 28) / (1024 * 1024 * 1024)
    bw_tikun = data_gb / (t_tikun / 1000)

    print(f"\n📊 COMPARATIVE SPEEDUPS (vs tikun {t_tikun:.2f} ms @ {bw_tikun:.2f} GB/s):")
    print(f"  • vs PyTorch (foreach=True):  tikun is {t_torch_foreach / t_tikun:.2f}x faster")
    print(f"  • vs PyTorch (foreach=False): tikun is {t_torch_std / t_tikun:.2f}x faster")
    print(f"  • vs Optax + JAX JIT:         tikun is {t_optax / t_tikun:.2f}x faster")

def main():
    run_grand_showdown(10_000_000, "10M Parameters")
    run_grand_showdown(50_000_000, "50M Parameters")
    run_grand_showdown(100_000_000, "100M Parameters")

if __name__ == "__main__":
    main()
