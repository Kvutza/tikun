import time
import sys
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def benchmark():
    print(f"==========================================================================")
    print(f"🚀 Benchmarking Complete Optimizer Suite: AdamW vs. Lion vs. SGD (JAX vs tikun)")
    print(f"==========================================================================")

    # 1. Model Structure (~2.6M parameters across 5 layers)
    key = jax.random.PRNGKey(42)
    params = {
        "embed": jax.random.normal(key, (1000, 1024), dtype=jnp.float32),
        "qkv_proj": jax.random.normal(key, (1024, 1024), dtype=jnp.float32),
        "out_proj": jax.random.normal(key, (1024, 512), dtype=jnp.float32),
        "norm_w": jax.random.normal(key, (1024,), dtype=jnp.float32),
        "norm_b": jax.random.normal(key, (1024,), dtype=jnp.float32),
    }

    grads = {
        "embed": jax.random.normal(key, (1000, 1024), dtype=jnp.float32),
        "qkv_proj": jax.random.normal(key, (1024, 1024), dtype=jnp.float32),
        "out_proj": jax.random.normal(key, (1024, 512), dtype=jnp.float32),
        "norm_w": jax.random.normal(key, (1024,), dtype=jnp.float32),
        "norm_b": jax.random.normal(key, (1024,), dtype=jnp.float32),
    }

    params_flat, _ = jax.tree_util.tree_flatten(params)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)

    params_views = [np.asarray(p) for p in params_flat]
    grads_views = [np.asarray(g) for g in grads_flat]
    m1_flat = [np.zeros_like(p) for p in params_flat]
    m2_flat = [np.zeros_like(p) for p in params_flat]

    # Pre-extract raw pointer vectors (True Zero-Copy Resident Memory)
    param_ptrs = [p.ctypes.data for p in params_views]
    grad_ptrs = [g.ctypes.data for g in grads_views]
    m1_ptrs = [m.ctypes.data for m in m1_flat]
    m2_ptrs = [m.ctypes.data for m in m2_flat]
    lengths = [p.size for p in params_views]

    # -------------------------------------------------------------
    # 1. AdamW Benchmark
    # -------------------------------------------------------------
    opt_adamw_jax = optax.adamw(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    state_adamw = opt_adamw_jax.init(params)
    @jax.jit
    def step_adamw_jax(p, s, g):
        u, s = opt_adamw_jax.update(g, s, p)
        return optax.apply_updates(p, u), s

    p_j, s_j = step_adamw_jax(params, state_adamw, grads)
    jax.block_until_ready(p_j)
    start = time.perf_counter()
    for _ in range(100):
        p_j, s_j = step_adamw_jax(p_j, s_j, grads)
    jax.block_until_ready(p_j)
    t_adamw_jax = (time.perf_counter() - start) * 1000 / 100

    # Warmup tikun
    tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, m2_ptrs, lengths, 1.0, "adamw", 1e-3, 0.9, 0.999, 1e-8, 0.01)
    start = time.perf_counter()
    for _ in range(100):
        tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, m2_ptrs, lengths, 1.0, "adamw", 1e-3, 0.9, 0.999, 1e-8, 0.01)
    t_adamw_tikun = (time.perf_counter() - start) * 1000 / 100

    # -------------------------------------------------------------
    # 2. Lion Benchmark
    # -------------------------------------------------------------
    opt_lion_jax = optax.lion(learning_rate=1e-4, b1=0.9, b2=0.99, weight_decay=0.01)
    state_lion = opt_lion_jax.init(params)
    @jax.jit
    def step_lion_jax(p, s, g):
        u, s = opt_lion_jax.update(g, s, p)
        return optax.apply_updates(p, u), s

    p_j, s_j = step_lion_jax(params, state_lion, grads)
    jax.block_until_ready(p_j)
    start = time.perf_counter()
    for _ in range(100):
        p_j, s_j = step_lion_jax(p_j, s_j, grads)
    jax.block_until_ready(p_j)
    t_lion_jax = (time.perf_counter() - start) * 1000 / 100

    # Warmup tikun
    tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, [], lengths, 1.0, "lion", 1e-4, 0.9, 0.99, 1e-8, 0.01)
    start = time.perf_counter()
    for _ in range(100):
        tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, [], lengths, 1.0, "lion", 1e-4, 0.9, 0.99, 1e-8, 0.01)
    t_lion_tikun = (time.perf_counter() - start) * 1000 / 100

    # -------------------------------------------------------------
    # 3. SGD with Momentum Benchmark
    # -------------------------------------------------------------
    opt_sgd_jax = optax.chain(
        optax.add_decayed_weights(0.01),
        optax.sgd(learning_rate=1e-2, momentum=0.9)
    )
    state_sgd = opt_sgd_jax.init(params)
    @jax.jit
    def step_sgd_jax(p, s, g):
        u, s = opt_sgd_jax.update(g, s, p)
        return optax.apply_updates(p, u), s

    p_j, s_j = step_sgd_jax(params, state_sgd, grads)
    jax.block_until_ready(p_j)
    start = time.perf_counter()
    for _ in range(100):
        p_j, s_j = step_sgd_jax(p_j, s_j, grads)
    jax.block_until_ready(p_j)
    t_sgd_jax = (time.perf_counter() - start) * 1000 / 100

    # Warmup tikun
    tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, [], lengths, 1.0, "sgd", 1e-2, 0.9, 0.0, 1e-8, 0.01)
    start = time.perf_counter()
    for _ in range(100):
        tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, [], lengths, 1.0, "sgd", 1e-2, 0.9, 0.0, 1e-8, 0.01)
    t_sgd_tikun = (time.perf_counter() - start) * 1000 / 100

    # -------------------------------------------------------------
    # Print Results Matrix
    # -------------------------------------------------------------
    print(f"\n📊 OPTIMIZER SUITE BENCHMARK MATRIX (2.6M Parameters):")
    print(f"--------------------------------------------------------------------------")
    print(f"  • AdamW:")
    print(f"      - JAX / Optax JIT: {t_adamw_jax:.3f} ms / step")
    print(f"      - tikun (Zero-Copy): {t_adamw_tikun:.3f} ms / step")
    print(f"      - Speedup: {t_adamw_jax / t_adamw_tikun:.2f}x faster")
    print(f"--------------------------------------------------------------------------")
    print(f"  • Lion (Sign Momentum):")
    print(f"      - JAX / Optax JIT: {t_lion_jax:.3f} ms / step")
    print(f"      - tikun (Zero-Copy): {t_lion_tikun:.3f} ms / step")
    print(f"      - Speedup: {t_lion_jax / t_lion_tikun:.2f}x faster")
    print(f"--------------------------------------------------------------------------")
    print(f"  • SGD (with Momentum):")
    print(f"      - JAX / Optax JIT: {t_sgd_jax:.3f} ms / step")
    print(f"      - tikun (Zero-Copy): {t_sgd_tikun:.3f} ms / step")
    print(f"      - Speedup: {t_sgd_jax / t_sgd_tikun:.2f}x faster")
    print(f"==========================================================================")

if __name__ == "__main__":
    benchmark()
