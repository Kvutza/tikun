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
    print(f"🚀 Benchmarking Contiguous Resident Memory Arena vs. Optax + JAX")
    print(f"==========================================================================")

    # 1. Total Model Parameters: 2,600,000 floats (~10 MB)
    total_elements = 2_600_000

    key = jax.random.PRNGKey(42)
    params_jax = jax.random.normal(key, (total_elements,), dtype=jnp.float32)
    grads_jax = jax.random.normal(key, (total_elements,), dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    opt_state = optimizer.init(params_jax)

    @jax.jit
    def step_optax(p, s, g):
        u, s = optimizer.update(g, s, p)
        return optax.apply_updates(p, u), s

    p_j, s_j = step_optax(params_jax, opt_state, grads_jax)
    jax.block_until_ready(p_j)

    start = time.perf_counter()
    for _ in range(100):
        p_j, s_j = step_optax(p_j, s_j, grads_jax)
    jax.block_until_ready(p_j)
    t_optax = (time.perf_counter() - start) * 1000 / 100
    print(f"⏱️ Optax + JAX JIT Time: {t_optax:.3f} ms / step")

    # 2. tikun Contiguous Memory Arena
    p_np = np.asarray(params_jax)
    g_np = np.asarray(grads_jax)
    m1_np = np.zeros(total_elements, dtype=np.float32)
    m2_np = np.zeros(total_elements, dtype=np.float32)

    p_ptr = p_np.ctypes.data
    g_ptr = g_np.ctypes.data
    m1_ptr = m1_np.ctypes.data
    m2_ptr = m2_np.ctypes.data

    # Warmup
    tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [total_elements], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)

    start = time.perf_counter()
    for _ in range(100):
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [total_elements], 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)
    t_tikun = (time.perf_counter() - start) * 1000 / 100
    print(f"⚡ tikun Contiguous Arena Time: {t_tikun:.3f} ms / step")

    speedup = t_optax / t_tikun
    print(f"\n🏆 Final Result: tikun is {speedup:.2f}x faster than Optax + JAX on Contiguous Memory Arena!")

if __name__ == "__main__":
    benchmark()
