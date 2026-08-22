import time
import sys
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, "target/release")
import _tikun as tikun

def run_scale_experiment(num_params: int, name: str):
    print(f"\n==========================================================================")
    print(f"🔥 Scaling Benchmark: {name} ({num_params:,} Parameters / {num_params * 16 / (1024*1024):.1f} MB RAM)")
    print(f"==========================================================================")

    # 1. Setup JAX arrays
    key = jax.random.PRNGKey(42)
    params_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)
    grads_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    opt_state = optimizer.init(params_jax)

    @jax.jit
    def step_optax(p, s, g):
        u, s = optimizer.update(g, s, p)
        return optax.apply_updates(p, u), s

    # Warmup Optax
    p_j, s_j = step_optax(params_jax, opt_state, grads_jax)
    jax.block_until_ready(p_j)

    # Time Optax JIT step
    start = time.perf_counter()
    for _ in range(10):
        p_j, s_j = step_optax(p_j, s_j, grads_jax)
    jax.block_until_ready(p_j)
    t_optax = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ Optax + JAX JIT Time: {t_optax:.2f} ms / step")

    # 2. tikun Contiguous Memory Arena
    p_np = np.asarray(params_jax)
    g_np = np.asarray(grads_jax)
    m1_np = np.zeros(num_params, dtype=np.float32)
    m2_np = np.zeros(num_params, dtype=np.float32)

    p_ptr = p_np.ctypes.data
    g_ptr = g_np.ctypes.data
    m1_ptr = m1_np.ctypes.data
    m2_ptr = m2_np.ctypes.data

    # Warmup tikun
    tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", 1, 0.001, 0.9, 0.999, 1e-8, 0.01)

    # Time tikun step
    start = time.perf_counter()
    for step in range(1, 11):
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 1.0, "adamw", step, 0.001, 0.9, 0.999, 1e-8, 0.01)
    t_tikun = (time.perf_counter() - start) * 1000 / 10
    print(f"⚡ tikun Arena Time:     {t_tikun:.2f} ms / step")

    data_gb = (num_params * 28) / (1024 * 1024 * 1024)
    bw_optax = data_gb / (t_optax / 1000)
    bw_tikun = data_gb / (t_tikun / 1000)

    speedup = t_optax / t_tikun
    print(f"📊 Memory Throughput: Optax: {bw_optax:.2f} GB/s | tikun: {bw_tikun:.2f} GB/s")
    print(f"🏆 Speedup at {name}: tikun is {speedup:.2f}x faster!")

def main():
    run_scale_experiment(10_000_000, "10M Parameters")
    run_scale_experiment(50_000_000, "50M Parameters")
    run_scale_experiment(100_000_000, "100M Parameters")

if __name__ == "__main__":
    main()
