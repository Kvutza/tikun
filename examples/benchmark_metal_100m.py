import time
import sys
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def benchmark_scale(num_params: int, name: str):
    print(f"\n==========================================================================")
    print(f"🚀 Scaling Benchmark: {name} ({num_params:,} Params / {num_params * 16 / (1024*1024):.1f} MB RAM)")
    print(f"==========================================================================")

    # 1. Setup JAX arrays
    key = jax.random.PRNGKey(42)
    params_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)
    grads_jax = jax.random.normal(key, (num_params,), dtype=jnp.float32)

    # 2. Optax + JAX JIT
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
    for _ in range(10):
        p_j, s_j = step_optax(p_j, s_j, grads_jax)
    jax.block_until_ready(p_j)
    t_optax = (time.perf_counter() - start) * 1000 / 10
    print(f"⏱️ Optax + JAX JIT (CPU):        {t_optax:.2f} ms / step")

    # 3. tikun CPU Engine (ARM NEON SIMD)
    p_np = np.asarray(params_jax)
    g_np = np.asarray(grads_jax)
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
    t_tikun_cpu = (time.perf_counter() - start) * 1000 / 10
    print(f"⚡ tikun CPU Engine (ARM NEON):   {t_tikun_cpu:.2f} ms / step")

    # 4. tikun Metal GPU Engine (Apple Silicon Unified Memory)
    # Warmup GPU
    tikun.step_metal_gpu(p_ptr, g_ptr, m1_ptr, m2_ptr, num_params, 1.0, 0.001, 0.9, 0.999, 1e-8, 0.01)
    start = time.perf_counter()
    for _ in range(10):
        tikun.step_metal_gpu(p_ptr, g_ptr, m1_ptr, m2_ptr, num_params, 1.0, 0.001, 0.9, 0.999, 1e-8, 0.01)
    t_tikun_gpu = (time.perf_counter() - start) * 1000 / 10
    print(f"🔥 tikun Metal GPU (Apple UMA):   {t_tikun_gpu:.2f} ms / step")

    data_gb = (num_params * 28) / (1024 * 1024 * 1024)
    bw_gpu = data_gb / (t_tikun_gpu / 1000)
    print(f"📊 Metal GPU Memory Bandwidth:   {bw_gpu:.2f} GB/s")

    speedup_cpu = t_optax / t_tikun_cpu
    speedup_gpu = t_optax / t_tikun_gpu
    print(f"🏆 Speedup vs Optax: CPU {speedup_cpu:.2f}x | Metal GPU {speedup_gpu:.2f}x FASTER!")

def main():
    benchmark_scale(10_000_000, "10M Parameters")
    benchmark_scale(50_000_000, "50M Parameters")
    benchmark_scale(100_000_000, "100M Parameters")

if __name__ == "__main__":
    main()
