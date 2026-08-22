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
    print(f"🚀 Benchmarking Full LLM Optimizer Pipeline: Clipped AdamW")
    print(f"   Recipe: optax.chain(clip_by_global_norm(1.0), add_decayed_weights, adam)")
    print(f"==========================================================================")

    # 1. Model Structure (5 layers of varying dimensions, total ~2.6M parameters)
    key = jax.random.PRNGKey(42)
    params = {
        "embed": jax.random.normal(key, (1000, 1024), dtype=jnp.float32),   # 1,024,000 floats
        "qkv_proj": jax.random.normal(key, (1024, 1024), dtype=jnp.float32),# 1,048,576 floats
        "out_proj": jax.random.normal(key, (1024, 512), dtype=jnp.float32), # 524,288 floats
        "norm_w": jax.random.normal(key, (1024,), dtype=jnp.float32),       # 1,024 floats
        "norm_b": jax.random.normal(key, (1024,), dtype=jnp.float32),       # 1,024 floats
    }

    grads = {
        "embed": jax.random.normal(key, (1000, 1024), dtype=jnp.float32),
        "qkv_proj": jax.random.normal(key, (1024, 1024), dtype=jnp.float32),
        "out_proj": jax.random.normal(key, (1024, 512), dtype=jnp.float32),
        "norm_w": jax.random.normal(key, (1024,), dtype=jnp.float32),
        "norm_b": jax.random.normal(key, (1024,), dtype=jnp.float32),
    }

    total_params = sum(p.size for p in params.values())
    print(f"📊 Total Model Size: {total_params:,} parameters (~{total_params * 4 / (1024*1024):.2f} MB weights)")

    # -------------------------------------------------------------
    # 2. Optax + JAX JIT Baseline
    # -------------------------------------------------------------
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step_optax(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Warmup JIT
    params_optax, opt_state = step_optax(params, opt_state, grads)
    jax.block_until_ready(params_optax)

    # Time Optax JIT step
    start = time.perf_counter()
    for _ in range(100):
        params_optax, opt_state = step_optax(params_optax, opt_state, grads)
    jax.block_until_ready(params_optax)
    optax_time = (time.perf_counter() - start) * 1000 / 100
    print(f"\n⏱️ Optax + JAX JIT Time: {optax_time:.3f} ms / step")

    # -------------------------------------------------------------
    # 3. tikun Two-Tier Engine (Direct Pointer Passing)
    # -------------------------------------------------------------
    params_flat, _ = jax.tree_util.tree_flatten(params)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)

    params_views = [np.asarray(p) for p in params_flat]
    grads_views = [np.asarray(g) for g in grads_flat]
    m1_flat = [np.zeros_like(p) for p in params_flat]
    m2_flat = [np.zeros_like(p) for p in params_flat]

    param_ptrs = [p.ctypes.data for p in params_views]
    grad_ptrs = [g.ctypes.data for g in grads_views]
    m1_ptrs = [m.ctypes.data for m in m1_flat]
    m2_ptrs = [m.ctypes.data for m in m2_flat]
    lengths = [p.size for p in params_views]

    # Warmup tikun
    clip_scale = tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, m2_ptrs, lengths, 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)

    # Time tikun step
    start = time.perf_counter()
    for _ in range(100):
        tikun.step_fast_buffers(param_ptrs, grad_ptrs, m1_ptrs, m2_ptrs, lengths, 1.0, "adamw", 0.001, 0.9, 0.999, 1e-8, 0.01)
    tikun_time = (time.perf_counter() - start) * 1000 / 100
    print(f"⚡ tikun Two-Tier Engine Time: {tikun_time:.3f} ms / step (Global Clip Scale: {clip_scale:.4f})")

    speedup = optax_time / tikun_time
    print(f"\n🏆 Final Result: tikun is {speedup:.2f}x faster than Optax + JAX on Full LLM Pipeline!")

if __name__ == "__main__":
    benchmark()
