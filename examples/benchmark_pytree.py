import time
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def benchmark():
    print(f"🚀 Running Multi-Buffer JAX PyTree Benchmark (TRUE ZERO-COPY)")

    # 1. Create a structured PyTree model (5 parameters of varying shapes)
    key = jax.random.PRNGKey(42)
    params = {
        "w1": jax.random.normal(key, (1000, 1000), dtype=jnp.float32), # 1,000,000 floats
        "b1": jax.random.normal(key, (1000,), dtype=jnp.float32),      # 1,000 floats
        "w2": jax.random.normal(key, (512, 1000), dtype=jnp.float32),  # 512,000 floats
        "b2": jax.random.normal(key, (512,), dtype=jnp.float32),       # 512 floats
        "w3": jax.random.normal(key, (10, 512), dtype=jnp.float32)     # 5,120 floats
    }

    grads = {
        "w1": jax.random.normal(key, (1000, 1000), dtype=jnp.float32),
        "b1": jax.random.normal(key, (1000,), dtype=jnp.float32),
        "w2": jax.random.normal(key, (512, 1000), dtype=jnp.float32),
        "b2": jax.random.normal(key, (512,), dtype=jnp.float32),
        "w3": jax.random.normal(key, (10, 512), dtype=jnp.float32)
    }

    # Flatten PyTrees to match leaves
    params_flat, treedef = jax.tree_util.tree_flatten(params)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)

    # Initialize moment states as numpy arrays (acting as resident memory buffers)
    m1_flat = [np.zeros_like(p) for p in params_flat]
    m2_flat = [np.zeros_like(p) for p in params_flat]

    # Convert JAX arrays to numpy views to access ctypes pointers
    params_views = [np.asarray(p) for p in params_flat]
    grads_views = [np.asarray(g) for g in grads_flat]

    # 2. Extract zero-copy buffer pointers
    buffers = []
    for p, g, m1, m2 in zip(params_views, grads_views, m1_flat, m2_flat):
        buffers.append({
            "param_ptr": p.ctypes.data,
            "grad_ptr": g.ctypes.data,
            "m1_ptr": m1.ctypes.data,
            "m2_ptr": m2.ctypes.data,
            "length": p.size
        })

    # 3. Execute step over all PyTree buffers in a single FFI dispatch!
    start = time.perf_counter()
    tikun.step_pytree(buffers, 0.01, 0.9, 0.999, 1e-8, 0.01)
    tikun_time = (time.perf_counter() - start) * 1000

    print(f"✅ tikun step_pytree Success!")
    print(f"  - Total buffers updated: {len(buffers)} model leaves")
    print(f"  - Total execution time: {tikun_time:.4f} ms")
    print(f"  - Sample b1 parameter[0] updated value: {params_views[1][0]:.6f}")

if __name__ == "__main__":
    benchmark()
