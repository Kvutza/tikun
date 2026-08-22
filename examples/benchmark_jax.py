import time
import sys
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, "target/release")
import tikun

def benchmark():
    num_elements = 1_000_000 # 1M parameters
    print(f"🚀 Benchmarking 1M Parameters: JAX/Optax vs. tikun + JAX (TRUE ZERO-COPY)")

    # 1. Setup JAX arrays
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (num_elements,), dtype=jnp.float32)
    grads = jax.random.normal(key, (num_elements,), dtype=jnp.float32)
    
    # Optax Setup
    optimizer = optax.adamw(learning_rate=0.01, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    opt_state = optimizer.init(params)

    # Compile the JAX JIT step first
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
    print(f"  - Optax JIT compiled step: {optax_time:.3f} ms")

    # 2. Setup tikun + JAX step (TRUE ZERO-COPY)
    # We retrieve the raw pointer addresses of the JAX memory buffers
    # by converting to numpy view (zero-copy) and accessing .ctypes.data
    params_view = np.asarray(params)
    grads_view = np.asarray(grads)
    m1_np = np.zeros(num_elements, dtype=np.float32)
    m2_np = np.zeros(num_elements, dtype=np.float32)

    params_ptr = params_view.ctypes.data
    grads_ptr = grads_view.ctypes.data
    m1_ptr = m1_np.ctypes.data
    m2_ptr = m2_np.ctypes.data

    # Time tikun step directly over JAX pointers!
    start = time.perf_counter()
    for _ in range(100):
        tikun.step_raw_pointers(
            params_ptr,
            grads_ptr,
            m1_ptr,
            m2_ptr,
            num_elements,
            0.01,
            0.9,
            0.999,
            1e-8,
            0.01
        )
    tikun_time = (time.perf_counter() - start) * 1000 / 100
    print(f"  - tikun + JAX zero-copy step: {tikun_time:.3f} ms")

    speedup = optax_time / tikun_time
    print(f"⚡ tikun + JAX speedup: {speedup:.2f}x faster!")

if __name__ == "__main__":
    benchmark()
