import time
import sys
import numpy as np

sys.path.insert(0, "target/debug")
import tikun

def unfused_adamw_python(params, grads, m, v, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, decay=0.01):
    # Pass 1: First moment
    m = b1 * m + (1.0 - b1) * grads
    # Pass 2: Second moment
    v = b2 * v + (1.0 - b2) * (grads ** 2)
    # Pass 3: Bias correction
    m_hat = m / (1.0 - b1)
    v_hat = v / (1.0 - b2)
    # Pass 4: Parameter update & weight decay
    update = (m_hat / (np.sqrt(v_hat) + eps)) + (decay * params)
    params = params - lr * update
    return params, m, v

def benchmark():
    num_elements = 10_000_000  # 10 Million Parameters (~40MB per buffer, 160MB total state)
    print(f"🚀 Running Benchmark: {num_elements:,} parameters (10M floats)")
    
    # Initialize random arrays
    np.random.seed(42)
    params_np = np.random.randn(num_elements).astype(np.float32)
    grads_np = np.random.randn(num_elements).astype(np.float32)
    m_np = np.zeros(num_elements, dtype=np.float32)
    v_np = np.zeros(num_elements, dtype=np.float32)

    params_list = params_np.tolist()
    grads_list = grads_np.tolist()
    m_list = m_np.tolist()
    v_list = v_np.tolist()

    # 1. Benchmark Unfused Python NumPy
    start = time.perf_counter()
    _ = unfused_adamw_python(params_np, grads_np, m_np, v_np)
    py_time = (time.perf_counter() - start) * 1000
    print(f"  - Unfused NumPy AdamW Time: {py_time:.2f} ms")

    # 2. Benchmark tikun Fused Engine
    start = time.perf_counter()
    _ = tikun.adamw_step(params_list, grads_list, m_list, v_list, 0.001, 0.9, 0.999, 1e-8, 0.01)
    tikun_time = (time.perf_counter() - start) * 1000
    print(f"  - tikun Fused SIMD Engine Time: {tikun_time:.2f} ms")

    if tikun_time > 0:
        speedup = py_time / tikun_time
        print(f"⚡ tikun Speedup: {speedup:.2f}x faster!")

if __name__ == "__main__":
    benchmark()
