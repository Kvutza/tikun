import time
import sys
import random

sys.path.insert(0, "target/debug")
import tikun

def benchmark():
    num_elements = 100_000  # 100,000 floats
    print(f"🚀 Running Pure Benchmark: {num_elements:,} parameters")

    random.seed(42)
    params = [random.random() for _ in range(num_elements)]
    grads = [random.random() for _ in range(num_elements)]
    m = [0.0] * num_elements
    v = [0.0] * num_elements

    # Benchmark tikun Fused Engine
    start = time.perf_counter()
    new_p, new_m, new_v = tikun.adamw_step(params, grads, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01)
    tikun_time = (time.perf_counter() - start) * 1000

    print(f"✅ tikun Fused Engine Execution Time: {tikun_time:.2f} ms")
    print(f"  - Parameters updated: {len(new_p):,} elements")
    print(f"  - Sample parameter[0]: {new_p[0]:.6f}")

if __name__ == "__main__":
    benchmark()
