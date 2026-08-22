import time
import sys
import struct

sys.path.insert(0, "target/debug")
import tikun

def benchmark():
    num_elements = 1_000_000  # 1 Million parameters (4MB per buffer)
    print(f"🚀 Running Zero-Copy Benchmark: {num_elements:,} parameters (1M floats)")

    # Create raw bytearray buffers in CPython
    params_bytes = bytearray(struct.pack(f"{num_elements}f", *[1.0] * num_elements))
    grads_bytes = bytearray(struct.pack(f"{num_elements}f", *[0.1] * num_elements))
    m1_bytes = bytearray(struct.pack(f"{num_elements}f", *[0.0] * num_elements))
    m2_bytes = bytearray(struct.pack(f"{num_elements}f", *[0.0] * num_elements))

    # Benchmark tikun Zero-Copy Fast Engine
    start = time.perf_counter()
    tikun.step_bytearray_fast(params_bytes, grads_bytes, m1_bytes, m2_bytes, 0.01, 0.9, 0.999, 1e-8, 0.01)
    tikun_time = (time.perf_counter() - start) * 1000

    # Unpack updated first float
    updated_param_0 = struct.unpack("f", params_bytes[:4])[0]

    print(f"⚡ tikun Zero-Copy Engine Time: {tikun_time:.2f} ms!")
    print(f"  - Parameters updated: {num_elements:,} elements in-place")
    print(f"  - Updated parameter[0]: {updated_param_0:.6f}")

if __name__ == "__main__":
    benchmark()
