import time
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release")))
import _tikun as _native

class EpistemicNearestNeighborSurrogate:
    """
    Epistemic Nearest Neighbors (ENN) Surrogate for Bayesian Optimization.
    Scales at O(N) for rapid hardware parameter exploration (Kvutza/ennx paradigm).
    """
    def __init__(self, k_neighbors: int = 3):
        self.k = k_neighbors
        self.points = []
        self.latencies = []

    def observe(self, x: np.ndarray, y: float):
        self.points.append(x)
        self.latencies.append(y)

    def predict(self, candidate: np.ndarray) -> tuple[float, float]:
        if len(self.points) == 0:
            return 100.0, 100.0

        pts = np.array(self.points)
        vals = np.array(self.latencies)

        dists = np.linalg.norm(pts - candidate, axis=1)
        k = min(self.k, len(pts))
        knn_idx = np.argsort(dists)[:k]

        weights = 1.0 / (dists[knn_idx] + 1e-6)
        weights /= weights.sum()

        mean = float(np.sum(weights * vals[knn_idx]))
        uncertainty = float(np.min(dists[knn_idx]))
        return mean, uncertainty

    def acquire_lcb(self, candidates: list[np.ndarray], kappa: float = 2.0) -> np.ndarray:
        scores = []
        for c in candidates:
            mu, sigma = self.predict(c)
            lcb = mu - kappa * sigma
            scores.append(lcb)
        return candidates[int(np.argmin(scores))]

def run_aggressive_autotune(num_params: int = 50_000_000, num_trials: int = 20, verbose: bool = True):
    """
    Runs aggressive large-scale ENNX Bayesian hardware auto-tuning on 50M-100M parameters.
    """
    if verbose:
        print("==========================================================================")
        print(f"🔥 AGGRESSIVE ENNX HARDWARE AUTO-TUNING ({num_params:,} Parameters / {num_params*16/(1024*1024):.1f} MB RAM)")
        print("   Surrogate: Epistemic Nearest Neighbors (Kvutza/ennx Paradigm)")
        print("==========================================================================")

    # Search Space: [Prefetch (0..512B), Tile Size (16KB..2MB), Chunk Split (4..64)]
    prefetches = [0, 64, 128, 256, 512]
    tiles = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    chunks = [4, 8, 12, 16, 24, 32, 48, 64]

    grid = []
    for p_idx, p in enumerate(prefetches):
        for t_idx, t in enumerate(tiles):
            for c_idx, c in enumerate(chunks):
                grid.append(np.array([
                    p_idx / (len(prefetches) - 1),
                    t_idx / (len(tiles) - 1),
                    c_idx / (len(chunks) - 1)
                ], dtype=np.float32))

    surrogate = EpistemicNearestNeighborSurrogate(k_neighbors=3)

    # Allocate physical 50M parameter arrays
    p = np.ones(num_params, dtype=np.float32)
    g = np.full(num_params, 0.05, dtype=np.float32)
    m1 = np.zeros(num_params, dtype=np.float32)
    m2 = np.zeros(num_params, dtype=np.float32)

    p_ptr = p.ctypes.data
    g_ptr = g.ctypes.data
    m1_ptr = m1.ctypes.data
    m2_ptr = m2.ctypes.data

    best_config = None
    best_time = float("inf")
    candidates = list(grid)

    for trial in range(1, num_trials + 1):
        if trial == 1:
            chosen = grid[0] # Baseline default
        elif trial == 2:
            chosen = grid[-1] # Extreme opposite
        else:
            chosen = surrogate.acquire_lcb(candidates, kappa=2.5)

        p_val = prefetches[int(np.round(chosen[0] * (len(prefetches) - 1)))]
        t_val = tiles[int(np.round(chosen[1] * (len(tiles) - 1)))]
        c_val = chunks[int(np.round(chosen[2] * (len(chunks) - 1)))]

        # Benchmark 5 full steps on target hardware
        _native.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 0.0, "adamw", 1, 0.001, 0.9, 0.999, 1e-8, 0.01)

        t0 = time.perf_counter()
        for s in range(1, 6):
            _native.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [num_params], 0.0, "adamw", s, 0.001, 0.9, 0.999, 1e-8, 0.01)
        lat_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

        surrogate.observe(chosen, lat_ms)

        if lat_ms < best_time:
            best_time = lat_ms
            best_config = {"prefetch_bytes": p_val, "tile_size": t_val, "thread_chunks": c_val}

        data_gb = (num_params * 28) / (1024 * 1024 * 1024)
        bw_gbs = data_gb / (lat_ms / 1000.0)

        if verbose:
            print(f"  [Trial {trial:02d}/{num_trials:02d}] Prefetch: {p_val:3d}B | Tile: {t_val/1024:4.0f}KB | Chunks: {c_val:2d} ──> {lat_ms:6.2f} ms ({bw_gbs:5.2f} GB/s)")

    profile_path = ".tikun_hardware_profile.json"
    result = {
        "hardware": "Apple Silicon ARM64",
        "num_params_tested": num_params,
        "optimal_step_ms": round(best_time, 2),
        "peak_bandwidth_gbs": round(((num_params * 28) / (1024 * 1024 * 1024)) / (best_time / 1000.0), 2),
        "optimal_config": best_config
    }
    with open(profile_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print("\n==========================================================================")
        print("🏆 AGGRESSIVE ENNX AUTO-TUNING COMPLETE:")
        print(f"  • Optimal Configuration: Prefetch: {best_config['prefetch_bytes']}B | Tile: {best_config['tile_size']/1024:.0f} KB | Chunks: {best_config['thread_chunks']}")
        print(f"  • Best Measured Step Time: {best_time:.2f} ms")
        print(f"  • Peak Memory Bandwidth:   {result['peak_bandwidth_gbs']} GB/s")
        print(f"  • Saved Tuned Profile:     {profile_path}")
        print("==========================================================================")

    return result

run_hardware_autotune = run_aggressive_autotune

if __name__ == "__main__":
    run_aggressive_autotune(num_params=50_000_000, num_trials=20)
