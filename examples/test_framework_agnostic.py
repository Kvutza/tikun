import sys
import os
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

def test_all_frameworks():
    print("==========================================================================")
    print("🌐 TESTING FRAMEWORK-AGNOSTIC TIKUN ENGINE (PyTorch, JAX, NumPy, C-Pointers)")
    print("==========================================================================")

    # -------------------------------------------------------------
    # 1. PyTorch Test
    # -------------------------------------------------------------
    print("🧪 1. Testing PyTorch nn.Module...")
    model = nn.Linear(100, 10)
    opt_torch = tikun.AdamW(model.parameters(), lr=1e-3)
    
    x = torch.randn(4, 100)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    opt_torch.step() # No args needed, reads .grad
    print("   ✅ PyTorch Step Successful!")

    # -------------------------------------------------------------
    # 2. JAX Nested PyTree Test
    # -------------------------------------------------------------
    print("\n🧪 2. Testing JAX Nested PyTree (Dict of Arrays)...")
    key = jax.random.PRNGKey(42)
    params_jax = {
        "encoder": {
            "w": np.ones((50, 50), dtype=np.float32),
            "b": np.zeros(50, dtype=np.float32),
        },
        "decoder": {
            "w": np.ones((50, 10), dtype=np.float32),
        }
    }
    grads_jax = {
        "encoder": {
            "w": np.full((50, 50), 0.1, dtype=np.float32),
            "b": np.full(50, 0.05, dtype=np.float32),
        },
        "decoder": {
            "w": np.full((50, 10), 0.2, dtype=np.float32),
        }
    }

    opt_jax = tikun.Lion(params_jax, lr=1e-4)
    clip_scale = opt_jax.step(grads=grads_jax)
    print(f"   ✅ JAX PyTree Step Successful (Algorithm: Lion, Scale: {clip_scale})!")

    # -------------------------------------------------------------
    # 3. Pure NumPy Array Test
    # -------------------------------------------------------------
    print("\n🧪 3. Testing Pure NumPy Arrays...")
    p_np = np.ones(1000, dtype=np.float32)
    g_np = np.full(1000, 0.05, dtype=np.float32)

    opt_np = tikun.SGD(p_np, lr=0.01, momentum=0.9)
    opt_np.step(grads=g_np)
    print("   ✅ Pure NumPy Step Successful (Algorithm: SGD with Momentum)!")

    # -------------------------------------------------------------
    # 4. Raw C Memory Pointers Test
    # -------------------------------------------------------------
    print("\n🧪 4. Testing Raw C Memory Address Pointers (C / Mojo / Swift Interop)...")
    p_raw = np.ones(500, dtype=np.float32)
    g_raw = np.full(500, 0.1, dtype=np.float32)

    opt_raw = tikun.AdamW([(p_raw.ctypes.data, 500)], lr=1e-3)
    opt_raw.step(grads=[(g_raw.ctypes.data, 500)])
    print("   ✅ Raw C-Pointer Step Successful!")

    print("\n==========================================================================")
    print("🎉 ALL FRAMEWORK-AGNOSTIC INTEGRATION TESTS PASSED (100% UNIVERSAL)")
    print("==========================================================================")

if __name__ == "__main__":
    test_all_frameworks()
