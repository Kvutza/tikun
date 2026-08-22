import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, "target/release")
import tikun

def verify_numerical_parity():
    print("==========================================================================")
    print("🔬 VERIFYING STRICT MATHEMATICAL PARITY: PyTorch vs. Optax vs. tikun")
    print("==========================================================================")

    np.random.seed(42)
    n = 10000
    p_init = np.random.randn(n).astype(np.float32)
    g_init = np.random.randn(n).astype(np.float32)

    lr = 0.01
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    decay = 0.01

    # 1. PyTorch Baseline
    p_torch = torch.tensor(p_init.copy(), requires_grad=True)
    p_torch.grad = torch.tensor(g_init.copy())
    opt_torch = torch.optim.AdamW([p_torch], lr=lr, betas=(b1, b2), eps=eps, weight_decay=decay)
    
    # 2. Optax / JAX Baseline
    p_jax = jnp.array(p_init.copy())
    g_jax = jnp.array(g_init.copy())
    opt_jax = optax.adamw(learning_rate=lr, b1=b1, b2=b2, eps=eps, weight_decay=decay)
    state_jax = opt_jax.init(p_jax)

    # 3. tikun Baseline
    p_tikun = p_init.copy()
    g_tikun = g_init.copy()
    m1_tikun = np.zeros(n, dtype=np.float32)
    m2_tikun = np.zeros(n, dtype=np.float32)

    p_ptr = p_tikun.ctypes.data
    g_ptr = g_tikun.ctypes.data
    m1_ptr = m1_tikun.ctypes.data
    m2_ptr = m2_tikun.ctypes.data

    print(f"{'Step':<6} | {'Max Diff (tikun vs Torch)':<28} | {'Max Diff (tikun vs Optax)':<28}")
    print("-" * 68)

    all_passed = True
    for step in range(1, 11):
        # Step PyTorch
        opt_torch.step()
        
        # Step Optax
        u_jax, state_jax = opt_jax.update(g_jax, state_jax, p_jax)
        p_jax = optax.apply_updates(p_jax, u_jax)

        # Step tikun with exact step count
        tikun.step_fast_buffers([p_ptr], [g_ptr], [m1_ptr], [m2_ptr], [n], 0.0, "adamw", step, lr, b1, b2, eps, decay)

        torch_res = p_torch.detach().numpy()
        jax_res = np.asarray(p_jax)

        diff_torch = np.max(np.abs(p_tikun - torch_res))
        diff_jax = np.max(np.abs(p_tikun - jax_res))

        print(f"{step:<6} | {diff_torch:<28.8e} | {diff_jax:<28.8e}")

        if diff_torch > 1e-5 or diff_jax > 1e-5:
            all_passed = False

    if all_passed:
        print("\n✅ PERFECT MATHEMATICAL PARITY: tikun matches PyTorch and Optax to float32 precision across all steps!")
    else:
        print("\n❌ Discrepancy detected.")

if __name__ == "__main__":
    verify_numerical_parity()
