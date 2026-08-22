import sys
import os
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

def demo():
    # -------------------------------------------------------------
    # 1. CPU Hardware SIMD Lowering Inspection (PyTorch Model)
    # -------------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, 512),
    )
    tikun.inspect_cpu_plan(model.parameters(), algorithm="adamw")

    # -------------------------------------------------------------
    # 2. JAX ClosedJaxpr IR Equation Inspection
    # -------------------------------------------------------------
    def jax_adamw_step(p, g, m1, m2, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, decay=0.01):
        m1_next = b1 * m1 + (1.0 - b1) * g
        m2_next = b2 * m2 + (1.0 - b2) * (g * g)
        m_hat = m1_next / (1.0 - b1)
        v_hat = m2_next / (1.0 - b2)
        step = (m_hat / (jnp.sqrt(v_hat) + eps)) + decay * p
        p_next = p - lr * step
        return p_next, m1_next, m2_next

    dummy_p = jnp.ones((4, 4), dtype=jnp.float32)
    dummy_g = jnp.full((4, 4), 0.1, dtype=jnp.float32)
    dummy_m1 = jnp.zeros((4, 4), dtype=jnp.float32)
    dummy_m2 = jnp.zeros((4, 4), dtype=jnp.float32)

    tikun.inspect_jaxpr(jax_adamw_step, dummy_p, dummy_g, dummy_m1, dummy_m2)

if __name__ == "__main__":
    demo()
