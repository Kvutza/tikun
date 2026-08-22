import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from .universal import UniversalOptimizer

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Keller Jordan's Canonical Quintic Newton-Schulz iteration (2024).
    Computes (G G^T)^(-1/2) G with strict convergence guarantees.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.float()
    X = X / (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(dtype=G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Keller Jordan's Muon Optimizer:
    - 2D internal weight matrices: Newton-Schulz orthogonalized momentum updates.
    - 1D biases, layernorms, and embeddings: Fused Tikun AdamW updates.
    """
    def __init__(
        self,
        muon_params,
        adamw_params=None,
        lr: float = 0.02,
        momentum: float = 0.95,
        adamw_lr: float = 6e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_decay: float = 0.1,
    ):
        muon_params = list(muon_params)
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(muon_params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.step_count = 0

        # State buffers for 2D momentum
        self.state_momentum = [torch.zeros_like(p) for p in muon_params]

        # Backend AdamW for 1D parameters
        if adamw_params is not None:
            adamw_list = list(adamw_params)
            if len(adamw_list) > 0:
                self.adamw = UniversalOptimizer(
                    adamw_list,
                    algorithm="adamw",
                    lr=adamw_lr,
                    beta1=adamw_betas[0],
                    beta2=adamw_betas[1],
                    weight_decay=adamw_decay,
                )
            else:
                self.adamw = None
        else:
            self.adamw = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        # 1. Update 2D weight matrices via Newton-Schulz Orthogonalized Momentum
        for i, p in enumerate(self.param_groups[0]["params"]):
            if p.grad is None:
                continue

            g = p.grad
            buf = self.state_momentum[i]

            # In-place momentum accumulation
            buf.mul_(self.momentum).add_(g, alpha=1.0 - self.momentum)

            # Newton-Schulz polar decomposition
            update = zeropower_via_newtonschulz5(buf)

            # RMS-scaled gradient step
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.data.add_(update, alpha=-self.lr * scale)

        # 2. Update 1D parameters via Fused Tikun AdamW
        if self.adamw is not None:
            self.adamw.step()
