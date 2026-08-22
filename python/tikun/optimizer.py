import torch
import numpy as np
import sys
import os
import importlib.util

# Load native Rust extension directly from target/release/tikun.so
_so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release/tikun.so"))
if not os.path.exists(_so_path):
    _so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release/libtikun.dylib"))

spec = importlib.util.spec_from_file_location("tikun", _so_path)
_native_tikun = importlib.util.module_from_spec(spec)
sys.modules["_native_tikun"] = _native_tikun
spec.loader.exec_module(_native_tikun)

class AdamW(torch.optim.Optimizer):
    """
    High-Performance Zero-Copy Drop-in AdamW Optimizer powered by the Tikun Native Engine.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_norm: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            beta_one=betas[0],
            beta_two=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            max_norm=max_norm,
        )
        super().__init__(params, defaults)

        self.step_count = 0
        self._param_ptrs = []
        self._grad_ptrs = []
        self._m1_ptrs = []
        self._m2_ptrs = []
        self._lengths = []
        self._moments1 = []
        self._moments2 = []

        # One-time initialization of memory buffers
        self._init_native_buffers()

    def _init_native_buffers(self):
        self._param_ptrs.clear()
        self._grad_ptrs.clear()
        self._m1_ptrs.clear()
        self._m2_ptrs.clear()
        self._lengths.clear()
        self._moments1.clear()
        self._moments2.clear()

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p_np = p.detach().numpy()
                    m1 = np.zeros_like(p_np)
                    m2 = np.zeros_like(p_np)

                    self._moments1.append(m1)
                    self._moments2.append(m2)
                    self._lengths.append(p.numel())
                    self._param_ptrs.append(p.data_ptr())
                    self._m1_ptrs.append(m1.ctypes.data)
                    self._m2_ptrs.append(m2.ctypes.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        group = self.param_groups[0]
        lr = group["lr"]
        b1 = group["beta_one"]
        b2 = group["beta_two"]
        eps = group["eps"]
        decay = group["weight_decay"]
        max_norm = group["max_norm"]

        # Collect current gradient pointers
        grad_ptrs = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    if p.grad is None:
                        raise RuntimeError(f"Gradient for parameter with shape {p.shape} is None.")
                    grad_ptrs.append(p.grad.data_ptr())

        # Execute in-register fused Rust SIMD kernel (Zero GIL retention)
        _native_tikun.step_fast_buffers(
            self._param_ptrs,
            grad_ptrs,
            self._m1_ptrs,
            self._m2_ptrs,
            self._lengths,
            max_norm,
            "adamw",
            self.step_count,
            lr,
            b1,
            b2,
            eps,
            decay,
        )

        return loss
