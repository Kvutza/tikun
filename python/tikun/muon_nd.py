import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release")))
import _tikun as _native
from .universal import _extract_pointer_and_len, _flatten_pytree

try:
    import torch
except ImportError:
    torch = None

class MultiDimensionalMuon:
    """
    Dimension-Aware Multi-Rank Tensor Optimizer:
      - 1D Tensors (LayerNorms, Biases): Fused in-register AdamW
      - 2D Tensors (Linear Weights): Pure Rust Quintic Newton-Schulz Polar Orthogonalization
      - 3D/4D Tensors (Multi-Head Attention, Conv/MoE): Batched Head-Wise Parallel Tensor Orthogonalization
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 1e-3,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_decay: float = 0.01,
    ):
        self.raw_params = _flatten_pytree(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adamw_lr = adamw_lr
        self.adamw_b1, self.adamw_b2 = adamw_betas
        self.adamw_decay = adamw_decay
        self.step_count = 0

        self.rank_1_params = []
        self.rank_2_params = []
        self.rank_nd_params = []

        self.momentum_state = {}
        self.adamw_m1 = {}
        self.adamw_m2 = {}

        for p in self.raw_params:
            ndim = getattr(p, "ndim", len(p.shape) if hasattr(p, "shape") else 1)
            if ndim < 2:
                self.rank_1_params.append(p)
            elif ndim == 2:
                self.rank_2_params.append(p)
            else:
                self.rank_nd_params.append(p)

    def zero_grad(self):
        for p in self.raw_params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad = None

    def step(self):
        self.step_count += 1

        # 1. Process 1D Parameters (Biases, 1D Vectors) with Fused In-Register AdamW
        if self.rank_1_params:
            p_ptrs, g_ptrs, m1_ptrs, m2_ptrs, lengths = [], [], [], [], []
            for p in self.rank_1_params:
                if not hasattr(p, "grad") or p.grad is None:
                    continue
                p_ptr, length = _extract_pointer_and_len(p)
                g_ptr, _ = _extract_pointer_and_len(p.grad)

                if p_ptr not in self.adamw_m1:
                    self.adamw_m1[p_ptr] = np.zeros(length, dtype=np.float32)
                    self.adamw_m2[p_ptr] = np.zeros(length, dtype=np.float32)

                p_ptrs.append(p_ptr)
                g_ptrs.append(g_ptr)
                m1_ptrs.append(self.adamw_m1[p_ptr].ctypes.data)
                m2_ptrs.append(self.adamw_m2[p_ptr].ctypes.data)
                lengths.append(length)

            if p_ptrs:
                _native.step_fast_buffers(
                    p_ptrs, g_ptrs, m1_ptrs, m2_ptrs, lengths,
                    0.0, "adamw", self.step_count,
                    self.adamw_lr, self.adamw_b1, self.adamw_b2, 1e-8, self.adamw_decay
                )

        # 2. Process 2D Parameters with Pure Rust Quintic Newton-Schulz Polar Decomposition
        for p in self.rank_2_params:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            p_ptr, length = _extract_pointer_and_len(p)
            g_ptr, _ = _extract_pointer_and_len(p.grad)
            rows, cols = p.shape[0], p.shape[1]

            if p_ptr not in self.momentum_state:
                self.momentum_state[p_ptr] = np.zeros((rows, cols), dtype=np.float32)

            buf = self.momentum_state[p_ptr]
            
            # Fast momentum update using PyTorch in-place ops if available
            if hasattr(p.grad, "data"):
                grad_arr = p.grad.data.cpu().numpy()
            else:
                grad_arr = np.asarray(p.grad)

            buf *= self.momentum
            buf += (1.0 - self.momentum) * grad_arr

            update_mat = (self.momentum * buf + (1.0 - self.momentum) * grad_arr) if self.nesterov else buf
            update_mat = np.ascontiguousarray(update_mat, dtype=np.float32)

            # Call pure Rust Newton-Schulz 2D Polar kernel
            _native.newton_schulz_2d_ffi(update_mat.ctypes.data, rows, cols, self.ns_steps)

            # Apply RMS-scaled update back to weights in-place
            scale = max(1.0, rows / cols) ** 0.5
            if hasattr(p, "data"):
                p.data.add_(torch.from_numpy(update_mat).to(p.device), alpha=-self.lr * scale)
            else:
                p -= self.lr * scale * update_mat

        # 3. Process 3D/4D Parameters with Batched Parallel Head-Wise Polar Decomposition
        for p in self.rank_nd_params:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            p_ptr, length = _extract_pointer_and_len(p)
            shape = p.shape
            num_heads = int(np.prod(shape[:-2]))
            head_out = shape[-2]
            head_in = shape[-1]

            if p_ptr not in self.momentum_state:
                self.momentum_state[p_ptr] = np.zeros(shape, dtype=np.float32)

            buf = self.momentum_state[p_ptr]
            if hasattr(p.grad, "data"):
                grad_arr = p.grad.data.cpu().numpy()
            else:
                grad_arr = np.asarray(p.grad)

            buf *= self.momentum
            buf += (1.0 - self.momentum) * grad_arr

            update_nd = (self.momentum * buf + (1.0 - self.momentum) * grad_arr) if self.nesterov else buf
            update_nd = np.ascontiguousarray(update_nd, dtype=np.float32)

            # Call pure Rust Batched 3D Head-Wise Newton-Schulz kernel
            _native.newton_schulz_3d_ffi(update_nd.ctypes.data, num_heads, head_out, head_in, self.ns_steps)

            scale = max(1.0, head_out / head_in) ** 0.5
            if hasattr(p, "data"):
                p.data.add_(torch.from_numpy(update_nd).to(p.device), alpha=-self.lr * scale)
            else:
                p -= self.lr * scale * update_nd
