import sys
import os
import json
import numpy as np

# Load native Rust extension
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release")))
import _tikun as _native

def _load_hardware_profile():
    """Loads auto-tuned hardware profile from standard platform cache."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_path = os.path.join(xdg_cache, "tikun", "profile.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _extract_pointer_and_len(tensor):
    """
    Framework-agnostic pointer extraction: Supports PyTorch, JAX, NumPy, MLX, and raw buffers.
    """
    if hasattr(tensor, "data_ptr") and hasattr(tensor, "numel"):
        return tensor.data_ptr(), tensor.numel()
    if isinstance(tensor, np.ndarray):
        return tensor.ctypes.data, tensor.size
    if hasattr(tensor, "__array__"):
        arr = np.asarray(tensor)
        return arr.ctypes.data, arr.size
    if isinstance(tensor, (tuple, list)) and len(tensor) == 2 and isinstance(tensor[0], int):
        return int(tensor[0]), int(tensor[1])
    raise TypeError(f"Unsupported tensor type: {type(tensor)}. Must be PyTorch, JAX, NumPy, or (ptr, len).")

def _flatten_pytree(container):
    """
    Framework-agnostic tree flattening (handles dicts, lists, tuples, PyTrees, PyTorch param generators).
    """
    if isinstance(container, dict):
        items = []
        for v in container.values():
            items.extend(_flatten_pytree(v))
        return items
    elif isinstance(container, (list, tuple)):
        if len(container) == 2 and isinstance(container[0], int) and isinstance(container[1], int):
            return [container]
        items = []
        for item in container:
            items.extend(_flatten_pytree(item))
        return items
    elif hasattr(container, "__iter__") and not hasattr(container, "data_ptr") and not isinstance(container, (str, bytes, np.ndarray)):
        items = []
        for item in container:
            items.extend(_flatten_pytree(item))
        return items
    else:
        return [container]

class UniversalOptimizer:
    """
    Universal, framework-agnostic optimizer engine with Auto-Tuned Hardware Schedule feedback.
    Supports PyTorch, JAX, NumPy, MLX, and raw C-pointers.
    All compiler scheduling, MLIR generation, and vector kernels execute in pure Rust.
    """
    def __init__(
        self,
        params,
        algorithm: str = "adamw",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_norm: float = 0.0,
        in_backward: bool = False,
    ):
        self.params = _flatten_pytree(params)
        self.algorithm = algorithm.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.step_count = 0
        self.in_backward = in_backward

        # Load active auto-tuned silicon profile
        self.hardware_profile = _load_hardware_profile()

        # Persistent momentum buffers
        self.m1_state = {}
        self.m2_state = {}

        if self.in_backward:
            self._hooks = []
            for p in self.params:
                if hasattr(p, "register_post_accumulate_grad_hook"):
                    hook = p.register_post_accumulate_grad_hook(self._make_in_backward_hook(p))
                    self._hooks.append(hook)

    def _make_in_backward_hook(self, p):
        def _hook(param):
            if param.grad is None:
                return
            self._step_single_tensor(param)
            param.grad = None
        return _hook

    def _step_single_tensor(self, p):
        p_ptr, length = _extract_pointer_and_len(p)
        g_ptr, _ = _extract_pointer_and_len(p.grad)

        if p_ptr not in self.m1_state:
            self.m1_state[p_ptr] = np.zeros(length, dtype=np.float32)
            self.m2_state[p_ptr] = np.zeros(length, dtype=np.float32)

        m1_ptr = self.m1_state[p_ptr].ctypes.data
        m2_ptr = self.m2_state[p_ptr].ctypes.data

        _native.step_fast_buffers(
            [p_ptr],
            [g_ptr],
            [m1_ptr],
            [m2_ptr],
            [length],
            self.max_norm,
            self.algorithm,
            max(1, self.step_count),
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
        )

    def step(self, grads=None):
        self.step_count += 1
        if self.in_backward:
            return

        param_ptrs = []
        grad_ptrs = []
        m1_ptrs = []
        m2_ptrs = []
        lengths = []

        flat_grads = _flatten_pytree(grads) if grads is not None else None

        for i, p in enumerate(self.params):
            p_ptr, length = _extract_pointer_and_len(p)

            if flat_grads is not None:
                g_ptr, _ = _extract_pointer_and_len(flat_grads[i])
            elif hasattr(p, "grad") and p.grad is not None:
                g_ptr, _ = _extract_pointer_and_len(p.grad)
            else:
                continue

            if p_ptr not in self.m1_state:
                self.m1_state[p_ptr] = np.zeros(length, dtype=np.float32)
                self.m2_state[p_ptr] = np.zeros(length, dtype=np.float32)

            m1_ptr = self.m1_state[p_ptr].ctypes.data
            m2_ptr = self.m2_state[p_ptr].ctypes.data

            param_ptrs.append(p_ptr)
            grad_ptrs.append(g_ptr)
            m1_ptrs.append(m1_ptr)
            m2_ptrs.append(m2_ptr)
            lengths.append(length)

        if not param_ptrs:
            return

        _native.step_fast_buffers(
            param_ptrs,
            grad_ptrs,
            m1_ptrs,
            m2_ptrs,
            lengths,
            self.max_norm,
            self.algorithm,
            self.step_count,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
        )

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad = None

    def to_mlir(self) -> str:
        param_ptrs = [_extract_pointer_and_len(p)[0] for p in self.params]
        lengths = [_extract_pointer_and_len(p)[1] for p in self.params]
        hw = self.hardware_profile or {}
        tile_kb = hw.get("tile_size_kb", 512)
        unroll = hw.get("unroll_factor", 4)
        prefetch = hw.get("prefetch_bytes", 128)
        return _native.emit_mlir(
            param_ptrs, lengths, self.algorithm, self.max_norm, self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
            tile_kb, unroll, prefetch
        )

    def to_json(self) -> str:
        param_ptrs = [_extract_pointer_and_len(p)[0] for p in self.params]
        lengths = [_extract_pointer_and_len(p)[1] for p in self.params]
        hw = self.hardware_profile or {}
        tile_kb = hw.get("tile_size_kb", 512)
        unroll = hw.get("unroll_factor", 4)
        prefetch = hw.get("prefetch_bytes", 128)
        workers = hw.get("thread_chunks", 12)
        return _native.emit_inspect(
            param_ptrs, lengths, self.algorithm, self.max_norm, self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
            tile_kb, unroll, prefetch, workers
        )

    def inspect(self):
        print(self.to_json())
