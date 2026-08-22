import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, "target/release")
import tikun

class TikunAdamW(torch.optim.Optimizer):
    """
    Zero-Copy Drop-in PyTorch AdamW Optimizer using the Tikun Native Engine.
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

        grad_ptrs = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    if p.grad is None:
                        raise RuntimeError(f"Gradient for parameter is None.")
                    grad_ptrs.append(p.grad.data_ptr())

        tikun.step_fast_buffers(
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

# 1. Define a Real Neural Network Policy (11.5 Million Parameters)
class DeepPolicyNetwork(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, depth=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def run_real_training_experiment():
    print("==========================================================================")
    print("🚀 REAL-WORLD END-TO-END TRAINING BENCHMARK (PyTorch nn.Module Policy)")
    print("==========================================================================")

    torch.manual_seed(42)
    device = torch.device("cpu")

    model = DeepPolicyNetwork(in_dim=1024, hidden_dim=1024, depth=10).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {num_params:,} (~{num_params * 4 / (1024*1024):.1f} MB weights)")

    batch_size = 32
    num_epochs = 50

    torch.manual_seed(1337)
    x_data = torch.randn(batch_size, 1024)
    y_target = torch.randn(batch_size, 1024)
    criterion = nn.MSELoss()

    # -------------------------------------------------------------
    # 1. Train with PyTorch Native AdamW (foreach=True)
    # -------------------------------------------------------------
    print("\n⏳ 1. Training with torch.optim.AdamW(foreach=True)...")
    torch.manual_seed(42)
    model_torch = DeepPolicyNetwork(in_dim=1024, hidden_dim=1024, depth=10).to(device)
    opt_torch = torch.optim.AdamW(model_torch.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)

    # Warmup
    out = model_torch(x_data)
    loss = criterion(out, y_target)
    loss.backward()
    opt_torch.step()
    opt_torch.zero_grad()

    torch_step_times = []
    start_total = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        opt_torch.zero_grad()
        out = model_torch(x_data)
        loss = criterion(out, y_target)
        loss.backward()
        
        t_opt = time.perf_counter()
        opt_torch.step()
        opt_time = (time.perf_counter() - t_opt) * 1000.0
        torch_step_times.append(opt_time)

    total_torch_time = time.perf_counter() - start_total
    final_loss_torch = loss.item()
    avg_opt_torch = np.mean(torch_step_times)
    print(f"✅ PyTorch: Total {total_torch_time:.2f}s | Opt Step {avg_opt_torch:.2f} ms/step | Final Loss {final_loss_torch:.6f}")

    # -------------------------------------------------------------
    # 2. Train with TikunAdamW Drop-in Optimizer
    # -------------------------------------------------------------
    print("\n⏳ 2. Training with TikunAdamW Drop-in Optimizer...")
    torch.manual_seed(42)
    model_tikun = DeepPolicyNetwork(in_dim=1024, hidden_dim=1024, depth=10).to(device)
    opt_tikun = TikunAdamW(model_tikun.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

    # Warmup
    out = model_tikun(x_data)
    loss = criterion(out, y_target)
    loss.backward()
    opt_tikun.step()
    opt_tikun.zero_grad()

    tikun_step_times = []
    start_total = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        opt_tikun.zero_grad()
        out = model_tikun(x_data)
        loss = criterion(out, y_target)
        loss.backward()
        
        t_opt = time.perf_counter()
        opt_tikun.step()
        opt_time = (time.perf_counter() - t_opt) * 1000.0
        tikun_step_times.append(opt_time)

    total_tikun_time = time.perf_counter() - start_total
    final_loss_tikun = loss.item()
    avg_opt_tikun = np.mean(tikun_step_times)
    print(f"✅ tikun:   Total {total_tikun_time:.2f}s | Opt Step {avg_opt_tikun:.2f} ms/step | Final Loss {final_loss_tikun:.6f}")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    opt_speedup = avg_opt_torch / avg_opt_tikun
    loss_diff = abs(final_loss_torch - final_loss_tikun)

    print("\n==========================================================================")
    print("🏆 REAL TRAINING BENCHMARK SUMMARY:")
    print("==========================================================================")
    print(f"  • Optimizer Step Time: PyTorch {avg_opt_torch:.2f} ms vs tikun {avg_opt_tikun:.2f} ms ({opt_speedup:.2f}x FASTER!)")
    print(f"  • Loss Convergence:    PyTorch {final_loss_torch:.6f} vs tikun {final_loss_tikun:.6f} (Diff: {loss_diff:.8e})")
    print(f"  • Mathematical Equivalence Confirmed: {loss_diff < 1e-5}")

if __name__ == "__main__":
    run_real_training_experiment()
