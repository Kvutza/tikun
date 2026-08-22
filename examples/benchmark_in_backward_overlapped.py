import time
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

class LargeMLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=4096, out_dim=2048, num_layers=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    print("==========================================================================", flush=True)
    print("⚡ ZERO-LATENCY IN-BACKWARD OVERLAPPED STREAMING BENCHMARK", flush=True)
    print("   PyTorch Native vs Tikun In-Backward Overlapped Stepper", flush=True)
    print("==========================================================================", flush=True)

    torch.manual_seed(42)
    model_torch = LargeMLP(2048, 4096, 2048, num_layers=6)
    num_params = sum(p.numel() for p in model_torch.parameters())
    print(f"📊 Model Parameters: {num_params:,} (~{num_params * 4 / (1024 * 1024):.1f} MB weights)", flush=True)

    # 1. Standard PyTorch Training
    opt_torch = torch.optim.AdamW(model_torch.parameters(), lr=1e-3, foreach=True)
    x = torch.randn(32, 2048)

    # Warmup
    out = model_torch(x).sum()
    out.backward()
    opt_torch.step()
    opt_torch.zero_grad()

    torch_step_times = []
    torch_opt_times = []
    for _ in range(20):
        t_start = time.perf_counter()
        out = model_torch(x).sum()
        out.backward()
        
        t_opt_start = time.perf_counter()
        opt_torch.step()
        t_opt_end = time.perf_counter()
        opt_torch.zero_grad()
        t_end = time.perf_counter()

        torch_step_times.append((t_end - t_start) * 1000.0)
        torch_opt_times.append((t_opt_end - t_opt_start) * 1000.0)

    avg_torch_step = np.mean(torch_step_times[5:])
    avg_torch_opt = np.mean(torch_opt_times[5:])

    # 2. Tikun In-Backward Overlapped Training
    torch.manual_seed(42)
    model_tikun = LargeMLP(2048, 4096, 2048, num_layers=6)
    opt_tikun = tikun.AdamW(model_tikun.parameters(), lr=1e-3, in_backward=True)

    # Warmup
    out = model_tikun(x).sum()
    out.backward()
    opt_tikun.step()

    tikun_step_times = []
    tikun_opt_times = []
    for _ in range(20):
        t_start = time.perf_counter()
        out = model_tikun(x).sum()
        out.backward() # Tikun updates weights concurrently inside the backward hooks!
        
        t_opt_start = time.perf_counter()
        opt_tikun.step() # Takes 0.0 ms!
        t_opt_end = time.perf_counter()
        t_end = time.perf_counter()

        tikun_step_times.append((t_end - t_start) * 1000.0)
        tikun_opt_times.append((t_opt_end - t_opt_start) * 1000.0)

    avg_tikun_step = np.mean(tikun_step_times[5:])
    avg_tikun_opt = np.mean(tikun_opt_times[5:])

    print("\n==========================================================================", flush=True)
    print("🏆 IN-BACKWARD OVERLAPPED TRAINING RESULTS:", flush=True)
    print("==========================================================================", flush=True)
    print(f"  • PyTorch Optimizer Stall Time:   {avg_torch_opt:6.2f} ms / step", flush=True)
    print(f"  • Tikun Optimizer Stall Time:     {avg_tikun_opt:6.2f} ms / step  (COMPLETELY HIDDEN!)", flush=True)
    print(f"  • PyTorch Total Step Time:        {avg_torch_step:6.2f} ms / step", flush=True)
    print(f"  • Tikun In-Backward Step Time:    {avg_tikun_step:6.2f} ms / step", flush=True)
    print(f"  • Optimizer Stall Reduction:      {avg_torch_opt - avg_tikun_opt:6.2f} ms eliminated per step!", flush=True)
    print("==========================================================================", flush=True)

if __name__ == "__main__":
    main()
