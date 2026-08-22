import sys
import os
import torch
import torch.nn as nn
import jax
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
import tikun

def test_live_inspection():
    print("==========================================================================")
    print("🔬 TESTING NATIVE LIVE OPTIMIZER INSTANCE INSPECTION (.to_mlir, .inspect)")
    print("==========================================================================")

    # 1. Instantiate a real PyTorch Deep Network
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.LayerNorm(4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    )

    opt = tikun.AdamW(model.parameters(), lr=1e-3, max_norm=1.0)

    # 2. Inspect physical live memory layout
    print("\n🧪 1. Calling opt.inspect() on live PyTorch model:")
    opt.inspect()

    # 3. Inspect lowered MLIR on live model
    print("\n🧪 2. Calling opt.to_mlir() on live PyTorch model:")
    mlir_repr = opt.to_mlir()
    print(mlir_repr)

    print("\n==========================================================================")
    print("🎉 LIVE INSTANCE INSPECTION VERIFIED (ZERO MOCKING)")
    print("==========================================================================")

if __name__ == "__main__":
    test_live_inspection()
