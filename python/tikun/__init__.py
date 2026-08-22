from .universal import UniversalOptimizer as Optimizer
from .muon import Muon
from .muon_nd import MultiDimensionalMuon as MuonND
from .diagnostics import inspect_cpu_plan, inspect_jaxpr
from .autotune import run_hardware_autotune as autotune

def AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, max_norm=0.0, in_backward=False):
    return Optimizer(params, algorithm="adamw", lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, max_norm=max_norm, in_backward=in_backward)

def Lion(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01, max_norm=0.0, in_backward=False):
    return Optimizer(params, algorithm="lion", lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, max_norm=max_norm, in_backward=in_backward)

def SGD(params, lr=1e-2, momentum=0.9, weight_decay=0.0, max_norm=0.0, in_backward=False):
    return Optimizer(params, algorithm="sgd", lr=lr, beta1=momentum, weight_decay=weight_decay, max_norm=max_norm, in_backward=in_backward)

__all__ = [
    "Optimizer",
    "AdamW",
    "Lion",
    "SGD",
    "Muon",
    "MuonND",
    "inspect_cpu_plan",
    "inspect_jaxpr",
    "autotune",
]
