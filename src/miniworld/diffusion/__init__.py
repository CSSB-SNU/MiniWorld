"""Diffusion utilities.

Structure:
    base/             — Abstract base classes (Diffuser, DiffusionScheduler, DiffusionSolver)
    edm/              — Euclidean EDM (eps-prediction)
    xpred/            — Euclidean x-prediction (VE, x0/sigma_data target)
    decoupled_xpred/  — Decoupled x-prediction (VE, separate R/T noise) — default
"""

from .base import AtomChainMap, Diffuser, DiffusionScheduler, DiffusionSolver, ModelFn
from .decoupled_xpred import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from .edm import AF3Solver, EDMScheduler, EuclideanDiffuser
from .xpred import XPredEuclideanDiffuser, XPredEulerSolver

__all__ = [
    "AF3Solver",
    "AtomChainMap",
    "DecoupledXPredScheduler",
    "Diffuser",
    "DiffusionScheduler",
    "DiffusionSolver",
    "EDMScheduler",
    "EuclideanDiffuser",
    "ModelFn",
    "XPredDecoupledDiffuser",
    "XPredDecoupledSolver",
    "XPredEuclideanDiffuser",
    "XPredEulerSolver",
]
