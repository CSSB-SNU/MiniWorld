"""Diffusion utilities.

The base abstractions and the concrete diffusers (EDM + decoupled x-prediction)
are promoted to :mod:`team_gm.diffusion`; the submodules here are thin
re-export shims. Diffuser configs stay in :mod:`miniworld.configs` since
team-gm cannot depend on them.

Structure:
    base/             — Abstract base classes (Diffuser, DiffusionScheduler, DiffusionSolver)
    edm/              — Euclidean EDM (eps-prediction) with Karras/AF3 preconditioning
    decoupled_xpred/  — Decoupled x-prediction (VE, separate R/T noise); the
                        diffuser used by the MiniWorld folding model + inference
"""

from .base import AtomChainMap, Diffuser, DiffusionScheduler, DiffusionSolver, ModelFn
from .decoupled_xpred import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from .edm import AF3Solver, EDMScheduler, EuclideanDiffuser

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
]
