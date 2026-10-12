"""Diffusion primitives promoted from MiniWorld.

base/             — abstract Diffuser / DiffusionScheduler / DiffusionSolver
edm/              — default Euclidean EDM (eps-prediction) with Karras/AF3
preconditioning (c_noise = log(sigma/sigma_data)/4)
decoupled_xpred/  — decoupled x-prediction (VE, separate R/T noise); the
diffuser used by the MiniWorld folding model + inference
"""

from miniworld.diffusion.base import (
    AtomChainMap,
    Diffuser,
    DiffusionScheduler,
    DiffusionSolver,
    ModelFn,
)
from miniworld.diffusion.decoupled_xpred import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from miniworld.diffusion.edm import AF3Solver, EDMScheduler, EuclideanDiffuser
from miniworld.diffusion.edm.preconditioned import PreconditionedEulerSolver
from miniworld.diffusion.edm.sampling import EulerSampler, PreconditionedDenoiser
from miniworld.diffusion.edm.schedules import LogNormalNoise, PowerLawSchedule

__all__ = [
    "AF3Solver",
    "AtomChainMap",
    "DecoupledXPredScheduler",
    "Diffuser",
    "DiffusionScheduler",
    "DiffusionSolver",
    "EDMScheduler",
    "EuclideanDiffuser",
    "EulerSampler",
    "LogNormalNoise",
    "ModelFn",
    "PowerLawSchedule",
    "PreconditionedDenoiser",
    "PreconditionedEulerSolver",
    "XPredDecoupledDiffuser",
    "XPredDecoupledSolver",
]
