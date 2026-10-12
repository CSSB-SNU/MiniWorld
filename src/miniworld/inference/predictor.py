"""Top-level inference wrapper.

:class:`Predictor` glues together:

  - the trunk (``Model.condition_forward``) — run once per
    :meth:`prepare` call
  - :func:`build_inference_cache` — builds the per-batch static cache
    from the trunk outputs (hoists ``DiffusionConditioning`` constants,
    atom_pair, scatter mapping, etc.)
  - :func:`build_step_schedule` — builds the per-step schedule + the
    precomputed ``token_single_cond`` stack for the requested
    ``timesteps``
  - :func:`sample_trajectory` — runs the actual diffusion solver,
    consulting the caches above so each per-step forward only does the
    x_t-dependent work

EMA / checkpoint / Fabric handling stays with the training
:class:`~miniworld.models.miniworld.client.Client` — call
:meth:`Predictor.from_client` after the client has loaded its weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch

from miniworld.inference.cache import (
    InferenceCache,
    StepSchedule,
    build_inference_cache,
    build_step_schedule,
)
from miniworld.inference.solver import sample_trajectory

if TYPE_CHECKING:
    from miniworld.data.features.batch import Batch
    from miniworld.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler
    from miniworld.diffusion.decoupled_xpred.solver import XPredDecoupledSolver
    from miniworld.models.miniworld.client import Client
    from miniworld.models.miniworld.model import Model


@dataclass
class PredictorOutput:
    """Mirror of :class:`miniworld.models.miniworld.model.InferenceOutput`.

    Kept structurally identical so the casp17 ``run_miniworld.py`` driver
    can swap between the legacy and inference paths without touching its
    cif-writing / trajectory-dumping code.
    """

    atom_pos_pred: torch.Tensor              # (n_samples, L_atom, 3)
    distogram_logit: torch.Tensor            # (B, L, L, n_distogram_bins)
    model_traj: np.ndarray                   # (n_samples, T, L_atom, 3)
    inter_traj: np.ndarray                   # (n_samples, T, L_atom, 3)
    input_traj: np.ndarray                   # (n_samples, T, L_atom, 3)


class Predictor:
    """Inference-only orchestrator. Stateless across batches.

    Build via :meth:`from_client` once a trained
    :class:`~miniworld.models.miniworld.client.Client` is loaded; reuse
    across many batches.
    """

    def __init__(
        self,
        model: "Model",
        scheduler: "DecoupledXPredScheduler",
        solver_cfg: "XPredDecoupledSolver.Config",
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.solver_cfg = solver_cfg

    @classmethod
    def from_client(cls, client: "Client") -> "Predictor":
        """Wrap an already-loaded Client (EMA shadow, fabric, compile all OK)."""
        from miniworld.models.miniworld.model import Model
        raw_model = cast("Model", getattr(client.model, "module", client.model))
        return cls(
            model=raw_model,
            scheduler=client.diffusion_scheduler,
            solver_cfg=client.solver.config,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def prepare(self, batch: "Batch") -> InferenceCache:
        """Run the trunk once, return a cache the diffusion loop reuses."""
        batch = batch.to(device=self.device)
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.model.condition_forward(
            msa=batch.msa,
            template=batch.template,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
        )
        return build_inference_cache(
            self.model,
            batch,
            token_single_input=token_single_input,
            token_single_trunk=token_single_trunk,
            token_pair_trunk=token_pair_trunk,
            distogram_logit=distogram_logit,
        )

    def _build_schedule(
        self,
        cache: InferenceCache,
        *,
        num_steps: int,
        start_sigma_y: float | None,
    ) -> StepSchedule:
        from miniworld.diffusion.decoupled_xpred.solver import XPredDecoupledSolver
        # We pass a transient solver instance only so build_step_schedule
        # can read its gamma_0 / gamma_min / step_scale config — it does
        # not call the solver's step().
        proxy = XPredDecoupledSolver(self.solver_cfg, self.scheduler)
        return build_step_schedule(
            self.model,
            cache,
            self.scheduler,
            proxy,
            num_steps=num_steps,
            start_sigma_y=start_sigma_y,
        )

    @torch.inference_mode()
    def sample(
        self,
        cache: InferenceCache,
        *,
        n_samples: int = 1,
        timesteps: int = 100,
        no_rt: bool = False,
        update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
        combine_all: bool = False,
        init_x0: torch.Tensor | None = None,
        start_sigma_y: float | None = None,
        return_intermediate: bool = True,
    ) -> PredictorOutput:
        """Run the diffusion sampler against a prepared cache.

        Returns a :class:`PredictorOutput` whose field layout matches the
        legacy :class:`InferenceOutput`. Trajectories are returned as
        ``(n_samples, T, ...)`` numpy arrays.
        """
        if n_samples < 1:
            msg = f"n_samples must be >= 1, got {n_samples}."
            raise ValueError(msg)

        if combine_all and init_x0 is not None:
            msg = (
                "init_x0 (flexible-docking warm start) is incompatible with "
                "combine_all=True: combine_all zeros the per-atom group ids."
            )
            raise ValueError(msg)
        if init_x0 is None and start_sigma_y is not None:
            msg = (
                "start_sigma_y is only meaningful together with init_x0 — "
                "the standard sampler always starts at sigma_y_max."
            )
            raise ValueError(msg)

        schedule = self._build_schedule(
            cache,
            num_steps=timesteps,
            start_sigma_y=start_sigma_y,
        )

        atom_to_combine = cache.batch.scheme.atom_to_chain_id
        if atom_to_combine.shape[0] == 1 and n_samples > 1:
            atom_to_combine = atom_to_combine.expand(n_samples, -1)
        if combine_all:
            atom_to_combine = torch.zeros_like(atom_to_combine)

        y_final, inter_traj, hat_list, input_list = sample_trajectory(
            self.model,
            cache,
            schedule,
            self.solver_cfg,
            self.scheduler,
            n_samples=n_samples,
            atom_to_combine=atom_to_combine,
            device=self.device,
            use_rt=not no_rt,
            update_rule=update_rule,
            init_x0=init_x0,
            return_intermediate=return_intermediate,
        )

        inter_np = [t.detach().cpu().numpy() for t in inter_traj]
        hat_np = [t.detach().cpu().numpy() for t in hat_list]
        input_np = [t.detach().cpu().numpy() for t in input_list]
        return PredictorOutput(
            atom_pos_pred=y_final,
            distogram_logit=cache.distogram_logit,
            model_traj=np.stack(hat_np, axis=1) if hat_np else np.zeros((n_samples, 0)),
            inter_traj=np.stack(inter_np, axis=1) if inter_np else np.zeros((n_samples, 0)),
            input_traj=np.stack(input_np, axis=1) if input_np else np.zeros((n_samples, 0)),
        )
