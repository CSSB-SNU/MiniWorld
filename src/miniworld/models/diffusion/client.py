"""phase 2 client: EDM diffusion-loss-only training over a FROZEN mini-SWA trunk.

Mirrors :mod:`miniworld.models.default_client` (EDM ``EuclideanDiffuser`` +
``EDMScheduler`` + ``AF3Solver``) but:

  * computes the DIFFUSION loss ONLY (no distogram / smooth-lDDT aux),
  * freezes + loads the epoch-900 trunk via the ``param_policy`` mechanism
    (reused from :mod:`miniworld.models.miniworld.client`): with
    ``default: freeze_loaded`` every checkpoint param (the trunk, whose keys
    match this model exactly) is loaded and frozen, while the brand-new
    ``to_token_single_trunk`` + ``diffusion_module`` params are re-initialized
    and left trainable.

Build the optimizer over :func:`miniworld.training.trainable_parameters` after
:meth:`maybe_apply_param_policy` so the frozen trunk is excluded.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from jaxtyping import Bool, Float
from lightning.fabric.wrappers import _FabricDataLoader
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.core.client import _SetEpochProtocol
from team_gm.diffusion import AF3Solver, EDMScheduler, EuclideanDiffuser
from torch.utils.data import DataLoader

from miniworld.configs import EDMDiffuserConfig
from miniworld.data.features.batch import Batch
from torch.utils.checkpoint import checkpoint

from miniworld.loss.smooth_lddt import cal_smooth_lddt
from miniworld.models.diffusion.model import (
    InferenceOutput,
    ModelWrapper,
    DiffusionModel,
)
from miniworld.training import ParamPolicyConfig

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class Client(BaseClient):
    """Client for phase 2: train ONLY the diffusion head with EDM diffusion loss."""

    class TrainConfig(BaseModel):
        """Configuration for training."""

        comment: str = "v1.0.0-phase2-diffusion"
        name: str = "MiniWorld-phase2"
        run_dir: str = "runs/v1.0.0/phase2"
        overfitting: bool = False
        overfitting_dir: str | None = None
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Adam"] = "Adam"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
        warmup_steps: int = int(5e3)
        decay_steps: int = int(5e6)
        decay_factor: float = 0.95
        compile: bool = False
        # EXPERIMENTAL (default "" = OFF): when non-empty (e.g. "reduce-overhead")
        # AND ``compile`` is True, the FROZEN trunk conditioning path is compiled
        # with this inductor mode (cudagraph-trees) and its outputs are cloned
        # before the grad diffusion path, while the diffusion module uses the
        # normal ``compile(dynamic=False)``. Empty string keeps the current
        # whole-model ``compile(dynamic=False)`` behaviour unchanged. Needs GPU
        # validation before use.
        trunk_compile_mode: str = ""
        num_augment: int = 48
        save_freq: int = 5
        eval_freq: int = 10
        eval_sample_num: int = 5
        eval_timesteps: int = 100
        eval_input_num: int = 50
        grad_clip_max_norm: float = 1.0
        grad_accum_steps: int = 256
        num_workers: int = 4
        prefetch_factor: int = 4
        seed: int = 0
        use_ema: bool = True
        ema_decay: float = 0.999

        bucket_msa_multiple: int | None = 128
        bucket_token_multiple: int | None = 128
        bucket_atom_multiple: int | None = 1024

        verbose: bool = False
        use_wandb: bool = False
        wandb_project: str = "MiniWorld"

        # Selective freeze / re-init / load-existing policy. For phase 2 set
        # enabled=True with default=freeze_loaded to load+freeze the trunk and
        # train only the new diffusion params.
        param_policy: ParamPolicyConfig = ParamPolicyConfig()

    class LossConfig(BaseModel):
        """Configuration for loss weights — diffusion loss ONLY."""

        diffusion_loss: float = 4.0
        # AF3 Eq.4 per-atom MSE up-weighting w_l = 1 + is_dna*alpha_dna +
        # is_rna*alpha_rna + is_ligand*alpha_ligand. AF3 defaults 5/5/10.
        alpha_dna: float = 5.0
        alpha_rna: float = 5.0
        alpha_ligand: float = 10.0
        # v1.0.1 auxiliary terms. BOTH DEFAULT TO 0.0 so that the v1.0.0 baseline
        # (loss/diffusion_only.yaml) is bit-identical even if a v1.0.0 job is requeued
        # onto this code. loss/diffusion_v101.yaml turns smooth lDDT on.
        #   smooth_lddt_loss : AF3 Algorithm 27; AF3 uses 1.0 from main training.
        #   bond_loss        : AF3 Eq.5 alpha_bond; 0 in main training, 1 in fine-tuning
        #                      (AF3 Table 6), so it stays 0 for phases 2a/2b.
        smooth_lddt_loss: float = 0.0
        bond_loss: float = 0.0

    class Config(BaseModel):
        """Configuration for the phase 2 client."""

        model: DiffusionModel.Config
        diffuser: EDMDiffuserConfig
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.train.seed)
        self.register_model(DiffusionModel(config.model))

        if config.train.use_ema:
            self.add_callback(ModelEMA(config.train.ema_decay))

        diffuser_method = config.diffuser.method
        if diffuser_method == "AF3":
            self.diffusion_scheduler = EDMScheduler(config.diffuser.scheduler)
            self.diffuser = EuclideanDiffuser(
                config=EuclideanDiffuser.EuclideanConfig(
                    seed=config.diffuser.seed,
                ),
                scheduler=self.diffusion_scheduler,
            )
            self.solver = AF3Solver(
                config=AF3Solver.SolverConfig(seed=config.diffuser.seed),
                scheduler=self.diffusion_scheduler,
            )
        else:
            msg = f"Diffuser method {diffuser_method} is not implemented yet."
            raise NotImplementedError(msg)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    # -- param policy (freeze + load epoch-900 trunk) -----------------------
    @staticmethod
    def _summarize_keys(title: str, keys: list[str], limit: int = 20) -> list[str]:
        if not keys:
            return [f"{title}: 0"]
        lines = [f"{title}: {len(keys)}"]
        lines.extend(f"  - {key}" for key in keys[:limit])
        if len(keys) > limit:
            lines.append(f"  ... and {len(keys) - limit} more")
        return lines

    def _warn_non_strict_model_load(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        incompatible_keys: list[tuple[str, Any, Any]],
    ) -> None:
        lines = [
            "=" * 88,
            "NON-STRICT CHECKPOINT LOAD: model architecture/state mismatch detected.",
            (
                "New, removed, or resized layers were not restored exactly. "
                "Missing or incompatible parameters will keep their current "
                "initialization."
            ),
            *self._summarize_keys("Missing model keys", missing_keys),
            *self._summarize_keys("Unexpected checkpoint keys", unexpected_keys),
            f"Shape-mismatched checkpoint keys: {len(incompatible_keys)}",
        ]
        if incompatible_keys:
            limit = 20
            lines.extend(
                (
                    f"  - {key} "
                    f"(checkpoint={checkpoint_shape}, current={current_shape})"
                )
                for key, checkpoint_shape, current_shape in incompatible_keys[:limit]
            )
            if len(incompatible_keys) > limit:
                lines.append(f"  ... and {len(incompatible_keys) - limit} more")
        lines.append("=" * 88)
        self.logger.warning("%s", "\n".join(lines))

    def maybe_apply_param_policy(
        self,
        state_dict: dict[str, Any] | None,
    ) -> dict[str, list[str]] | None:
        """Apply ``self.config.train.param_policy`` to the model in-place.

        Returns ``None`` (and the caller falls back to the standard
        ``load_state_dict`` path) when the policy is disabled. Otherwise loads +
        freezes checkpoint params, re-inits the rest, and restores epoch/step.
        Build the optimizer over :func:`trainable_parameters` afterwards.
        """
        policy = self.config.train.param_policy
        if not policy.enabled:
            return None

        from miniworld.training import apply_param_policy, format_summary

        # Initialize + freeze the phase 1b trunk from the RAW model_state_dict.
        # NOTE: do NOT use ema_state_dict here — the phase 1b EMA is broken (it did
        # not track the live weights under compiled/CUDA-graph training and sits
        # near the untrained init: measured distogram loss 3.75 (~ln n_bins) vs
        # 0.86 for raw, worse on 256/256 val items). The raw weights are correct.
        ckpt_model_sd = (
            state_dict.get("model_state_dict") if state_dict is not None else None
        )
        summary = apply_param_policy(
            self.model, ckpt_model_sd, policy, log=self.logger,
        )
        self.logger.info("Param policy applied:\n%s", format_summary(summary))

        if state_dict is not None:
            self._epoch = state_dict.get("epoch", 0)
            self._global_step = state_dict.get("global_step", 0)

        return summary

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        model_only: bool = False,
        strict: bool = True,
    ) -> None:
        """Load model (and optionally optimizer/scheduler) state.

        With ``strict=False`` only key- and shape-matching params load; the rest
        keep their current init and a strong warning is emitted.
        """
        model_state_dict = state_dict["model_state_dict"]
        if strict:
            self.model.load_state_dict(model_state_dict)
        else:
            current_model_state = self.model.state_dict()
            filtered_model_state: dict[str, Any] = {}
            unexpected_keys: list[str] = []
            incompatible_keys: list[tuple[str, Any, Any]] = []

            for key, value in model_state_dict.items():
                if key not in current_model_state:
                    unexpected_keys.append(key)
                    continue
                current_value = current_model_state[key]
                if (
                    isinstance(value, torch.Tensor)
                    and isinstance(current_value, torch.Tensor)
                    and value.shape != current_value.shape
                ):
                    incompatible_keys.append(
                        (key, tuple(value.shape), tuple(current_value.shape)),
                    )
                    continue
                filtered_model_state[key] = value

            missing_keys = [
                key for key in current_model_state if key not in model_state_dict
            ]

            self.model.load_state_dict(filtered_model_state, strict=False)
            if missing_keys or unexpected_keys or incompatible_keys:
                self._warn_non_strict_model_load(
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                    incompatible_keys=incompatible_keys,
                )

        self._epoch = state_dict["epoch"]
        self._global_step = state_dict["global_step"]

        if not model_only:
            optimizer_state = state_dict.get("optimizer_state_dict")
            scheduler_state = state_dict.get("scheduler_state_dict")
            if optimizer_state is not None and self._optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)

        self.call_callbacks("on_load_state_dict", state_dict)
        self.logger.info(
            "Loaded checkpoint (epoch=%d, step=%d)",
            self.epoch,
            self.global_step,
        )

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        *,
        strict: bool = True,
        **extra_kwargs: Any,
    ) -> Client:
        """Restore model weights from a checkpoint file."""
        state_dict = torch.load(filepath, map_location="cpu")
        config = cls.Config.model_validate(state_dict["config"])
        client = cls(config, **extra_kwargs)
        client.load_state_dict(state_dict, model_only=True, strict=strict)
        return client

    # -- EDM diffusion training (loss = diffusion ONLY) ---------------------
    def loss_fn(
        self,
        batch: Batch,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        t_emb: Float[torch.Tensor, ...],
        sigma: Float[torch.Tensor, ...],
        x_mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the EDM diffusion loss (no distogram / lddt aux)."""
        atom_pos_update = self.model.forward(
            msa=batch.msa,
            template=batch.template,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
            x_t=x_input,
            x_mask=x_mask,
            t_emb=t_emb,
        )

        # AF3 Eq.4 per-atom weight w_l = 1 + is_dna*a_dna + is_rna*a_rna
        # + is_ligand*a_ligand, gathered from chain entity_type to atoms.
        # entity_type ints: RNA=3, DNA=4, NA=5, LIGAND=6, BRANCHED=7.
        et = batch.chain.entity_type  # [B, L_chain]
        lc = self.config.loss
        w_chain = (
            1.0
            + lc.alpha_dna * (et == 4).to(atom_pos_update.dtype)
            + lc.alpha_rna * ((et == 3) | (et == 5)).to(atom_pos_update.dtype)
            + lc.alpha_ligand * ((et == 6) | (et == 7)).to(atom_pos_update.dtype)
        )  # [B, L_chain]
        atom_weight = torch.gather(
            w_chain, dim=1, index=batch.scheme.atom_to_chain_id,
        )  # [B, L_atom] -> broadcasts over the augment dim

        structure_loss = self.diffuser.cal_loss(
            x0=x0,
            x_input=x_input,
            x_update=atom_pos_update,
            sigma=sigma,
            mask=x_mask,
            atom_weight=atom_weight,
        )

        # ---- v1.0.1 auxiliary terms (both no-ops when their weights are 0.0) ----
        smooth_lddt_loss = torch.tensor(0.0, device=atom_pos_update.device)
        bond_loss = torch.tensor(0.0, device=atom_pos_update.device)
        if lc.smooth_lddt_loss > 0 or lc.bond_loss > 0:
            # Same prediction the MSE term uses; both aux losses are distance-based and
            # therefore rigid-invariant, so no alignment is needed.
            x_pred = self.diffuser.get_x_pred(
                x_input=x_input, x_update=atom_pos_update, sigma=sigma,
            )

        if lc.smooth_lddt_loss > 0:
            # Nucleotide atoms get the wider 30A inclusion radius (entity_type 3/4/5).
            chain_is_nuc = ((et == 3) | (et == 4) | (et == 5)).bool()
            atom_is_nuc = torch.gather(
                chain_is_nuc, dim=1, index=batch.scheme.atom_to_chain_id,
            )
            # Checkpointed per augment: A x [L_atom, L_atom] pair tensors would otherwise
            # dominate memory at num_augment 48.
            smooth_lddt_loss = torch.stack(
                [
                    checkpoint(
                        cal_smooth_lddt,
                        x_pred[a],
                        x0[a],
                        atom_is_nuc,
                        batch.structure.atom_pos_mask,
                        use_reentrant=False,
                    )
                    for a in range(x_pred.shape[0])
                ],
            ).mean()

        # AF3 Eq.5 bonded-ligand bond loss: MSE of bonded-atom-pair distances vs GT,
        #   L_bond = mean_{(l,m) in B} ( ||x_l - x_m|| - ||x^GT_l - x^GT_m|| )^2
        # Distance-based, so it needs no rigid alignment (x0 may stay unaligned).
        #
        # AF3 Eq.6 puts it INSIDE the noise-level weight, next to the MSE term:
        #   L_diffusion = w(sigma) * (L_MSE + alpha_bond * L_bond) + L_smooth_lddt
        # so it is weighted per augment by the same w(sigma) the MSE term gets (the EDM
        # diffuser folds that into structure_loss internally). Weighting it here keeps the
        # two terms on the same footing at every noise level.
        bond_pairs = batch.structure.bond_atom_pairs
        if lc.bond_loss > 0 and bond_pairs is not None and bond_pairs.shape[1] > 0:
            bi = bond_pairs[0, :, 0].long()
            bj = bond_pairs[0, :, 1].long()
            d_pred = (x_pred[..., bi, :] - x_pred[..., bj, :]).norm(dim=-1)
            d_gt = (x0[..., bi, :] - x0[..., bj, :]).norm(dim=-1)
            per_augment = (d_pred - d_gt).pow(2).mean(dim=-1)  # [A, B]
            w_sigma = (
                self.diffuser.scheduler.loss_weight(sigma)
                .to(dtype=per_augment.dtype, device=per_augment.device)
                .reshape(per_augment.shape)
            )
            bond_loss = (w_sigma * per_augment).mean()

        # AF3 Eq.6 + Eq.15: L_total gets alpha_diffusion * L_diffusion, and L_diffusion is
        #   w(sigma) * (L_MSE + alpha_bond * L_bond) + L_smooth_lddt
        # so BOTH auxiliary terms sit inside the alpha_diffusion factor. w(sigma) is already
        # inside structure_loss and (above) inside bond_loss; smooth lDDT is outside it in
        # AF3, which is why it is added here unweighted by sigma.
        loss = lc.diffusion_loss * (
            structure_loss
            + lc.bond_loss * bond_loss
            + lc.smooth_lddt_loss * smooth_lddt_loss
        )

        return loss, {
            "diffusion_loss": structure_loss.item(),
            "smooth_lddt_loss": smooth_lddt_loss.item(),
            "bond_loss": bond_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the diffusion head on a batch."""
        num_augment = self.config.train.num_augment
        # Use atom_pos_mask (resolved atoms), NOT atom_mask (all-True). Unresolved
        # atoms have their coords set to (0,0,0) in convert.py; passing all-True
        # atom_mask puts them in the diffusion + loss mask, training the model to
        # predict unresolved atoms AT THE ORIGIN — the observed origin-collapse of
        # disordered termini / His-tags. atom_pos_mask excludes them (correct).
        x0, x_input, x_mask, t_emb, sigma = self.diffuser.sample(
            batch.structure.atom_pos,
            num_augment=num_augment,
            mask=batch.structure.atom_pos_mask,
        )

        loss, loss_dict = self.loss_fn(
            batch=batch,
            x0=x0,
            x_input=x_input,
            t_emb=t_emb,
            sigma=sigma,
            x_mask=x_mask,
        )

        self.backward(loss)
        del loss
        return loss_dict

    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Measure inference quality (best-of-N) on a single-item batch."""
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        batch = batch.duplicate(self.config.train.eval_sample_num)
        output = self.inference(batch, timesteps=self.config.train.eval_timesteps)
        return self.test_inference_quality(batch, output)

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch."""
        if not isinstance(dataloader, _FabricDataLoader) and isinstance(
            dataloader.sampler,
            _SetEpochProtocol,
        ):
            dataloader.sampler.set_epoch(self.epoch)
            self.model.set_seed(self.config.train.seed + self.epoch)  # pyright: ignore[reportCallIssue]

        self.model.train()
        self.call_callbacks("on_train_epoch_start")

        try:
            for batch_idx, _batch in enumerate(dataloader):
                batch = cast("Batch", _batch)
                batch = batch.to(device=self.device)
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.call_callbacks("on_train_step_start", batch, batch_idx)
                self.call_callbacks("on_train_batch_start", batch, batch_idx)
                is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0
                with self.fabric.no_backward_sync(
                    self.model,  # pyright: ignore[reportArgumentType]
                    enabled=is_accumulating,
                ):
                    loss_dict = self.training_step(batch)
                self.call_callbacks(
                    "on_train_batch_end",
                    batch,
                    batch_idx,
                    loss_dict,
                )
                if not is_accumulating:
                    self._optimizer_step()
                    self.call_callbacks("on_train_step_end", batch, batch_idx, loss_dict)
                yield loss_dict
        finally:
            self.optimizer.zero_grad()
            self._epoch += 1
            self.call_callbacks("on_train_epoch_end")

    @torch.no_grad()
    def test_inference_quality(
        self,
        batch: Batch,
        output: InferenceOutput,
    ) -> dict[str, float]:
        """Best RMSD / lDDT over the sampled structures."""
        from miniworld.loss import metrics

        batch = batch.to(device=self.device)
        max_lddt, min_rmsd = 0.0, float("inf")
        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_pos_mask[0],  # resolved atoms only (not all-True atom_mask)
        )
        max_lddt = max(max_lddt, lddt)
        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_pos_mask[0],  # resolved atoms only
        )
        min_rmsd = min(min_rmsd, rmsd)
        return {"best_rmsd": min_rmsd, "best_lddt": max_lddt}

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
        n_samples: int = 1,
        n_recycle: int | None = None,
        sample_batch: bool = False,
    ) -> InferenceOutput:
        """Inference using the EDM diffusion solver.

        The trunk conditioning runs ONCE on ``batch`` (which must be B=1 — the
        trimul cute kernel in the template/pairformer only supports B=1), then
        the diffusion head is sampled ``n_samples`` times reusing that same
        condition. This is both correct (no per-sample trunk recompute) and
        avoids the B>1 kernel limit that a ``batch.duplicate(n)`` up front hits.
        ``n_recycle`` overrides the trunk recycle depth (default: model's
        ``n_recycle_max``) for this call.

        ``sample_batch`` draws all ``n_samples`` trajectories in ONE solver run
        instead of looping. ``ModelWrapper.forward`` already takes a leading
        structure axis -- it is the same axis training fills with ``num_augment``
        (48) -- so the samples ride it directly. The loop leaves a small target's
        kernels far too small to fill an H100 (measured: 25% utilisation on the
        128-token bucket, where the 5 sequential 200-step rollouts are essentially
        the whole cost), which is why AF3 vmaps its 5 samples instead.
        """
        raw_model = getattr(self.model, "module", self.model)
        raw_model = cast("DiffusionModel", raw_model)
        if n_recycle is not None:
            raw_model._forced_n_recycle = int(n_recycle)
        model_wrapper = ModelWrapper(raw_model)
        batch = batch.to(device=self.device)
        model_wrapper.prepare_condition(
            msa=batch.msa,
            template=batch.template,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
        )
        shape = batch.structure.atom_pos.shape  # [B, L, 3], B == 1
        preds: list[torch.Tensor] = []
        inter_traj0: list[np.ndarray] | None = None
        model_traj0: list[np.ndarray] | None = None
        n_runs = 1 if sample_batch else n_samples
        if sample_batch:
            shape = (n_samples, *shape[1:])  # [n_samples, L, 3]: one batched rollout
        for _ in range(n_runs):
            atom_pos_pred, inter_traj, model_traj = self.solver.sample(
                model_fn=model_wrapper,
                shape=shape,
                num_steps=timesteps,
                device=self.device,
                # atom_mask, not atom_pos_mask: every atom of the target is being
                # predicted; the mask is only here to keep the padding slots out of
                # the per-step centring (see AF3Solver._centre_random_augmentation).
                mask=batch.structure.atom_mask,
                return_intermediate=True,
            )
            preds.append(atom_pos_pred)  # [B or n_samples, L, 3], independent draws
            if inter_traj0 is None:  # keep only the first sample's trajectory
                inter_traj0 = [x.detach().cpu().numpy() for x in inter_traj]
                model_traj0 = [x.detach().cpu().numpy() for x in model_traj]
        atom_pos_pred = torch.cat(preds, dim=0)  # [n_samples * B, L, 3]
        return InferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj0, axis=1),
            inter_traj=np.stack(inter_traj0, axis=1),
        )
