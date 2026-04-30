from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from lightning.fabric.wrappers import _FabricDataLoader
from pydantic import BaseModel
from team_gm import BaseClient, typecheck
from team_gm.core.callbacks import ModelEMA
from team_gm.core.client import _SetEpochProtocol
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from miniworld.configs import XPredDecoupledDiffuserConfig
from miniworld.data.features.batch import Batch
from miniworld.diffusion import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from miniworld.loss import metrics
from miniworld.loss.auxiliary import (
    cal_atom_distogram_loss,
)
from miniworld.models.miniworld.model import (
    InferenceOutput,
    Model,
    ModelWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from jaxtyping import Bool, Float


@torch.compile
@typecheck
def cal_smooth_lddt(
    pred_coord: Float[torch.Tensor, "... N 3"],
    gt_coord: Float[torch.Tensor, "... N 3"],
    is_nucleotide: Bool[torch.Tensor, "... N"],
    mask: Bool[torch.Tensor, "... N"],
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    nucleotide_cutoff: float = 30.0,
    non_nucleotide_cutoff: float = 15.0,
) -> Float[torch.Tensor, ""]:
    """Smooth lDDT loss (AF3 Algorithm 27).

    Computes sigmoid-smoothed lDDT at multiple distance thresholds. Inclusion radius is
    per atom-i: 30A for nucleotides, 15A for others.

    Parameters
    ----------
    pred_coord
        Predicted atom coordinates.
    gt_coord
        Ground-truth atom coordinates.
    is_nucleotide
        Per-atom flag for nucleotide atoms (DNA/RNA), which use a wider inclusion
        radius.
    mask
        Valid atom mask.
    distance_bins
        Distance thresholds for sigmoid scoring.
    nucleotide_cutoff
        Inclusion radius for nucleotide atom pairs.
    non_nucleotide_cutoff
        Inclusion radius for non-nucleotide atom pairs.

    """
    pred_dist = torch.cdist(pred_coord, pred_coord)
    gt_dist = torch.cdist(gt_coord, gt_coord)

    dist_diff = torch.abs(pred_dist - gt_dist)
    score = sum(torch.sigmoid(thres - dist_diff) for thres in distance_bins)
    score = score / len(distance_bins)

    is_nuc = is_nucleotide.unsqueeze(-1)
    cutoff_mask = (gt_dist < nucleotide_cutoff) & is_nuc
    cutoff_mask = cutoff_mask | ((gt_dist < non_nucleotide_cutoff) & ~is_nuc)

    diag_mask = ~torch.eye(mask.shape[-1], dtype=torch.bool, device=mask.device)
    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    mask_2d = mask_2d & cutoff_mask & diag_mask

    score = score * mask_2d
    lddt = score.sum(dim=(-1, -2)) / mask_2d.float().sum(dim=(-1, -2)).clamp(min=1)
    return (1 - lddt).mean()


class Client(BaseClient):
    """Client for training and inference of MiniWorld (VE x-prediction)."""

    class TrainConfig(BaseModel):
        """Configuration for trains."""

        comment: str = "default"
        name: str = "AF3Like-PSK-2"
        run_dir: str = "runs/af3like"
        overfitting: bool = False
        overfitting_dir: str | None = None  # Directory for overfitting mode
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Adam"] = "AdamW"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
        warmup_steps: int = int(5e3)
        decay_steps: int = int(5e6)
        decay_factor: float = 0.95
        compile: bool = False
        num_augment: int = 8
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
        bce_pos_weight: float = 2.0
        long_range_min_seq_sep: int | None = None
        long_range_sigmoid_k: float | None = None
        long_range_sigmoid_amp: float = 3.0

        bucket_msa_multiple: int | None = 128
        bucket_token_multiple: int | None = 128
        bucket_atom_multiple: int | None = 1024

        verbose: bool = False
        use_wandb: bool = False
        wandb_project: str = "MiniWorld"

    class LossConfig(BaseModel):
        """Configuration for loss weights."""

        diffusion_loss: float = 4.0
        distogram_loss: float = 0.03
        smooth_lddt_loss: float = 1.0

    class Config(BaseModel):
        """Configuration for the MiniWorld client."""

        model: Model.Config
        diffuser: XPredDecoupledDiffuserConfig
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.train.seed)
        self.register_model(Model(config.model))

        if config.train.use_ema:
            self.add_callback(ModelEMA(config.train.ema_decay))
        self.diffusion_scheduler = DecoupledXPredScheduler(config.diffuser.scheduler)
        self.diffuser = XPredDecoupledDiffuser(
            config=XPredDecoupledDiffuser.DecoupledXPredConfig(
                seed=config.diffuser.seed,
                translation_noise=config.diffuser.translation_noise,
            ),
            scheduler=self.diffusion_scheduler,
        )
        self.solver = XPredDecoupledSolver(
            config=XPredDecoupledSolver.Config(seed=config.diffuser.seed),
            scheduler=self.diffusion_scheduler,
        )

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _summarize_keys(
        title: str,
        keys: list[str],
        limit: int = 20,
    ) -> list[str]:
        """Format checkpoint key summaries for warning logs."""
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
        """Emit a strong warning when model checkpoint loading is non-strict."""
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

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        model_only: bool = False,
        strict: bool = True,
    ) -> None:
        """Load model and optimizer state from a checkpoint file.

        With ``strict=False``, model parameters are loaded only when both key and
        shape match. Missing, unexpected, or resized parameters are skipped with a
        strong warning, so newly added layers stay at their current initialization.
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

            if strict:
                if optimizer_state is None:
                    msg = "Optimizer state not found in checkpoint."
                    raise ValueError(msg)
                if self._optimizer is None:
                    msg = "Optimizer is not set in the client."
                    raise ValueError(msg)
                if scheduler_state is not None and self.scheduler is None:
                    msg = (
                        "Scheduler state found in checkpoint, but no scheduler is set "
                        "in the client."
                    )
                    raise ValueError(msg)
                if scheduler_state is None and self.scheduler is not None:
                    msg = "Scheduler state not found in checkpoint."
                    raise ValueError(msg)

            if optimizer_state is not None and self._optimizer is not None:
                if strict:
                    self.optimizer.load_state_dict(optimizer_state)
                else:
                    try:
                        self.optimizer.load_state_dict(optimizer_state)
                    except ValueError as exc:
                        self.logger.warning(
                            "%s",
                            "\n".join(
                                [
                                    "=" * 88,
                                    "NON-STRICT CHECKPOINT LOAD: optimizer state mismatch.",
                                    "Optimizer state was skipped and current optimizer state is kept.",
                                    f"  - {type(exc).__name__}: {exc}",
                                    "=" * 88,
                                ],
                            ),
                        )

            if scheduler_state is not None and self.scheduler is not None:
                if strict:
                    self.scheduler.load_state_dict(scheduler_state)
                else:
                    try:
                        self.scheduler.load_state_dict(scheduler_state)
                    except (KeyError, ValueError) as exc:
                        self.logger.warning(
                            "%s",
                            "\n".join(
                                [
                                    "=" * 88,
                                    "NON-STRICT CHECKPOINT LOAD: scheduler state mismatch.",
                                    "Scheduler state was skipped and current scheduler state is kept.",
                                    f"  - {type(exc).__name__}: {exc}",
                                    "=" * 88,
                                ],
                            ),
                        )

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

    def loss_fn(
        self,
        batch: Batch,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        t_emb: Float[torch.Tensor, ...],
        sigma_y: Float[torch.Tensor, ...],
        rotation_matrix: Float[torch.Tensor, ...],
        translation_vector: Float[torch.Tensor, ...],
        atom_to_combine: Int[torch.Tensor, ...],
        x_mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the loss given a noisy batch."""
        atom_pos_update, distogram_logit = self.model.forward(
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

        x_pred = self.diffuser.get_x0_hat(
            x0=x0,
            x_input=x_input,
            x_update=atom_pos_update,
            sigma_y=sigma_y,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            atom_to_combine=atom_to_combine,
            mask=x_mask,
        )

        structure_loss = self.diffuser.cal_loss(
            x0=x0,
            x_pred=x_pred,
            sigma_y=sigma_y,
            mask=x_mask,
            dtype=atom_pos_update.dtype,
        )

        distogram_loss = cal_atom_distogram_loss(
            distogram_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
        )

        # Smooth lDDT (checkpointed per augment)
        if self.config.loss.smooth_lddt_loss > 0:
            # nuc tag : 3,4,5
            chain_is_nuc = torch.where(
                (batch.chain.entity_type == 3)
                | (batch.chain.entity_type == 4)
                | (batch.chain.entity_type == 5),
                torch.ones_like(batch.chain.entity_type),
                torch.zeros_like(batch.chain.entity_type),
            ).bool()
            atom_is_nuc = torch.gather(
                chain_is_nuc,
                dim=1,
                index=batch.scheme.atom_to_chain_id,
            )
            smooth_lddt_loss = torch.stack(
                [
                    checkpoint(
                        cal_smooth_lddt,
                        x_pred[a, None],
                        x0[a],
                        atom_is_nuc,
                        batch.structure.atom_pos_mask,
                        use_reentrant=False,
                    )
                    for a in range(x_pred.shape[0])
                ],  # pyright: ignore[reportArgumentType]
            ).mean()
        else:
            smooth_lddt_loss = torch.tensor(0.0, device=x_pred.device)

        loss = (
            self.config.loss.diffusion_loss * structure_loss
            + self.config.loss.distogram_loss * distogram_loss
            + self.config.loss.smooth_lddt_loss * smooth_lddt_loss
        )

        return loss, {
            "diffusion_loss": structure_loss.item(),
            "distogram_loss": distogram_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        num_augment = self.config.train.num_augment

        (
            x0,
            x_input,
            x_mask,
            t_emb,
            sigma_y,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
        ) = self.diffuser.sample(
            x0=batch.structure.atom_pos,
            mask=batch.structure.atom_pos_mask,
            atom_to_chain_idx=batch.scheme.atom_to_chain_id,
            num_augment=num_augment,
        )

        loss, loss_dict = self.loss_fn(
            batch=batch,
            x0=x0,
            x_input=x_input,
            t_emb=t_emb,
            x_mask=x_mask,
            sigma_y=sigma_y,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            atom_to_combine=atom_to_combine,
        )

        self.backward(loss)
        del loss
        return loss_dict

    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Valdiate the model on a batch."""
        # Note that when doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        batch = batch.duplicate(self.config.train.eval_sample_num)
        output = self.inference(batch, timesteps=self.config.train.eval_timesteps)

        return self.test_inference_quality(
            batch,
            output,
        )

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
        """Test the inference quality of the model on a batch."""
        batch = batch.to(device=self.device)

        distogram_logit = output.distogram_logit
        distogram_loss = cal_atom_distogram_loss(
            distogram_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
        )

        max_lddt, min_rmsd = 0, float("inf")

        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        max_lddt = max(max_lddt, lddt)

        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        min_rmsd = min(min_rmsd, rmsd)

        return {
            "best_rmsd": min_rmsd,
            "best_lddt": max_lddt,
            "vald_distogram_loss": distogram_loss.item(),
        }

    @torch.no_grad()
    def prepare(self, batch: Batch) -> tuple[ModelWrapper, Batch]:
        """Run the trunk (msa_module + pairformer + recycling) once.

        Returns a ``ModelWrapper`` whose conditioning is cached, plus the
        device-resident ``batch`` (so the caller can reuse its scheme /
        structure tensors when invoking :meth:`sample`). The returned
        wrapper can be fed to :meth:`sample` multiple times — each call
        produces an independent set of diffusion samples without rerunning
        the trunk.
        """
        raw_model = getattr(self.model, "module", self.model)
        raw_model = cast("Model", raw_model)
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
        return model_wrapper, batch

    @torch.no_grad()
    def sample(
        self,
        wrapper: ModelWrapper,
        batch: Batch,
        *,
        n_samples: int = 1,
        timesteps: int = 100,
        no_rt: bool = False,
        update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
        combine_all: bool = False,
    ) -> InferenceOutput:
        """Run the diffusion solver on a prepared trunk conditioning.

        ``n_samples`` is broadcast along the model's augmentation axis
        (``x_t: A B L 3`` with ``A = n_samples``), so all samples share one
        forward pass per step. The returned tensors carry that augmentation
        dimension as their leading axis — ``atom_pos_pred`` is
        ``(n_samples, L, 3)`` and each trajectory is ``(n_samples, T, L, 3)``.
        """
        if n_samples < 1:
            msg = f"n_samples must be >= 1, got {n_samples}."
            raise ValueError(msg)
        _, n_atoms, three = batch.structure.atom_pos.shape
        shape = torch.Size((n_samples, n_atoms, three))
        # ``apply_chain_rt`` requires atom_to_combine.shape == x.shape[:2].
        # batch.scheme.atom_to_chain_id is (1, N_atom); expand along the
        # augmentation axis so it matches the (n_samples, N_atom) batch
        # produced by the solver.
        atom_to_combine = batch.scheme.atom_to_chain_id
        if atom_to_combine.shape[0] == 1 and n_samples > 1:
            atom_to_combine = atom_to_combine.expand(n_samples, -1)
        atom_pos_pred, inter_traj, model_traj, input_traj = self.solver.sample(
            model_fn=wrapper,
            shape=shape,
            atom_to_combine=atom_to_combine,
            num_steps=timesteps,
            device=self.device,
            use_rt=not no_rt,
            mask=batch.structure.atom_pos_mask.bool(),
            update_rule=update_rule,
            return_intermediate=True,
            combine_all=combine_all,
        )
        inter_traj = [x.detach().cpu().numpy() for x in inter_traj]
        model_traj = [x.detach().cpu().numpy() for x in model_traj]
        input_traj = [x.detach().cpu().numpy() for x in input_traj]
        distogram_logit = wrapper.condition["distogram_logit"]
        return InferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            input_traj=np.stack(input_traj, axis=1),
            distogram_logit=distogram_logit,
        )

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
        no_rt: bool = False,
        update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered",
        combine_all: bool = False,
    ) -> InferenceOutput:
        """Single-shot inference: prepare trunk, then sample once.

        Convenience wrapper around :meth:`prepare` + :meth:`sample`. Uses
        ``batch.structure.atom_pos.shape[0]`` as the augmentation count to
        preserve historical behaviour (B=1 -> 1 sample). For best-of-N
        scaling, drive :meth:`prepare` and :meth:`sample` directly so the
        trunk is reused across diffusion samples.
        """
        wrapper, batch = self.prepare(batch)
        n_samples = int(batch.structure.atom_pos.shape[0])
        return self.sample(
            wrapper,
            batch,
            n_samples=n_samples,
            timesteps=timesteps,
            no_rt=no_rt,
            update_rule=update_rule,
            combine_all=combine_all,
        )
