"""Phase4 client: train ONLY the confidence head over a FROZEN phase3 structure model.

Reuses the phase3 client's checkpoint/param-policy/EMA machinery (subclass) but:

  * registers :class:`~miniworld.models.phase4.model.Phase4Model` (trunk + diffusion
    frozen, confidence head trainable),
  * ``training_step`` runs the frozen structure model to get a predicted structure
    (:meth:`predict_structure` — the diffusion-step SEAM), builds pLDDT / PDE / (PAE)
    targets vs the ground truth under ``no_grad``, then trains the confidence head
    with per-bin cross-entropy,
  * loss = confidence losses ONLY.

The diffusion rollout used to produce the predicted structure is isolated in
:meth:`predict_structure`; how many steps / whether it runs inline per step or reads
a precomputed cache is a deliberate open decision — swap it there without touching the
rest of the training format.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from pydantic import BaseModel
from team_gm.core.callbacks import ModelEMA
from team_gm.diffusion import AF3Solver, EDMScheduler, EuclideanDiffuser

from miniworld.configs import EDMDiffuserConfig
from miniworld.data.features.batch import Batch
from miniworld.loss import confidence as conf
from miniworld.models.phase3.client import Client as Phase3Client
from miniworld.models.phase3.model import ModelWrapper
from miniworld.models.phase4.model import ConfidenceOutput, Phase4Model
from miniworld.training import ParamPolicyConfig


class Client(Phase3Client):
    """Client for phase4: train ONLY the confidence head."""

    class TrainConfig(BaseModel):
        """Configuration for training (mirrors phase3, + prediction/eval knobs)."""

        comment: str = "phase4-confidence"
        name: str = "MiniWorld-phase4"
        run_dir: str = "runs/phase4"
        overfitting: bool = False
        overfitting_dir: str | None = None
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Adam"] = "Adam"
        max_lr: float = 1e-3
        min_lr: float = 1e-4
        weight_decay: float = 0.01
        warmup_steps: int = int(1e3)
        decay_steps: int = int(5e4)
        decay_factor: float = 0.95
        compile: bool = False
        trunk_compile_mode: str = ""
        # DIFFUSION-STEP SEAM: reverse steps for the frozen rollout in predict_structure.
        # Open decision (inline count / precompute); change here.
        predict_timesteps: int = 200
        save_freq: int = 5
        eval_freq: int = 10
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

        # Load + FREEZE the phase3 structure model; train only the confidence head.
        # On requeue a phase4 checkpoint contains confidence_head too, so keep it
        # TRAINABLE via load_existing (else freeze_loaded would freeze it).
        param_policy: ParamPolicyConfig = ParamPolicyConfig(
            enabled=True,
            default="freeze_loaded",
            load_existing=["confidence_head"],
        )

    class LossConfig(BaseModel):
        """Confidence loss weights. PAE off by default (Stage B: needs token frames)."""

        plddt_loss: float = 1.0
        pde_loss: float = 1.0
        pae_loss: float = 0.0

    class Config(BaseModel):
        """Configuration for the phase4 client."""

        model: Phase4Model.Config
        diffuser: EDMDiffuserConfig
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        # Bypass Phase3Client.__init__ (which registers a Phase3Model); replicate its
        # setup but register the Phase4Model instead.
        from team_gm import BaseClient

        BaseClient.__init__(self, config)
        self.config = config
        self.set_seed(config.train.seed)
        self.register_model(Phase4Model(config.model))

        if config.train.use_ema:
            self.add_callback(ModelEMA(config.train.ema_decay))

        if config.diffuser.method != "AF3":
            msg = f"Diffuser method {config.diffuser.method} is not implemented yet."
            raise NotImplementedError(msg)
        self.diffusion_scheduler = EDMScheduler(config.diffuser.scheduler)
        self.diffuser = EuclideanDiffuser(
            config=EuclideanDiffuser.EuclideanConfig(seed=config.diffuser.seed),
            scheduler=self.diffusion_scheduler,
        )
        self.solver = AF3Solver(
            config=AF3Solver.SolverConfig(seed=config.diffuser.seed),
            scheduler=self.diffusion_scheduler,
        )

    # -- diffusion-step SEAM -------------------------------------------------
    @torch.no_grad()
    def predict_structure(
        self,
        batch: Batch,
        timesteps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frozen structure model -> predicted coords + cached trunk conditioning.

        Runs the trunk once and a full diffusion rollout (no grad). Returns
        ``(x_pred [B, L_atom, 3], token_single_input [B, L, d_in], token_pair
        [B, L, L, d_pair])``. **This is the diffusion-step seam** — the rollout step
        count (``timesteps``) and whether to run inline vs. read a precomputed cache
        are open decisions; change them here only.
        """
        steps = timesteps if timesteps is not None else self.config.train.predict_timesteps
        raw_model = cast("Phase4Model", getattr(self.model, "module", self.model))
        wrapper = ModelWrapper(raw_model)
        batch = batch.to(device=self.device)
        wrapper.prepare_condition(
            msa=batch.msa,
            template=batch.template,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
        )
        atom_pos_pred, _, _ = self.solver.sample(
            model_fn=wrapper,
            shape=batch.structure.atom_pos.shape,
            num_steps=steps,
            device=self.device,
            return_intermediate=True,
        )
        token_single_input = wrapper.condition["token_single_input"]
        token_pair = wrapper.condition["token_pair_trunk"]
        return atom_pos_pred, token_single_input, token_pair

    # -- targets + confidence loss ------------------------------------------
    def _confidence_targets_and_logits(
        self,
        batch: Batch,
        x_pred: torch.Tensor,          # [B, L_atom, 3]
        token_single_input: torch.Tensor,  # [B, L, d_in]
        token_pair: torch.Tensor,      # [B, L, L, d_pair]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build head logits + bucketed targets + masks (targets under no_grad)."""
        structure = batch.structure
        scheme = batch.scheme
        token_mask = structure.token_mask                 # [B, L]
        atom_to_token = scheme.atom_to_token_idx_map       # [B, L_atom]
        atom_pos_mask = structure.atom_pos_mask.bool()     # [B, L_atom]
        atom_mask = structure.atom_mask                    # [B, L_atom]
        atom_is_rep = structure.atom_is_rep                # [B, L_atom] | None
        token_num = int(token_mask.shape[1])
        n = x_pred.shape[0]

        cfg = self.config.model.confidence

        with torch.no_grad():
            # Representative positions (predicted + ground truth).
            if atom_is_rep is None:
                msg = "structure.atom_is_rep is required for PDE/PAE targets."
                raise ValueError(msg)
            pred_rep_pos, tok_valid = conf.representative_positions(
                x_pred, atom_pos_mask, atom_to_token, atom_is_rep, token_num,
            )
            gt_rep_pos, _ = conf.representative_positions(
                structure.atom_pos, atom_pos_mask, atom_to_token, atom_is_rep, token_num,
            )
            pred_rep_dist = conf.pred_rep_distance(
                pred_rep_pos, tok_valid, cfg.dist_min, cfg.dist_max,
            )
            # pLDDT (per-atom lDDT vs GT).
            lddt, atom_valid = conf.per_atom_lddt(
                x_pred, structure.atom_pos[0], atom_mask[0],
            )
            plddt_bins = conf.plddt_target_bins(lddt, cfg.n_plddt_bins)
            plddt_mask = atom_valid.unsqueeze(0).expand(n, -1)
            # PDE (representative distance error).
            pde_bins, pde_mask = conf.pde_target_bins(
                pred_rep_pos, gt_rep_pos, tok_valid, cfg.n_pde_bins, cfg.pde_max,
            )
            # PAE (frame-aligned) — Stage B seam; None until token frames exist.
            pae = conf.pae_target_bins(
                pred_rep_pos, gt_rep_pos, tok_valid,
                token_frame=None, n_bins=cfg.n_pae_bins, pae_max=cfg.pae_max,
            )

        # Route through self.model (the DDP/Fabric wrapper) so confidence-head grads
        # are all-reduced across ranks. Phase4Model.forward IS the confidence head.
        logits = self.model(
            token_single_input,
            token_pair,
            pred_rep_dist,
            token_mask.expand(n, -1),
            atom_to_token.expand(n, -1),
        )
        targets = {"plddt": plddt_bins, "pde": pde_bins}
        masks = {"plddt": plddt_mask, "pde": pde_mask}
        if pae is not None:
            targets["pae"], masks["pae"] = pae
        return logits, targets, masks

    def loss_fn(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Weighted per-bin cross-entropy over the confidence heads."""
        w = self.config.loss
        loss = torch.zeros((), device=self.device)
        logs: dict[str, float] = {}

        plddt_ce = conf.masked_ce(logits["plddt"], targets["plddt"], masks["plddt"])
        loss = loss + w.plddt_loss * plddt_ce
        logs["plddt_loss"] = plddt_ce.item()

        pde_ce = conf.masked_ce(logits["pde"], targets["pde"], masks["pde"])
        loss = loss + w.pde_loss * pde_ce
        logs["pde_loss"] = pde_ce.item()

        if "pae" in targets and w.pae_loss > 0:
            pae_ce = conf.masked_ce(logits["pae"], targets["pae"], masks["pae"])
            loss = loss + w.pae_loss * pae_ce
            logs["pae_loss"] = pae_ce.item()

        logs["total_loss"] = loss.item()
        logs["main_loss"] = loss.item()
        return loss, logs

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the confidence head on one item."""
        # Frozen structure prediction (no grad) — the diffusion-step seam.
        x_pred, token_single_input, token_pair = self.predict_structure(batch)
        logits, targets, masks = self._confidence_targets_and_logits(
            batch, x_pred, token_single_input, token_pair,
        )
        loss, loss_dict = self.loss_fn(logits, targets, masks)
        self.backward(loss)
        del loss
        return loss_dict

    # training_epoch is inherited from Phase3Client (identical loop).

    @torch.no_grad()
    def inference(self, batch: Batch, timesteps: int | None = None) -> ConfidenceOutput:
        """Predict a structure, then score it with the confidence head."""
        x_pred, token_single_input, token_pair = self.predict_structure(batch, timesteps)
        structure = batch.structure
        scheme = batch.scheme
        token_mask = structure.token_mask
        atom_to_token = scheme.atom_to_token_idx_map
        n = x_pred.shape[0]
        raw_model = cast("Phase4Model", getattr(self.model, "module", self.model))

        if structure.atom_is_rep is None:
            msg = "structure.atom_is_rep is required for confidence inference."
            raise ValueError(msg)
        rep_pos, tok_valid = conf.representative_positions(
            x_pred, structure.atom_pos_mask.bool(), atom_to_token,
            structure.atom_is_rep, int(token_mask.shape[1]),
        )
        cfg = raw_model.config.confidence
        pred_rep_dist = conf.pred_rep_distance(rep_pos, tok_valid, cfg.dist_min, cfg.dist_max)
        logits = raw_model.confidence_forward(
            token_single_input, token_pair, pred_rep_dist,
            token_mask.expand(n, -1), atom_to_token.expand(n, -1),
        )
        head = raw_model.confidence_head
        return ConfidenceOutput(
            plddt=head.expected_plddt(logits["plddt"]),
            pae=head.expected_pae(logits["pae"]),
            plddt_logits=logits["plddt"],
            pae_logits=logits["pae"],
            pde_logits=logits["pde"],
        )

    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Confidence loss on a single-item validation batch."""
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        x_pred, token_single_input, token_pair = self.predict_structure(batch)
        logits, targets, masks = self._confidence_targets_and_logits(
            batch, x_pred, token_single_input, token_pair,
        )
        _, loss_dict = self.loss_fn(logits, targets, masks)
        return loss_dict
