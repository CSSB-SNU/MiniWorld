import random
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.modules import Pairformer
from team_gm.utils import data_utils as du
from torch import nn

from miniworld.data.dataloader.dataloader_multistate_contam import (
    CropConfig,
    KmerFastAlignConfig,
    MSAConfig,
    MultistateConfig,
    MultiStatedbConfig,
)
from miniworld.data.features.features_multistate import Batch, NoisyBatch
from miniworld.loss import metrics
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.diffusion_module import (
    DiffusionModule,
)
from miniworld.modules.feature_embedder import InputFeatureEmbedder
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.diffusion.diffuser import EuclideanDiffuser
from miniworld.utils.diffusion.scheduler import EDMScheduler
from miniworld.utils.diffusion.solver import AF3Solver
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager
from miniworld.utils.structure.distance import (
    get_shortest_distances_from_multistructures,
)


class SdistEmbedder(nn.Module):
    """Shortest Distogram Embedder."""

    class Config(BaseModel):
        """Configuration for SdistEmbedder."""

        min_distance: float = 2.0
        max_distance: float = 22.0
        num_bins: int = 64
        pairformer: Pairformer.Config

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.to_pair = Linear(config.num_bins, 128, bias=False)
        pairformer_config = config.pairformer
        # use_single is False
        pairformer_config.use_single = False
        self.pairformer = Pairformer(config.pairformer)


    def forward(
        self,
        shortest_distogram: torch.Tensor,  # (B, L, L, D)
        token_mask: torch.Tensor,                 # (B, L), bool
    ) -> torch.Tensor:
        """Forward pass of SdistEmbedder."""
        token_pair = self.to_pair(shortest_distogram)  # (B, L, L, d_pair)
        # token_pair, _ = self.pairformer.forward(
        #     token_pair,
        #     single=None,
        #     mask=token_mask,
        # )  # (B, L, L, d_pair), None
        return token_pair  # (B, L, L, d_pair)

class Sdist2StrModel(nn.Module):
    """Shortest Distogram to Structure Model."""

    class ConditionConfig(BaseModel):
        """Configuration for the trunk condition module."""

        pairformer: Pairformer.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for Sdist2StrModel."""

        common: CommonConfig
        distogram: "SdistEmbedder.Config"
        trunk: "Sdist2StrModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization
        self.input_feature_embedder = InputFeatureEmbedder(
            config.common, config.diffusion,
        )

        # embed sdistogram
        self.sdist_embedder = SdistEmbedder(config.distogram)
        self.sdist_to_pair = nn.Sequential(
            LayerNorm(
                config.common.d_token_pair,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_token_pair,
                config.common.d_token_pair,
                init="zero",
            ),
        )

        # Recycle layers
        self.add_pair_recycle = nn.Sequential(
            LayerNorm(
                config.common.d_token_pair,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_token_pair,
                config.common.d_token_pair,
                init="zero",
            ),
        )
        self.add_single_recycle = nn.Sequential(
            LayerNorm(
                config.common.d_token_single,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_token_single,
                config.common.d_token_single,
                init="zero",
            ),
        )

        # Trunk forward
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)

        # Diffusion module
        self.diffusion_module = DiffusionModule(config.common, config.diffusion)

    def condition_forward(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass of the trunk condition module."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        if noisy_batch.msa.aligned_sequences.shape[1] != self.n_recycle_max:
            msg = "The number of MSA sequences should match the number of recycle steps."
            raise ValueError(msg)

        # gen sdistogram
        with torch.no_grad():
            residue_dists, residue_pair_mask = get_shortest_distances_from_multistructures(atom_pos, noisy_batch.structure.atom_pos_mask, noisy_batch.scheme.atom_to_residue_idx_map)
            edges = torch.linspace(
                self.config.distogram.min_distance,
                self.config.distogram.max_distance,
                self.config.distogram.num_bins - 1,
                device=atom_pos.device,
            )
            if self.training:
                contam_residue_dists, contam_residue_pair_mask = get_shortest_distances_from_multistructures(
                    noisy_batch.contam.atom_pos,
                    noisy_batch.contam.atom_pos_mask,
                    noisy_batch.contam.atom_to_residue_idx_map,
                )
                contam_start = noisy_batch.contam_bias # list of int where contamination starts
                L_contam = contam_residue_dists.shape[1]
                for b, start in enumerate(contam_start):
                    end = start + L_contam

                    main_block = residue_dists[b, start:end, start:end]           # (L_contam, L_contam)

                    contam_block = contam_residue_dists[b, :L_contam, :L_contam]  # (L_contam, L_contam)
                    contam_mask_block = contam_residue_pair_mask[b, :L_contam, :L_contam]

                    merged_block = torch.minimum(main_block, contam_block)
                    updated_block = torch.where(contam_mask_block, merged_block, main_block)

                    residue_dists[b, start:end, start:end] = updated_block

            shortest_distogram = torch.bucketize(residue_dists, edges)  # (*, L, L), int64 in [0, D-1]
            shortest_distogram_onehot = nn.functional.one_hot(
                shortest_distogram,
                num_classes=self.config.distogram.num_bins,
            ).to(torch.float32)  # (*, L, L, D)
            # mask out invalid pairs
            shortest_distogram_onehot = shortest_distogram_onehot * residue_pair_mask.unsqueeze(-1).to(torch.float32)

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(noisy_batch)

        # embed sdistogram to pair token
        token_pair_sdist = self.sdist_embedder(shortest_distogram_onehot, noisy_batch.structure.residue_mask)
        token_pair_sdist = self.sdist_to_pair(token_pair_sdist)
        token_pair_init = token_pair_init + token_pair_sdist

        token_pair = torch.zeros_like(token_pair_init)
        token_single = torch.zeros_like(token_single_init)
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)
                token_single = token_single_init + self.add_single_recycle(token_single)

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair, token_single, noisy_batch.structure.residue_mask,
                )


        return (
            token_single_input,
            token_single,
            token_pair,
        )

    def diffusion_forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_input: torch.Tensor,
        token_single_trunk: torch.Tensor,
        token_pair_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the diffusion module.

        Parameters
        ----------
        noisy_batch: NoisyBatch
            Batch of noisy data.
        token_single_input: FloatTensor, (B, L, d_single)
            Input single representation.
        token_single_trunk: FloatTensor, (B, L, d_single)
            Single representation after trunk forward.
        token_pair_trunk: FloatTensor, (B, L, L, d_pair)
            Pair representation after trunk forward.
        atom_single_cond: FloatTensor, (B, L, d_atom_single)
            Atom single condition representation.
        atom_pair: FloatTensor, (B, L, L, d_atom_pair)
            Atom pair representation.

        """
        return self.diffusion_module(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

    def forward(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of Sdist2StrModel."""
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        ) = self.condition_forward(noisy_batch, atom_pos)
        # Diffusion forward
        return self.diffusion_forward(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

class Sdist2StrModelWrapper(nn.Module):
    """Wrapper for Sdist2StrModel to handle the input and output using solver."""

    def __init__(self, model: Sdist2StrModel, use_self_condition: bool = True) -> None:
        super().__init__()
        self.batch_loaded = False
        self.conditioned_forwarded = False
        self.model = model
        self.use_self_condition = use_self_condition
        self.z_sc = None  # Placeholder for self-conditioned input

    def load_batch(self, batch: Batch) -> None:
        """Load a new batch to the model."""
        self.batch = batch
        self.z_sc = None
        self.batch_loaded = True

    def prepare_condition(self, batch: Batch, atom_pos: torch.Tensor) -> None:
        """Prepare the model for conditioned forward pass."""
        if self.use_self_condition:
            msg = "Batch is already loaded. Cannot prepare condition again."
            raise ValueError(msg)
        if self.conditioned_forwarded:
            msg = "Conditioned forward is already done. Cannot prepare condition again."
            raise ValueError(msg)

        # Load the batch and prepare the model for conditioned forward pass
        self.load_batch(batch)
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        ) = self.model.condition_forward(self.batch, atom_pos)
        self.conditioned_forwarded = True

        self.condition = {
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
        }

    def forward(self, z_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model wrapper."""
        if not self.batch_loaded:
            msg = "Batch must be loaded before forward pass."
            raise ValueError(msg)
        if not self.conditioned_forwarded:
            msg = "Conditioned forward must be called before forward pass."
            raise ValueError(msg)

        n_str = z_i.shape[0]
        t_emb = t_emb[None,None,None,None].repeat(n_str, 1, 1, 1)

        noisy_batch = NoisyBatch(
            **self.batch.__dict__,
            x_t=z_i,
            t=t_emb,
            x_sc=self.z_sc,
        )

        z_update = self.model.diffusion_forward(
            noisy_batch,
            self.condition["token_single_input"],
            self.condition["token_single_trunk"],
            self.condition["token_pair_trunk"],
        )
        if self.use_self_condition:
            self.z_sc = z_update

        return z_update


@dataclass
class Sdist2StrInferenceOutput:
    """Output of Sdist2Str inference."""

    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray | None # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray | None # (B, T, L, 3)

    batch: Batch


class Sdist2StrClient(BaseClient):
    """Client for training and inference of Sdist2StrModel."""

    class DataConfig(BaseModel):
        """Configuration for data loading."""

        crop: CropConfig
        msa: MSAConfig
        kmer_fast_align: KmerFastAlignConfig
        multistate: MultistateConfig
        train_preprocessing : MultiStatedbConfig
        valid_preprocessing : MultiStatedbConfig

    class ExperimentsConfig(BaseModel):
        """Configuration for experiment settings."""

        comment: str = "default"
        name: str = "Sdist2Str-PSK"
        overfitting: bool = False
        overfitting_dir: str | None = None  # Directory for overfitting mode
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        warmup_steps: int = 5e3
        decay_steps: int = 5e6
        decay_factor: float = 0.95
        self_condition: bool = True
        compile: bool = False
        num_augment: int = 8
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
        ema_decay: float = 0.9999


    class DiffuserConfig(BaseModel):
        """Configuration for the diffuser."""

        seed: int = 0
        scheduler: EDMScheduler.EDMSchedulerConfig
        method: Literal["AF3", "EDM"] = "AF3"

    class Config(BaseModel):
        """Configuration for Sdist2StrClient."""

        data: "Sdist2StrClient.DataConfig"
        model: Sdist2StrModel.Config
        experiment: "Sdist2StrClient.ExperimentsConfig"
        diffuser: "Sdist2StrClient.DiffuserConfig"


    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(Sdist2StrModel(config.model))

        if config.experiment.use_ema:
            self.add_callback(ModelEMA(config.experiment.ema_decay))
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

    def loss_fn(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> tuple[torch.Tensor, Mapping]:
        """Compute the loss for a batch."""
        atom_pos_update = self.model.forward(noisy_batch, atom_pos)

        structure_loss = self.diffuser.cal_loss(atom_pos_update)

        loss = 4.0 * structure_loss  # + aux_losses

        return loss, {"EDMLoss": loss.item()}

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        with precision_manager(self.model, self.config.model.precision):
            num_augment = self.config.experiment.num_augment
            noisy_atom_pos, t_emb = self.diffuser.sample(
                batch.structure.atom_pos,
                num_augment=num_augment,
                mask=batch.structure.atom_pos_mask,
            )
            noisy_batch = NoisyBatch(**batch.__dict__, t=t_emb, x_t=noisy_atom_pos)

            if self.config.experiment.self_condition and random.random() > 0.5:
                with torch.no_grad():
                    atom_pos_update = self.model.forward(noisy_batch, batch.structure.atom_pos)
                    noisy_batch.x_sc = atom_pos_update
            loss, loss_dict = self.loss_fn(noisy_batch, batch.structure.atom_pos)

            self.backward(loss)
            return loss_dict


    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Validate the model on a batch."""
        # Note that when doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        return self.test_inference_quality(
            batch, self.config.experiment.eval_timesteps, n_sample=self.config.experiment.eval_sample_num,
        )

    @torch.no_grad()
    def test_inference_quality(
        self,
        batch: Batch,
        timesteps: int = 100,
        n_sample: int = 48,
    ) -> dict[str, float]:
        """Test the inference quality of the model on a batch."""
        batch = batch.to(device=self.device)
        output = self.best_of_n_sample(batch, timesteps=timesteps, n_sample=n_sample)

        max_lddt, min_rmsd = 0, float("inf")

        N_sample = output.atom_pos_pred.shape[0]
        N_str = batch.structure.atom_pos.shape[1]

        lddt_list = []
        rmsd_list = []

        for sample_idx in range(N_sample):
            max_lddt = 0
            min_rmsd = float("inf")
            for true_idx in range(N_str):
                lddt = metrics.cal_atom_lddt(
                    output.atom_pos_pred[sample_idx,0],
                    batch.structure.atom_pos[0,true_idx],
                    batch.structure.atom_pos_mask[0,true_idx],
                )
                max_lddt = max(max_lddt, lddt)

                rmsd = metrics.cal_aligned_rmsd(
                    output.atom_pos_pred[sample_idx,0],
                    batch.structure.atom_pos[0,true_idx],
                    batch.structure.atom_pos_mask[0,true_idx],
                )
                min_rmsd = min(min_rmsd, rmsd)

            lddt_list.append(max_lddt)
            rmsd_list.append(min_rmsd)
        lddt_list = np.array(lddt_list)
        rmsd_list = np.array(rmsd_list)

        max_lddt, min_lddt = lddt_list.max(), lddt_list.min()
        max_rmsd, min_rmsd = rmsd_list.max(), rmsd_list.min()

        return {
            "max_lddt": max_lddt,
            "min_lddt": min_lddt,
            "max_rmsd": max_rmsd,
            "min_rmsd": min_rmsd,
        }

    @torch.no_grad()
    def best_of_n_sample(
        self,
        batch: Batch,
        timesteps: int = 100,
        n_sample: int = 48,
        batch_size: int = 48,
        return_traj: bool = False,
    ) -> Sdist2StrInferenceOutput:
        """Generate best of N samples for a batch."""
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = Sdist2StrModelWrapper(
            raw_model, use_self_condition=self.config.experiment.self_condition,
        )
        batch = batch.to(device=self.device)

        atom_pos = batch.structure.atom_pos
        B, _, L, _ = atom_pos.shape
        model_wrapper.prepare_condition(batch, atom_pos)

        cursor = 0
        atom_pos_pred_stacked = []
        inter_traj_stacked = []
        model_traj_stacked = []

        while cursor < n_sample:
            print(f"Sampling {min(n_sample - cursor, batch_size)} / {n_sample} structures")  # noqa: T201
            shape = (min(batch_size, n_sample - cursor), B, L, 3)
            atom_pos_pred, inter_traj, model_traj = self.solver.sample(
                model_fn=model_wrapper,
                shape=shape,
                num_steps=timesteps,
                device=self.device,
                return_intermediate=True,
            )
            atom_pos_pred_stacked.append(atom_pos_pred)
            if return_traj:
                inter_traj = [du.to_numpy(x) for x in inter_traj] # list of (N_str, B, L, 3)
                model_traj = [du.to_numpy(x) for x in model_traj]
                inter_traj = np.stack(inter_traj, axis=1)  # (N_str, T, B, L, 3)
                model_traj = np.stack(model_traj, axis=1)  # (N_str, T, B, L, 3)
                inter_traj_stacked.append(inter_traj)
                model_traj_stacked.append(model_traj)
            cursor += shape[0]
        atom_pos_pred = torch.cat(atom_pos_pred_stacked, dim=0)  # (N_sample, B, L, 3)
        if return_traj:
            inter_traj = np.concatenate(inter_traj_stacked, axis=0)  # (N_sample, T, B, L, 3)
            model_traj = np.concatenate(model_traj_stacked, axis=0)  # (N_sample, T, B, L, 3)
        else:
            inter_traj = None
            model_traj = None

        return Sdist2StrInferenceOutput(
                atom_pos_pred=atom_pos_pred,
                model_traj=model_traj,
                inter_traj=inter_traj,
                batch=batch,
        )


