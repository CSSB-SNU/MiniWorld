import numpy as np
import random
import torch
import torch.nn as nn

from typing import Literal
from collections.abc import Mapping
from dataclasses import dataclass
from contextlib import ExitStack

from team_gm import BaseClient
from pydantic import BaseModel

from MiniWorld.data.features.features_multistate_contam import Batch, NoisyBatch
from MiniWorld.data.dataloader.dataloader_multistate_contam import (
    CropConfig,
    KmerFastAlignConfig,
    MSAConfig,
    MultistateConfig,
    BioMolMonomerPreProcessingConfig,
)

from team_gm.utils import metrics  # , losses
from team_gm.utils import data_utils as du
from MiniWorld.utils.diffuser import EuclideanDiffuser
from MiniWorld.utils.structure.sdist import get_shortest_distances
from team_gm.utils.scheduler import EDM_Scheduler
from team_gm.utils.solver import AF3Solver

from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from team_gm.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from team_gm.modules.diffusion_module import (
    DiffusionModule,
)
from team_gm.modules.feature_embedder import InputFeatureEmbedder
from team_gm.modules import Pairformer
from team_gm.utils.precision_manager import PrecisionConfig, precision_manager

class SdistEmbedder(nn.Module):
    """Shortest Distogram Embedder."""

    class Config(BaseModel):
        min_distance: float = 2.0
        max_distance: float = 22.0
        num_bins: int = 64
        pairformer: Pairformer.Config

    def __init__(self, common_config: CommonConfig, config: Config):
        super().__init__()

        self.to_pair = Linear(config.num_bins, 128, bias=False)
        pairformer_config = config.pairformer
        # use_single is False
        pairformer_config.use_single = False
        self.pairformer = Pairformer(common_config, config.pairformer)


    def forward(
        self,
        shortest_distogram: torch.Tensor,  # (B, L, L, D)
        token_mask: torch.Tensor,                 # (B, L), bool
    ) -> torch.Tensor:
        token_pair = self.to_pair(shortest_distogram)  # (B, L, L, d_pair)
        token_pair, _ = self.pairformer.forward(
            token_pair,
            single=None,
            mask=token_mask,
        )  # (B, L, L, d_pair), None
        return token_pair  # (B, L, L, d_pair)

class Sdist2StrModel(nn.Module):
    """Shortest Distogram to Structure Model."""

    class ConditionConfig(BaseModel):
        pairformer: Pairformer.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        common: CommonConfig
        distogram: "SdistEmbedder.Config"
        trunk: "Sdist2StrModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization
        self.input_feature_embedder = InputFeatureEmbedder(
            config.common, config.diffusion
        )

        # embed sdistogram
        self.sdist_embedder = SdistEmbedder(config.common, config.distogram)
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
        self.pairformer_blocks = Pairformer(config.common, config.trunk.pairformer)

        # Diffusion module
        self.diffusion_module = DiffusionModule(config.common, config.diffusion)

    @torch.no_grad()
    def mini_rollout(self):
        pass  # TODO

    def condition_forward(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        assert (
            noisy_batch.msa.aligned_sequences.shape[1] == self.n_recycle_max
        ), "The number of MSA sequences should match the number of recycle steps."

        # gen sdistogram
        with torch.no_grad():
            residue_dists, residue_pair_mask = get_shortest_distances(atom_pos, noisy_batch.structure.atom_pos_mask, noisy_batch.scheme.atom_to_residue_idx_map)
            edges = torch.linspace(
                self.config.distogram.min_distance,
                self.config.distogram.max_distance,
                self.config.distogram.num_bins - 1,
                device=atom_pos.device,
            )
            if self.training:
                contam_residue_dists, contam_residue_pair_mask = get_shortest_distances(
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
                    token_pair, token_single, noisy_batch.structure.residue_mask
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
        x_update = self.diffusion_module(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )
        return x_update

    def forward(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> torch.Tensor:
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        ) = self.condition_forward(noisy_batch, atom_pos)
        # Diffusion forward
        atom_pos_update = self.diffusion_forward(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

        # TODO confidence head & mini_rollout
        return atom_pos_update


class Sdist2StrModelWrapper(nn.Module):
    """Wrapper for Sdist2StrModel to handle the input and output using solver."""

    def __init__(self, model: Sdist2StrModel, use_self_condition: bool = True):
        super().__init__()
        self.batch_loaded = False
        self.conditioned_forwarded = False
        self.model = model
        self.use_self_condition = use_self_condition
        self.z_sc = None  # Placeholder for self-conditioned input

    def load_batch(self, batch: Batch):
        """Load a new batch to the model."""
        self.batch = batch
        self.z_sc = None
        self.batch_loaded = True

    def prepare_condition(self, batch: Batch, atom_pos: torch.Tensor):
        """Prepare the model for conditioned forward pass."""
        assert not self.batch_loaded, "Batch is already loaded."
        assert not self.conditioned_forwarded, "Conditioned forward is already done."

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
        assert self.batch_loaded, "Batch must be loaded before forward pass."
        assert (
            self.conditioned_forwarded
        ), "Conditioned forward must be called before forward pass."

        n_str = z_i.shape[0]
        t_emb = t_emb[None,None,None,None].repeat(n_str, 1, 1, 1)

        noisy_batch = NoisyBatch(
            **self.batch.__dict__,
            # x_t=z_i.unsqueeze(0),  # (B, L, 3) -> (1, B, L, 3)
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
        # z_update = z_update.squeeze(0)  # (1, B, L, 3) -> (B, L, 3)
        if self.use_self_condition:
            self.z_sc = z_update

        return z_update


@dataclass
class Sdist2StrInferenceOutput:
    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray | None # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray | None # (B, T, L, 3)

    batch: Batch


class Sdist2StrClient(BaseClient):
    class DataConfig(BaseModel):
        crop: CropConfig
        msa: MSAConfig
        kmer_fast_align: KmerFastAlignConfig
        multistate: MultistateConfig
        train_preprocessing : BioMolMonomerPreProcessingConfig
        valid_preprocessing : BioMolMonomerPreProcessingConfig

    class LossConfig(BaseModel):
        # TODO
        t_normalize_clip: float = 0.9
        translation_loss_weight: float = 2.0
        aux_loss_weight: float = 1.0

        all_atom_loss_weight: float = 1.0
        all_atom_loss_t_filter: float = 0.25
        dist_mat_threshold: float = 6.0
        dist_mat_loss_weight: float = 1.0
        dist_mat_loss_t_filter: float = 0.25
        atom_clash_loss_weight: float = 0.0
        atom_clash_loss_t_filter: float = 0.25
        bond_length_loss_weight: float = 1.0
        bond_length_loss_t_filter: float = 0.25

    class ExperimentsConfig(BaseModel):
        comment: str = "default"
        name: str = "Sdist2Str-PSK-2"
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

        # loss: "Sdist2StrClient.LossConfig" TODO

    class DiffuserConfig(BaseModel):
        seed: int = 0
        scheduler: EDM_Scheduler.EDM_SchedulerConfig
        method: Literal["AF3", "EDM"] = "AF3"  # TODO

    class Config(BaseModel):
        data: "Sdist2StrClient.DataConfig"
        model: Sdist2StrModel.Config
        experiment: "Sdist2StrClient.ExperimentsConfig"
        diffuser: "Sdist2StrClient.DiffuserConfig"

    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def get_step_decay_scheduler_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1e3,
        decay_steps: int = 5e4,
        decay_factor: float = 0.95,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Return a LambdaLR scheduler that
        1) linearly warms up from 0 → 1 over the first `warmup_steps`
        2) thereafter, multiplies the lr by `decay_factor` every `decay_steps`
        The scheduler multiplies the optimizer's base_lr by the returned factor.
        """

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # warmup: 0 -> 1
                return step / float(warmup_steps)
            else:
                # step decay: factor ** floor((step - warmup_steps) / decay_steps)
                num_decays = (step - warmup_steps) // decay_steps
                return decay_factor**num_decays

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def __init__(self, config: Config, name: str = "Sdist2Str-PSK-2"):
        super().__init__()
        self.config = config
        self.set_seed(config.experiment.seed)
        self.model = Sdist2StrModel(config.model)
        if config.experiment.compile:
            self.model = torch.compile(self.model)

        self.model = self.setup_model(self.model)
        # diffuser setup
        diffuser_method = config.diffuser.method
        if diffuser_method == "AF3":
            self.diffusion_scheduler = EDM_Scheduler(config.diffuser.scheduler)
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
            raise NotImplementedError(
                f"Diffuser method {diffuser_method} is not implemented yet."
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), config.experiment.max_lr)
        model_scheduler = self.get_step_decay_scheduler_with_warmup(
            optimizer,
            config.experiment.warmup_steps,
            config.experiment.decay_steps,
            config.experiment.decay_factor,
        )
        self.setup(
            config=config,
            optimizer=optimizer,
            scheduler=model_scheduler,
            clip_max_norm=config.experiment.grad_clip_max_norm,
            accum_steps=config.experiment.grad_accum_steps,
            name=name,
        )

    def loss_fn(self, noisy_batch: NoisyBatch, atom_pos: torch.Tensor) -> tuple[torch.Tensor, Mapping]:
        # TODO : implement other losses like smooth lddt or distogram loss etc.
        # loss_config = self.config.experiment.loss
        atom_pos_update = self.model.forward(noisy_batch, atom_pos)

        structure_loss = self.diffuser.cal_loss(atom_pos_update)
        # aux_losses = None TODO

        loss = 4.0 * structure_loss  # + aux_losses

        return loss, {"EDMLoss": loss.item()}

    def training_step(self, batch: Batch):
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

            self.log_metrics(
                {"train/total_loss": loss.item()},
                on_step=True,
                on_epoch=True,
            )

            # self.log_message(batch.name[0])
            self.backward(loss)

    def validation_step(self, batch: Batch):
        # Note that when doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        assert batch.shape[0] == 1
        valid_dict = self.test_inference_quality(
            batch, self.config.experiment.eval_timesteps, n_sample=self.config.experiment.eval_sample_num
        )
        self.log_metrics(
            {"valid/" + k: v for k, v in valid_dict.items()}, on_epoch=True
        )
        return valid_dict

    @torch.no_grad()
    def test_inference_quality(
        self,
        batch: Batch,
        timesteps: int = 100,
        n_sample: int = 48,
    ) -> dict[str, float]:
        batch = batch.to(device=self.device)
        output = self.best_of_N_sample(batch, timesteps=timesteps, n_sample=n_sample)

        max_lddt, min_rmsd = 0, float("inf")

        N_sample = output.atom_pos_pred.shape[0]
        N_str = batch.structure.atom_pos.shape[1]

        lddt_list = []
        rmsd_list = []

        # TODO : vectorize
        for sample_idx in range(N_sample):
            max_lddt = 0
            min_rmsd = float("inf")
            for true_idx in range(N_str):
                lddt = metrics.cal_atom_lddt(
                    output.atom_pos_pred[sample_idx,0],
                    batch.structure.atom_pos[0,true_idx],
                    batch.structure.atom_pos_mask[0,true_idx],
                )
                if max_lddt < lddt:
                    max_lddt = lddt

                rmsd = metrics.cal_aligned_rmsd(
                    output.atom_pos_pred[sample_idx,0],
                    batch.structure.atom_pos[0,true_idx],
                    batch.structure.atom_pos_mask[0,true_idx],
                )
                if min_rmsd > rmsd:
                    min_rmsd = rmsd

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
    def best_of_N_sample(
        self,
        batch: Batch,
        timesteps: int = 100,
        n_sample: int = 48,
        batch_size: int = 48,
        return_traj: bool = False,
    ) -> Sdist2StrInferenceOutput:
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = Sdist2StrModelWrapper(
            raw_model, use_self_condition=self.config.experiment.self_condition
        )
        batch = batch.to(device=self.device)

        # TODO for now atom_pos shape : (B, N_str, L, 3) -> (N_str, B, L, 3)
        atom_pos = batch.structure.atom_pos
        B, _, L, _ = atom_pos.shape
        model_wrapper.prepare_condition(batch, atom_pos)

        cursor = 0
        atom_pos_pred_stacked = []
        inter_traj_stacked = []
        model_traj_stacked = []

        while cursor < n_sample:
            print(f"Sampling {min(n_sample - cursor, batch_size)} / {n_sample} structures")
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


    @torch.no_grad()
    def SPELL_sample(
        self,
        batch: Batch,
        timesteps: int = 100,
        N_sample: int = 10,
        radius: float = 5.0,
    ) -> list[Sdist2StrInferenceOutput]:
        from MiniWorld.utils.solver import SPELLSolver
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = Sdist2StrModelWrapper(
            raw_model, use_self_condition=self.config.experiment.self_condition
        )
        batch = batch.to(device=self.device)

        # TODO for now atom_pos shape : (B, N_str, L, 3) -> (N_str, B, L, 3)
        atom_pos = batch.structure.atom_pos
        B, N_str, L, _ = atom_pos.shape
        shape = (N_str, B, L, 3)
        model_wrapper.prepare_condition(batch, atom_pos)

        self.solver = SPELLSolver(
            config=SPELLSolver.SolverConfig(seed=self.config.diffuser.seed, radius=radius),
            scheduler=self.diffusion_scheduler,
        )

        results = []
        for _ in range(N_sample):
            atom_pos_pred, inter_traj, model_traj = self.solver.sample(
                model_fn=model_wrapper,
                shape=shape,
                num_steps=timesteps,
                device=self.device,
                return_intermediate=True,
            )
            inter_traj = [du.to_numpy(x) for x in inter_traj]
            model_traj = [du.to_numpy(x) for x in model_traj]
            results.append(Sdist2StrInferenceOutput(
                atom_pos_pred=atom_pos_pred,
                model_traj=np.stack(model_traj, axis=1),
                inter_traj=np.stack(inter_traj, axis=1),
                batch=batch,
            ))
        return results

        pass  # TODO

    def sample(self):
        pass  # TODO
