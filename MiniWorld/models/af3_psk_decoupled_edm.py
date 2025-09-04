from pyexpat import model
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

from team_gm.data.features_BioMol import Batch, NoisyBatch
from team_gm.data.dataloader_BioMol import (
    BioMolPreProcessing,
    CropConfig,
    MSAConfig,
    MolTypeConfig,
)

from team_gm.utils import metrics  # , losses
from team_gm.utils import data_utils as du
from MiniWorld.utils.diffuser import DecoupledEDMDiffuser
from MiniWorld.utils.scheduler import DecoupledEDMScheduler
from MiniWorld.utils.solver import DecoupledEDMSolver
from team_gm.losses.auxiliary import cal_distogram_loss


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
from team_gm.modules.msa_module import MSAModule
from team_gm.modules.head import DistogramHead
from team_gm.utils.precision_manager import PrecisionConfig, precision_manager


class AF3Model(nn.Module):
    """Structure AF3 model"""

    class ConditionConfig(BaseModel):
        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        common: CommonConfig
        trunk: "AF3Model.ConditionConfig"
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
        self.msa_module = MSAModule(config.common, config.trunk.msa_module)
        # self.template_embedder = TemplateEmbedder(config.diffusion.embedder) # TODO
        self.pairformer_blocks = Pairformer(config.common, config.trunk.pairformer)

        # Diffusion module
        self.diffusion_module = DiffusionModule(config.common, config.diffusion)

        # Distogram prediction
        self.distogram_head = DistogramHead(config.common)

    @torch.no_grad()
    def mini_rollout(self):
        pass  # TODO

    def condition_forward(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, ...]:
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        assert (
            noisy_batch.msa.aligned_sequences.shape[1] == self.n_recycle_max
        ), "The number of MSA sequences should match the number of recycle steps."

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(noisy_batch)

        token_pair = torch.zeros_like(token_pair_init)
        token_single = torch.zeros_like(token_single_init)
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)
                token_pair = token_pair + self.msa_module(
                    noisy_batch,
                    i_cycle,
                    token_pair,
                    token_single_input,
                    noisy_batch.structure.residue_mask,
                )
                token_single = token_single_init + self.add_single_recycle(token_single)

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair, token_single, noisy_batch.structure.residue_mask
                )

        distogram_logit = self.distogram_head(token_pair)
        return (
            token_single_input,
            token_single,
            token_pair,
            distogram_logit,
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

    def forward(self, noisy_batch: NoisyBatch) -> torch.Tensor:
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.condition_forward(noisy_batch)
        # Diffusion forward
        atom_pos_update = self.diffusion_forward(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

        # TODO confidence head & mini_rollout
        return atom_pos_update, distogram_logit


class AF3ModelWrapper(nn.Module):
    """Wrapper for AF3Model to handle the input and output using solver."""

    def __init__(self, model: AF3Model, use_self_condition: bool = True):
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

    def prepare_condition(self, batch: Batch):
        """Prepare the model for conditioned forward pass."""
        assert not self.batch_loaded, "Batch is already loaded."
        assert not self.conditioned_forwarded, "Conditioned forward is already done."

        # Load the batch and prepare the model for conditioned forward pass
        self.load_batch(batch)
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            distogram_logit,
        ) = self.model.condition_forward(self.batch)
        self.conditioned_forwarded = True

        self.condition = {
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
            "distogram_logit": distogram_logit,
        }

    def forward(self, z_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        assert self.batch_loaded, "Batch must be loaded before forward pass."
        assert (
            self.conditioned_forwarded
        ), "Conditioned forward must be called before forward pass."

        noisy_batch = NoisyBatch(
            **self.batch.__dict__,
            x_t=z_i.unsqueeze(0),  # (B, L, 3) -> (1, B, L, 3)
            t=t_emb[None, None, None, None],  # (,) -> (1, 1, 1, 1)
            x_sc=self.z_sc,
        )

        z_update = self.model.diffusion_forward(
            noisy_batch,
            self.condition["token_single_input"],
            self.condition["token_single_trunk"],
            self.condition["token_pair_trunk"],
        )
        z_update = z_update.squeeze(0)  # (1, B, L, 3) -> (B, L, 3)
        if self.use_self_condition:
            self.z_sc = z_update

        return z_update


@dataclass
class AF3InferenceOutput:
    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Distogram logits
    distogram_logit: torch.Tensor  # (B, L, L, D)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray  # (B, T, L, 3)

    batch: Batch


class AF3Client(BaseClient):
    class DataConfig(BaseModel):
        DB_PATH: str
        meta: BioMolPreProcessing.MetaConfig
        train: BioMolPreProcessing.PipelineConfig
        valid: BioMolPreProcessing.PipelineConfig
        crop: CropConfig
        msa: MSAConfig
        mol_types: MolTypeConfig

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
        name: str = "AF3-PSK-2"
        overfitting: bool = False
        overfitting_dir: str | None = None  # Directory for overfitting mode
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Muon"] = "AdamW"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
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

        # loss: "AF3Client.LossConfig" TODO

    class DiffuserConfig(BaseModel):
        seed: int = 0
        scheduler: DecoupledEDMScheduler.DecoupledEDMSchedulerConfig
        method: Literal["AF3", "EDM"] = "AF3"  # TODO

    class Config(BaseModel):
        data: "AF3Client.DataConfig"
        model: AF3Model.Config
        experiment: "AF3Client.ExperimentsConfig"
        diffuser: "AF3Client.DiffuserConfig"

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

    def __init__(self, config: Config, name: str = "AF3-PSK-2"):
        super().__init__()
        self.set_seed(config.experiment.seed)
        self.model = AF3Model(config.model)
        if config.experiment.compile:
            self.model = torch.compile(self.model)

        self.model = self.setup_model(self.model)
        # diffuser setup
        diffuser_method = config.diffuser.method
        if diffuser_method == "AF3":
            self.diffusion_scheduler = DecoupledEDMScheduler(config.diffuser.scheduler)
            self.diffuser = DecoupledEDMDiffuser(
                config=DecoupledEDMDiffuser.DecoupledEDMDiffuserConfig(
                    seed=config.diffuser.seed,
                ),
                scheduler=self.diffusion_scheduler,
            )
            self.solver = DecoupledEDMSolver(
                config=DecoupledEDMSolver.DecoupledEDMSolverConfig(
                    seed=config.diffuser.seed
                ),
                scheduler=self.diffusion_scheduler,
            )
        else:
            raise NotImplementedError(
                f"Diffuser method {diffuser_method} is not implemented yet."
            )
        if config.experiment.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), config.experiment.max_lr
            )
        elif config.experiment.optimizer == "Muon":
            from muon import MuonWithAuxAdam

            hidden_weights = [p for p in self.model.parameters() if p.ndim >= 2]
            other_params = [p for p in self.model.parameters() if p.ndim < 2]

            param_groups = [
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": config.experiment.max_lr,
                    "weight_decay": config.experiment.weight_decay,
                },
                {
                    "params": other_params,
                    "use_muon": False,
                    "lr": config.experiment.max_lr,
                    "betas": (0.9, 0.999),
                    "weight_decay": config.experiment.weight_decay,
                },
            ]

            optimizer = MuonWithAuxAdam(param_groups)

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

    def loss_fn(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, Mapping]:
        # TODO : implement other losses like smooth lddt or distogram loss etc.
        # loss_config = self.config.experiment.loss
        atom_pos_update, distogram_logit = self.model.forward(noisy_batch)

        structure_loss = self.diffuser.cal_loss(atom_pos_update)
        # aux_losses = None TODO

        distogram_loss = cal_distogram_loss(
            distogram_logit,
            noisy_batch.structure.residue_pos,
            noisy_batch.structure.residue_mask,
        )

        loss = 4.0 * structure_loss + 0.03 * distogram_loss

        return loss, {
            "EDMLoss": structure_loss.item(),
            "DistogramLoss": distogram_loss.item(),
        }

    def training_step(self, batch: Batch):
        with precision_manager(self.model, self.config.model.precision):
            num_augment = self.config.experiment.num_augment
            noisy_atom_pos, t_emb = self.diffuser.sample(
                batch.structure.atom_pos,
                num_augment=num_augment,
                mask=batch.structure.atom_mask,
            )
            noisy_batch = NoisyBatch(**batch.__dict__, t=t_emb, x_t=noisy_atom_pos)

            if self.config.experiment.self_condition and random.random() > 0.5:
                with torch.no_grad():
                    atom_pos_update = self.model.forward(noisy_batch)
                    noisy_batch.x_sc = atom_pos_update
            loss, loss_dict = self.loss_fn(noisy_batch)

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
        batch = batch.duplicate(self.config.experiment.eval_sample_num)
        valid_dict = self.test_inference_quality(
            batch, self.config.experiment.eval_timesteps
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
    ) -> dict[str, float]:
        batch = batch.to(device=self.device)
        output = self.inference(batch, timesteps=timesteps)

        max_lddt, min_rmsd = 0, float("inf")

        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        if max_lddt < lddt:
            max_lddt = lddt

        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        if min_rmsd > rmsd:
            min_rmsd = rmsd
        # TODO: use dataclass
        return {
            "best_rmsd": min_rmsd,
            "best_lddt": max_lddt,
        }

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> AF3InferenceOutput:
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = AF3ModelWrapper(
            raw_model, use_self_condition=self.config.experiment.self_condition
        )
        batch = batch.to(device=self.device)
        model_wrapper.prepare_condition(batch)
        shape = batch.structure.atom_pos.shape

        atom_pos_pred, inter_traj, model_traj = self.solver.sample(
            model_fn=model_wrapper,
            shape=shape,
            num_steps=timesteps,
            device=self.device,
            return_intermediate=True,
        )
        inter_traj = [du.to_numpy(x) for x in inter_traj]
        model_traj = [du.to_numpy(x) for x in model_traj]
        distogram_logit = model_wrapper.condition["distogram_logit"]
        return AF3InferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            batch=batch,
            distogram_logit=distogram_logit,
        )

    def sample(self):
        pass  # TODO
