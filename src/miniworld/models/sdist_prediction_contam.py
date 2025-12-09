import random
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.modules import Pairformer
from torch import nn

from miniworld.data.dataloader.dataloader_multistate import (
    BioMolMonomerPreProcessingConfig,
    CropConfig,
    KmerFastAlignConfig,
    MSAConfig,
    MultistateConfig,
)
from miniworld.data.features.features_multistate import Batch, NoisyBatch
from miniworld.loss.multistate import cal_shortest_distogram_loss
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.feature_embedder import RelativePositionEmbedding
from miniworld.modules.head import DistogramHead
from miniworld.modules.msa_module import MSAModule
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager
from miniworld.utils.structure.distance import (
    get_shortest_distances_from_multistructures,
)


class InputFeatureEmbedder(nn.Module):
    def __init__(
        self,
        common_config: CommonConfig,
    ):
        super().__init__()
        self.num_res_class = common_config.num_res_class
        self.use_checkpoint = common_config.use_checkpoint
        self.d_token_pair = common_config.d_token_pair
        d_init = common_config.d_token_single_input
        self.to_token_pair_left = Linear(
            d_init,
            common_config.d_token_pair,
            init="default",
            bias=False,
        )
        self.to_token_pair_right = Linear(
            d_init,
            common_config.d_token_pair,
            init="default",
            bias=False,
        )
        self.relative_position_embedder = RelativePositionEmbedding(
            d_hidden=common_config.d_token_pair,
            r_max=common_config.r_max,
            s_max=common_config.s_max,
        )

    def forward(
        self, noisy_batch: NoisyBatch
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_token_single_input"],
        Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ]:
        token_single_input = torch.concat(
            [
                noisy_batch.msa.profile,
                noisy_batch.msa.deletion_mean.unsqueeze(-1),
            ],
            dim=-1,
        )

        token_left = self.to_token_pair_left(token_single_input)
        token_right = self.to_token_pair_right(token_single_input)
        token_pair_init = rearrange(token_left, "b l d -> b l 1 d") + rearrange(
            token_right, "b l d -> b 1 l d"
        )

        token_pair_init = token_pair_init + self.relative_position_embedder(noisy_batch)

        return (
            token_single_input,
            token_pair_init,
        )



class SdistModel(nn.Module):
    """Predict shortest distance."""

    class ConditionConfig(BaseModel):
        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        common: CommonConfig
        trunk: "SdistModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization
        self.input_feature_embedder = InputFeatureEmbedder(
            config.common,
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

        # Trunk forward
        self.msa_module = MSAModule(config.common, config.trunk.msa_module)
        # self.template_embedder = TemplateEmbedder(config.diffusion.embedder) # TODO
        self.pairformer_blocks = Pairformer(config.common, config.trunk.pairformer)
        self.distogram_head = DistogramHead(config.common)

    @torch.no_grad()
    def mini_rollout(self):
        pass  # TODO

    def forward(self, noisy_batch: NoisyBatch) -> torch.Tensor:
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
            token_pair_init,
        ) = self.input_feature_embedder(noisy_batch)

        token_pair = torch.zeros_like(token_pair_init)

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

                token_pair, _ = self.pairformer_blocks.forward(
                    token_pair, None, noisy_batch.structure.residue_mask
                )

        distogram_logit = self.distogram_head(token_pair)

        return distogram_logit



@dataclass
class SdistInferenceOutput:
    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray  # (B, T, L, 3)

    batch: Batch


class SdistClient(BaseClient):
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
        name: str = "SDIST-PSK"
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



    class Config(BaseModel):
        data: "SdistClient.DataConfig"
        model: SdistModel.Config
        experiment: "SdistClient.ExperimentsConfig"

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

    def __init__(self, config: Config, name: str = "SDIST-PSK-2"):
        super().__init__()
        self.set_seed(config.experiment.seed)
        self.model = SdistModel(config.model)
        if config.experiment.compile:
            self.model = torch.compile(self.model)

        self.model = self.setup_model(self.model)
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

    def loss_fn(self, batch: Batch) -> tuple[torch.Tensor, Mapping]:
        # TODO : implement other losses like smooth lddt or distogram loss etc.
        # loss_config = self.config.experiment.loss
        distogram_logit = self.model.forward(batch)

        distogram_loss = cal_shortest_distogram_loss(
            distogram_logit,
            atom_pos=batch.structure.atom_pos,
            atom_pos_mask=batch.structure.atom_pos_mask,
            atom_to_res_idx=batch.scheme.atom_to_residue_idx_map,
            min_distance=2.0,
            max_distance=22.0,
        )

        return distogram_loss, {"distogram_loss": distogram_loss.item()}

    def training_step(self, batch: Batch):
        with precision_manager(self.model, self.config.model.precision):
            loss, loss_dict = self.loss_fn(batch)

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
        loss, loss_dict = self.loss_fn(batch)
        self.log_metrics(
            {"valid/" + k: v for k, v in loss_dict.items()}, on_epoch=True
        )
        return loss_dict

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distogram_logit = self.model.forward(batch) # (B, L, L, D)
        D = distogram_logit.shape[-1]
        prob = torch.nn.functional.softmax(distogram_logit, dim=-1)  # (B, L, L, D)
        edges = torch.linspace(2.0, 22.0, D, device=distogram_logit.device)
        expected_dist = (prob * edges).sum(dim=-1)  # (B, L, L)
        residue_dists, residue_pair_mask = get_shortest_distances_from_multistructures(
            atom_pos=batch.structure.atom_pos,
            atom_pos_mask=batch.structure.atom_pos_mask,
            atom_to_res_idx=batch.scheme.atom_to_residue_idx_map,
            min_distance=2.0,
            max_distance=22.0,
        )  # residue_dists: (*, L, L), residue_pair_mask: (*, L, L)
        return expected_dist, residue_dists, residue_pair_mask
