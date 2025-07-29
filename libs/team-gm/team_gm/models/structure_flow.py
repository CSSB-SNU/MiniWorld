import numpy as np
import random
import torch
import torch.nn as nn

from typing import Literal, overload
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from jaxtyping import Float, Bool

from team_gm import BaseClient, losses, typecheck
from team_gm.data import chemical
from team_gm.data.features import Batch, SequenceFeatures, StructureFeatures
from team_gm.utils import metrics, so3_utils
from team_gm.utils import data_utils as du
from team_gm.utils.frame_utils import (
    local_to_global,
    global_to_local,
    cal_distogram,
    cal_unit_vector,
    centered_gaussian,
    uniform_so3,
)
from team_gm.utils.rigid_utils import Rigid
from team_gm.utils.interpolant import sample_rigid_path, RigidPath
from team_gm.modules import Pairformer, StructureModule, AllAtomModule
from team_gm.modules.primitives import (
    Transition,
    RelativePositionEmbedding,
    Linear,
    TimestepEmbedder,
    LayerNorm,
)


class EmbedConfig(BaseModel):
    d_single: int = 384
    d_pair: int = 128
    num_res_class: int = len(chemical.PROT_RES_NAMES)
    relpos_bins: int = 32
    implementation: Literal["pytorch", "triton"] = "pytorch"


class SingleEmbedder(nn.Module):
    def __init__(self, config: EmbedConfig):
        super().__init__()
        self.config = config

        self.res_embedder = nn.Embedding(config.num_res_class, config.d_single)
        self.transition = Transition(config.d_single, implementation="pytorch")

    @typecheck
    def forward(self, sequence: SequenceFeatures) -> Float[torch.Tensor, "B L d_single"]:
        single = self.res_embedder(sequence.res_type)
        single = single + self.transition(single)
        return single


class PairEmbedder(nn.Module):
    def __init__(self, config: EmbedConfig):
        super().__init__()
        self.config = config

        self.seq_embedder = nn.Embedding(config.num_res_class, 2 * config.d_pair)
        self.relpos_embedder = RelativePositionEmbedding(
            config.relpos_bins, config.d_pair
        )
        self.transition = Transition(config.d_pair, implementation=config.implementation)

    @typecheck
    def forward(self, sequence: SequenceFeatures) -> Float[torch.Tensor, "B L L d_pair"]:
        seq_embed = self.seq_embedder.forward(sequence.res_type)
        seq_embed_i, seq_embed_j = seq_embed.chunk(2, dim=-1)
        pair = seq_embed_i.unsqueeze(-2) + seq_embed_j.unsqueeze(-3)
        pair = pair + self.relpos_embedder(sequence.seq_idx)
        pair = pair + self.transition(pair)
        return pair


class TemplateEmbedder(nn.Module):
    class Config(BaseModel):
        d_pair: int = 128
        dgram_bins: int = 22
        implementation: Literal["pytorch", "triton"] = "pytorch"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.bins = config.dgram_bins

        self.mlp = nn.Sequential(
            Linear((config.dgram_bins + 3) * 2, config.d_pair * 2, bias=False),
            nn.ReLU(),
            LayerNorm(
                config.d_pair * 2, bias=False, implementation=config.implementation
            ),
            Linear(config.d_pair * 2, config.d_pair, bias=False),
            nn.ReLU(),
            LayerNorm(config.d_pair, bias=False, implementation=config.implementation),
        )

    @typecheck
    def forward(self, rigid_path: RigidPath) -> Float[torch.Tensor, "B L L d_pair"]:
        dgram_t = cal_distogram(rigid_path.trans_t, num_bins=self.bins)
        unitvec_t = cal_unit_vector(rigid_path.rigid_t)
        if rigid_path.rigid_sc is not None:
            dgram_sc = cal_distogram(rigid_path.rigid_sc.get_trans(), num_bins=self.bins)
            unitvec_sc = cal_unit_vector(rigid_path.rigid_sc)
        else:
            dgram_sc = torch.zeros_like(dgram_t)
            unitvec_sc = torch.zeros_like(unitvec_t)

        pair = torch.cat([dgram_t, unitvec_t, dgram_sc, unitvec_sc], dim=-1)
        pair = self.mlp(pair)
        return pair


class StructureFlowModel(nn.Module):
    """Structure Flow matching model"""

    class TrunkConfig(BaseModel):
        embedder: EmbedConfig
        pairformer: Pairformer.Config

        @field_validator("embedder")
        def validate_num_res_class(cls, v: EmbedConfig) -> EmbedConfig:
            if v.num_res_class != len(chemical.PROT_RES_NAMES):
                raise ValueError(
                    f"Invalid num_res_class: {v.num_res_class}, "
                    f"expected {len(chemical.PROT_RES_NAMES)}"
                )
            return v

    class FlowConfig(BaseModel):
        embedder: TemplateEmbedder.Config
        structure_module: StructureModule.Config
        allatom_module: AllAtomModule.Config

        @field_validator("allatom_module")
        def validate_atom_num(cls, v: AllAtomModule.Config) -> AllAtomModule.Config:
            if v.atom_num != chemical.RES_ATOM_NUM:
                raise ValueError(
                    f"Invalid atom_num: {v.atom_num}, expected {chemical.RES_ATOM_NUM}"
                )
            return v

    class Config(BaseModel):
        d_single: int = 384
        d_pair: int = 128
        implementation: Literal["pytorch", "triton"] = "pytorch"
        trunk: "StructureFlowModel.TrunkConfig"
        flow: "StructureFlowModel.FlowConfig"

    @typecheck
    @dataclass
    class Output:
        rigid_pred: Rigid
        atom_pos_pred: Float[torch.Tensor, "B L N 3"]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Trunk forward
        self.pair_embedder = PairEmbedder(config.trunk.embedder)
        self.single_embedder = SingleEmbedder(config.trunk.embedder)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)
        self.ln_pair = LayerNorm(self.config.d_pair, bias=False)
        self.ln_single = LayerNorm(self.config.d_single, bias=False)

        # Flow module
        self.t_embedder = TimestepEmbedder(config.d_single)
        self.single_transition = Transition(config.d_single, implementation="pytorch")
        self.template_embedder = TemplateEmbedder(config.flow.embedder)
        self.structure_module_blocks = StructureModule(config.flow.structure_module)
        self.allatom_module = AllAtomModule(config.flow.allatom_module)
        # TODO: add dgram prediction head

    @typecheck
    def trunk_forward(
        self,
        sequence: SequenceFeatures,
        res_mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B L L d_pair"], Float[torch.Tensor, "B L d_single"]]:
        """Forward pass of the trunk module.

        Parameters
        ----------
        sequence: SequenceFeatures
            Sequence features containing residue types and sequence indices.
        res_mask: torch.Tensor | None
            Residue mask indicating valid residues in the sequence.

        Returns
        -------
        pair: FloatTensor, (B, L, L d_pair)
            Pair representation.
        single: FloatTensor, (B, L, d_single)
            Single representation.
        """
        pair = self.pair_embedder.forward(sequence)
        single = self.single_embedder.forward(sequence)
        pair, single = self.pairformer_blocks.forward(pair, single, res_mask)
        pair = self.ln_pair(pair)
        single = self.ln_single(single)
        return pair, single

    @typecheck
    def flow_forward(
        self,
        rigid_path: RigidPath,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"],
    ) -> Output:
        """Forward pass of the flow module.

        Parameters
        ----------
        rigid_path: RigidPath
            Rigid path containing the rigid transformation and time information.
        pair: FloatTensor, (B, L, L d_pair)
            Pair representation.
        single: FloatTensor, (B, L, d_single)
            Single representation.

        Returns
        -------
        Output: StructureFlowModel.Output
            Output containing predicted rigid, atom positions.
        """
        # Add timestep & self condition
        single = single + self.t_embedder(rigid_path.t[:, None])
        single = single + self.single_transition(single)
        pair = pair + self.template_embedder.forward(rigid_path)

        # Structure update
        init_single = single.clone()
        rigid_pred, single = self.structure_module_blocks.forward(
            single, pair, rigid_path.rigid_t, rigid_path.res_mask
        )
        local_atom_pos_pred = self.allatom_module(single, init_single)
        atom_pos_pred = local_to_global(rigid_pred, local_atom_pos_pred)
        return StructureFlowModel.Output(
            rigid_pred=rigid_pred,
            atom_pos_pred=atom_pos_pred,
        )

    @overload
    def __call__(
        self,
        sequence: SequenceFeatures,
        res_mask: Bool[torch.Tensor, "B L"] | None = None,
        mode: Literal["trunk"] = "trunk",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        rigid_path: RigidPath,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"],
        mode: Literal["flow"] = "flow",
    ) -> Output: ...

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        *args,
        mode: Literal["trunk", "flow"] = "flow",
        **kwargs,
    ):
        if mode == "trunk":
            return self.trunk_forward(*args, **kwargs)
        elif mode == "flow":
            return self.flow_forward(*args, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'trunk' or 'flow'.")


class LossFunction(nn.Module):
    class Config(BaseModel):
        t_normalize_clip: float = 0.9
        translation_loss_weight: float = 2.0
        rotation_loss_weight: float = 1.0
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

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(
        self,
        structure: StructureFeatures,
        sequence: SequenceFeatures,  # Need for clash / bond length loss
        rigid_path: RigidPath,
        output: StructureFlowModel.Output,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        t_clip = self.config.t_normalize_clip
        norm_scale = (1 - torch.min(rigid_path.t, torch.tensor(t_clip))) ** 2

        trans_vf_loss = losses.cal_trans_vf_loss(
            structure.rigid.get_trans(),
            output.rigid_pred.get_trans(),
            structure.res_mask,
        )
        trans_vf_loss *= du.ANG_TO_NM_SCALE**2
        trans_vf_loss /= norm_scale
        trans_vf_loss *= self.config.translation_loss_weight
        trans_vf_loss = torch.clamp(trans_vf_loss, max=5)  # TODO: test this

        rots_vf_loss = losses.cal_rots_vf_loss(
            rigid_path.rot_mats_t,
            structure.rigid.get_rots().get_rot_mats(),
            output.rigid_pred.get_rots().get_rot_mats(),
            structure.res_mask,
        )
        rots_vf_loss /= norm_scale
        rots_vf_loss *= self.config.rotation_loss_weight

        if self.config.all_atom_loss_weight == 0:
            all_atom_loss = torch.zeros_like(trans_vf_loss)
        else:
            all_atom_loss = losses.cal_all_atom_loss(
                structure.atom_pos,
                output.atom_pos_pred,
                structure.atom_mask,
                structure.res_mask,
            )
            all_atom_loss *= du.ANG_TO_NM_SCALE**2
            all_atom_loss /= norm_scale
            all_atom_loss *= rigid_path.t > self.config.all_atom_loss_t_filter
            all_atom_loss *= self.config.all_atom_loss_weight
            all_atom_loss *= self.config.aux_loss_weight

        if self.config.dist_mat_loss_weight == 0:
            dist_mat_loss = torch.zeros_like(trans_vf_loss)
        else:
            dist_mat_loss = losses.cal_dist_mat_loss(
                structure.atom_pos,
                output.atom_pos_pred,
                structure.atom_mask,
                structure.res_mask,
                self.config.dist_mat_threshold,
            )
            dist_mat_loss *= du.ANG_TO_NM_SCALE**2
            dist_mat_loss /= norm_scale
            dist_mat_loss *= rigid_path.t > self.config.dist_mat_loss_t_filter
            dist_mat_loss *= self.config.dist_mat_loss_weight
            dist_mat_loss *= self.config.aux_loss_weight

        loss_dict = {
            "trans_vf_loss": trans_vf_loss,
            "rots_vf_loss": rots_vf_loss,
            "all_atom_loss": all_atom_loss,
            "dist_mat_loss": dist_mat_loss,
        }
        loss_dict = {k: v.mean(0) for k, v in loss_dict.items()}
        total_loss = sum(loss_dict.values())
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        if torch.isnan(total_loss).any():
            raise ValueError("NaN in loss")
        return total_loss, loss_dict


@typecheck
@dataclass
class StructureFlowInferenceOutput:
    # Tensor of final predicted atom coordinate.
    atom_pos_pred: Float[torch.Tensor, "B L N 3"]

    # Predicted rigid tensor of each frame.
    rigid_pred: Rigid  # (B, L)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: Float[np.ndarray, "B T L N 3"]

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: Float[np.ndarray, "B T L N 3"]

    # Sequence features of the input batch.
    sequence: SequenceFeatures

    @torch.no_grad()
    def evaluate_quality(self, structure: StructureFeatures) -> dict[str, float]:
        max_lddt, min_rmsd = 0, float("inf")
        for i in range(structure.batch_size):
            lddt = metrics.cal_atom_lddt(
                self.atom_pos_pred[i],
                structure.atom_pos[i],
                structure.atom_mask[i],
            )
            if max_lddt < lddt:
                max_lddt = lddt

            rmsd = metrics.cal_aligned_rmsd(
                self.atom_pos_pred[i, :, chemical.PROT_ATOM_CENTER_IDX],
                structure.atom_pos[i, :, chemical.PROT_ATOM_CENTER_IDX],
                structure.res_mask[i],
            )
            if min_rmsd > rmsd:
                min_rmsd = rmsd

        return {
            "best_rmsd": min_rmsd,
            "best_lddt": max_lddt,
        }

    @typecheck
    def build_atom_mask(self) -> Bool[torch.Tensor, "B L N"]:
        """Build atom mask from the atom positions."""
        device = self.atom_pos_pred.device
        return chemical.RES_TYPE_ATOM_MASK.to(device)[self.sequence.res_type]


class StructureFlowClient(BaseClient):
    class DataConfig(BaseModel):
        data_pkl_path: str | None = None
        input_dir_path: str | None = None

    class ExperimentsConfig(BaseModel):
        comment: str = "default"
        num_batch: int = 1
        num_epoch: int = 1000
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        warmup_steps: int = 5e3
        decay_steps: int = 5e6
        self_condition: bool = True
        crop_length: int = 256
        compile: bool = False
        num_augment: int = 8
        augment: bool = True
        equal_timestep: bool = False
        eval_freq: int = 10
        eval_sample_num: int = 5
        eval_timesteps: int = 100
        eval_input_num: int = 50
        eval_crop_length: int = 256
        grad_clip_max_norm: float | None = None
        grad_accum_steps: int = 256
        loss: LossFunction.Config

    class Config(BaseModel):
        data: "StructureFlowClient.DataConfig"
        model: StructureFlowModel.Config
        experiment: "StructureFlowClient.ExperimentsConfig"

    def __init__(self, config: Config, name: str | None = None):
        super().__init__()
        self.config = config
        self.model = StructureFlowModel(config.model)
        if config.experiment.compile:
            self.model = torch.compile(self.model)
        self.model = self.setup_model(self.model)
        self.loss_fn = LossFunction(config.experiment.loss)

        optimizer = torch.optim.AdamW(self.model.parameters(), config.experiment.max_lr)
        self.setup(
            config=config,
            optimizer=optimizer,
            clip_max_norm=config.experiment.grad_clip_max_norm,
            accum_steps=config.experiment.grad_accum_steps,
            name=name,
        )

    def training_step(self, batch: Batch):
        # For now, we only support training with batch size 1.
        if batch.batch_size != 1:
            raise ValueError(f"Batch size must be 1, but got {batch.batch_size}.")

        pair, single = self.model(batch.sequence, batch.structure.res_mask, mode="trunk")
        num_aug = self.config.experiment.num_augment
        pair = pair.expand((num_aug,) + pair.shape[1:])
        single = single.expand((num_aug,) + single.shape[1:])
        structure = batch.structure.duplicate(num_aug)

        if self.config.experiment.augment:
            structure = structure.augmentation()

        rigid_path = sample_rigid_path(
            rigid=structure.rigid,
            res_mask=structure.res_mask,
            equal_timestep=self.config.experiment.equal_timestep,
        )

        if self.config.experiment.self_condition and random.random() > 0.5:
            with torch.no_grad():
                output_sc = self.model(rigid_path, pair, single, mode="flow")
                rigid_path.rigid_sc = output_sc.rigid_pred

        output = self.model(rigid_path, pair, single, mode="flow")
        total_loss, avg_loss = self.loss_fn(
            structure=structure,
            sequence=batch.sequence,
            rigid_path=rigid_path,
            output=output,
        )
        self.backward(total_loss)

        self.log_metrics(
            {"train/total_loss": total_loss.item()},
            on_step=True,
            on_epoch=True,
        )
        self.log_metrics(
            {"train/" + k: v for k, v in avg_loss.items()},
            on_step=True,
            on_epoch=True,
        )

    def validation_step(self, batch: Batch):
        # NOTE: When doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        if batch.batch_size != 1:
            raise ValueError(f"Batch size must be 1, but got {batch.batch_size}.")

        output = self.inference(
            sequence=batch.sequence,
            res_mask=batch.structure.res_mask,
            timesteps=self.config.experiment.eval_timesteps,
            num_flow_sample=self.config.experiment.eval_sample_num,
        )

        structure = batch.structure.duplicate(self.config.experiment.eval_sample_num)
        valid_dict = output.evaluate_quality(structure)
        self.log_metrics({"valid/" + k: v for k, v in valid_dict.items()}, on_epoch=True)

    @torch.no_grad()
    def inference(
        self,
        sequence: SequenceFeatures,
        res_mask: torch.Tensor | None = None,
        timesteps: int = 100,  # TODO: rename to num_timesteps
        num_flow_sample: int = 1,
    ) -> StructureFlowInferenceOutput:
        self.model.eval()
        sequence = sequence.to(device=self.device)
        if res_mask is None:
            res_mask = torch.ones_like(sequence.res_type, dtype=torch.bool)
        res_mask = res_mask.to(self.device)

        # Embed single and pair representation
        pair, single = self.model(sequence, res_mask, mode="trunk")
        pair = pair.expand((num_flow_sample,) + pair.shape[1:])
        single = single.expand((num_flow_sample,) + single.shape[1:])
        res_mask = res_mask.repeat(num_flow_sample, 1)
        sequence = sequence.duplicate(num_flow_sample)

        # Set-up noise
        trans_0 = centered_gaussian(res_mask) * du.NM_TO_ANG_SCALE
        rot_mats_0 = uniform_so3(res_mask)
        rigid_path = RigidPath(
            t=torch.zeros(sequence.batch_size, device=self.device),
            trans_0=trans_0,
            trans_t=trans_0,
            rot_mats_0=rot_mats_0,
            rot_mats_t=rot_mats_0,
            res_mask=res_mask,
        )

        # Sample with Euler-Maruyama method
        model_traj = []
        inter_traj = []
        t1_list = torch.linspace(0, 1.0, timesteps)
        t2_list = torch.cat([t1_list[1:], t1_list[-1:]])
        for t1, t2 in zip(t1_list, t2_list):
            rigid_path.t[:] = t1
            output = self.model(rigid_path, pair, single, mode="flow")

            d_t = t2 - t1
            if not d_t == 0:
                trans_vf = (output.rigid_pred.get_trans() - rigid_path.trans_t) / (
                    1 - t1
                )
                rigid_path.trans_t = rigid_path.trans_t + trans_vf * d_t
                rigid_path.rot_mats_t = so3_utils.geodesic_t(
                    10 * d_t,
                    output.rigid_pred.get_rots().get_rot_mats(),
                    rigid_path.rot_mats_t,
                )
                local_atom_pos = global_to_local(output.rigid_pred, output.atom_pos_pred)
                atom_pos_pred_t = local_to_global(rigid_path.rigid_t, local_atom_pos)
            else:
                atom_pos_pred_t = output.atom_pos_pred

            if self.config.experiment.self_condition:
                rigid_path.rigid_sc = output.rigid_pred

            model_traj.append(du.to_numpy(output.atom_pos_pred))
            inter_traj.append(du.to_numpy(atom_pos_pred_t))

        return StructureFlowInferenceOutput(
            atom_pos_pred=output.atom_pos_pred,
            rigid_pred=output.rigid_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            sequence=sequence,
        )

    def sample(
        self,
        sequence: str,
        seq_idx: np.ndarray | torch.Tensor | None = None,
        num_sample: int = 1,
        timesteps: int = 100,
    ) -> StructureFlowInferenceOutput:
        res_name = np.array(list(sequence))
        L = len(res_name)

        if seq_idx is None:
            seq_idx = torch.arange(L)
        elif isinstance(seq_idx, np.ndarray):
            seq_idx = torch.from_numpy(seq_idx)
        if seq_idx.shape != (L,):
            raise ValueError(f"Invalid seq_idx shape: {seq_idx.shape}")

        mapping = lambda x: chemical.RES_NAMES.index(chemical.RESCHAR_TO_RESNAME[x])
        res_type = np.vectorize(mapping)(res_name)
        res_type = torch.from_numpy(res_type)

        sequence = SequenceFeatures.from_sample(
            res_type=res_type,
            seq_idx=seq_idx,
        )
        return self.inference(
            sequence=sequence,
            timesteps=timesteps,
            num_flow_sample=num_sample,
        )
