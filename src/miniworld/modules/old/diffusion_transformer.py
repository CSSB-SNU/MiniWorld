import numbers
from collections import abc
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float
from team_gm import typecheck
from torch import nn

from miniworld.data.features.batch_edge_backprop import NoisyBatch

# from .attentions import AugmentedAttention
from .configs import CommonConfig, DiffusionConfig
from .primitives import (
    ConditionedMoETransition,
    ConditionedTransition,
    LayerNorm,
    Linear,
)


def mask_mean(
    mask: Bool[torch.Tensor, "..."],
    value: Float[torch.Tensor, "..."],
    axis: numbers.Integral | None = None,
    keepdims: bool = False,
    eps: float = 1e-9,
) -> Float[torch.Tensor, "..."]:
    """Compute the mean of `value` over `axis`, masked by `mask`."""
    is_torch = isinstance(value, torch.Tensor)

    mask_shape = mask.shape
    value_shape = value.shape

    if len(mask_shape) != len(value_shape):
        msg = f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}"
        raise ValueError(msg)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    if not isinstance(axis, abc.Iterable):
        msg = f"axis must be an integer or an iterable of integers, got {type(axis)}"
        raise TypeError(msg)

    broadcast_factor = 1.0

    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        elif mask_size != value_size:
            msg = f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}"
            raise ValueError(msg)

        if is_torch:
            sum_fn = lambda x, ax, kd: torch.sum(x, dim=ax, keepdim=kd)
        else:
            sum_fn = lambda x, ax, kd: np.sum(x, axis=ax, keepdims=kd)

        numerator = sum_fn(value * mask, axis, keepdims)
        denom = sum_fn(mask, axis, keepdims) * broadcast_factor

        if is_torch:
            eps_t = torch.tensor(eps, dtype=denom.dtype, device=denom.device)
            safe_denom = torch.maximum(denom, eps_t)
        else:
            safe_denom = np.maximum(denom, eps)

    return numerator / safe_denom


class DiffusionTransformerBlock(nn.Module):
    """A single block of the Diffusion Transformer model."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
        level: Literal["token", "atom"] = "token",
    ) -> None:
        super().__init__()
        if level not in ["token", "atom"]:
            msg = f"Invalid level: {level}. Choose 'token' or 'atom'."
            raise ValueError(msg)
        implementation = diffusion_config.implementation

        if level == "token":
            d_single_rep = common_config.d_token_single_diffusion
            d_single_cond = common_config.d_token_single
            d_pair_cond = common_config.d_token_pair
            n_head = diffusion_config.n_head_token
            experts = diffusion_config.token_single_moe_experts
            topk = diffusion_config.token_single_moe_topk
        else:  # level == "atom"
            d_single_rep = common_config.d_atom_single
            d_single_cond = common_config.d_atom_single
            d_pair_cond = common_config.d_atom_pair
            n_head = diffusion_config.n_head_atom
            experts = diffusion_config.atom_single_moe_experts
            topk = diffusion_config.atom_single_moe_topk
        # if common_config.use_checkpoint and level == "token":
        if common_config.use_checkpoint:
            self.use_checkpoint = True
        else:
            self.use_checkpoint = common_config.use_checkpoint

        # self.atom_attention_pair_bias = AugmentedAttention(
        #     d_single_rep=d_single_rep,
        #     d_single_cond=d_single_cond,
        #     d_pair_cond=d_pair_cond,
        #     n_head=n_head,
        #     level=level,
        #     use_beta=diffusion_config.use_beta,
        #     implementation=implementation,
        #     to_bias_init=common_config.to_bias_init,
        # )
        # if experts > 1:
        #     self.conditioned_transition = ConditionedMoETransition(
        #         d_rep=d_single_rep,
        #         d_cond=d_single_cond,
        #         experts=experts,
        #         topk=topk,
        #         implementation=implementation,
        #         use_checkpoint=not self.use_checkpoint,
        #     )
        # else:
        #     self.conditioned_transition = ConditionedTransition(
        #         d_rep=d_single_rep,
        #         d_cond=d_single_cond,
        #         implementation=implementation,
        #         use_checkpoint=not self.use_checkpoint,
        #     )

    @typecheck
    def _forward(
        self,
        noisy_batch: NoisyBatch,
        # atom_single_rep: Float[torch.Tensor, "B L d_atom_single_rep"],
        # atom_single_cond: Float[torch.Tensor, "B L d_atom_single_cond"] | None = None,
        atom_pair: Float[torch.Tensor, "B L L d_atom_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> torch.Tensor:
        # atom_single_rep = atom_single_rep + self.atom_attention_pair_bias(
        #     noisy_batch,
        #     atom_single_rep,
        #     atom_single_cond,
        #     atom_pair,
        #     mask,
        # )

        # return atom_single_rep + self.conditioned_transition(
        #     atom_single_rep,
        #     atom_single_cond,
        # )

    @typecheck
    def forward(
        self,
        noisy_batch: NoisyBatch,
        # atom_single_rep: Float[torch.Tensor, "B L d_atom_single_rep"],
        # atom_single_cond: Float[torch.Tensor, "B L d_atom_single_cond"] | None = None,
        atom_pair: Float[torch.Tensor, "B L L d_atom_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # if self.use_checkpoint:
        #     atom_single_rep = torch.utils.checkpoint.checkpoint(
        #         self._forward,
        #         noisy_batch,
        #         atom_single_rep,
        #         atom_single_cond,
        #         atom_pair,
        #         mask,
        #         use_reentrant=False,
        #     )
        # else:
        #     atom_single_rep = self._forward(
        #         noisy_batch,
        #         atom_single_rep,
        #         atom_single_cond,
        #         atom_pair,
        #         mask,
        #     )
        # return atom_single_rep


class DiffusionTransformer(nn.Module):
    """A stack of Diffusion Transformer blocks."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
        level: Literal["token", "atom"] = "token",
    ) -> None:
        super().__init__()
        n_block = (
            diffusion_config.n_block_token
            if level == "token"
            else diffusion_config.n_block_atom
        )
        # self.blocks = nn.ModuleList(
        #     [
        #         DiffusionTransformerBlock(
        #             common_config=common_config,
        #             diffusion_config=diffusion_config,
        #             level=level,
        #         )
        #         for _ in range(n_block)
        #     ],
        # )

    @typecheck
    def forward(
        self,
        noisy_batch: NoisyBatch,
        atom_pair: Float[torch.Tensor, "B L L d_atom_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # for block in self.blocks:
        #     atom_single_rep = block(
        #         noisy_batch,
        #         atom_single_rep,
        #         atom_single_cond,
        #         atom_pair,
        #         mask,
        #     )
        # return atom_single_rep


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.common_config = common_config
        self.diffusion_config = diffusion_config
        d_atom_single = common_config.d_atom_single
        d_atom_pair = common_config.d_atom_pair
        self.d_token_single = common_config.d_token_single_diffusion
        d_token_pair = common_config.d_token_pair

        self.use_beta = diffusion_config.use_beta
        self.use_checkpoint = common_config.use_checkpoint

        self.to_atom_single_cond = Linear(6, common_config.d_atom_single, bias=False)

        self.to_atom_pair = Linear(5, common_config.d_atom_pair, bias=False)

        self.token_single_to_atom_single_cond = nn.Sequential(
            LayerNorm(
                common_config.d_token_single,
                implementation=diffusion_config.implementation,
            ),
            Linear(
                common_config.d_token_single,
                d_atom_single,
                bias=False,
                init="zero",
            ),
        )
        self.token_pair_to_atom_pair = nn.Sequential(
            LayerNorm(d_token_pair, implementation=diffusion_config.implementation),
            Linear(d_token_pair, d_atom_pair, bias=False, init="zero"),
        )
        self.noisy_to_atom_single_rep = Linear(
            3, d_atom_single, bias=True,
        )  # bias set to true for missing atoms

        self.atom_single_to_pair_left = nn.Sequential(
            nn.ReLU(),
            Linear(d_atom_single, d_atom_pair, bias=False),
        )

        self.atom_single_to_pair_right = nn.Sequential(
            nn.ReLU(),
            Linear(d_atom_single, d_atom_pair, bias=False),
        )

        self.mlp_atom_pair = nn.Sequential(
            Linear(d_atom_pair, d_atom_pair, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_atom_pair, d_atom_pair, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_atom_pair, d_atom_pair, init="zero", bias=False),
        )

        self.atom_transformer = DiffusionTransformer(
            common_config=common_config,
            diffusion_config=diffusion_config,
            level="atom",
        )

        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_atom_single, self.d_token_single, bias=False),
            nn.ReLU(),
        )

    @typecheck
    def _add_trunk_info(
        self,
        noisy_batch: NoisyBatch,
        atom_single_cond: Float[torch.Tensor, "B L d_atom_single_cond"],
        atom_pair: Float[torch.Tensor, "B L L d_atom_pair"],
        token_single_cond: Float[torch.Tensor, "B L_token d_token_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L d_atom_single_rep"],
        Float[torch.Tensor, "B L_token d_token_single_cond"],
        Float[torch.Tensor, "B L_token L_token d_token_pair_cond"],
    ]:
        device = noisy_batch.device
        B, L_atom = noisy_batch.shape
        A = noisy_batch.x_t.shape[0]  # Number of augmentations

        _to_add_single = self.token_single_to_atom_single_cond(token_single_cond)
        _to_add_pair = self.token_pair_to_atom_pair(token_pair_cond)

        atom_to_residue_idx_map = noisy_batch.scheme.atom_to_residue_idx_map  # (B, L)
        batch_1D_idx = torch.arange(B, device=device).view(B, 1).expand(-1, L_atom)
        atom_single_cond = (
            atom_single_cond + _to_add_single[batch_1D_idx, atom_to_residue_idx_map]
        )
        if not self.use_beta:
            batch_2D_idx = (
                torch.arange(B, device=device).view(B, 1, 1).expand(-1, L_atom, L_atom)
            )
            atom_pair = (
                atom_pair
                + _to_add_pair[
                    batch_2D_idx,
                    atom_to_residue_idx_map,
                    atom_to_residue_idx_map,
                ]
            )
            # augmentation
            atom_single_rep = atom_single_cond.unsqueeze(0)
            to_add = self.noisy_to_atom_single_rep(
                noisy_batch.x_t.to(torch.float32),
            )  # (A, B, L_atom, d_atom_single)
            to_add = to_add * noisy_batch.x_mask.unsqueeze(-1)
            atom_single_rep = atom_single_rep + to_add  # (A, B, L_atom, d_atom_single)

            _left = self.atom_single_to_pair_left(atom_single_cond)
            _right = self.atom_single_to_pair_right(atom_single_cond)
            atom_single_cond = atom_single_cond.unsqueeze(0).expand(A, -1, -1, -1)

        else:
            msg = "Beta version does not support trunk information addition yet."
            raise NotImplementedError(msg)

        atom_pair = atom_pair + _left[..., None, :] + _right[..., None, :, :]
        atom_pair = atom_pair + self.mlp_atom_pair(atom_pair)

        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def _get_input_feature(
        self,
        noisy_batch: NoisyBatch,
    ) -> tuple[
        Float[torch.Tensor, "B L d_atom_single_cond"],
        Float[torch.Tensor, "B L L d_atom_pair"],
    ]:
        """Get input feature for atom single and pair embedding.

        Parameters
        ----------
        noisy_batch: NoisyBatch
            Batch of data.

        Returns
        -------
        atom_single_cond: FloatTensor, (B, L, d_atom_single)
            Atom single condition representation.
        atom_pair: FloatTensor, (B, L, L, d_atom_pair)
            Atom pair representation.

        """
        ref_infos = torch.cat(
            [
                noisy_batch.reference.pos,
                noisy_batch.reference.mask.unsqueeze(-1),
                noisy_batch.reference.element.unsqueeze(-1),
                torch.arcsinh(noisy_batch.reference.charge).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_single_cond = self.to_atom_single_cond(ref_infos)
        atom_single_cond = atom_single_cond * noisy_batch.reference.mask.unsqueeze(-1)
        if not self.use_beta:
            d_lm = (
                noisy_batch.reference.pos[:, :, None]
                - noisy_batch.reference.pos[:, None, :]
            )
            v_lm = (
                noisy_batch.reference.space_uid[:, :, None]
                == noisy_batch.reference.space_uid[:, None, :]
            )  # (B, L, L)

        else:
            msg = "Beta version does not support input feature generation yet."
            raise NotImplementedError(msg)
        v_lm = v_lm[..., None]
        arctan_d_lm = 1 / (1 + d_lm.norm(dim=-1) ** 2)
        arctan_d_lm = arctan_d_lm.unsqueeze(-1)
        d_lm = torch.cat([d_lm, arctan_d_lm, v_lm], dim=-1)
        atom_pair = d_lm * v_lm
        atom_pair = self.to_atom_pair(atom_pair)

        return atom_single_cond, atom_pair

    @typecheck
    def _scatter_atom_to_token(
        self,
        noisy_batch: NoisyBatch,
        atom_single_rep: Float[torch.Tensor, "B L d_atom_single_rep"],
    ) -> Float[torch.Tensor, "B L_token d_token_single"]:
        """Scatter atom single representation to token single representation."""
        atom_mask = noisy_batch.structure.atom_mask  # (B, L_atom)
        atom_single_rep = torch.where(
            atom_mask.unsqueeze(-1),
            atom_single_rep,
            torch.zeros_like(atom_single_rep),
        )
        to_add_token_single_rep = self.atom_single_rep_to_token_single(atom_single_rep)

        # Convert back to token-atom layout and aggregate to tokens
        if not self.use_beta:
            atom_to_residue_idx_map = (
                noisy_batch.scheme.atom_to_residue_idx_map
            )  # (B, L_atom)
            atom_mask = noisy_batch.structure.atom_mask  # (B, L_atom)

            B = noisy_batch.shape[0]
            L_token = noisy_batch.residue_length
            count = torch.zeros(
                (B, L_token),
                device=noisy_batch.device,
                dtype=noisy_batch.dtype,
            )
            count.scatter_add_(
                1,
                atom_to_residue_idx_map,
                torch.ones_like(atom_to_residue_idx_map, dtype=noisy_batch.dtype)
                * atom_mask,
            )
            token_single_rep = torch.zeros(
                (
                    noisy_batch.shape[0],
                    noisy_batch.residue_length,
                    self.d_token_single,
                ),
                device=noisy_batch.device,
            )  # (B, L_atom, d_token_single)
            to_add_token_single_rep = self.atom_single_rep_to_token_single(
                atom_single_rep,
            )  # (B, L_atom, d_single) or (A, B, L_atom, d_single)
            # apply augmentation
            A = atom_single_rep.shape[0]  # Number of augmentations
            token_single_rep = token_single_rep.unsqueeze(1).expand(
                A,
                -1,
                -1,
                -1,
            )  # (A, B, L_atom, d_token_single)
            atom_to_residue_idx_map = atom_to_residue_idx_map.unsqueeze(1).unsqueeze(-1)
            atom_to_residue_idx_map = atom_to_residue_idx_map.expand(
                A,
                -1,
                -1,
                to_add_token_single_rep.shape[-1],
            )
            atom_mask = atom_mask.unsqueeze(1).unsqueeze(-1)
            to_add_token_single_rep = to_add_token_single_rep * atom_mask
            token_single_rep = token_single_rep.scatter_add(
                2,
                atom_to_residue_idx_map,
                to_add_token_single_rep,
            )
            token_single_rep = token_single_rep / count.unsqueeze(1).unsqueeze(
                -1,
            ).clamp(
                min=1.0,
            )
        else:
            msg = "Beta version does not support scatter atom to token yet."
            raise NotImplementedError(msg)

        return token_single_rep

    @typecheck
    def _before_atom_transformer(
        self,
        noisy_batch: NoisyBatch,
        token_single_cond: Float[torch.Tensor, "B L_token d_token_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L d_atom_single_rep"],
        Float[torch.Tensor, "B L d_atom_single_cond"],
        Float[torch.Tensor, "B L L d_atom_pair_cond"],
    ]:
        atom_single_cond, atom_pair = self._get_input_feature(noisy_batch)
        atom_single_rep, atom_single_cond, atom_pair = self._add_trunk_info(
            noisy_batch,
            atom_single_cond,
            atom_pair,
            token_single_cond,
            token_pair_cond,
        )
        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_cond: Float[torch.Tensor, "B L_token d_token_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_token_single_rep"],
        Float[torch.Tensor, "B L d_atom_single_rep"],
        Float[torch.Tensor, "B L d_atom_single_cond"],
        Float[torch.Tensor, "B L L d_atom_pair"],
    ]:
        """Forward pass."""
        if self.use_checkpoint:
            atom_single_rep, atom_single_cond, atom_pair = (
                torch.utils.checkpoint.checkpoint(
                    self._before_atom_transformer,
                    noisy_batch,
                    token_single_cond,
                    token_pair_cond,
                    use_reentrant=False,
                )
            )
        else:
            atom_single_rep, atom_single_cond, atom_pair = (
                self._before_atom_transformer(
                    noisy_batch,
                    token_single_cond,
                    token_pair_cond,
                )
            )
        atom_single_rep = self.atom_transformer(
            noisy_batch,
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            noisy_batch.structure.atom_mask,
        )

        if self.use_checkpoint:
            token_single_rep = torch.utils.checkpoint.checkpoint(
                self._scatter_atom_to_token,
                noisy_batch,
                atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(noisy_batch, atom_single_rep)
        return token_single_rep, atom_single_rep, atom_single_cond, atom_pair


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.common_config = common_config
        self.diffusion_config = diffusion_config
        d_atom_single = common_config.d_atom_single
        d_token_single = common_config.d_token_single_diffusion

        self.add_token_info = Linear(d_token_single, d_atom_single, bias=False)

        self.atom_transformer = DiffusionTransformer(
            common_config=common_config,
            diffusion_config=diffusion_config,
            level="atom",
        )

        self.final_denoising = nn.Sequential(
            LayerNorm(d_atom_single, implementation=diffusion_config.implementation),
            Linear(
                d_atom_single,
                3,
                bias=False,
                init="zero",
            ),
        )

    @typecheck
    def forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_rep: Float[torch.Tensor, "A B L_token d_token_single"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_atom_single_rep"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_atom_single_cond"],
        atom_pair: Float[torch.Tensor, "A B L_atom L_atom d_atom_pair"],
    ) -> Float[torch.Tensor, "A B L_atom 3"]:
        """Forward pass."""
        A, B, L_atom = atom_single_rep.shape[:3]
        device = noisy_batch.device
        batch_1D_idx = (
            torch.arange(B, device=device).view(1, B, 1).expand(A, -1, L_atom)
        )
        aug_1D_idx = torch.arange(A, device=device).view(A, 1, 1).expand(-1, B, L_atom)
        atom_to_residue_idx_map = noisy_batch.scheme.atom_to_residue_idx_map  # (B, L)
        atom_to_residue_idx_map = atom_to_residue_idx_map.unsqueeze(0).expand(A, -1, -1)

        _to_add_single = self.add_token_info(token_single_rep)
        atom_single_rep = (
            atom_single_rep
            + _to_add_single[aug_1D_idx, batch_1D_idx, atom_to_residue_idx_map]
        )

        atom_single_rep = self.atom_transformer(
            noisy_batch,
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            mask=noisy_batch.structure.atom_mask,
        )
        return self.final_denoising(atom_single_rep)
