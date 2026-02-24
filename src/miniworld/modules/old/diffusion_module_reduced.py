import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer
from team_gm.modules.layers import Transition
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn
from torch.utils.checkpoint import checkpoint

from miniworld.configs import SharedConfig
from miniworld.data.features.batch_edge_backprop import (
    ReferenceFeatures,
    SchemeFeatures,
    StructureFeatures,
)
from miniworld.modules.embeddings import RelativePositionEmbedding, fourier_embedding


@typecheck
@torch.no_grad
def init_atom_features(
    reference: ReferenceFeatures,
) -> tuple[
    Float[torch.Tensor, "B L_atom d_single_atom_cond"],
    Float[torch.Tensor, "B L_atom L_atom d_pair_atom"],
]:
    """Get input feature for atom single and pair embedding."""
    atom_single_init = torch.cat(
        [
            reference.pos,
            reference.mask.unsqueeze(-1),
            reference.element.unsqueeze(-1),
            torch.arcsinh(reference.charge).unsqueeze(-1),
        ],
        dim=-1,
    )
    atom_single_init = atom_single_init * reference.mask.unsqueeze(-1)

    d_lm = (
        reference.pos[:, :, None]
        - reference.pos[:, None, :]
    )
    v_lm = (
        reference.space_uid[:, :, None]
        == reference.space_uid[:, None, :]
    )

    v_lm = v_lm[..., None].to(d_lm.dtype)
    arctan_d_lm = 1 / (1 + d_lm.norm(dim=-1) ** 2)
    arctan_d_lm = arctan_d_lm.unsqueeze(-1)
    d_lm = torch.cat([d_lm, arctan_d_lm, v_lm], dim=-1)
    atom_pair_init = d_lm * v_lm

    return atom_single_init, atom_pair_init


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning module."""

    class Config(BaseModel):
        """Configuration for DiffusionConditioning."""

        n_expand: int = 2
        n_blocks: int = 2

    def __init__(
        self,
        shared_config: SharedConfig,
        dit_cond_config: Config,
    ) -> None:
        super().__init__()
        d_pair = shared_config.d_pair
        d_time = shared_config.d_time
        self.relative_position_embedder = RelativePositionEmbedding(
            d_hidden=d_pair,
            r_max=shared_config.r_max,
            s_max=shared_config.s_max,
        )

        self.linear_token_pair = nn.Sequential(
            LayerNorm(
                2 * d_pair,
            ),
            Linear(2 * d_pair, d_pair, bias=False),
        )
        self.pair_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_pair,
                    n=dit_cond_config.n_expand,
                )
                for _ in range(dit_cond_config.n_blocks)
            ],
        )
        self.to_atom_single_cond = Linear(6, shared_config.d_single_atom, bias=False)
        self.linear_atom_single_init = nn.Sequential(
            LayerNorm(
                shared_config.d_single_atom,
            ),
            Linear(
                shared_config.d_single_atom,
                shared_config.d_single_atom,
                bias=False,
            ),
        )
        self.add_time_embedding = nn.Sequential(
            LayerNorm(
                d_time,
            ),
            Linear(d_time, shared_config.d_single_atom, bias=False),
        )
        self.single_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=shared_config.d_single_atom,
                    n=dit_cond_config.n_expand,
                )
                for _ in range(dit_cond_config.n_blocks)
            ],
        )

    @typecheck
    def forward(
        self,
        scheme: SchemeFeatures,
        t_emb: Float[torch.Tensor, "A B"],
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_atom d_single_atom"],
        Float[torch.Tensor, "B L_token L_token d_pair"],
    ]:
        """Forward pass of the diffusion conditioning module."""
        rel_emb = self.relative_position_embedder(
            asym_id = scheme.residue_asym_id,
            residue_idx = scheme.residue_idx,
            entity_id = scheme.residue_entity_id,
            sym_id = scheme.residue_sym_id,
        )
        token_pair = torch.cat([token_pair_trunk, rel_emb],dim=-1)
        token_pair = self.linear_token_pair(token_pair)

        for transition in self.pair_transitions:
            token_pair = token_pair + transition(token_pair)

        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_single_cond = self.linear_atom_single_init(atom_single_cond)
        time_embedding = fourier_embedding(t_emb)
        time_embedding = time_embedding.squeeze(-2)
        atom_single_cond = atom_single_cond + self.add_time_embedding(time_embedding)

        for transition in self.single_transitions:
            atom_single_cond = atom_single_cond + transition(atom_single_cond)

        return atom_single_cond, token_pair


class Token2AtomMapper(nn.Module):
    """Module to map token features to atom features."""

    def __init__(
        self,
        shared_config: SharedConfig,
    ) -> None:
        super().__init__()
        self.use_checkpoint = shared_config.use_checkpoint
        self.to_atom_single_cond = Linear(6, shared_config.d_single_atom, bias=False)
        self.to_atom_pair = Linear(5, shared_config.d_pair_atom, bias=False)

        self.token_single_input_to_atom_single_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single_token_input,
            ),
            Linear(
                shared_config.d_single_token_input,
                shared_config.d_single_atom,
                bias=False,
                init="zero",
            ),
        )

        self.token_single_cond_to_atom_single_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single,
            ),
            Linear(
                shared_config.d_single,
                shared_config.d_single_atom,
                bias=False,
                init="zero",
            ),
        )
        self.token_pair_to_atom_pair = nn.Sequential(
            LayerNorm(shared_config.d_pair),
            Linear(shared_config.d_pair, shared_config.d_pair_atom, bias=False, init="zero"),
        )
        self.noisy_to_atom_single_rep = Linear(
            3, shared_config.d_single_atom, bias=True,
        )  # bias set to true for missing atoms

        self.atom_single_to_pair_left = nn.Sequential(
            nn.ReLU(),
            Linear(shared_config.d_single_atom, shared_config.d_pair_atom, bias=False),
        )

        self.atom_single_to_pair_right = nn.Sequential(
            nn.ReLU(),
            Linear(shared_config.d_single_atom, shared_config.d_pair_atom, bias=False),
        )

        self.mlp_atom_pair = nn.Sequential(
            Linear(shared_config.d_pair_atom, shared_config.d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(shared_config.d_pair_atom, shared_config.d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(shared_config.d_pair_atom, shared_config.d_pair_atom, init="zero", bias=False),
        )


    @typecheck
    def _forward(
        self,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        atom_single_init: Float[torch.Tensor, "B L_atom 6"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_single_atom"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
        atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom_cond"],
    ]:
        """Forward pass of the token to atom mapper."""
        atom_single_init = self.to_atom_single_cond(atom_single_init)

        atom_pair = self.to_atom_pair(atom_pair_init)

        device = x_t.device
        num_aug, batch_size, atom_length = x_t.shape[:3]

        _to_add_single_input = self.token_single_input_to_atom_single_cond(token_single_input)
        _to_add_single_cond = self.token_single_cond_to_atom_single_cond(token_single_cond)
        _to_add_pair = self.token_pair_to_atom_pair(token_pair_cond)

        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(batch_size, 1).expand(-1, atom_length)
        _to_add_atom_single = (
            _to_add_single_input[batch_1d_idx, atom_to_residue_idx_map]
            + _to_add_single_cond[batch_1d_idx, atom_to_residue_idx_map]
        )
        atom_single_init = atom_single_init + _to_add_atom_single
        atom_single_cond = atom_single_cond + _to_add_atom_single

        batch_2d_idx = torch.arange(batch_size, device=device)
        batch_2d_idx = batch_2d_idx.view(batch_size, 1, 1).expand(
            -1, atom_length, atom_length,
        )
        atom_pair = (
            atom_pair
            + _to_add_pair[
                batch_2d_idx,
                atom_to_residue_idx_map,
                atom_to_residue_idx_map,
            ]
        )
        # augmentation
        atom_single_rep = atom_single_cond
        to_add = self.noisy_to_atom_single_rep(
            x_t.to(torch.float32),
        )
        to_add = to_add * x_mask.unsqueeze(-1)
        atom_single_rep = atom_single_rep + to_add
        _left = self.atom_single_to_pair_left(atom_single_init)
        _right = self.atom_single_to_pair_right(atom_single_init)

        atom_pair = atom_pair + _left[..., None, :] + _right[..., None, :, :]
        atom_pair = atom_pair + self.mlp_atom_pair(atom_pair)
        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def forward(
        self,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        atom_single_init: Float[torch.Tensor, "B L_atom 6"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_single_atom"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
        atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom_cond"],
    ]:
        """Forward pass with optional checkpointing."""
        if self.use_checkpoint:
            return checkpoint(
                self._forward,
                x_t,
                x_mask,
                atom_single_init,
                atom_single_cond,
                atom_pair_init,
                atom_to_residue_idx_map,
                token_single_input,
                token_single_cond,
                token_pair_cond,
            ) # pyright: ignore[reportReturnType]
        return self._forward(
            x_t,
            x_mask,
            atom_single_init,
            atom_single_cond,
            atom_pair_init,
            atom_to_residue_idx_map,
            token_single_input,
            token_single_cond,
            token_pair_cond,
            )


class DiffusionModule(nn.Module):
    """Diffusion module for processing input features."""

    def __init__(
        self,
        shared_config: SharedConfig,
        atom_dit_config: DiffusionTransformer.Config,
        dit_cond_config: DiffusionConditioning.Config,
    ) -> None:
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(
            shared_config=shared_config,
            dit_cond_config=dit_cond_config,
        )

        self.token2atom_mapper = Token2AtomMapper(
            shared_config=shared_config,
        )

        self.atom_transformer = DiffusionTransformer(config=atom_dit_config)

        self.final_denoising = nn.Sequential(
            LayerNorm(shared_config.d_single_atom),
            Linear(
                shared_config.d_single_atom,
                3,
                bias=False,
                init="zero",
            ),
        )

    @typecheck
    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module."""
        atom_single_init, atom_pair_init = init_atom_features(reference)
        atom_single_cond, token_pair_cond = self.diffusion_conditioning(
            scheme,
            t_emb,
            atom_single_init,
            token_pair_trunk,
        )

        atom_single_rep, atom_single_cond, atom_pair = self.token2atom_mapper(
            x_t,
            x_mask,
            atom_single_init,
            atom_single_cond,
            atom_pair_init,
            scheme.atom_to_residue_idx_map,
            token_single_input,
            token_single_trunk,
            token_pair_cond,
        )

        atom_single_rep = self.atom_transformer(
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            structure.atom_mask,
        )

        return self.final_denoising(atom_single_rep)

