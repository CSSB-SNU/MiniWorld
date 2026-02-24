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

    d_lm = reference.pos[:, :, None] - reference.pos[:, None, :]
    v_lm = reference.space_uid[:, :, None] == reference.space_uid[:, None, :]

    v_lm = v_lm[..., None].to(d_lm.dtype)
    arctan_d_lm = 1 / (1 + d_lm.norm(dim=-1) ** 2)
    arctan_d_lm = arctan_d_lm.unsqueeze(-1)
    d_lm = torch.cat([d_lm, arctan_d_lm, v_lm], dim=-1)
    atom_pair_init = d_lm * v_lm

    return atom_single_init, atom_pair_init


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
    ) -> None:
        super().__init__()
        self.shared_config = shared_config
        self.diffusion_config = diffusion_config
        d_single_atom = shared_config.d_single_atom
        d_pair_atom = shared_config.d_pair_atom
        self.d_single_token = shared_config.d_single_token
        d_pair = shared_config.d_pair

        self.use_checkpoint = shared_config.use_checkpoint

        self.to_atom_single_cond = Linear(6, shared_config.d_single_atom, bias=False)

        self.to_atom_pair = Linear(5, shared_config.d_pair_atom, bias=False)

        self.token_single_to_atom_single_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single,
            ),
            Linear(
                shared_config.d_single,
                d_single_atom,
                bias=False,
                init="zero",
            ),
        )
        self.token_pair_to_atom_pair = nn.Sequential(
            LayerNorm(d_pair),
            Linear(d_pair, d_pair_atom, bias=False, init="zero"),
        )
        self.noisy_to_atom_single_rep = Linear(
            3,
            d_single_atom,
            bias=True,
        )  # bias set to true for missing atoms

        self.atom_single_to_pair_left = nn.Sequential(
            nn.ReLU(),
            Linear(d_single_atom, d_pair_atom, bias=False),
        )

        self.atom_single_to_pair_right = nn.Sequential(
            nn.ReLU(),
            Linear(d_single_atom, d_pair_atom, bias=False),
        )

        self.mlp_atom_pair = nn.Sequential(
            Linear(d_pair_atom, d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_pair_atom, d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_pair_atom, d_pair_atom, init="zero", bias=False),
        )

        self.atom_transformer = DiffusionTransformer(config=diffusion_config)

        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_single_atom, self.d_single_token, bias=False),
            nn.ReLU(),
        )

    @typecheck
    def _before_atom_transformer(
        self,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom_init"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom_cond"],
    ]:
        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_pair = self.to_atom_pair(atom_pair_init)

        device = x_t.device
        num_aug, batch_size, atom_length = x_t.shape[:3]

        _to_add_single = self.token_single_to_atom_single_cond(token_single_cond)
        _to_add_pair = self.token_pair_to_atom_pair(token_pair_cond)

        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(batch_size, 1).expand(-1, atom_length)
        atom_single_cond = (
            atom_single_cond + _to_add_single[batch_1d_idx, atom_to_token_idx_map]
        )
        batch_2d_idx = torch.arange(batch_size, device=device)
        batch_2d_idx = batch_2d_idx.view(batch_size, 1, 1).expand(
            -1,
            atom_length,
            atom_length,
        )
        atom_pair = (
            atom_pair
            + _to_add_pair[
                batch_2d_idx,
                atom_to_token_idx_map,
                atom_to_token_idx_map,
            ]
        )
        # augmentation
        atom_single_rep = atom_single_cond.unsqueeze(0)
        to_add = self.noisy_to_atom_single_rep(
            x_t.to(torch.float32),
        )
        to_add = to_add * x_mask.unsqueeze(-1)
        atom_single_rep = atom_single_rep + to_add
        _left = self.atom_single_to_pair_left(atom_single_cond)
        _right = self.atom_single_to_pair_right(atom_single_cond)
        atom_single_cond = atom_single_cond.unsqueeze(0).expand(num_aug, -1, -1, -1)

        atom_pair = atom_pair + _left[..., None, :] + _right[..., None, :, :]
        atom_pair = atom_pair + self.mlp_atom_pair(atom_pair)
        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def _scatter_atom_to_token(
        self,
        token_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Scatter atom single representation to token single representation."""
        dtype = atom_single_rep.dtype

        atom_single_rep = torch.where(
            atom_mask.unsqueeze(-1),
            atom_single_rep,
            torch.zeros_like(atom_single_rep),
        )

        token_length = int(token_idx.shape[1])

        # one-hot assignment: (B, L_atom, L_token)
        mapping = torch.nn.functional.one_hot(
            atom_to_token_idx_map,
            num_classes=token_length,
        ).to(dtype)
        mask_f = atom_mask.to(dtype)  # (B, L_atom)
        count = torch.einsum("bal,ba->bl", mapping, mask_f)

        # project atoms -> token feature dim: (A, B, L_atom, d_single_token)
        to_add_single_token_rep = self.atom_single_rep_to_token_single(atom_single_rep)

        # apply mask AFTER projection (prevents bias leakage if projection has bias)
        atom_mask = atom_mask.unsqueeze(0).unsqueeze(-1)  # (1, B, L_atom, 1)
        to_add_single_token_rep = (
            to_add_single_token_rep * atom_mask
        )  # (A, B, L_atom, d)

        # einsum over atoms -> token sum: (A, B, L_token, d)
        token_single_rep = torch.einsum(
            "bal,nbac->nblc",
            mapping,
            to_add_single_token_rep,
        )
        # Explanation of labels:
        # A:    (B, L_atom, L_token) -> "bal" (b=batch, a=atom, l=token)
        # to_add (A, B, L_atom, d)   -> "abac" where c=d, reuse a=atom
        # out:  (A, B, L_token, d)   -> "ablc"

        return token_single_rep / count.unsqueeze(0).unsqueeze(-1).clamp(min=1.0)

    @typecheck
    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single_token_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "A B L_atom L_atom d_pair_atom"],
    ]:
        """Forward pass."""
        atom_single_init, atom_pair_init = init_atom_features(reference)
        atom_to_token_idx_map = scheme.atom_to_token_idx_map

        if self.use_checkpoint:
            atom_single_rep, atom_single_cond, atom_pair = checkpoint(
                self._before_atom_transformer,
                x_t,
                x_mask,
                atom_single_init,
                atom_pair_init,
                atom_to_token_idx_map,
                token_single_cond,
                token_pair_cond,
                use_reentrant=False,
            )  # pyright: ignore[reportGeneralTypeIssues]
        else:
            atom_single_rep, atom_single_cond, atom_pair = self._before_atom_transformer(
                x_t,
                x_mask,
                atom_single_init,
                atom_pair_init,
                atom_to_token_idx_map,
                token_single_cond,
                token_pair_cond,
            )
        atom_single_rep = self.atom_transformer(
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            structure.atom_mask,
        )

        if self.use_checkpoint:
            token_single_rep = checkpoint(
                self._scatter_atom_to_token,
                scheme.token_idx,
                structure.atom_mask,
                atom_to_token_idx_map,
                atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(
                scheme.token_idx,  # pyright: ignore[reportCallIssue]
                structure.atom_mask,  # pyright: ignore[reportCallIssue]
                atom_to_token_idx_map,
                atom_single_rep,
            )
        return token_single_rep, atom_single_rep, atom_single_cond, atom_pair  # pyright: ignore[reportReturnType]


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
    ) -> None:
        super().__init__()
        self.shared_config = shared_config
        self.diffusion_config = diffusion_config
        d_single_atom = shared_config.d_single_atom
        d_single_token = shared_config.d_single_token

        self.add_token_info = Linear(d_single_token, d_single_atom, bias=False)

        self.atom_transformer = DiffusionTransformer(config=diffusion_config)

        self.final_denoising = nn.Sequential(
            LayerNorm(d_single_atom),
            Linear(
                d_single_atom,
                3,
                bias=False,
                init="zero",
            ),
        )

    @typecheck
    def forward(
        self,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        token_single_rep: Float[torch.Tensor, "A B L_token d_single_token"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        atom_pair: Float[torch.Tensor, "A B L_atom L_atom d_pair_atom"],
    ) -> Float[torch.Tensor, "A B L_atom 3"]:
        """Forward pass."""
        num_augment, batch_size, atom_length = atom_single_rep.shape[:3]
        device = atom_single_rep.device
        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(1, batch_size, 1).expand(
            num_augment,
            -1,
            atom_length,
        )
        aug_1d_idx = torch.arange(num_augment, device=device)
        aug_1d_idx = aug_1d_idx.view(num_augment, 1, 1).expand(
            -1,
            batch_size,
            atom_length,
        )
        atom_to_token_idx_map = scheme.atom_to_token_idx_map
        atom_to_token_idx_map = atom_to_token_idx_map.unsqueeze(0).expand(
            num_augment,
            -1,
            -1,
        )

        _to_add_single = self.add_token_info(token_single_rep)
        atom_single_rep = (
            atom_single_rep
            + _to_add_single[aug_1d_idx, batch_1d_idx, atom_to_token_idx_map]
        )

        atom_single_rep = self.atom_transformer(
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            mask=structure.atom_mask,
        )
        return self.final_denoising(atom_single_rep)


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
        d_single = shared_config.d_single
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
        self.linear_token_single = nn.Sequential(
            LayerNorm(
                shared_config.d_single_token_input + shared_config.d_single,
            ),
            Linear(
                shared_config.d_single_token_input + shared_config.d_single,
                d_single,
                bias=False,
            ),
        )
        self.add_time_embedding = nn.Sequential(
            LayerNorm(
                d_time,
            ),
            Linear(d_time, d_single, bias=False),
        )
        self.single_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_single,
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
        token_single_input: Float[torch.Tensor, "B L_token d_single_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single"],
        Float[torch.Tensor, "B L_token L_token d_pair"],
    ]:
        """Forward pass of the diffusion conditioning module."""
        rel_emb = self.relative_position_embedder(
            asym_id=scheme.token_asym_id,
            token_residue_idx=scheme.token_residue_idx,
            token_idx=scheme.token_idx,
            entity_id=scheme.token_entity_id,
            sym_id=scheme.token_sym_id,
        )
        token_pair = torch.cat([token_pair_trunk, rel_emb], dim=-1)
        token_pair = self.linear_token_pair(token_pair)

        for transition in self.pair_transitions:
            token_pair = token_pair + transition(token_pair)

        token_single = torch.cat([token_single_input, token_single_trunk], dim=-1)

        token_single = self.linear_token_single(token_single)
        time_embedding = fourier_embedding(t_emb)
        time_embedding = time_embedding.squeeze(-2)
        token_single = token_single + self.add_time_embedding(time_embedding)

        for transition in self.single_transitions:
            token_single = token_single + transition(token_single)

        return token_single, token_pair


class DiffusionModule(nn.Module):
    """Diffusion module for processing input features."""

    def __init__(
        self,
        shared_config: SharedConfig,
        atom_dit_config: DiffusionTransformer.Config,
        token_dit_config: DiffusionTransformer.Config,
        dit_cond_config: DiffusionConditioning.Config,
    ) -> None:
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(
            shared_config=shared_config,
            dit_cond_config=dit_cond_config,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            shared_config=shared_config,
            diffusion_config=atom_dit_config,
        )
        self.add_single_token_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single,
            ),
            Linear(
                shared_config.d_single,
                shared_config.d_single_token,
                bias=False,
                init="zero",
            ),
        )
        self.diffusion_transformer = DiffusionTransformer(config=token_dit_config)
        self.ln_token_single_rep = LayerNorm(
            shared_config.d_single_token,
        )
        self.atom_attention_decoder = AtomAttentionDecoder(
            shared_config=shared_config,
            diffusion_config=atom_dit_config,
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
        token_single_cond, token_pair_cond = self.diffusion_conditioning(
            scheme,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )
        token_single_rep, atom_single_rep, atom_single_cond, atom_pair = (
            self.atom_attention_encoder(
                reference,
                scheme,
                structure,
                x_t,
                x_mask,
                token_single_trunk,
                token_pair_cond,
            )
        )
        token_single_rep = token_single_rep + self.add_single_token_cond(
            token_single_cond,
        )
        token_single_rep = self.diffusion_transformer(
            token_single_rep,
            token_single_cond,
            token_pair_cond,
        )

        token_single_rep = self.ln_token_single_rep(token_single_rep)
        return self.atom_attention_decoder(
            scheme,
            structure,
            token_single_rep,
            atom_single_rep,
            atom_single_cond,
            atom_pair,
        )
