import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer
from team_gm.modules.primitives import Linear
from torch import nn
from torch.utils.checkpoint import checkpoint

from miniworld.configs import SharedConfig
from miniworld.data.features.batch_edge_backprop import (
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.modules.embeddings import RelativePositionEmbedding


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


class InputAtomAttentionEncoder(nn.Module):
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
        self.d_single_token = shared_config.d_single

        self.use_checkpoint = shared_config.use_checkpoint

        self.to_atom_single_cond = Linear(
            6,
            shared_config.d_single_atom,
            init="default",
            bias=False,
        )

        self.to_atom_pair = Linear(
            5,
            shared_config.d_pair_atom,
            init="default",
            bias=False,
        )

        self.atom_single_to_pair_left = nn.Sequential(
            nn.ReLU(), Linear(d_single_atom, d_pair_atom, bias=False),
        )

        self.atom_single_to_pair_right = nn.Sequential(
            nn.ReLU(), Linear(d_single_atom, d_pair_atom, bias=False),
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
            Linear(
                d_single_atom,
                self.d_single_token,
                init="default",
                bias=False,
            ),
            nn.ReLU(),
        )

    def _before_atom_transformer(
        self,
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom_init"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
    ) -> tuple[
            Float[torch.Tensor, "B L_atom d_single_atom_rep"],
            Float[torch.Tensor, "B L_atom d_single_atom_cond"],
            Float[torch.Tensor, "B L_atom L_atom d_pair_atom"],
        ]:
        """Prepare atom single representation before transformer."""
        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_single_rep = atom_single_cond
        atom_pair = self.to_atom_pair(atom_pair_init)


        left = self.atom_single_to_pair_left(atom_single_cond)
        right = self.atom_single_to_pair_right(atom_single_cond)

        atom_pair = atom_pair + left[..., None, :] + right[..., None, :, :]
        atom_pair = atom_pair + self.mlp_atom_pair(atom_pair)
        return atom_single_rep, atom_single_cond, atom_pair


    @typecheck
    def _scatter_atom_to_token(
        self,
        residue_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "B L_atom d_single_atom"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Scatter atom single representation to token single representation."""
        batch_size = atom_single_rep.shape[0]
        device = atom_single_rep.device
        atom_single_rep = atom_single_rep * atom_mask[..., None]
        to_add_single_token_rep = self.atom_single_rep_to_token_single(atom_single_rep)

        # Convert back to token-atom layout and aggregate to tokens
        token_length = int(residue_idx.shape[1])

        # A[b, a, t] = 1 if atom a maps to token t else 0
        mapping = torch.nn.functional.one_hot(atom_to_residue_idx_map, num_classes=token_length).to(to_add_single_token_rep.dtype)  # (B, L_atom, L_token)

        # token sums: (B, L_token, d) = einsum_{a}(A[b,a,t] * to_add[b,a,d])
        token_sum = torch.einsum("bat,bad->btd", mapping, to_add_single_token_rep)

        # counts: (B, L_token) = einsum_{a}(A[b,a,t] * mask[b,a])
        mask_f = atom_mask.to(to_add_single_token_rep.dtype)
        count = torch.einsum("bat,ba->bt", mapping, mask_f)

        return token_sum / count.unsqueeze(-1).clamp(min=1.0)


        count = torch.zeros((batch_size, token_length),device=device,dtype=torch.long)
        count.scatter_add_(
            1,
            atom_to_residue_idx_map,
            torch.ones_like(atom_to_residue_idx_map).long() * atom_mask,
        )

        token_single_rep = torch.zeros(
            (
                batch_size,
                token_length,
                self.d_single_token,
            ),
            device=device,
        )
        token_single_rep = token_single_rep.scatter_add(
            1,
            atom_to_residue_idx_map.unsqueeze(-1).expand(
                -1, -1, to_add_single_token_rep.shape[-1],
            ),
            to_add_single_token_rep,
        )
        return token_single_rep / count.unsqueeze(-1).clamp(min=1.0)

    @torch.compiler.disable
    def forward( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Forward pass."""
        atom_single_init, atom_pair_init = init_atom_features(reference)
        if self.use_checkpoint:
            atom_single_rep, atom_single_cond, atom_pair = checkpoint(
                self._before_atom_transformer,
                atom_single_init, atom_pair_init, use_reentrant=False,
            ) # pyright: ignore[reportGeneralTypeIssues]
        else:
            atom_single_rep, atom_single_cond, atom_pair = self._before_atom_transformer(
                atom_single_init, atom_pair_init,
            )
        atom_single_rep = self.atom_transformer(
            atom_single_rep.unsqueeze(0),
            atom_single_cond.unsqueeze(0),
            atom_pair,
        )
        atom_single_rep = atom_single_rep.squeeze(0)

        if self.use_checkpoint:
            token_single_rep = checkpoint(
                self._scatter_atom_to_token,
                scheme.residue_idx,
                structure.atom_mask,
                scheme.atom_to_residue_idx_map,
                atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(
                scheme.residue_idx,
                structure.atom_mask,
                scheme.atom_to_residue_idx_map,
                atom_single_rep,
            )

        return token_single_rep # pyright: ignore[reportReturnType]


class InputFeatureEmbedder(nn.Module):
    """Input feature embedder module."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
    ) -> None:
        super().__init__()
        self.num_res_class = shared_config.num_res_class
        self.use_checkpoint = shared_config.use_checkpoint
        self.d_pair = shared_config.d_pair
        self.atom_attention_encoder = InputAtomAttentionEncoder(
            shared_config=shared_config,
            diffusion_config=diffusion_config,
        )
        d_init = shared_config.d_single_token_input
        self.to_token_init = Linear(
            d_init,
            shared_config.d_single,
            init="default",
            bias=False,
        )
        self.to_token_pair_left = Linear(
            d_init,
            shared_config.d_pair,
            init="default",
            bias=False,
        )
        self.to_token_pair_right = Linear(
            d_init,
            shared_config.d_pair,
            init="default",
            bias=False,
        )
        self.relative_position_embedder = RelativePositionEmbedding(
            d_hidden=shared_config.d_pair,
            r_max=shared_config.r_max,
            s_max=shared_config.s_max,
        )
        self.add_token_bond = Linear(
            2,
            shared_config.d_pair,
            init="default",
            bias=False,
        )
        # self.add_atom_bond = Linear(2, config.d_pair_atom, init="default") TODO

    @torch.no_grad()
    def _gen_bond_feature(
        self,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token L_token 2"]:
        #  -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_length = structure.residue_mask.shape[:2]
        device = structure.residue_bond.device
        residue_bond = structure.residue_bond.long()  # (batch_size, n_residue_bond, 3)
        token_bond = torch.zeros(
            (batch_size, token_length, token_length),
            device=device,
        )
        residue_bond_i, residue_bond_j, residue_bond_type = (
            residue_bond[:, :, 0],
            residue_bond[:, :, 1],
            residue_bond[:, :, 2],
        )

        # use only canonical bond where residue_bond_type == 0
        batch_idx, ij = torch.where(residue_bond_type == 0)
        residue_bond_i = residue_bond_i[batch_idx, ij]
        residue_bond_j = residue_bond_j[batch_idx, ij]
        token_bond[batch_idx, residue_bond_i, residue_bond_j] = 1
        token_bond[batch_idx, residue_bond_j, residue_bond_i] = 1

        return torch.nn.functional.one_hot(
            token_bond.long(),
            num_classes=2,
        )

    def forward(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single_token_input"],
        Float[torch.Tensor, "B L_token d_single_token_init"],
        Float[torch.Tensor, "B L_token L_token d_pair"],
    ]:
        """Forward pass."""
        token_single_input = self.atom_attention_encoder(reference,scheme,structure)

        residue_type = torch.nn.functional.one_hot(
            sequence.residue_type.long(), num_classes=self.num_res_class,
        ).to(token_single_input.device, dtype=token_single_input.dtype)

        token_single_input = torch.concat(
            [
                token_single_input,
                residue_type,
                msa.profile.to(dtype=token_single_input.dtype),
                msa.deletion_mean.unsqueeze(-1).to(dtype=token_single_input.dtype),
            ],
            dim=-1,
        )

        token_single_init = self.to_token_init(token_single_input)
        token_left = self.to_token_pair_left(token_single_input)
        token_right = self.to_token_pair_right(token_single_input)
        token_pair_init = rearrange(token_left, "b l d -> b l 1 d") + rearrange(
            token_right, "b l d -> b 1 l d",
        )

        token_pair_init = token_pair_init + self.relative_position_embedder(
            asym_id = scheme.residue_asym_id,
            residue_idx = scheme.residue_idx,
            entity_id = scheme.residue_entity_id,
            sym_id = scheme.residue_sym_id,
        )
        token_pair_init = token_pair_init + self.add_token_bond(
            self._gen_bond_feature(structure).to(dtype=token_pair_init.dtype),
        )

        return (
            token_single_input,
            token_single_init,
            token_pair_init,
        )



