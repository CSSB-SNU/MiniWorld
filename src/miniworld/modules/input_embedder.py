
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer
from team_gm.modules.primitives import Linear
from torch import nn
from torch.utils.checkpoint import checkpoint

from miniworld.configs import SharedConfig
from miniworld.data.features.features import (
    ReferenceFeatures,
    SchemeFeatures,
    StructureFeatures,
)
from team_gm.modules.layers import RelativePositionEmbedding


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

    # Inference-time chunking constant. Splits the first L_atom axis so the
    # broadcast add and the pair MLP each materialise only
    # [B, _ATOM_CHUNK, L_atom, d] temporaries instead of the full
    # [B, L_atom, L_atom, d] (which OOMs once L_atom is in the 14k+ range —
    # H1340 hit this at L_atom=15819 trying to allocate 14.92 GiB).
    _ATOM_CHUNK: int = 1024

    @typecheck
    def _before_atom_transformer_chunked(
        self,
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom_init"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
    ) -> tuple[
        Float[torch.Tensor, "B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom"],
    ]:
        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_single_rep = atom_single_cond
        atom_pair = self.to_atom_pair(atom_pair_init)

        left = self.atom_single_to_pair_left(atom_single_cond)
        right = self.atom_single_to_pair_right(atom_single_cond)

        atom_length = atom_pair.shape[1]
        chunk = self._ATOM_CHUNK
        for s in range(0, atom_length, chunk):
            e = min(s + chunk, atom_length)
            pair_slice = atom_pair[:, s:e]
            pair_slice.add_(left[:, s:e].unsqueeze(2))
            pair_slice.add_(right.unsqueeze(1))

        for s in range(0, atom_length, chunk):
            e = min(s + chunk, atom_length)
            atom_pair[:, s:e].add_(self.mlp_atom_pair(atom_pair[:, s:e]))

        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def _scatter_atom_to_token(
        self,
        token_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "B L_atom d_single_atom"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Scatter atom single representation to token single representation."""
        atom_single_rep = atom_single_rep * atom_mask[..., None]
        to_add_single_token_rep = self.atom_single_rep_to_token_single(atom_single_rep)

        # Convert back to token-atom layout and aggregate to tokens
        token_length = int(token_idx.shape[1])

        # A[b, a, t] = 1 if atom a maps to token t else 0
        mapping = torch.nn.functional.one_hot(
            atom_to_token_idx_map,
            num_classes=token_length,
        ).to(to_add_single_token_rep.dtype)  # (B, L_atom, L_token)

        # token sums: (B, L_token, d) = einsum_{a}(A[b,a,t] * to_add[b,a,d])
        token_sum = torch.einsum("bat,bad->btd", mapping, to_add_single_token_rep)

        # counts: (B, L_token) = einsum_{a}(A[b,a,t] * mask[b,a])
        mask_f = atom_mask.to(to_add_single_token_rep.dtype)
        count = torch.einsum("bat,ba->bt", mapping, mask_f)

        return token_sum / count.unsqueeze(-1).clamp(min=1.0)

    @torch.compiler.disable
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Forward pass."""
        atom_single_init, atom_pair_init = init_atom_features(reference)
        # Inference chunking is opt-in via env var, mirroring diffusion_module.
        # Same parity guarantee — the chunked path is bit-exact with the
        # canonical (see tests/test_input_embedder_chunked.py). Enabled when
        # the canonical's [B, L_atom, L_atom, d] broadcast OOMs (~14k+ atoms).
        # Default fallback: chunk the atom attention whenever not training (the canonical
        # [B, L_atom, L_atom, d] broadcast OOMs at ~14k+ atoms); never chunk in training.
        use_chunked_inference = not self.training
        if use_chunked_inference:
            atom_single_rep, atom_single_cond, atom_pair = (
                self._before_atom_transformer_chunked(
                    atom_single_init,
                    atom_pair_init,
                )
            )
            del atom_single_init, atom_pair_init
        elif self.use_checkpoint:
            atom_single_rep, atom_single_cond, atom_pair = checkpoint(
                self._before_atom_transformer,
                atom_single_init,
                atom_pair_init,
                use_reentrant=False,
            )  # pyright: ignore[reportGeneralTypeIssues]
        else:
            atom_single_rep, atom_single_cond, atom_pair = self._before_atom_transformer(
                atom_single_init,
                atom_pair_init,
            )
        atom_single_rep = self.atom_transformer(
            atom_single_rep.unsqueeze(0),
            atom_single_cond.unsqueeze(0),
            atom_pair,
            mask=structure.atom_mask.unsqueeze(0),
        )
        atom_single_rep = atom_single_rep.squeeze(0)

        if self.use_checkpoint:
            token_single_rep = checkpoint(
                self._scatter_atom_to_token,
                scheme.token_idx,
                structure.atom_mask,
                scheme.atom_to_token_idx_map,
                atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(
                scheme.token_idx,
                structure.atom_mask,
                scheme.atom_to_token_idx_map,
                atom_single_rep,
            )

        return token_single_rep  # pyright: ignore[reportReturnType]


class InputFeatureEmbedder(nn.Module):
    """Input feature embedder module."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
        *,
        produce_single_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_res_class = shared_config.num_res_class
        self.use_checkpoint = shared_config.use_checkpoint
        self.d_pair = shared_config.d_pair
        self.produce_single_init = produce_single_init
        self.atom_attention_encoder = InputAtomAttentionEncoder(
            shared_config=shared_config,
            diffusion_config=diffusion_config,
        )
        d_init = shared_config.d_single_token_input

        if produce_single_init:
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
    @torch.compiler.disable  # we are not bucketing token_bond, so the input size can be very different across batches. disable torch.compile for now to avoid overhead of recompilation.
    def _gen_bond_feature(
        self,
        structure: StructureFeatures,
    ) -> Float[torch.Tensor, "B L_token L_token 2"]:
        #  -> tuple[torch.Tensor, torch.Tensor]:
        # The bond feature is a pure function of the (fixed-per-input) token_bond and has a
        # STATIC output shape (B, L, L, 2) even though token_bond is variable-length. Its
        # construction uses boolean-mask indexing (device->host sync) + variable shapes that
        # are illegal inside a CUDA-graph capture, so memoise on token_bond identity: it runs
        # once (eagerly, warm-up) and a graph replay reuses the cached static tensor.
        _tb = structure.token_bond
        _c = getattr(self, "_bond_feat_cache", None)
        if _c is not None and _c[0] is _tb:
            return _c[1]
        batch_size, token_length = structure.token_mask.shape[:2]
        device = structure.token_bond.device
        token_bond = structure.token_bond.long()  # (batch_size, n_token_bond, 3)
        token_bond_i, token_bond_j = (
            token_bond[:, :, 0],
            token_bond[:, :, 1],
        )
        # remove diagonal bonds if exist
        mask = token_bond_i != token_bond_j
        # Expand batch_idx to (B, n_bond) BEFORE masking so the per-bond batch
        # identity survives the flatten. Without this, batch_idx stays (B, 1)
        # and broadcasts every surviving bond into every batch row.
        batch_idx = (
            torch.arange(batch_size, device=device)[:, None]
            .expand_as(token_bond_i)
        )
        batch_idx = batch_idx[mask]
        token_bond_i = token_bond_i[mask]
        token_bond_j = token_bond_j[mask]
        token_bond_feature = torch.zeros(
            (batch_size, token_length, token_length),
            device=device,
        )

        token_bond_feature[batch_idx, token_bond_i, token_bond_j] = 1
        token_bond_feature[batch_idx, token_bond_j, token_bond_i] = 1

        _out = torch.nn.functional.one_hot(
            token_bond_feature.long(),
            num_classes=2,
        )
        self._bond_feat_cache = (_tb, _out)
        return _out

    def forward(
        self,
        token_single_msa: Float[torch.Tensor, "B L_token d_single_token_init"],
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single_token_input"],
        Float[torch.Tensor, "B L_token d_single_token_init"] | None,
        Float[torch.Tensor, "B L_token L_token d_pair"],
    ]:
        """Forward pass."""
        token_single_input = self.atom_attention_encoder(reference, scheme, structure)

        token_single_input = torch.concat(
            [
                token_single_input,
                token_single_msa,
            ],
            dim=-1,
        )

        token_single_init = (
            self.to_token_init(token_single_input) if self.produce_single_init else None
        )
        token_left = self.to_token_pair_left(token_single_input)
        token_right = self.to_token_pair_right(token_single_input)
        token_pair_init = rearrange(token_left, "b l d -> b l 1 d") + rearrange(
            token_right,
            "b l d -> b 1 l d",
        )

        token_pair_init = token_pair_init + self.relative_position_embedder(
            asym_id=scheme.token_asym_id,
            token_residue_idx=scheme.token_residue_idx,
            token_idx=scheme.token_idx,
            entity_id=scheme.token_entity_id,
            sym_id=scheme.token_sym_id,
        )

        # Prefer the dataloader-precomputed dense adjacency (fixed shape -> the
        # captured forward stays graph-legal and updates correctly per replay).
        # Fall back to the in-forward scatter for legacy/eager batches that don't
        # carry ``token_bond_feat``.
        if structure.token_bond_feat is not None:
            bond_onehot = torch.nn.functional.one_hot(
                structure.token_bond_feat.long(), num_classes=2,
            )
        else:
            bond_onehot = self._gen_bond_feature(structure)
        token_pair_init = token_pair_init + self.add_token_bond(
            bond_onehot.to(dtype=token_pair_init.dtype),
        )

        return (
            token_single_input,
            token_single_init,
            token_pair_init,
        )
