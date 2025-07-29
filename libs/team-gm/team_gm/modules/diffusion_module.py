import torch
import torch.nn as nn
from jaxtyping import Float, Bool
from team_gm.data.features import NoisyBatch

from .primitives import (
    Linear,
    LayerNorm,
    Transition,
)

from .diffusion_transformer import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    DiffusionTransformer,
)
from .feature_embedder import FourierEmbedding
from .feature_embedder import RelativePositionEmbedding
from .configs import CommonConfig, DiffusionConfig


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning module."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
    ):
        super().__init__()
        d_token_single = common_config.d_token_single
        d_token_pair = common_config.d_token_pair
        d_time = common_config.d_time
        self.relative_position_embedder = RelativePositionEmbedding(
            d_hidden=d_token_pair,
            r_max=common_config.r_max,
            s_max=common_config.s_max,
        )

        implementation = diffusion_config.implementation

        self.linear_token_pair = nn.Sequential(
            LayerNorm(
                2 * d_token_pair,
                implementation=implementation,
                # implementation="pytorch",
            ),
            Linear(2 * d_token_pair, d_token_pair, bias=False),
        )
        self.pair_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_token_pair,
                    n=diffusion_config.n_transition_expand,
                    implementation=implementation,
                )
                for _ in range(diffusion_config.n_transition_block)
            ]
        )
        self.linear_token_single = nn.Sequential(
            LayerNorm(
                common_config.d_token_single_input + common_config.d_token_single,
                implementation=implementation,
                # implementation="pytorch",
            ),
            Linear(
                common_config.d_token_single_input + common_config.d_token_single,
                d_token_single,
                bias=False,
            ),
        )
        self.add_time_embedding = nn.Sequential(
            LayerNorm(
                d_time,
                implementation=implementation,
                # implementation="pytorch",
            ),
            Linear(d_time, d_token_single, bias=False),
        )
        self.single_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_token_single,
                    n=diffusion_config.n_transition_expand,
                    implementation="pytorch",
                )
                for _ in range(diffusion_config.n_transition_block)
            ]
        )

    def forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_input: Float[torch.Tensor, "B L d_single_input"],
        token_single_trunk: Float[torch.Tensor, "B L d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L L d_pair"],
    ) -> tuple[Float[torch.Tensor, "B L d_token_single"],
               Float[torch.Tensor, "B L L d_token_pair"]]:
        """Forward pass of the diffusion conditioning module.

        Parameters
        ----------
        noisy_batch: NoisyBatch
            Batch of noisy data.
        token_single_input: FloatTensor, (B, L, d_single)
            Input single representation.
        token_single_trunk: FloatTensor, (B, L, d_single)
            Single representation.
        token_pair_trunk: FloatTensor, (B, L, L, d_pair)
            Pair representation.
        """
        token_pair = torch.cat(
            [token_pair_trunk, self.relative_position_embedder(noisy_batch)],
            dim=-1,
        )  # (B, L, L, 2 * d_pair)
        token_pair = self.linear_token_pair(token_pair)

        for transition in self.pair_transitions:
            token_pair = token_pair + transition(token_pair)

        token_single = torch.cat([token_single_input, token_single_trunk], dim=-1)

        token_single = self.linear_token_single(token_single)
        time_embedding = FourierEmbedding(noisy_batch.t)
        time_embedding = time_embedding.squeeze(-2)
        token_single = token_single + self.add_time_embedding(time_embedding)

        for transition in self.single_transitions:
            token_single = token_single + transition(token_single)

        return token_single, token_pair


class DiffusionModule(nn.Module):
    """Diffusion module for processing input features."""

    def __init__(self, common_config: CommonConfig, diffusion_config: DiffusionConfig):
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(
            common_config=common_config,
            diffusion_config=diffusion_config,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            common_config=common_config,
            diffusion_config=diffusion_config,
        )
        self.add_token_single_cond = nn.Sequential(
            LayerNorm(
                common_config.d_token_single,
                implementation=diffusion_config.implementation,
                # implementation="pytorch",
            ),
            Linear(
                common_config.d_token_single,
                common_config.d_token_single_diffusion,
                bias=False,
                init="final",
            ),
        )
        self.diffusion_transformer = DiffusionTransformer(
            common_config=common_config,
            diffusion_config=diffusion_config,
            level="token",  # Token level for diffusion transformer
        )
        self.ln_token_single_rep = LayerNorm(
            common_config.d_token_single_diffusion,
            implementation=diffusion_config.implementation,
            # implementation="pytorch",
        )
        self.atom_attention_decoder = AtomAttentionDecoder(
            common_config=common_config,
            diffusion_config=diffusion_config,
        )

    def forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_input: Float[torch.Tensor, "B L_token d_token_single_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_token_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module.

        Parameters
        ----------
        noisy_batch: NoisyBatch
            Batch of noisy data.
        token_single_input: FloatTensor, (B, L_token, d_single)
            Input single representation.
        token_single_trunk: FloatTensor, (B, L_token, d_single)
            Single representation.
        token_pair_trunk: FloatTensor, (B, L_token, L_token, d_pair)
            Pair representation.
        atom_single_cond: FloatTensor, (B, L_atom, d_atom_single)
            Atom single condition representation.
        atom_pair: FloatTensor, (B, L_atom, L_atom, d_atom_pair, bf16)
            Atom pair representation.
        """

        token_single_cond, token_pair_cond = self.diffusion_conditioning(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )
        token_single_rep, atom_single_rep, atom_single_cond, atom_pair = (
            self.atom_attention_encoder(
                noisy_batch,
                token_single_trunk,
                token_pair_cond,
            )
        )  # (A, B, L_token, d_token_single), (A, B, L_atom, d_atom_single),
        token_single_rep = token_single_rep + self.add_token_single_cond(
            token_single_cond
        )
        token_single_rep = self.diffusion_transformer(
            noisy_batch, token_single_rep, token_single_cond, token_pair_cond
        )

        token_single_rep = self.ln_token_single_rep(token_single_rep)

        x_denoised = self.atom_attention_decoder(
            noisy_batch,
            token_single_rep,
            atom_single_rep,
            atom_single_cond,
            atom_pair,
        )

        return x_denoised
