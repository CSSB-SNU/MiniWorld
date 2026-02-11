import math

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from team_gm.modules.primitives import (
    Linear,
)
from torch import nn

from miniworld.data.features.batch_edge_backprop import NoisyBatch

from .configs import CommonConfig, DiffusionConfig
from .diffusion_transformer import DiffusionTransformer

_WEIGHT = [
    -1.1589,
    1.4807,
    0.7262,
    1.3044,
    0.6426,
    1.8878,
    -0.1465,
    0.7416,
    -0.3565,
    -0.6110,
    0.6655,
    -0.7138,
    -0.7535,
    0.0494,
    0.1794,
    1.6631,
    -2.3644,
    0.1177,
    1.2897,
    -0.5229,
    1.7306,
    -0.7403,
    -0.0762,
    -1.0188,
    -0.2480,
    0.6329,
    -0.9951,
    0.3340,
    -0.5763,
    2.4806,
    -0.8525,
    -0.7035,
    -0.8966,
    0.1351,
    0.1460,
    0.0361,
    -0.8752,
    0.9624,
    0.7519,
    1.6343,
    -0.7335,
    -0.3643,
    -0.0600,
    0.1521,
    -0.3653,
    1.4547,
    -0.3140,
    -1.7813,
    -0.3936,
    1.9698,
    0.8503,
    0.8260,
    0.0833,
    -0.9798,
    -1.3805,
    -0.6103,
    -0.6822,
    2.4680,
    -0.1125,
    -1.2157,
    -2.1034,
    0.6673,
    -0.4207,
    -0.2562,
    0.8737,
    0.5097,
    0.4577,
    0.6198,
    0.3605,
    -0.2447,
    -1.0918,
    -0.4810,
    0.2123,
    -0.9405,
    0.2234,
    0.6086,
    -0.1535,
    0.5058,
    -1.7518,
    -0.1994,
    -1.8199,
    -1.7643,
    0.5369,
    0.8609,
    1.5825,
    -0.1063,
    -0.3740,
    -1.0754,
    -0.4132,
    1.1038,
    1.5608,
    0.1073,
    -0.2388,
    2.8097,
    -2.1987,
    1.1067,
    -0.2794,
    -0.8661,
    -0.7962,
    -1.8509,
    -1.3860,
    -0.8742,
    -3.0911,
    0.6652,
    -0.6121,
    0.9059,
    -0.7976,
    0.0305,
    2.0485,
    -0.6105,
    0.2266,
    -1.0364,
    1.6659,
    -0.3239,
    -0.6981,
    -0.9190,
    0.9563,
    -0.9255,
    -0.1570,
    1.8202,
    -0.3919,
    -1.0554,
    0.3473,
    -0.7071,
    -1.0972,
    0.6956,
    0.1403,
    -0.8596,
    -0.2272,
    -0.7880,
    -0.0045,
    -1.0988,
    0.6069,
    0.8678,
    -0.5890,
    -0.6310,
    -1.3496,
    0.8744,
    -0.9512,
    -0.2902,
    0.7098,
    -1.4746,
    -1.2565,
    0.9278,
    1.2793,
    0.2097,
    -0.2348,
    -0.7039,
    -0.6304,
    -1.4228,
    0.8491,
    0.5395,
    0.8440,
    0.2323,
    0.7833,
    0.4665,
    0.7819,
    -1.2969,
    0.7206,
    -0.7932,
    2.3619,
    -0.1087,
    0.5218,
    -0.0716,
    1.3572,
    1.8727,
    -1.2118,
    -0.8358,
    -1.1375,
    -1.1205,
    0.1123,
    0.8100,
    1.0739,
    1.7904,
    0.7583,
    -1.0359,
    1.5144,
    -0.3864,
    1.3583,
    -0.0583,
    -0.5507,
    -0.8598,
    0.4309,
    -1.9506,
    -0.5559,
    0.8420,
    -0.6507,
    3.3478,
    -0.7853,
    1.6784,
    -1.0035,
    -0.0962,
    0.1253,
    1.0577,
    -0.3417,
    -0.1534,
    0.1866,
    1.5311,
    -1.1470,
    -2.4486,
    0.3169,
    -1.1714,
    0.7569,
    0.3302,
    -1.9390,
    -0.1006,
    -0.8827,
    -0.1644,
    -0.7730,
    0.3290,
    1.9623,
    0.5125,
    -1.4730,
    0.6633,
    0.1830,
    -1.4353,
    -0.5513,
    -0.0752,
    0.6699,
    -0.8696,
    -0.1894,
    0.0845,
    -0.3084,
    0.0685,
    0.3570,
    -1.7642,
    1.4354,
    -0.4957,
    0.1688,
    0.4841,
    -2.3683,
    -1.8467,
    -0.1852,
    -0.5323,
    -0.5766,
    0.5890,
    0.5705,
    -0.3292,
    1.8812,
    -1.0640,
    -1.0277,
    1.0256,
    -1.2382,
    0.6019,
    -0.2316,
    0.6242,
    -1.4421,
    1.4296,
    -0.8979,
    -2.1853,
    1.4251,
    0.7186,
    0.0644,
    0.5098,
    -1.2769,
    -0.0987,
]
_BIAS = [
    -1.4190,
    -1.0065,
    0.0362,
    1.1636,
    1.1115,
    0.6089,
    -0.0211,
    -0.5830,
    -1.8058,
    0.0243,
    -1.6017,
    0.7425,
    1.1244,
    1.0739,
    0.9154,
    0.1145,
    -0.8459,
    1.1712,
    -1.4933,
    -0.4954,
    -0.7849,
    -3.1017,
    0.2436,
    -1.4313,
    -0.7459,
    0.5631,
    0.5681,
    -0.2503,
    -0.1774,
    -0.9234,
    0.0393,
    0.0625,
    -0.1277,
    -0.1806,
    -1.8476,
    0.8164,
    -0.3174,
    -0.3205,
    0.7008,
    -0.5756,
    0.6807,
    1.8730,
    -0.3832,
    -1.5800,
    1.1597,
    -0.3843,
    0.0205,
    2.1178,
    -0.9236,
    1.8892,
    -0.1891,
    1.3272,
    -0.3843,
    0.1069,
    -1.3175,
    -2.4995,
    0.7618,
    0.8227,
    -0.6093,
    -0.2384,
    -0.8309,
    -0.2393,
    -1.2441,
    -0.0734,
    0.3302,
    -0.4147,
    -1.2927,
    -0.3335,
    -0.4191,
    1.3443,
    0.8877,
    0.7347,
    1.2962,
    -0.3946,
    0.7831,
    0.4926,
    -1.9702,
    0.6418,
    -1.3820,
    -0.8366,
    2.4390,
    -0.5834,
    1.6482,
    0.9362,
    -0.2113,
    -0.5718,
    0.1988,
    -0.7303,
    -1.4876,
    0.5828,
    1.8084,
    0.0479,
    0.1199,
    -0.6434,
    -0.4102,
    0.8611,
    1.2832,
    0.0302,
    -2.6942,
    -0.2834,
    0.0483,
    1.5292,
    -1.5802,
    -1.0741,
    -0.6141,
    -1.0296,
    0.4388,
    -1.6312,
    -0.7918,
    0.7881,
    1.2290,
    -1.0167,
    1.3910,
    0.9654,
    0.9372,
    0.6062,
    -1.9253,
    1.2297,
    -0.1572,
    -0.0205,
    -0.0170,
    0.5851,
    -0.8096,
    -0.9248,
    -0.2621,
    -1.9889,
    1.3132,
    0.7940,
    -0.8714,
    -1.1346,
    0.5938,
    1.2530,
    -0.1799,
    0.5836,
    -0.4587,
    1.1356,
    0.1612,
    -1.1111,
    -0.6099,
    -1.4074,
    -0.6229,
    0.0145,
    0.7609,
    0.2248,
    1.5537,
    -0.2796,
    0.0800,
    0.0235,
    -1.6223,
    0.5634,
    -0.7584,
    0.4526,
    0.4029,
    0.7784,
    0.7047,
    -0.5739,
    -1.0243,
    0.0605,
    0.2671,
    0.4763,
    -1.3214,
    0.8809,
    -0.4152,
    -1.0702,
    -3.2218,
    0.4255,
    -1.6404,
    -2.2027,
    -0.2414,
    0.5740,
    -0.6045,
    0.2962,
    -1.3391,
    0.5869,
    -1.5150,
    -0.5294,
    -1.1596,
    1.1069,
    -0.0623,
    1.3868,
    0.5474,
    0.6112,
    1.4193,
    0.3986,
    -0.3027,
    -0.9390,
    0.3705,
    0.5854,
    -1.2239,
    0.1300,
    -0.7828,
    0.1169,
    0.8018,
    -1.9469,
    0.6882,
    -0.0783,
    -0.1827,
    -0.2774,
    1.1396,
    -1.1318,
    0.2418,
    1.0324,
    -0.2324,
    -0.5123,
    1.3504,
    0.0091,
    -0.2387,
    1.2060,
    -0.9270,
    0.2123,
    0.7567,
    -0.6940,
    0.2930,
    0.9979,
    -0.2107,
    0.7412,
    -0.8158,
    0.7478,
    0.3334,
    -0.4809,
    0.0800,
    0.8701,
    1.3877,
    -0.0225,
    -0.9060,
    -1.1622,
    -0.4295,
    2.0665,
    -1.5668,
    1.0471,
    1.0771,
    -0.5370,
    -0.9477,
    0.7880,
    0.4338,
    -0.9888,
    0.7968,
    0.6770,
    -1.4634,
    -0.3976,
    0.7776,
    1.0420,
    0.9116,
    0.3766,
    0.2317,
    0.4907,
    -0.4076,
    0.0104,
    -0.1422,
    0.2210,
    0.6038,
    0.4143,
    -0.0416,
    -3.2148,
    -0.7869,
    -0.3049,
]


def fourier_embedding(sigma_nosie_level: torch.Tensor) -> torch.Tensor:
    """Return Fourier noise level embeddings for diffusion model."""
    sigma_nosie_level = sigma_nosie_level.to(dtype=torch.float32)
    weight = torch.tensor(_WEIGHT, dtype=torch.float32, device=sigma_nosie_level.device)
    bias = torch.tensor(_BIAS, dtype=torch.float32, device=sigma_nosie_level.device)
    embeddings = sigma_nosie_level[..., None] * weight + bias
    return torch.cos(2 * math.pi * embeddings)


class InputAtomAttentionEncoder(nn.Module):
    """Atom attention encoder."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.common_config = common_config
        self.diffusion_config = diffusion_config
        self.use_beta = diffusion_config.use_beta
        d_atom_single = common_config.d_atom_single
        d_atom_pair = common_config.d_atom_pair
        self.d_token_single = common_config.d_token_single

        self.use_checkpoint = common_config.use_checkpoint

        # self.to_atom_single_cond = Linear(
        #     6,
        #     common_config.d_atom_single,
        #     init="default",
        #     bias=False,
        # )

        self.to_atom_pair = Linear(
            5,
            common_config.d_atom_pair,
            init="default",
            bias=False,
        )

        # self.atom_single_to_pair_left = nn.Sequential(
        #     nn.ReLU(), Linear(d_atom_single, d_atom_pair, bias=False),
        # )

        # self.atom_single_to_pair_right = nn.Sequential(
        #     nn.ReLU(), Linear(d_atom_single, d_atom_pair, bias=False),
        # )

        self.mlp_atom_pair = nn.Sequential(
            Linear(d_atom_pair, d_atom_pair, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_atom_pair, d_atom_pair, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_atom_pair, d_atom_pair, init="zero", bias=False),
        )

        # self.atom_transformer = DiffusionTransformer(
        #     common_config=common_config,
        #     diffusion_config=diffusion_config,
        #     level="atom",
        # )

        # self.atom_single_rep_to_token_single = nn.Sequential(
        #     Linear(
        #         d_atom_single,
        #         self.d_token_single,
        #         init="default",
        #         bias=False,
        #     ),
        #     nn.ReLU(),
        # )

    def _get_input_feature(
        self, noisy_batch: NoisyBatch,
    ) -> Float[torch.Tensor, "B L L d_atom_pair"]:
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
        # ref_infos = torch.cat(
        #     [
        #         noisy_batch.reference.pos,
        #         noisy_batch.reference.mask.unsqueeze(-1),
        #         noisy_batch.reference.element.unsqueeze(-1),
        #         torch.arcsinh(noisy_batch.reference.charge).unsqueeze(-1),
        #     ],
        #     dim=-1,
        # )
        with torch.no_grad():
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
                msg = "Beta version does not support get input feature yet."
                raise NotImplementedError(msg)
            v_lm = v_lm[..., None]
            arctan_d_lm = 1 / (1 + d_lm.norm(dim=-1) ** 2)
            arctan_d_lm = arctan_d_lm.unsqueeze(-1)
            d_lm = torch.cat([d_lm, arctan_d_lm, v_lm], dim=-1)
            atom_pair = d_lm * v_lm
        # atom_single_cond = self.to_atom_single_cond(ref_infos)
        # atom_single_cond = atom_single_cond * noisy_batch.reference.mask.unsqueeze(-1)
        atom_pair = self.to_atom_pair(atom_pair)

        return atom_pair
        # return atom_single_cond, atom_pair

    def _scatter_single_to_pair(
        self,
        noisy_batch: NoisyBatch,  # noqa: ARG002
        # atom_single_cond: Float[torch.Tensor, "B L d_atom_single"],
        atom_pair: Float[torch.Tensor, "B L L d_atom_pair"],
    ) -> Float[torch.Tensor, "B L L d_atom_pair"]:
        # if not self.use_beta:
        #     _left = self.atom_single_to_pair_left(atom_single_cond)
        #     _right = self.atom_single_to_pair_right(atom_single_cond)

        # else:
        #     msg = "Beta version does not support scatter single to pair yet."
        #     raise NotImplementedError(msg)

        # atom_pair = atom_pair + _left[..., None, :] + _right[..., None, :, :]
        return atom_pair + self.mlp_atom_pair(atom_pair)

    def _scatter_atom_to_token(
        self,
        noisy_batch: NoisyBatch,
        # atom_single_rep: Float[torch.Tensor, "B L d_atom_single"],
    ) -> Float[torch.Tensor, "B L_token d_token_single"]:
        """Scatter atom single representation to token single representation."""
        atom_mask = noisy_batch.structure.atom_mask  # (B, L_atom)
        # atom_single_rep = atom_single_rep * atom_mask[..., None]
        # to_add_token_single_rep = self.atom_single_rep_to_token_single(atom_single_rep) # linear, relu

        # Convert back to token-atom layout and aggregate to tokens
        if not self.use_beta:
            atom_to_residue_idx_map = (
                noisy_batch.scheme.atom_to_residue_idx_map
            )  # (B, L_atom)
            count = torch.zeros_like(noisy_batch.sequence.residue_type).long()
            count.scatter_add_(
                1,
                atom_to_residue_idx_map,
                torch.ones_like(atom_to_residue_idx_map).long() * atom_mask,
            )

            token_single_rep = torch.zeros(
                (
                    noisy_batch.shape[0],
                    noisy_batch.residue_length,
                    self.d_token_single,
                ),
                device=noisy_batch.device,
            )
            # token_single_rep = token_single_rep.scatter_add(
            #     1,
            #     atom_to_residue_idx_map.unsqueeze(-1).expand(
            #         -1, -1, self.d_token_single,
            #     ),
            #     to_add_token_single_rep,
            # )
            # token_single_rep = token_single_rep / count.unsqueeze(-1).clamp(min=1.0)
        else:
            msg = "Beta version does not support scatter atom to token yet."
            raise NotImplementedError(msg)

        return token_single_rep

    def _before_atom_transformer(self, noisy_batch: NoisyBatch) -> torch.Tensor:
        """Prepare atom single representation before transformer."""
        atom_pair = self._get_input_feature(noisy_batch)
        # atom_single_rep = atom_single_cond
        atom_pair = self._scatter_single_to_pair(
            noisy_batch,
            # atom_single_cond,
            atom_pair,
        )
        return atom_pair

    def forward(
        self,
        noisy_batch: NoisyBatch,
    ) -> Float[torch.Tensor, "B L_token d_token_single"]:
        """Forward pass."""
        # if self.use_checkpoint:
        #     atom_pair = (
        #         torch.utils.checkpoint.checkpoint(
        #             self._before_atom_transformer, noisy_batch, use_reentrant=False,
        #         )
        #     )
        # else:
        #     atom_pair = self._before_atom_transformer(
        #         noisy_batch,
        #     )
        # atom_single_rep = self.atom_transformer(
        #     noisy_batch, atom_single_rep, atom_single_cond, atom_pair,
        # )

        if self.use_checkpoint:
            token_single_rep = torch.utils.checkpoint.checkpoint(
                self._scatter_atom_to_token,
                noisy_batch,
                # atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(noisy_batch)#, atom_single_rep)

        return token_single_rep


class RelativePositionEmbedding(nn.Module):
    """Relative position embedding module.

    Relative position embedding is used to encode the relative position of residues.

    Parameters
    ----------
    d_hidden: int
        Dimension of hidden layer.
    bins: int
        Number of bins for relative position encoding.
    init_std: float
        Standard deviation for embedding initialization.

    """

    def __init__(
        self,
        d_hidden: int,
        r_max: int = 32,
        s_max: int = 2,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.r_max = r_max
        self.s_max = s_max
        self.embed_rel_pos = Linear(73, d_hidden, init="default", bias=False)

    def forward(self, noisy_batch: NoisyBatch) -> Float[torch.Tensor, "B L L d_hidden"]:
        """Forward pass."""
        with torch.no_grad():
            asym_id = noisy_batch.scheme.residue_asym_id
            residue_idx = noisy_batch.scheme.residue_idx
            entity_id = noisy_batch.scheme.residue_entity_id
            sym_id = noisy_batch.scheme.residue_sym_id
            b_same_chain = asym_id[:, :, None] == asym_id[:, None, :]
            b_same_entity = entity_id[:, :, None] == entity_id[:, None, :]
            d_residue = (
                residue_idx[:, :, None] - residue_idx[:, None, :] + self.r_max
            )  # (B, L, L)
            d_residue = torch.clamp(d_residue, 0, 2 * self.r_max) * b_same_chain
            d_residue = d_residue + ~b_same_chain * (2 * self.r_max + 1)
            d_chain = sym_id[:, :, None] - sym_id[:, None, :] + self.s_max  # (B, L, L)
            d_chain = torch.clamp(d_chain, 0, 2 * self.s_max) * b_same_entity
            d_chain = d_chain + ~b_same_entity * (2 * self.s_max + 1)
            a_rel_pos = F.one_hot(
                d_residue.long(), num_classes=2 * self.r_max + 2,
            )  # (B, L, L, 2 * r_max + 1)
            a_rel_chain = F.one_hot(
                d_chain.long(), num_classes=2 * self.s_max + 2,
            )  # (B, L, L, 2 * s_max + 1)
            token_pair = torch.cat(
                [a_rel_pos, a_rel_chain, b_same_entity.unsqueeze(-1)], dim=-1,
            )  # (B, L, L, 2 * r_max + 2 + 2 * s_max + 2 + 1 = 73)
            token_pair = token_pair.float()
        return self.embed_rel_pos(token_pair)  # (B, L, L, d_hidden)


class InputFeatureEmbedder(nn.Module):
    """Input feature embedder module."""

    def __init__(
        self,
        common_config: CommonConfig,
        diffusion_config: DiffusionConfig,
    ) -> None:
        super().__init__()
        self.num_res_class = common_config.num_res_class
        self.use_checkpoint = common_config.use_checkpoint
        self.d_token_pair = common_config.d_token_pair
        self.atom_attention_encoder = InputAtomAttentionEncoder(
            common_config=common_config,
            diffusion_config=diffusion_config,
        )
        d_init = common_config.d_token_single_input
        self.to_token_init = Linear(
            d_init,
            common_config.d_token_single,
            init="default",
            bias=False,
        )
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
        self.add_token_bond = Linear(
            2,
            common_config.d_token_pair,
            init="default",
            bias=False,
        )
        # self.add_atom_bond = Linear(2, config.d_atom_pair, init="default") TODO

    @torch.no_grad()
    def _gen_bond_feature(
        self, noisy_batch: NoisyBatch,
    ) -> Float[torch.Tensor, "B L_token L_token 2"]:
        #  -> tuple[torch.Tensor, torch.Tensor]:
        B = noisy_batch.shape[0]
        L_token = noisy_batch.residue_length
        residue_bond = noisy_batch.structure.residue_bond.long()  # (B, n_residue_bond, 3)
        token_bond = torch.zeros(
            (B, L_token, L_token),
            device=noisy_batch.device,
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

        return F.one_hot(
            token_bond.long(),
            num_classes=2,
        ).to(dtype=noisy_batch.dtype)


    def forward(
        self, noisy_batch: NoisyBatch,
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_token_single_input"],
        Float[torch.Tensor, "B L_token d_token_single_init"],
        Float[torch.Tensor, "B L_token L_token d_token_pair"],
    ]:
        """Forward pass."""
        token_single_input = self.atom_attention_encoder(noisy_batch)

        residue_type = F.one_hot(
            noisy_batch.sequence.residue_type.long(), num_classes=self.num_res_class,
        ).to(token_single_input.device, dtype=token_single_input.dtype)

        token_single_input = torch.concat(
            [
                token_single_input,
                residue_type,
                # noisy_batch.msa.profile.to(dtype=token_single_input.dtype),
                noisy_batch.msa.deletion_mean.unsqueeze(-1).to(dtype=token_single_input.dtype),
            ],
            dim=-1,
        )

        token_single_init = self.to_token_init(token_single_input)
        token_left = self.to_token_pair_left(token_single_input)
        token_right = self.to_token_pair_right(token_single_input)
        token_pair_init = rearrange(token_left, "b l d -> b l 1 d") + rearrange(
            token_right, "b l d -> b 1 l d",
        )

        token_pair_init = token_pair_init + self.relative_position_embedder(noisy_batch)
        token_pair_init = token_pair_init + self.add_token_bond(
            self._gen_bond_feature(noisy_batch),
        )

        return (
            token_single_input,
            token_single_init,
            token_pair_init,
        )
