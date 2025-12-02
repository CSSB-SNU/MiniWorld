from typing import Literal

import torch
from jaxtyping import Float
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules.primitives import Linear
from torch import nn

from miniworld.modules.primitives import LayerNorm


class AllAtomModule(nn.Module):
    """All-atom update resnet module."""

    class Config(BaseModel):
        """Configuration for AllAtomModule.

        ----------
        n_block: int
            Number of residual blocks.
        d_single: int
            Dimension of single representation.
        d_hidden: int
            Dimension of hidden layer.
        atom_num: int
            Number of all-atoms.
        implementation: str
            Implementation type, either "pytorch" or "triton".
        """

        n_block: int = 2
        d_single: int = 384
        d_hidden: int = 128
        atom_num: int = 27  # 27 atoms in RNA all-atom
        implementation: Literal["pytorch", "triton"] = "pytorch"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.linear_in = Linear(config.d_single, config.d_hidden)
        self.linear_initial = Linear(config.d_single, config.d_hidden)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(config.d_hidden, implementation=config.implementation),
                    nn.GELU(),
                    Linear(config.d_hidden, config.d_hidden, init="relu"),
                    nn.GELU(),
                    Linear(config.d_hidden, config.d_hidden, init="zero"),
                )
                for _ in range(config.n_block)
            ],
        )
        self.mlp_out = nn.Sequential(
            LayerNorm(config.d_hidden, implementation=config.implementation),
            nn.GELU(),
            Linear(config.d_hidden, config.atom_num * 3),
        )

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "* L d_single"],
        init_single: Float[torch.Tensor, "* L d_single"],
    ) -> Float[torch.Tensor, "* L N 3"]:
        """Forward pass."""
        single = self.linear_in(self.relu(single))
        single = single + self.linear_initial(self.relu(init_single))

        for block in self.blocks:
            single = single + block(single)
        single = self.mlp_out(self.relu(single))
        return single.view(single.shape[:-1] + (-1, 3))
