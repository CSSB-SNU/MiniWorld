import torch
import torch.nn.functional as F
from jaxtyping import Float
from team_gm import typecheck
from team_gm.modules.primitives import Linear
from torch import nn

from .diffusion_module import CommonConfig


class ContactMapHead(nn.Module):
    """ContactMap prediction head."""

    def __init__(self, config: CommonConfig) -> None:
        super().__init__()
        self.linear = Linear(config.d_token_pair, 2, bias=False)

    @typecheck
    def _symmetrize(
        self,
        x: Float[torch.Tensor, "B L L C"],
    ) -> Float[torch.Tensor, "B L L C"]:
        return (x + x.transpose(-2, -3)) / 2

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L C"],
    ) -> Float[torch.Tensor, "B L L D"]:
        """Forward pass."""
        pair = self._symmetrize(pair)
        x = self.linear(pair.float())
        return F.log_softmax(x, dim=-1)


class DistogramHead(nn.Module):
    """Distogram prediction head."""

    def __init__(self, config: CommonConfig) -> None:
        super().__init__()
        self.linear = Linear(config.d_token_pair, config.num_distogram_bins, bias=False)

    @typecheck
    def _symmetrize(
        self,
        x: Float[torch.Tensor, "B L L C"],
    ) -> Float[torch.Tensor, "B L L C"]:
        return (x + x.transpose(-2, -3)) / 2

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L C"],
    ) -> Float[torch.Tensor, "B L L D"]:
        """Forward pass."""
        pair = self._symmetrize(pair)
        x = self.linear(pair.float())
        return F.log_softmax(x, dim=-1)
