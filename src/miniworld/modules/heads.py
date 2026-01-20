import torch
from jaxtyping import Float
from team_gm import typecheck
from team_gm.modules.primitives import Linear
from torch import nn


class ContactMapHead(nn.Module):
    """ContactMap prediction head."""

    def __init__(self, d_token_pair: int) -> None:
        super().__init__()
        self.linear = Linear(d_token_pair, 2, bias=False, init="zero")

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
        return self.linear(pair.float())


class DistogramHead(nn.Module):
    """Distogram prediction head."""

    def __init__(self, d_token_pair: int, num_distogram_bins: int) -> None:
        super().__init__()
        self.linear = Linear(d_token_pair, num_distogram_bins, bias=False, init="zero")

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
        return self.linear(pair.float())
