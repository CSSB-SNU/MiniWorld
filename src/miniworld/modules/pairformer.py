import torch
from jaxtyping import Bool, Float
from torch import nn

from team_gm import typecheck

from team_gm.modules.pairformer import PairformerBlock

class PairformerBlockGroup(nn.Module):
    """A group of Pairformer blocks with shared configurations."""

    class Config(PairformerBlock.Config):
        """Configuration for PairformerBlockGroup module."""

        n_block: int = 4

    def __init__(self, config: Config, use_checkpoint: bool) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.pairformer_blocks = nn.ModuleList(
            [PairformerBlock(config) for _ in range(config.n_block)],
        )

    def _forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        for block in self.pairformer_blocks:
            pair, single = block(pair, single, mask)
        return pair, single

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""

        if self.use_checkpoint:
            pair, single = torch.utils.checkpoint.checkpoint(
                self._forward,
                pair,
                single,
                mask,
            )
        else:
            for block in self.pairformer_blocks:
                pair, single = block(pair, single, mask)
        return pair, single

class Pairformer(nn.Module):
    """The Pairformer stack consisting of multiple Pairformer blocks."""

    class Config(PairformerBlock.Config):
        """Configuration for Pairformer module."""

        n_block: int = 4
        n_block_per_group: int = 4
        use_checkpoint: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()

        print(f"test pairformer config: {config}")

        if config.n_block % config.n_block_per_group != 0:
            msg = "n_block must be divisible by n_block_per_group"
            raise ValueError(msg)
        self.n_block_groups = config.n_block // config.n_block_per_group
        self.pairformer_groups = nn.ModuleList(
            [
                PairformerBlockGroup(
                    PairformerBlockGroup.Config(
                        **config.model_dump(exclude={"n_block"}),
                    ),
                    use_checkpoint=config.use_checkpoint,
                )
                for _ in range(self.n_block_groups)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        for group in self.pairformer_groups:
            pair, single = group(pair, single, mask)
        return pair, single
