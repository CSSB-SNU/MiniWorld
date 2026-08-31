"""AF3-style confidence head (pLDDT / PAE / PDE) for MiniWorld phase4.

Given the FROZEN pair-only trunk's pair representation ``z`` (``d_pair``) and input
single ``s_in`` (``d_single_token_input``), plus a *predicted* structure's per-token
representative-atom distances, this head:

  1. re-injects the input single into the pair (AF3 ``Linear(s_i)+Linear(s_j)``),
  2. adds a binned-distance embedding of the predicted structure (the structure
     signal the confidence head scores),
  3. seeds a single track ``s = Linear(s_in)`` and refines BOTH single and pair with a
     full :class:`~team_gm.modules.Pairformer` (``use_single=True``) — the confidence
     mini-trunk, AF3-style (updates ``s_i`` and ``z_ij`` per block),
  4. reads three heads:
       * **PAE** — per token-pair predicted aligned error (asymmetric), ``n_pae_bins``.
       * **PDE** — per token-pair predicted distance error (symmetric), ``n_pde_bins``.
       * **pLDDT** — per-atom predicted lDDT, ``n_plddt_bins``, read from the refined
         single ``s_i`` (``Linear``) then broadcast token -> atom.

The frozen trunk is pair-only (no ``d_single`` single track), so the confidence single
is *seeded* from the input single ``s_in`` (``to_single_init``) and then refined by the
Pairformer — recovering AF3's single-carrying confidence trunk without a trunk single.
Everything is trained ONLY through this head; the structure model that produced ``z`` /
the predicted coordinates stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules.primitives import Linear
from torch import nn

from miniworld.modules.mini_pairformer import MiniPairformer


class ConfidenceHead(nn.Module):
    """pLDDT / PAE / PDE confidence head over a frozen pair representation."""

    class Config(BaseModel):
        """Configuration for the confidence head."""

        d_pair: int = 128
        d_single: int = 384        # confidence single track (seeded from d_single_input)
        d_single_input: int = 441  # d_single_token_input (input-embedder single)
        # confidence mini-trunk: MiniPairformer WITH its single track (use_single) —
        # lightweight pair core (bidir trimul) + AttentionPairBias single per block,
        # so pLDDT reads a refined single (AF3-style) rather than a pair reduction.
        n_block: int = 4
        n_head_attention: int = 16
        n_checkpoint_segments: int | None = None
        p_drop: float = 0.0
        # output bins
        n_plddt_bins: int = 50   # lDDT in [0, 1] -> 50 bins (0.02 each)
        n_pae_bins: int = 64     # aligned error [0, pae_max] Å
        n_pde_bins: int = 64     # distance error [0, pde_max] Å
        pae_max: float = 32.0
        pde_max: float = 32.0
        # predicted-structure distance embedding (input signal)
        n_dist_bins: int = 64
        dist_min: float = 2.0
        dist_max: float = 22.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        d_pair = config.d_pair
        d_single = config.d_single

        # AF3 single re-injection into the pair track.
        self.to_pair_single_i = Linear(config.d_single_input, d_pair, bias=False, init="normal")
        self.to_pair_single_j = Linear(config.d_single_input, d_pair, bias=False, init="normal")
        # Predicted-structure distance embedding.
        self.to_pair_dist = Linear(config.n_dist_bins, d_pair, bias=False, init="normal")
        # Seed the confidence single track from the input single (the pair-only trunk
        # has no d_single track, so we project s_inputs -> d_single here).
        self.to_single_init = Linear(config.d_single_input, d_single, bias=False, init="normal")

        # Confidence mini-trunk: MiniPairformer with its single track enabled — updates
        # both single s and pair z per block. The engine trimul/attention ops hold bf16
        # weights, so cast the stack to bf16 (as the trunk does) and cast s/z back to
        # fp32 for the output projections.
        self.pairformer = MiniPairformer(
            MiniPairformer.Config(
                d_pair=d_pair,
                d_single=d_single,
                use_single=True,
                n_head_attention=config.n_head_attention,
                p_drop=config.p_drop,
                n_block=config.n_block,
                n_checkpoint_segments=config.n_checkpoint_segments,
            ),
        ).to(torch.bfloat16)

        # Heads. PAE asymmetric (no symmetrize), PDE symmetric, pLDDT from the single.
        self.to_pae = Linear(d_pair, config.n_pae_bins, bias=False, init="zero")
        self.to_pde = Linear(d_pair, config.n_pde_bins, bias=False, init="zero")
        self.to_plddt = Linear(d_single, config.n_plddt_bins, bias=False, init="zero")

        self._dist_edges: torch.Tensor
        self.register_buffer(
            "_dist_edges",
            torch.linspace(config.dist_min, config.dist_max, config.n_dist_bins - 1),
            persistent=False,
        )

    def _embed_pred_distance(
        self,
        pred_rep_dist: Float[torch.Tensor, "N L L"],
        pair_mask: Bool[torch.Tensor, "N L L"],
    ) -> Float[torch.Tensor, "N L L d_pair"]:
        """Bin the predicted representative-atom distances and embed them."""
        edges = self._dist_edges.to(pred_rep_dist.device)
        binned = torch.bucketize(pred_rep_dist, edges)  # [N, L, L] in [0, n_dist_bins-1]
        onehot = F.one_hot(binned, num_classes=self.config.n_dist_bins).to(pred_rep_dist.dtype)
        onehot = onehot * pair_mask.unsqueeze(-1)
        return self.to_pair_dist(onehot)

    @typecheck
    def forward(
        self,
        token_pair: Float[torch.Tensor, "N L L d_pair"],
        token_single_input: Float[torch.Tensor, "N L d_single_input"],
        pred_rep_dist: Float[torch.Tensor, "N L L"],
        token_mask: Bool[torch.Tensor, "N L"],
        atom_to_token_idx: Int[torch.Tensor, "N L_atom"],
    ) -> dict[str, torch.Tensor]:
        """Confidence logits from frozen pair + predicted-structure distances.

        Returns ``{"plddt": [N, L_atom, n_plddt_bins], "pae": [N, L, L, n_pae_bins],
        "pde": [N, L, L, n_pde_bins]}``.
        """
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]  # [N, L, L]

        s_in = token_single_input.float()
        z = token_pair.float()
        z = z + self.to_pair_single_i(s_in)[:, :, None, :]
        z = z + self.to_pair_single_j(s_in)[:, None, :, :]
        z = z + self._embed_pred_distance(pred_rep_dist, pair_mask)
        z = z * pair_mask.unsqueeze(-1)
        s = self.to_single_init(s_in)                     # seed single [N, L, d_single]

        # The Pairformer engine ops hold bf16 weights (as in the trunk); run the
        # mini-trunk in bf16 and cast s/z back to fp32 for the head projections + CE.
        z, s = self.pairformer(z.to(torch.bfloat16), s.to(torch.bfloat16), token_mask)
        z = z.float()
        s = s.float()  # [N, L, d_single] — refined single track

        pae_logits = self.to_pae(z)                       # asymmetric
        pde_logits = self.to_pde((z + z.transpose(-2, -3)) / 2)  # symmetric
        plddt_tok = self.to_plddt(s)                      # [N, L, n_plddt_bins] from single

        # Broadcast per-token pLDDT logits to atoms.
        idx = atom_to_token_idx.clamp(min=0, max=plddt_tok.shape[1] - 1)  # [N, L_atom]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, plddt_tok.shape[-1])
        plddt_atom = torch.gather(plddt_tok, 1, gather_idx)  # [N, L_atom, n_plddt_bins]

        return {"plddt": plddt_atom, "pae": pae_logits, "pde": pde_logits}

    # -- expected-value decoders (inference-time scalar confidences) ---------
    @torch.no_grad()
    def expected_plddt(self, plddt_logits: Float[torch.Tensor, "N L_atom B"]) -> torch.Tensor:
        """Expected per-atom lDDT in [0, 100] from the pLDDT logits."""
        n = self.config.n_plddt_bins
        centers = (torch.arange(n, device=plddt_logits.device) + 0.5) / n  # bin centers in [0,1]
        p = torch.softmax(plddt_logits.float(), dim=-1)
        return (p * centers).sum(dim=-1) * 100.0

    @torch.no_grad()
    def expected_pae(self, pae_logits: Float[torch.Tensor, "N L L B"]) -> torch.Tensor:
        """Expected aligned error (Å) per token pair from the PAE logits."""
        n = self.config.n_pae_bins
        step = self.config.pae_max / n
        centers = (torch.arange(n, device=pae_logits.device) + 0.5) * step
        p = torch.softmax(pae_logits.float(), dim=-1)
        return (p * centers).sum(dim=-1)
