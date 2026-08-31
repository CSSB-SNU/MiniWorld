"""Phase4 model: FROZEN phase3 structure model + trainable confidence head.

``Phase4Model`` subclasses :class:`~miniworld.models.phase3.model.Phase3Model`, so it
inherits the exact trunk + diffusion submodules (their keys match the phase3
checkpoint). It adds a single new trainable module:

  * ``confidence_head`` — :class:`~miniworld.modules.confidence_head.ConfidenceHead`
    (pLDDT / PAE / PDE) over the frozen trunk's pair + the predicted structure.

Both the trunk AND the diffusion module are frozen (loaded from the phase3 checkpoint
and kept in eval mode); only ``confidence_head`` trains. The predicted structure that
the head scores is produced by the frozen diffusion model — see
``phase4.client.Client.predict_structure`` (the diffusion-step seam). This model does
NOT run the diffusion rollout itself; it only exposes :meth:`confidence_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

from miniworld.models.phase3.model import Phase3Model
from miniworld.modules.confidence_head import ConfidenceHead


class Phase4Model(Phase3Model):
    """Frozen phase3 structure model + trainable confidence head."""

    # Freeze the diffusion head too (phase3 only froze the trunk). These modules are
    # kept in eval mode; the client's param_policy sets requires_grad=False on them.
    _TRUNK_MODULE_NAMES = (
        *Phase3Model._TRUNK_MODULE_NAMES,  # noqa: SLF001
        "to_token_single_trunk",
        "diffusion_module",
    )

    class Config(Phase3Model.Config):
        """Phase4 config = phase3 config + confidence head."""

        confidence: ConfidenceHead.Config = ConfidenceHead.Config()

    def __init__(self, config: Config) -> None:
        # Build the phase3 structure model (trunk + diffusion) via the parent, from
        # the phase3 subset of the config. freeze_trunk semantics + eval mode extend
        # to the diffusion module through the overridden _TRUNK_MODULE_NAMES.
        phase3_config = Phase3Model.Config(
            shared=config.shared,
            input_feat_embbeder=config.input_feat_embbeder,
            atom_swa=config.atom_swa,
            trunk=config.trunk,
            diffusion=config.diffusion,
            freeze_trunk=config.freeze_trunk,
        )
        super().__init__(phase3_config)
        self.config = config

        self.confidence_head = ConfidenceHead(config.confidence).to(torch.float32)

    @typecheck
    def forward(
        self,
        token_single_input: Float[torch.Tensor, "N L d_single_input"],
        token_pair: Float[torch.Tensor, "N L L d_pair"],
        pred_rep_dist: Float[torch.Tensor, "N L L"],
        token_mask: Bool[torch.Tensor, "N L"],
        atom_to_token_idx: Int[torch.Tensor, "N L_atom"],
    ) -> dict[str, torch.Tensor]:
        """Run ONLY the confidence head over the frozen conditioning + prediction.

        This overrides :meth:`Phase3Model.forward` (the diffusion path), which phase4
        never calls — the frozen structure prediction goes through
        :class:`~miniworld.models.phase3.model.ModelWrapper` (``condition_forward`` +
        ``diffusion_module``), not ``forward``. Routing the trainable path through
        ``forward`` keeps DDP gradient sync intact for the confidence head.
        """
        return self.confidence_head(
            token_pair,
            token_single_input,
            pred_rep_dist,
            token_mask,
            atom_to_token_idx,
        )

    def confidence_forward(
        self,
        token_single_input: torch.Tensor,
        token_pair: torch.Tensor,
        pred_rep_dist: torch.Tensor,
        token_mask: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Alias for :meth:`forward` (used at inference, off the DDP path)."""
        return self.forward(
            token_single_input, token_pair, pred_rep_dist, token_mask, atom_to_token_idx,
        )


# Convenience alias so the client/entrypoint can ``import Model``.
Model = Phase4Model


@dataclass
class ConfidenceOutput:
    """Confidence-head inference output (expected-value scalars + raw logits)."""

    plddt: torch.Tensor        # [N, L_atom] expected per-atom lDDT (0-100)
    pae: torch.Tensor          # [N, L, L] expected aligned error (Å)
    plddt_logits: torch.Tensor
    pae_logits: torch.Tensor
    pde_logits: torch.Tensor
