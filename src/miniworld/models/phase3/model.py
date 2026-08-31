"""Phase3 model: FROZEN pair-only mini-SWA trunk + trainable EDM diffusion head.

``Phase3Model`` subclasses the phase2 :class:`MiniSWAModel` so the trunk
submodules keep the *exact* attribute/parameter names of the epoch-900
checkpoint (``input_feature_embedder``, ``add_pair_recycle``, ``temp_embedder``,
``msa_module``, ``pairformer_blocks``, ``distogram_head``). The epoch-900
``model_state_dict`` therefore loads into these inherited submodules with no
missing/unexpected keys.

On top of the trunk it adds:

  * ``to_token_single_trunk`` — a trainable Linear projecting the trunk's input
    single embedding (``token_single_input``, ``d_single_token_input``) to
    ``d_single``. The pair-only trunk has no single track, so this projection
    supplies the diffusion single-conditioning that AF3 would otherwise take
    from the trunk single (used both as the ``DiffusionConditioning`` single
    slot and as the atom encoder's ``enc_token_single``).
  * ``diffusion_module`` — the ESMFold2 SWA atom DiT (``swa_atom_config``) +
    AF3 token DiT (``token_dit``) diffusion module.

Only ``to_token_single_trunk`` + ``diffusion_module`` train; the trunk is frozen
(``requires_grad=False`` via the client's param policy) and kept in eval mode so
its dropout/recycle behaviour is deterministic.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, Field
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer, SWAAtomTransformer
from team_gm.modules.primitives import Linear
from torch import nn

from miniworld.configs import SharedConfig
from miniworld.configs.models import AtomSWAConfig
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.modules.diffusion_module import DiffusionConditioning, DiffusionModule

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Bool, Float

    from miniworld.data.features import (
        MSAFeatures,
        ReferenceFeatures,
        SchemeFeatures,
        SequenceFeatures,
        StructureFeatures,
        TemplateFeatures,
    )


logger = logging.getLogger(__name__)


class Phase3Model(MiniSWAModel):
    """Pair-only mini-SWA trunk (frozen) + EDM diffusion module (trainable)."""

    # Inherited (phase2) trunk submodules — their keys match the epoch-900
    # checkpoint and they stay frozen + eval in phase3. ``distogram_head`` is
    # loaded (present in the checkpoint) but unused by the diffusion head.
    _TRUNK_MODULE_NAMES = (
        "input_feature_embedder",
        "add_pair_recycle",
        "temp_embedder",
        "msa_module",
        "pairformer_blocks",
        "distogram_head",
    )

    class DiffusionConfig(BaseModel):
        """Configuration for the diffusion module.

        Phase3 default: atom DiT = ESMFold2-style SWA + 3D RoPE (``atom_swa``),
        token DiT = AF3-style pair-bias ``DiffusionTransformer`` (``token_dit``).
        ``atom_dit`` is only used when ``atom_swa`` is null (AF3 pair-bias atom
        attention fallback); with ``atom_swa`` set it is parsed but unused.
        """

        atom_dit: DiffusionTransformer.Config
        token_dit: DiffusionTransformer.Config
        dit_cond: DiffusionConditioning.Config
        atom_swa: SWAAtomTransformer.Config | None = Field(
            default_factory=SWAAtomTransformer.Config,
        )

    class Config(BaseModel):
        """Configuration for the phase3 model."""

        shared: SharedConfig
        # non-atom parts of the input feature embedder still use this config
        input_feat_embbeder: DiffusionTransformer.Config
        # ESMFold2-style atom SWA/3D-RoPE front-end for the trunk input embedder
        atom_swa: AtomSWAConfig
        trunk: MiniSWAModel.TrunkConfig
        diffusion: Phase3Model.DiffusionConfig
        # Freeze the trunk (requires_grad handled by the client's param policy;
        # this flag keeps the trunk modules in eval mode and runs the trunk under
        # ``torch.no_grad`` during forward).
        freeze_trunk: bool = True

    def __init__(self, config: Config) -> None:
        # Build the exact phase2 trunk via the parent, so submodule/param names
        # match the epoch-900 checkpoint.
        trunk_config = MiniSWAModel.Config(
            shared=config.shared,
            input_feat_embbeder=config.input_feat_embbeder,
            atom_swa=config.atom_swa,
            trunk=config.trunk,
        )
        super().__init__(trunk_config)
        # Replace the trunk-only config the parent stored with the phase3 config.
        self.config = config
        self.freeze_trunk = config.freeze_trunk

        # Optional compiled (inductor cudagraph-trees) frozen-trunk conditioning
        # callable. None = OFF (default): the normal eager/whole-model-compile path
        # is used. Set via :meth:`enable_trunk_cudagraph`; when set, the trunk
        # outputs are cloned before flowing into the grad diffusion path so a later
        # cudagraph replay never overwrites a tensor the grad path still reads.
        self._trunk_compiled = None

        # Pair-only trunk has no single track: derive the diffusion single
        # conditioning from the trunk input single embedding.
        self.to_token_single_trunk = Linear(
            config.shared.d_single_token_input,
            config.shared.d_single,
            bias=False,
            init="default",
        ).to(torch.float32)

        # ESMFold2 3D-RoPE atom DiT (swa_atom_config) + AF3 token DiT (token_dit).
        self.diffusion_module = DiffusionModule(
            config.shared,
            config.diffusion.atom_dit,
            config.diffusion.token_dit,
            config.diffusion.dit_cond,
            swa_atom_config=config.diffusion.atom_swa,
        ).to(torch.float32)

        if self.freeze_trunk:
            self._set_trunk_eval()

    # -- trunk freeze helpers ------------------------------------------------
    def _trunk_modules(self):
        for name in self._TRUNK_MODULE_NAMES:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                yield mod

    def _set_trunk_eval(self) -> None:
        for mod in self._trunk_modules():
            mod.eval()

    def train(self, mode: bool = True) -> Phase3Model:  # noqa: FBT001, FBT002
        """Keep the frozen trunk in eval even when the model is set to train."""
        super().train(mode)
        if self.freeze_trunk:
            self._set_trunk_eval()
        return self

    # -- trunk cudagraph (optional, flag-gated) -----------------------------
    def enable_trunk_cudagraph(self, mode: str = "reduce-overhead") -> None:
        """Compile the FROZEN trunk conditioning path with inductor cudagraphs.

        Sets ``self._trunk_compiled`` to a ``torch.compile`` wrapper of
        :meth:`_condition_impl` using ``mode`` (typically ``"reduce-overhead"``,
        i.e. cudagraph-trees). :meth:`forward` then clones the trunk outputs
        before handing them to the grad diffusion path (see note in ``forward``).

        No-op when ``freeze_trunk`` is False: gradients must flow through the
        trunk in that case, so a cudagraph (which requires ``no_grad`` static
        replay) is inappropriate — the caller should fall back to normal compile.
        """
        if not self.freeze_trunk:
            logger.warning(
                "enable_trunk_cudagraph is a no-op when freeze_trunk=False "
                "(grad must flow through the trunk); falling back to normal "
                "compile. Requested mode=%s.",
                mode,
            )
            return
        self._trunk_compiled = torch.compile(
            self._condition_impl,
            mode=mode,
            dynamic=False,
        )

    # -- trunk (condition) forward ------------------------------------------
    def condition_forward(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
        template: TemplateFeatures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk conditioning, dispatching to the compiled trunk if set.

        With ``self._trunk_compiled is None`` (default, flag OFF) this calls
        :meth:`_condition_impl` directly — behaviour is byte-for-byte unchanged.
        When the frozen-trunk cudagraph is enabled it routes through the compiled
        callable instead; the caller (:meth:`forward` / ``ModelWrapper``) is
        responsible for cloning the returned tensors before they reach a grad
        path that outlives the next cudagraph replay.
        """
        if self._trunk_compiled is not None:
            return self._trunk_compiled(
                msa,
                reference,
                scheme,
                sequence,
                structure,
                template,
            )
        return self._condition_impl(
            msa,
            reference,
            scheme,
            sequence,
            structure,
            template,
        )

    def _condition_impl(
        self,
        msa: MSAFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
        template: TemplateFeatures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the pair-only mini trunk and return (token_single_input, token_pair).

        Mirrors :meth:`MiniSWAModel.forward` (same ``_embed`` / ``_trunk_step``
        recycle loop) but returns the trunk conditioning tensors instead of
        distogram logits. The frozen trunk always uses the full recycle depth for
        the best, deterministic conditioning (warmup can still pin a fixed count
        via ``_forced_n_recycle``).
        """
        n_recycle = (
            self._forced_n_recycle
            if self._forced_n_recycle is not None
            else self.n_recycle_max
        )

        (
            token_pair_init_bf16,
            token_single_input_bf16,
            msa_feat,
            msa_mask,
            token_mask,
        ) = self._embed(msa, reference, scheme, sequence, structure)

        token_pair = torch.zeros_like(token_pair_init_bf16)
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                token_pair = self._trunk_step(
                    token_pair,
                    token_pair_init_bf16,
                    token_single_input_bf16,
                    msa_feat,
                    msa_mask,
                    token_mask,
                    scheme.token_asym_id,
                    template,
                )
        return (
            token_single_input_bf16.to(torch.float32),
            token_pair.to(torch.float32),
        )

    def diffusion_forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Project the single conditioning and run the diffusion module."""
        token_single_trunk = self.to_token_single_trunk(token_single_input)
        return self.diffusion_module(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

    def forward(
        self,
        msa: MSAFeatures,
        template: TemplateFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Frozen trunk conditioning -> diffusion denoising update."""
        trunk_ctx = torch.no_grad() if self.freeze_trunk else nullcontext()
        with trunk_ctx:
            token_single_input, token_pair_trunk = self.condition_forward(
                msa,
                reference,
                scheme,
                sequence,
                structure,
                template,
            )
        # When the trunk runs under inductor cudagraph-trees, its outputs are
        # cudagraph-managed buffers that a later replay would overwrite. Clone
        # them before they flow into the (grad) diffusion path so the grad path
        # never reads a tensor that has been overwritten by a subsequent replay.
        if self._trunk_compiled is not None:
            token_single_input = token_single_input.clone()
            token_pair_trunk = token_pair_trunk.clone()
        return self.diffusion_forward(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            t_emb,
            token_single_input,
            token_pair_trunk,
        )


# Convenience alias so the entrypoint/client can ``import Model`` like the other
# model packages (af3_like / miniworld).
Model = Phase3Model


class ModelWrapper(nn.Module):
    """Wrapper for :class:`Phase3Model` to drive the EDM diffusion solver."""

    def __init__(self, model: Phase3Model) -> None:
        super().__init__()
        self.conditioned_forwarded = False
        self.model = model

    @torch.no_grad()
    def prepare_condition(
        self,
        msa: MSAFeatures,
        template: TemplateFeatures,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        sequence: SequenceFeatures,
        structure: StructureFeatures,
    ) -> None:
        """Run the trunk once and cache the diffusion conditioning tensors."""
        if self.conditioned_forwarded:
            msg = "Conditioned forward is already done."
            raise ValueError(msg)

        token_single_input, token_pair_trunk = self.model.condition_forward(
            msa,
            reference,
            scheme,
            sequence,
            structure,
            template,
        )
        # Same cudagraph-managed-output guard as ``Phase3Model.forward``: these
        # conditioning tensors are cached and reused across every solver step, so
        # clone them off any cudagraph buffer before caching.
        if self.model._trunk_compiled is not None:  # noqa: SLF001
            token_single_input = token_single_input.clone()
            token_pair_trunk = token_pair_trunk.clone()
        token_single_trunk = self.model.to_token_single_trunk(token_single_input)
        self.conditioned_forwarded = True
        self.condition = {
            "reference": reference,
            "scheme": scheme,
            "structure": structure,
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
        }

    def forward(
        self,
        x_t: Float[torch.Tensor, "N_str L 3"],
        t_emb: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "N_str L 3"]:
        """Diffusion denoising step over the cached conditioning."""
        if not self.conditioned_forwarded:
            msg = "Conditioned forward must be called before forward pass."
            raise ValueError(msg)

        n_str = x_t.shape[0]
        atom_mask = self.condition["structure"].atom_mask  # (B=1, L_atom)
        x_mask = atom_mask.unsqueeze(0).expand(n_str, -1, -1)  # (A, B=1, L_atom)
        x_update = self.model.diffusion_module(
            self.condition["reference"],
            self.condition["scheme"],
            self.condition["structure"],
            x_t.unsqueeze(1),  # (N_str, L, 3) -> (A=N_str, B=1, L, 3)
            x_mask,
            t_emb[None, None, None, None],  # (,) -> (1, 1, 1, 1), broadcasts A,B
            self.condition["token_single_input"],
            self.condition["token_single_trunk"],
            self.condition["token_pair_trunk"],
        )
        return x_update.squeeze(1)  # (A=N_str, B=1, L, 3) -> (N_str, L, 3)


@dataclass
class InferenceOutput:
    """Output of the phase3 model inference."""

    atom_pos_pred: torch.Tensor  # (N_str, L, 3)
    model_traj: np.ndarray  # (N_str, T, L, 3)
    inter_traj: np.ndarray  # (N_str, T, L, 3)
