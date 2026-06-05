"""Selective freeze / re-init / load policy for model parameters.

Used to continue training from a pre-trained checkpoint while

  * **freezing** a subset of layers (no gradient, excluded from optimizer),
  * **re-initializing** another subset from scratch (e.g. layers whose old
    weights are stale after a semantic bug fix),
  * keeping the rest at their checkpoint values and continuing to train them.

Patterns use **prefix-glob matching**: a pattern matches a parameter name
iff ``name == pattern`` or ``name.startswith(pattern + ".")``. Naming a
module (``foo.bar``) covers every parameter beneath it (``foo.bar.weight``,
``foo.bar.0.bias``, ...). Naming a single parameter (``foo.bar.weight``)
covers just that tensor.

If a single parameter is matched by patterns in two different categories
the policy raises :class:`ParamPolicyConflictError`. Parameters with no
matching pattern fall back to ``default`` (``load_existing`` by default).
Parameters in ``load_existing`` whose checkpoint entry is missing or
shape-mismatched fall back to ``reinit`` and emit a warning.

The ``default`` may also be set to ``freeze_loaded``: every parameter that
is present in the checkpoint (with a matching shape) is loaded and
**frozen**, while every parameter missing from the checkpoint is
re-initialized and left **trainable**. This is the natural policy when
attaching a new head/module to a frozen pre-trained trunk — the loaded
trunk weights freeze, the brand-new parameters train — without having to
enumerate module names. Explicit ``freeze``/``reinit``/``load_existing``
patterns still take precedence over the ``freeze_loaded`` default.

Apply the policy between model construction and optimizer construction,
then build the optimizer over :func:`trainable_parameters` so frozen
weights are excluded from the optimizer state entirely.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import nn

logger = logging.getLogger(__name__)

ActionKey = Literal["freeze", "reinit", "load_existing"]
# The ``default`` may additionally request ``freeze_loaded`` (resolved
# per-parameter against the checkpoint inside :func:`apply_param_policy`).
DefaultKey = Literal["freeze", "reinit", "load_existing", "freeze_loaded"]


class ParamPolicyConfig(BaseModel):
    """Per-parameter policy: which params to freeze, re-init, or load.

    Set ``enabled=False`` (default) to skip the policy entirely — the run
    script falls back to the standard ``load_state_dict``-everything path.
    """

    enabled: bool = False
    freeze: list[str] = Field(default_factory=list)
    reinit: list[str] = Field(default_factory=list)
    load_existing: list[str] = Field(default_factory=list)
    default: DefaultKey = "load_existing"


class ParamPolicyConfigError(ValueError):
    """Raised for invalid policy configuration (typo, missing checkpoint, etc.)."""


class ParamPolicyConflictError(ParamPolicyConfigError):
    """A parameter is matched by patterns in multiple action categories."""


def _matches(name: str, pattern: str) -> bool:
    """Prefix-glob: ``name == pattern`` or ``name.startswith(pattern + '.')``."""
    pattern = pattern.rstrip(".")
    if not pattern:
        return False
    return name == pattern or name.startswith(pattern + ".")


def _classify_one(name: str, policy: ParamPolicyConfig) -> DefaultKey:
    matched: list[ActionKey] = []
    if any(_matches(name, p) for p in policy.freeze):
        matched.append("freeze")
    if any(_matches(name, p) for p in policy.reinit):
        matched.append("reinit")
    if any(_matches(name, p) for p in policy.load_existing):
        matched.append("load_existing")

    if len(matched) > 1:
        msg = (
            f"Parameter {name!r} is matched by patterns in multiple action "
            f"categories: {matched}. Each parameter must belong to exactly "
            f"one category (or none, to fall back to the default)."
        )
        raise ParamPolicyConflictError(msg)
    if matched:
        return matched[0]
    return policy.default


def classify_params(
    model: nn.Module,
    policy: ParamPolicyConfig,
) -> dict[str, DefaultKey]:
    """Return ``{param_name: action}`` for every parameter in ``model``.

    The action is an explicit ``freeze``/``reinit``/``load_existing`` when a
    pattern matches, otherwise ``policy.default`` (which may be the
    checkpoint-dependent ``freeze_loaded`` — resolved in
    :func:`apply_param_policy`).
    """
    return {name: _classify_one(name, policy) for name, _ in model.named_parameters()}


def validate_policy(
    model: nn.Module,
    policy: ParamPolicyConfig,
) -> None:
    """Sanity-check the policy against the model.

    Raises if any pattern matches zero parameters (likely a typo). Also
    runs the conflict check by exercising :func:`classify_params`.
    """
    if not policy.enabled:
        return

    all_names = [name for name, _ in model.named_parameters()]
    unmatched: list[tuple[str, str]] = []
    for action in ("freeze", "reinit", "load_existing"):
        for pat in getattr(policy, action):
            if not any(_matches(n, pat) for n in all_names):
                unmatched.append((action, pat))

    if unmatched:
        lines = ["Pattern(s) matched zero parameters (typo?):"]
        for action, pat in unmatched:
            lines.append(f"  {action}: {pat!r}")
        raise ParamPolicyConfigError("\n".join(lines))

    # Will raise on conflict
    classify_params(model, policy)


def _reinit_param_inplace(model: nn.Module, name: str) -> None:
    """Re-initialize a single parameter in-place using the project's init."""
    # Lazy imports — the policy module shouldn't hard-depend on team_gm
    # at import time (callers may not have it installed for tests).
    from team_gm.modules.primitives import (
        InitType,
        LayerNorm as ProjectLayerNorm,
        Linear as ProjectLinear,
    )

    parts = name.split(".")
    module: nn.Module = model
    for p in parts[:-1]:
        module = getattr(module, p)
    leaf = parts[-1]

    with torch.no_grad():
        if isinstance(module, ProjectLinear):
            if leaf == "weight":
                InitType[module.init.upper()].apply(module.weight)
            elif leaf == "bias" and module.bias is not None:
                if module.init == "gating":
                    InitType["ONE"].apply(module.bias)
                elif module.init in {"zero", "one"}:
                    InitType["ZERO"].apply(module.bias)
                else:
                    # Match ``nn.Linear.reset_parameters`` bias init
                    # (uniform within +/- 1/sqrt(fan_in)).
                    bound = 1.0 / math.sqrt(module.in_features)
                    nn.init.uniform_(module.bias, -bound, bound)
            else:
                msg = (
                    f"Unexpected leaf {leaf!r} on Linear (param: {name}); "
                    f"expected 'weight' or 'bias'."
                )
                raise NotImplementedError(msg)

        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, ProjectLayerNorm)):
            if leaf == "weight":
                nn.init.ones_(module.weight)
            elif leaf == "bias" and getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            else:
                msg = f"Unexpected leaf {leaf!r} on norm (param: {name})"
                raise NotImplementedError(msg)

        elif isinstance(module, nn.Linear):
            # Vanilla torch Linear (not project's). Both weight and bias get
            # reset together — we call ``reset_parameters`` and rely on the
            # outer loop to skip the sibling param when it comes up.
            module.reset_parameters()

        else:
            reset = getattr(module, "reset_parameters", None)
            if reset is None:
                msg = (
                    f"No re-init recipe for {type(module).__name__}.{leaf} "
                    f"(param: {name}). Add a case to _reinit_param_inplace()."
                )
                raise NotImplementedError(msg)
            reset()


def apply_param_policy(
    model: nn.Module,
    ckpt_state_dict: dict[str, torch.Tensor] | None,
    policy: ParamPolicyConfig,
    *,
    log: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Apply the policy to ``model`` in-place. Returns a summary."""
    if log is None:
        log = logger

    if not policy.enabled:
        msg = "apply_param_policy called with enabled=False"
        raise ParamPolicyConfigError(msg)

    validate_policy(model, policy)
    classifications = classify_params(model, policy)

    named_params = dict(model.named_parameters())
    ckpt = ckpt_state_dict or {}

    summary: dict[str, list[str]] = {"loaded": [], "reinit": [], "frozen": []}
    reinit_targets: list[str] = []
    to_load: dict[str, torch.Tensor] = {}
    # Params that must end up frozen (filled as we resolve actions below).
    freeze_targets: set[str] = set()

    def _ckpt_loadable(name: str, param: torch.Tensor) -> bool:
        ckpt_value = ckpt.get(name)
        return ckpt_value is not None and ckpt_value.shape == param.shape

    for name, action in classifications.items():
        param = named_params[name]
        ckpt_value = ckpt.get(name)

        # ``freeze_loaded`` resolves per-parameter against the checkpoint:
        # present (loadable) -> behave as ``freeze``; missing/mismatched ->
        # reinit and keep trainable. This differs from a plain ``freeze``
        # miss (which reinits *and freezes*).
        if action == "freeze_loaded":
            if _ckpt_loadable(name, param):
                action = "freeze"
            else:
                log.info(
                    "param %r (freeze_loaded) absent/mismatched in checkpoint; "
                    "re-initializing and leaving trainable",
                    name,
                )
                reinit_targets.append(name)
                summary["reinit"].append(name)
                continue

        if action in ("load_existing", "freeze"):
            if action == "freeze":
                freeze_targets.add(name)
            if ckpt_value is None:
                log.warning(
                    "param %r assigned %s but missing from checkpoint; "
                    "falling back to reinit (then %s)",
                    name, action, action,
                )
                reinit_targets.append(name)
            elif ckpt_value.shape != param.shape:
                log.warning(
                    "param %r assigned %s but checkpoint shape %s != model "
                    "shape %s; falling back to reinit (then %s)",
                    name, action, tuple(ckpt_value.shape),
                    tuple(param.shape), action,
                )
                reinit_targets.append(name)
            else:
                to_load[name] = ckpt_value
                if action == "load_existing":
                    summary["loaded"].append(name)
        elif action == "reinit":
            reinit_targets.append(name)
            summary["reinit"].append(name)

    for name, tensor in to_load.items():
        with torch.no_grad():
            named_params[name].copy_(tensor.to(named_params[name].dtype))

    for name in reinit_targets:
        _reinit_param_inplace(model, name)
        if name not in summary["reinit"]:
            summary["reinit"].append(name)

    # Apply requires_grad LAST so freezes survive any earlier in-place writes
    for name, p in model.named_parameters():
        if name in freeze_targets:
            p.requires_grad_(False)
            summary["frozen"].append(name)
        else:
            p.requires_grad_(True)

    return summary


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters with ``requires_grad=True`` (use for building the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def format_summary(summary: dict[str, list[str]], *, max_per_group: int = 5) -> str:
    """Pretty-print the summary returned by :func:`apply_param_policy`."""
    lines = []
    for key in ("loaded", "reinit", "frozen"):
        names = summary.get(key, [])
        lines.append(f"  {key}: {len(names)} params")
        for n in names[:max_per_group]:
            lines.append(f"    - {n}")
        if len(names) > max_per_group:
            lines.append(f"    ... ({len(names) - max_per_group} more)")
    return "\n".join(lines)
