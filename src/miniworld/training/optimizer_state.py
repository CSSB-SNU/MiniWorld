"""Re-key a checkpoint's optimizer state to the current model's parameter order.

``torch.optim.Optimizer.state_dict()`` keys the per-parameter state (Adam ``exp_avg`` / ``exp_avg_sq`` / ``step``) by the parameter's
POSITION in the optimizer's groups, i.e. by the model's ``parameters()`` order, which is the order the modules registered them in.
miniworld-engine 3.x registers some parameters in another order than 2.4.x did (the bias-only attention: ``ada_ln_in, to_value, to_gate,
ln_pair, to_bias, ...`` instead of ``ada_ln_in, ln_pair, to_bias, to_value, ...``); the state-dict KEYS and shapes are unchanged, so
the weights load by name, but an optimizer checkpoint written under the old order would hand every moved parameter the moments of
another one (a shape error at the first step, or silent corruption when the shapes agree).

The checkpoint's own ``model_state_dict`` key order is the old registration order, so the old position of each parameter is known:
:func:`remap_optimizer_state` moves every state entry to the position its parameter has in the current optimizer.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from torch import nn
from torch.optim import Optimizer

_log = logging.getLogger(__name__)

_WRAPPER_PREFIXES = ("_orig_mod.", "_forward_module.", "_original_module.")


def _plain(name: str) -> str:
    """A parameter name without the torch.compile / Fabric wrapper prefixes."""
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def remap_optimizer_state(
    optimizer_state: dict[str, Any],
    checkpoint_model_state: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
) -> dict[str, Any]:
    """``optimizer_state`` re-keyed to ``optimizer``'s parameter order (a deep-enough copy; the input is not modified).

    ``checkpoint_model_state`` is the checkpoint's ``model_state_dict`` (only its key ORDER is read). The optimizer's parameters are
    identified by their names in ``model``; each group keeps the parameters it has now (the same name predicate built it before).
    Returns ``optimizer_state`` unchanged when the order already agrees, and raises ``ValueError`` when the checkpoint cannot be mapped.
    """
    name_of = {id(p): _plain(n) for n, p in model.named_parameters()}
    current = [[name_of[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    saved_groups = optimizer_state["param_groups"]
    if len(saved_groups) != len(current) or any(len(sg["params"]) != len(cg) for sg, cg in zip(saved_groups, current, strict=True)):
        msg = (
            f"optimizer state has groups of {[len(g['params']) for g in saved_groups]} parameters, the optimizer "
            f"{[len(c) for c in current]}: not the same parameter set, cannot re-key by name"
        )
        raise ValueError(msg)
    ckpt_names = list(checkpoint_model_state.keys())
    new_state: dict[int, Any] = {}
    new_groups = copy.deepcopy(saved_groups)
    moved = 0
    new_id = 0
    for sg, ng, cur in zip(saved_groups, new_groups, current, strict=True):
        members = set(cur)
        old_order = [k for k in ckpt_names if k in members]
        if len(old_order) != len(cur):
            missing = sorted(members - set(old_order))[:5]
            msg = f"the checkpoint's model_state_dict lacks {len(cur) - len(old_order)} parameters of this optimizer group, e.g. {missing}"
            raise ValueError(msg)
        old_id_of = dict(zip(old_order, sg["params"], strict=True))
        ids = []
        for name in cur:
            old_id = old_id_of[name]
            if old_id in optimizer_state["state"]:
                new_state[new_id] = optimizer_state["state"][old_id]
            moved += old_id != new_id
            ids.append(new_id)
            new_id += 1
        ng["params"] = ids
    if moved == 0:
        return optimizer_state
    _log.info("[resume] optimizer state re-keyed by parameter name: %d of %d entries moved (engine registration order differs)", moved, new_id)
    out = dict(optimizer_state)
    out["state"] = new_state
    out["param_groups"] = new_groups
    return out
