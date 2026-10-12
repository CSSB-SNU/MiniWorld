from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from lightning.fabric.wrappers import _FabricModule
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import override

from .callback import Callback

if TYPE_CHECKING:
    from miniworld.training.client import BaseClient


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Strip Fabric and DDP wrappers to reach the raw nn.Module."""
    while isinstance(module, (_FabricModule, DistributedDataParallel)):
        module = module.module
    return module


class ModelEMA(Callback):
    """Maintain exponential moving average of model parameters.

    Adapted from
    https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/optim/ema.py

    Parameters
    ----------
    decay : float
        Decay rate for exponential moving average. Default is 0.999.
    update_after_step : int
        Number of steps to wait before starting EMA updates. Default is 0.
    update_every_n_steps : int
        Update EMA parameters every n steps. Default is 1.
    enable_ema : bool
        If True, apply EMA parameters to the model, except during training.
        Default is True.

    """

    def __init__(
        self,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every_n_steps: int = 1,
        *,
        enable_ema: bool = True,
    ) -> None:
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every_n_steps = update_every_n_steps
        self.enable_ema = enable_ema

        self._ema_params: dict[str, torch.Tensor] | None = None
        self._original_params: dict[str, torch.Tensor] | None = None
        self._is_replaced: bool = False
        self._dbg_step: int = 0  # for periodic EMA health logging

    @property
    def ema_params(self) -> dict[str, torch.Tensor]:
        """EMA parameters."""
        if self._ema_params is None:
            msg = "EMA parameters not initialized. Run training first."
            raise RuntimeError(msg)
        return self._ema_params

    @override
    def on_train_epoch_start(self, client: BaseClient) -> None:
        if self._ema_params is None:
            self._initialize_ema(client)
        elif self.enable_ema:
            self._restore_original_params(client)
        if next(iter(self.ema_params.values())).device != client.device:
            self._ema_params = {
                name: param.to(client.device) for name, param in self.ema_params.items()
            }

    @override
    def on_train_step_end(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
        results: Any,
    ) -> None:
        step = client.global_step - self.update_after_step
        if step < 0:
            return
        if (step - 1) % self.update_every_n_steps == 0:
            self._update_ema(client)

    @override
    def on_train_epoch_end(self, client: BaseClient) -> None:
        if self.enable_ema:
            self._replace_ema_params(client)

    @override
    def on_save_checkpoint(self, client: BaseClient, state_dict: dict[str, Any]) -> None:
        state_dict["ema_state_dict"] = self.ema_params
        if self.enable_ema:
            self._restore_original_params(client)
            with torch.no_grad():
                state_dict["model_state_dict"] = {
                    k: v.clone().detach()
                    for k, v in _unwrap(client.model).state_dict().items()
                }
            self._replace_ema_params(client)

    @override
    def on_load_state_dict(self, client: BaseClient, state_dict: dict[str, Any]) -> None:
        ema_params = state_dict.get("ema_state_dict")
        if ema_params is None:
            client.logger.warning(
                "No EMA parameters found in checkpoint. Skipping EMA loading.",
            )
            return
        # Shadows written before the diffusion transformer block took the engine's
        # attribute name spell its attention ``attention_pair_bias.``; read them as
        # the live model names them. (Imported here: the blocks import the engine.)
        from miniworld.modules.blocks.diffusion_transformer import (  # noqa: PLC0415
            upgrade_legacy_keys,
        )

        ema_params = upgrade_legacy_keys(_unwrap(client.model), ema_params)
        # A checkpoint can carry an EMA shadow for a parameter the current model no
        # longer has -- an engine bump that removes a dead parameter is the usual
        # cause, and the model's own load path drops such keys in
        # `_load_from_state_dict` while the EMA dict is read verbatim. A shadow with
        # nothing to replace is simply dropped; keeping it would fail the restore.
        live = set(_unwrap(client.model).state_dict())
        dropped = [k for k in ema_params if k not in live]
        if dropped:
            client.logger.warning(
                "[ema] dropping %d EMA key(s) absent from the model (e.g. %s)",
                len(dropped),
                dropped[0],
            )
        # Promote to FP32 master (old checkpoints may hold BF16 EMA tensors).
        self._ema_params = {
            k: v.detach().float() for k, v in ema_params.items() if k in live
        }
        if self.enable_ema:
            self._replace_ema_params(client)

    def _initialize_ema(self, client: BaseClient) -> None:
        if self._ema_params is not None:
            return
        raw = _unwrap(client.model)
        # All ranks must iterate the EMA dict in the same order so dist.broadcast
        # pairs the right tensors. Build keys deterministically from named_parameters.
        trainable_keys = [n for n, p in raw.named_parameters() if p.requires_grad]
        sd = raw.state_dict()
        with torch.no_grad():
            # Keep the EMA master in FP32. If the model is BF16, a BF16 EMA buffer
            # rounds away the decay update (ema += (1-decay)*param, ~0.001 near 1.0
            # is below BF16's ~0.0078 spacing), so the EMA never moves. Accumulate
            # in FP32 and cast back to the model dtype only when applying.
            self._ema_params = {
                k: sd[k].detach().clone().float() for k in trainable_keys
            }
        client.logger.info(
            "EMA initialized (%d params, fp32 master)", len(self._ema_params)
        )

    def _update_ema(self, client: BaseClient) -> None:
        if client.is_global_zero:
            raw = _unwrap(client.model)
            # Read the LIVE trainable leaf tensors the optimizer actually writes
            # (named_parameters), not a possibly-detached state_dict view. Keys
            # match self.ema_params (built from named_parameters in _initialize_ema).
            live = dict(raw.named_parameters())
            with torch.no_grad():
                drift = 0.0
                for k, ema in self.ema_params.items():
                    p = live.get(k)
                    if p is None:  # buffer / non-param key -> fall back
                        p = raw.state_dict()[k]
                    p = p.detach().float()
                    if self._dbg_step % 500 == 0:
                        drift += float((ema - p).abs().mean())
                    # ema is FP32; accumulate in FP32 so small updates aren't lost.
                    ema.mul_(self.decay).add_(p, alpha=1 - self.decay)
                # Health check: EMA must actually move with the model. If it is
                # (near-)frozen it will read as a near-uniform / degenerate model
                # at eval — log the drift so this never ships silently again.
                if self._dbg_step % 500 == 0 and self.ema_params:
                    drift /= len(self.ema_params)
                    client.logger.info(
                        "[ema] step %d mean|ema-live|=%.5f",
                        self._dbg_step,
                        drift,
                    )
                self._dbg_step += 1
        if client.fabric.world_size > 1:
            for key, p in self.ema_params.items():
                if not p.is_contiguous():
                    # The EMA copy keeps the strides of its parameter (``clone`` preserves them) and engine parameter
                    # layouts can be strided; NCCL broadcasts dense buffers only. Densify once, in place of the entry.
                    p = self.ema_params[key] = p.contiguous()
                dist.broadcast(p, src=0)

    def _replace_ema_params(self, client: BaseClient, *, log: bool = True) -> None:
        """Replace model parameters with EMA parameters."""
        if self._is_replaced:
            return
        self._is_replaced = True

        raw = _unwrap(client.model)
        sd = raw.state_dict()
        missing = [k for k in self.ema_params if k not in sd]
        if missing:
            msg = f"EMA keys not in model state_dict: {missing[:5]} ..."
            raise RuntimeError(msg)

        with torch.no_grad():
            self._original_params = {k: sd[k].clone().detach() for k in self.ema_params}
            # EMA master is FP32; cast each tensor to the live param's dtype so a
            # BF16 model receives BF16 weights (load_state_dict would cast anyway,
            # but be explicit).
            casted = {k: v.to(sd[k].dtype) for k, v in self.ema_params.items()}
        raw.load_state_dict(casted, strict=False)

        if log:
            client.logger.info(
                "Replaced %d params with EMA.",
                len(self.ema_params),
            )

    def _restore_original_params(
        self,
        client: BaseClient,
        *,
        log: bool = True,
    ) -> None:
        if not self._is_replaced:
            return
        self._is_replaced = False
        if self._original_params is None:
            return
        _unwrap(client.model).load_state_dict(self._original_params, strict=False)
        if log:
            client.logger.info("Restored original params.")
