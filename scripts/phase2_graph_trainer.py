"""Phase-2 training with the whole micro-step captured in one CUDA graph (``train.cuda_graph=true``).

The graph holds the GPU-side sampling (augmentation rotation, noise level), the frozen trunk, the diffusion head, the EDM loss and
the backward; the gradient accumulates into the static ``.grad`` over the ``grad_accum_steps`` replays of an optimizer step.
Outside the graph, once per micro-batch: the copy of the batch into the static inputs. Once per optimizer step: the NCCL gradient
average (bucketed), clipping, Adam, the scheduler and the EMA.

Why not DDP: the captured graph owns ``backward``, so there is no wrapper whose hooks could fire inside it. ``GraphFabric`` is the
slice of ``lightning.Fabric`` the client and ``MetricsAggregator`` use, without a strategy; gradients are averaged by hand.

The capture-safe sampling and loss live in ``miniworld.training.phase2_graph_safe``; the batch copy, the template gating and the
gradient average are the phase-1 graph trainer's (``random_recycle_graph_trainer``). The step has ONE static shape (the bucket
multiples must equal the crop, as phase 1 requires), and one trunk pass: with ``model.trunk.diffusion`` set there is no recycle count
to bake into the graph.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from random_recycle_graph_trainer import (
    average_gradients,
    configure_graph_cublas,
    copy_static,
    prepare_template_graph,
    validate_empty_template,
)

from miniworld.configs import TemplateConfig
from miniworld.training.phase2_graph_safe import cal_loss_gs, clear_pack_caches, graph_safe_sampling
from miniworld.training.precision import model_autocast

__all__ = ["GraphFabric", "Phase2GraphTrainer", "configure_graph_cublas"]


class GraphFabric:
    """What the client uses of ``lightning.Fabric``, with one process per GPU, no DDP and a manual gradient average."""

    def __init__(self) -> None:
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device("cuda", self.local_rank)

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    def launch(self) -> None:
        torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", device_id=self.device)

    def seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed + self.global_rank)  # independent augmentation / noise draws per rank
        np.random.seed(seed)
        random.seed(seed)

    def setup_module(self, module):
        return module.to(self.device)

    def setup_optimizers(self, optimizer):
        return optimizer

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def all_reduce(self, data, reduce_op: str = "mean"):
        """Mean over ranks of a float, a tensor or a ``dict[str, float]``."""
        if self.world_size == 1:
            return data
        if isinstance(data, dict):
            values = torch.tensor(list(data.values()), device=self.device, dtype=torch.float64)
            dist.all_reduce(values)
            return dict(zip(data, (values / self.world_size).tolist()))
        tensor = torch.as_tensor(data, device=self.device, dtype=torch.float32)
        dist.all_reduce(tensor)
        return tensor / self.world_size

    @contextlib.contextmanager
    def no_backward_sync(self, module, enabled: bool = True):  # noqa: ARG002
        yield


class Phase2GraphTrainer:
    """Builds the step graph on the first batch and runs ``epoch`` generators with the interface of ``Client.training_epoch``."""

    def __init__(self, client, cfg) -> None:
        self.client, self.cfg = client, cfg
        self.fabric = client.fabric
        self.device, self.world = self.fabric.device, self.fabric.world_size
        self.graph = None

    # ------------------------------------------------------------------ build
    def _fill_atom_weight(self, batch) -> None:
        """AF3 Eq.4 per-atom weight (entity type -> atom), outside the graph: the chain axis of a batch is not static."""
        lc = self.client.config.loss
        et = batch.chain.entity_type
        w_chain = (
            1.0
            + lc.alpha_dna * (et == 4).float()
            + lc.alpha_rna * ((et == 3) | (et == 5)).float()
            + lc.alpha_ligand * ((et == 6) | (et == 7)).float()
        )
        self.atom_weight.copy_(torch.gather(w_chain, dim=1, index=batch.scheme.atom_to_chain_id))

    def _body(self):
        """One micro-step on the static inputs; runs eagerly for the warm-up and the check, and once under capture."""
        client, b = self.client, self.static
        lc = client.config.loss
        with graph_safe_sampling(client.diffuser):
            x0, x_input, x_mask, t_emb, sigma = client.diffuser.sample(
                b.structure.atom_pos, num_augment=client.config.train.num_augment, mask=b.structure.atom_pos_mask,
            )
        with model_autocast(client.model):
            # __call__, not .forward: the compiled implementation of the module (train.compile) is behind __call__
            update = client.model(
                msa=b.msa, template=b.template, reference=b.reference, scheme=b.scheme, sequence=b.sequence,
                structure=b.structure, x_t=x_input, x_mask=x_mask, t_emb=t_emb,
            )
        loss = lc.diffusion_loss * cal_loss_gs(client.diffuser, x0, x_input, update.float(), sigma, x_mask, self.atom_weight)
        (loss / client.gradient_accumulation_steps).backward()
        return loss.detach()

    def _clear_grads(self) -> None:
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

    def _build(self, first) -> None:
        client = self.client
        lc = client.config.loss
        if lc.smooth_lddt_loss != 0 or lc.bond_loss != 0:
            raise ValueError("The graph-safe loss covers the EDM term only (phase 2a / 2b weights)")
        raw = client.model
        first_dev = first.to(device=self.device)
        self.static = copy.deepcopy(first_dev)
        self.unused_if_empty = prepare_template_graph(raw, self.static, TemplateConfig().n_templates)
        validate_empty_template(raw, self.static, first_dev)
        self.atom_weight = torch.empty(self.static.structure.atom_pos.shape[:2], device=self.device, dtype=torch.float32)
        self._fill_atom_weight(first_dev)
        self.params = [p for p in raw.parameters() if p.requires_grad]
        self.stream = torch.cuda.Stream()

        t0 = time.perf_counter()
        # warm-up on the side stream: compiles, builds the autotune picks, and gives every trainable parameter a .grad
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self._clear_grads()
                self._body()
        caller.wait_stream(self.stream)
        torch.cuda.synchronize()
        if any(p.grad is None for p in self.params):
            missing = [n for n, p in raw.named_parameters() if p.requires_grad and p.grad is None][:5]
            raise RuntimeError(f"Graph capture needs a .grad on every trainable parameter; none for {missing} ...")

        # reference step (eager, same random numbers) for the replay check below
        rng = torch.cuda.get_rng_state()
        self._clear_grads()
        with torch.cuda.stream(self.stream):
            ref_loss = self._body()
        torch.cuda.synchronize()
        ref_grads = [p.grad.detach().clone() for p in self.params]

        clear_pack_caches()
        self._clear_grads()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = self._body()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        clear_pack_caches()
        self._clear_grads()
        self.pointers = [p.grad.data_ptr() for p in self.params]

        torch.cuda.set_rng_state(rng)  # the same draws as the reference step
        self.graph.replay()
        torch.cuda.synchronize()
        num = sum(float((g.float() - p.grad.float()).pow(2).sum()) for g, p in zip(ref_grads, self.params))
        den = sum(float(g.float().pow(2).sum()) for g in ref_grads)
        rel = (num / max(den, 1e-30)) ** 0.5
        self._clear_grads()
        client.logger.info(
            "[graph] captured the phase-2 micro-step in %.0fs; replay vs eager: loss %.6f vs %.6f, grad rel. error %.2e; "
            "allocated %.1f GiB",
            time.perf_counter() - t0, float(self.loss), float(ref_loss), rel, torch.cuda.memory_allocated() / 2**30,
        )
        if rel > 5e-2:
            client.logger.warning("[graph] the replayed gradients differ from the eager step by %.2e", rel)

    # ------------------------------------------------------------------ step
    def _optimizer_step(self, has_template: bool) -> None:
        client = self.client
        average_gradients(self.params, self.world)
        if client.gradient_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self.params, client.gradient_clip_norm)
            if not torch.isfinite(norm):  # the one host read of the optimizer step
                raise RuntimeError("Non-finite gradient norm")
        present = torch.tensor(int(has_template), device=self.device)
        if self.world > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
        held = []
        if not bool(present):  # no template anywhere in the global batch: keep Adam off the template parameters
            held = [(p, p.grad) for p in self.unused_if_empty]
            for p, _ in held:
                p.grad = None
        client.optimizer.step()
        for p, grad in held:
            p.grad = grad
        self._clear_grads()
        if client.scheduler is not None:
            client.scheduler.step()
        client._global_step += 1  # noqa: SLF001
        if self.pointers != [p.grad.data_ptr() for p in self.params]:
            raise RuntimeError("A gradient buffer moved: the captured graph would write into freed memory")

    def epoch(self, dataloader):
        """Yield one result dict per micro-batch, like ``Client.training_epoch``; the optimizer step runs every ``ga`` of them."""
        client = self.client
        ga = client.gradient_accumulation_steps
        iterator = iter(dataloader)
        if self.graph is None:
            first = next(iterator)
            self._build(first)
            iterator = itertools.chain([first], iterator)
        client.model.train()
        client.call_callbacks("on_train_epoch_start")
        weight = client.config.loss.diffusion_loss
        losses = torch.zeros(ga, device=self.device)
        has_template = False
        started = time.perf_counter()
        try:
            for batch_idx, batch in enumerate(iterator):
                slot = batch_idx % ga
                if slot == 0:
                    client.call_callbacks("on_train_step_start", batch, batch_idx)
                    has_template = False
                    started = time.perf_counter()
                batch = batch.to(device=self.device)
                has_template = has_template or batch.template.mask.shape[1] > 0
                copy_static(self.static, batch)
                self._fill_atom_weight(batch)
                self.graph.replay()
                losses[slot].copy_(self.loss)
                if slot != ga - 1:
                    continue
                self._optimizer_step(has_template)
                values = losses.tolist()  # one host sync per optimizer step
                result = {
                    "diffusion_loss": 0.0, "smooth_lddt_loss": 0.0, "bond_loss": 0.0, "total_loss": 0.0, "main_loss": 0.0,
                }
                client.call_callbacks(
                    "on_train_step_end", batch, batch_idx, {**result, "total_loss": values[-1], "main_loss": values[-1]},
                )
                if client.global_step % 50 == 0:
                    client.logger.info(
                        "[graph] step %d: %.0f ms per optimizer step", client.global_step, (time.perf_counter() - started) * 1000,
                    )
                for v in values:
                    yield {
                        "diffusion_loss": v / weight, "smooth_lddt_loss": 0.0, "bond_loss": 0.0, "total_loss": v, "main_loss": v,
                    }
        finally:
            self._clear_grads()
            client._epoch += 1  # noqa: SLF001
            client.call_callbacks("on_train_epoch_end")
