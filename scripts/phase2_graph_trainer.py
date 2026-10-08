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
from miniworld.training.phase2_graph_safe import (
    bond_loss_gs, cal_loss_gs, clear_pack_caches, graph_safe_sampling, weighted_align_gs,
)
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
        self.graphs: dict[int | None, torch.cuda.CUDAGraph] = {}  # recycle depth (None: no recycle loop) -> graph
        self.losses: dict[int | None, torch.Tensor] = {}
        self.recycles: list[int | None] = [None]
        self.use_bond = False
        self._stash = None  # MW_P2_ALIGN_DIAG only: the tensors of one eager micro-step

    # ------------------------------------------------------------------ build
    def _fill_atom_weight(self, batch) -> None:
        """AF3 Eq.4 per-atom weight (entity type -> atom), outside the graph: the chain axis of a batch is not static."""
        lc = self.client.config.loss
        et = batch.chain.entity_type.to(self.device, non_blocking=True)
        atom_to_chain = batch.scheme.atom_to_chain_id.to(self.device, non_blocking=True)
        w_chain = (
            1.0
            + lc.alpha_dna * (et == 4).float()
            + lc.alpha_rna * ((et == 3) | (et == 5)).float()
            + lc.alpha_ligand * ((et == 6) | (et == 7)).float()
        )
        self.atom_weight.copy_(torch.gather(w_chain, dim=1, index=atom_to_chain))

    def _fill_bond_pairs(self, batch) -> None:
        """AF3 Eq.5 bonded atom pairs of the micro-batch into the static, padded index buffers, outside the graph: their number
        changes from sample to sample (``copy_static`` skips ``bond_atom_pairs``). The buffers hold one slot per atom, so nothing
        is ever cut off."""
        pairs = batch.structure.bond_atom_pairs
        n = 0 if pairs is None else int(pairs.shape[1])
        if n > self.bond_valid.numel():
            raise RuntimeError(f"{n} bond pairs for {self.bond_valid.numel()} atoms")
        self.bond_i.zero_()
        self.bond_j.zero_()
        self.bond_valid.zero_()
        if n:
            pairs = pairs[0].to(self.device, non_blocking=True)
            self.bond_i[:n] = pairs[:, 0]
            self.bond_j[:n] = pairs[:, 1]
            self.bond_valid[:n] = True

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
        update = update.float()
        if self._stash is not None:
            self._stash.update(x0=x0.detach(), x_input=x_input.detach(), update=update.detach(), sigma=sigma.detach(), x_mask=x_mask)
        structure = cal_loss_gs(
            client.diffuser, x0, x_input, update, sigma, x_mask, self.atom_weight, align_atom_weight=lc.align_atom_weight,
        )
        bond = torch.zeros((), device=self.device)
        if self.use_bond:
            bond = bond_loss_gs(client.diffuser, x0, x_input, update, sigma, self.bond_i, self.bond_j, self.bond_valid)
        loss = lc.diffusion_loss * (structure + lc.bond_loss * bond)
        (loss / client.gradient_accumulation_steps).backward()
        return torch.stack([loss, structure, bond]).detach()  # total, EDM term, bond term

    def _clear_grads(self) -> None:
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

    def _build(self, first) -> None:
        client = self.client
        lc = client.config.loss
        if lc.smooth_lddt_loss != 0:
            raise ValueError("The graph-safe loss covers the EDM and the bond terms; the smooth lDDT term is not implemented for graphs")
        self.use_bond = lc.bond_loss != 0
        raw = client.model
        first_dev = first.to(device=self.device)
        self.static = copy.deepcopy(first_dev)
        self.unused_if_empty = prepare_template_graph(raw, self.static, TemplateConfig().n_templates)
        validate_empty_template(raw, self.static, first_dev)
        self.atom_weight = torch.empty(self.static.structure.atom_pos.shape[:2], device=self.device, dtype=torch.float32)
        self._fill_atom_weight(first_dev)
        n_atom = self.static.structure.atom_pos.shape[1]
        self.bond_i = torch.zeros(n_atom, dtype=torch.long, device=self.device)
        self.bond_j = torch.zeros(n_atom, dtype=torch.long, device=self.device)
        self.bond_valid = torch.zeros(n_atom, dtype=torch.bool, device=self.device)
        if self.use_bond:
            self._fill_bond_pairs(first_dev)
        self.params = [p for p in raw.parameters() if p.requires_grad]
        self.stream = torch.cuda.Stream()

        self.recycles = recycles = self._recycle_depths(raw)
        t0 = time.perf_counter()
        # One graph per recycle depth, all in ONE memory pool (a pool per graph holds the activations of every depth at once and
        # does not fit a B200): the graphs are replayed one at a time and each rewrites everything it reads, and the loss is copied
        # out right after the replay. One shared static .grad. The reverse-order replay check below guards the sharing.
        # Per depth: warm-up on the side stream (compiles, builds the autotune picks, gives every trainable parameter a .grad),
        # an eager reference step with the same random numbers, the capture, and the replay check.
        caller = torch.cuda.current_stream()
        rng = torch.cuda.get_rng_state()
        pool = torch.cuda.graph_pool_handle()
        rels: dict[int | None, float] = {}
        ref_losses: dict[int | None, float] = {}
        replay_losses: dict[int | None, float] = {}
        ref_grads_by_depth: dict[int | None, list[torch.Tensor]] = {}
        for r in recycles:
            raw._forced_n_recycle = r  # noqa: SLF001
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

            torch.cuda.set_rng_state(rng)
            self._clear_grads()
            with torch.cuda.stream(self.stream):
                ref_loss = self._body()
            torch.cuda.synchronize()
            ref_grads = [p.grad.detach().clone() for p in self.params]

            clear_pack_caches()
            self._clear_grads()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream), torch.cuda.graph(graph, pool=pool, stream=self.stream):
                loss = self._body()
            torch.cuda.current_stream().wait_stream(self.stream)
            torch.cuda.synchronize()
            clear_pack_caches()
            self._clear_grads()
            self.graphs[r], self.losses[r] = graph, loss

            torch.cuda.set_rng_state(rng)  # the same draws as the reference step
            graph.replay()
            torch.cuda.synchronize()
            num = sum(float((g.float() - p.grad.float()).pow(2).sum()) for g, p in zip(ref_grads, self.params))
            den = sum(float(g.float().pow(2).sum()) for g in ref_grads)
            rels[r] = (num / max(den, 1e-30)) ** 0.5
            ref_losses[r] = float(ref_loss[0])
            replay_losses[r] = float(loss[0])
            ref_grads_by_depth[r] = ref_grads
            self._clear_grads()
        # The pool is shared: replay in the reverse of the capture order and compare with the eager gradients again.
        reverse_rels: dict[int | None, float] = {}
        for r in reversed(recycles):
            torch.cuda.set_rng_state(rng)
            self.graphs[r].replay()
            torch.cuda.synchronize()
            ref = ref_grads_by_depth[r]
            num = sum(float((g.float() - p.grad.float()).pow(2).sum()) for g, p in zip(ref, self.params))
            den = sum(float(g.float().pow(2).sum()) for g in ref)
            reverse_rels[r] = (num / max(den, 1e-30)) ** 0.5
            self._clear_grads()
        del ref_grads_by_depth
        if self.use_bond:
            self._check_bond_term(raw, rng, recycles[-1], first_dev)
        if os.environ.get("MW_P2_ALIGN_DIAG"):
            self._align_diagnostic(raw, rng, recycles[-1])
        raw._forced_n_recycle = None  # noqa: SLF001
        self.pointers = [p.grad.data_ptr() for p in self.params]
        client.logger.info(
            "[graph] captured %d phase-2 micro-step graph(s), recycle depths %s, in %.0fs; allocated %.1f GiB",
            len(recycles), recycles, time.perf_counter() - t0, torch.cuda.memory_allocated() / 2**30,
        )
        for r in recycles:
            client.logger.info(
                "[graph] recycle %s: replay vs eager: loss %.6f vs %.6f, grad rel. error %.2e (reverse-order replay %.2e)",
                r, replay_losses[r], ref_losses[r], rels[r], reverse_rels[r],
            )
        worst = max(*rels.values(), *reverse_rels.values())
        if worst > 5e-2:
            client.logger.warning("[graph] the replayed gradients differ from the eager step by %.2e", worst)

    def _align_diagnostic(self, raw, rng, depth) -> None:
        """MW_P2_ALIGN_DIAG=1: the Eq.4 weights of the first batch's resolved atoms, and the same micro-step's loss with the alignment
        weighted by mask only vs by mask * w_l (same random numbers, eager)."""
        log, lc = self.client.logger, self.client.config.loss
        mask = self.static.structure.atom_pos_mask.reshape(-1).bool()
        w = self.atom_weight.reshape(-1)[mask]
        values, counts = torch.unique(w, return_counts=True)
        log.info("[align-diag] resolved atoms %d; w_l value -> atoms: %s", int(mask.sum()),
                 {round(float(v), 3): int(c) for v, c in zip(values, counts)})
        raw._forced_n_recycle = depth  # noqa: SLF001
        flag, out = lc.align_atom_weight, {}
        self._stash = {}
        for on in (False, True):
            lc.align_atom_weight = on
            torch.cuda.set_rng_state(rng)
            self._clear_grads()
            with torch.cuda.stream(self.stream):
                out[on] = self._body().tolist()
            torch.cuda.synchronize()
        lc.align_atom_weight = flag
        st, self._stash = self._stash, None
        with torch.no_grad():  # structure RMSD of the denoised prediction against the ground truth aligned both ways
            xp = self.client.diffuser.get_x_pred(x_input=st["x_input"], x_update=st["update"], sigma=st["sigma"])
            m = st["x_mask"].expand(xp.shape[:-1])
            x0s, ps = torch.where(m[..., None], st["x0"].to(xp.dtype), 0.0), torch.where(m[..., None], xp, 0.0)
            wa = self.atom_weight.to(xp.dtype)
            heavy = m & (wa > 1.0)
            res = {}
            for name, aw in (("mask-only", m.to(xp.dtype)), ("mask*w_l", m.to(xp.dtype) * wa)):
                sq = (ps - weighted_align_gs(x0s, ps, aw)).pow(2).sum(-1)
                rm_all = ((sq * m).sum(-1) / m.sum(-1).clamp_min(1)).sqrt().mean()
                rm_hv = ((sq * heavy).sum(-1) / heavy.sum(-1).clamp_min(1)).sqrt().mean() if bool(heavy.any()) else float("nan")
                res[name] = (float(rm_all), float(rm_hv))
        log.info("[align-diag] RMSD (A) of the denoised prediction vs aligned GT, mean over %d augments, mean sigma %.2f, %d nucleic/ligand atoms: "
                 "all atoms %.3f -> %.3f | nucleic/ligand atoms %.3f -> %.3f (mask-only -> mask*w_l)",
                 xp.shape[0], float(st["sigma"].mean()), int(heavy[0, 0].sum()), res["mask-only"][0], res["mask*w_l"][0],
                 res["mask-only"][1], res["mask*w_l"][1])
        raw._forced_n_recycle = None  # noqa: SLF001
        self._clear_grads()
        log.info("[align-diag] recycle %s, same draws: EDM term mask-only %.6f vs mask*w_l %.6f (total %.6f vs %.6f)",
                 depth, out[False][1], out[True][1], out[False][0], out[True][0])

    def _check_bond_term(self, raw, rng, depth, first_dev) -> None:
        """Replay vs eager on synthetic bond pairs (atoms k, k+1, both resolved), then restore the pairs of the first batch."""
        mask = self.static.structure.atom_pos_mask.reshape(-1)
        k = (mask[:-1] & mask[1:]).nonzero().flatten()[:64]
        self.bond_i.zero_()
        self.bond_j.zero_()
        self.bond_valid.zero_()
        self.bond_i[: k.numel()], self.bond_j[: k.numel()], self.bond_valid[: k.numel()] = k, k + 1, True
        raw._forced_n_recycle = depth  # noqa: SLF001
        torch.cuda.set_rng_state(rng)
        self._clear_grads()
        with torch.cuda.stream(self.stream):
            ref = self._body()
        torch.cuda.synchronize()
        ref_grads = [p.grad.detach().clone() for p in self.params]
        self._clear_grads()
        torch.cuda.set_rng_state(rng)
        self.graphs[depth].replay()
        torch.cuda.synchronize()
        got = self.losses[depth].tolist()
        num = sum(float((g.float() - p.grad.float()).pow(2).sum()) for g, p in zip(ref_grads, self.params))
        rel = (num / max(sum(float(g.float().pow(2).sum()) for g in ref_grads), 1e-30)) ** 0.5
        ref = ref.tolist()
        self.client.logger.info(
            "[graph] bond term, %d synthetic pairs, recycle %s: total %.6f vs %.6f, bond %.6f vs %.6f (replay vs eager), "
            "grad rel. error %.2e", k.numel(), depth, got[0], ref[0], got[2], ref[2], rel,
        )
        if rel > 5e-2 or not ref[2] > 0:
            self.client.logger.warning("[graph] the bond term of the replay does not match the eager step (grad rel. error %.2e)", rel)
        del ref_grads
        self._clear_grads()
        self._fill_bond_pairs(first_dev)

    @staticmethod
    def _recycle_depths(raw) -> list[int | None]:
        """Recycle depths to capture: 1..n_recycle_max when the frozen trunk draws a random depth per micro-step in training
        (``train_recycle: random``, the phase-1 rule); otherwise one graph (``max`` depth, or the distogram-diffusion trunk,
        which has no recycle loop)."""
        if raw.config.train_recycle == "random" and getattr(raw, "diffusion", None) is None:
            return list(range(1, raw.n_recycle_max + 1))
        return [None]

    def _draw_recycle(self) -> int | None:
        """The depth of the next micro-step: the model's own draw, so the distribution is the eager path's."""
        return None if self.recycles == [None] else self.client.model._draw_train_recycle()  # noqa: SLF001

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

    def _profile_replay(self) -> None:
        """MW_P2_PROFILE=1: GPU time per kernel family of one replay on the current batch (debugging aid, rank 0, once)."""
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            self.graphs[self.recycles[-1]].replay()
            torch.cuda.synchronize()
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        total = sum(e.device_time for e in events) / 1000
        by_name: dict[str, list[float]] = {}
        for e in events:
            row = by_name.setdefault(e.name, [0.0, 0])
            row[0] += e.device_time / 1000
            row[1] += 1
        self.client.logger.info("[graph-profile] one replay: %d kernels, GPU busy %.1f ms", len(events), total)
        for name, (ms, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:25]:
            self.client.logger.info("[graph-profile] %8.3f ms x%-4d %s", ms, n, name[:110])

    def epoch(self, dataloader):
        """Yield one result dict per micro-batch, like ``Client.training_epoch``; the optimizer step runs every ``ga`` of them."""
        client = self.client
        ga = client.gradient_accumulation_steps
        iterator = iter(dataloader)
        if not self.graphs:
            first = next(iterator)
            self._build(first)
            iterator = itertools.chain([first], iterator)
        client.model.train()
        client.call_callbacks("on_train_epoch_start")
        losses = torch.zeros(ga, 3, device=self.device)  # per micro-step: total, EDM term, bond term
        has_template = False
        started = time.perf_counter()
        spent = {"data": 0.0, "copy": 0.0, "replay": 0.0}  # host seconds of the current optimizer step
        epoch_started, steps_done = time.perf_counter(), 0
        marks = []  # CUDA events around the copy and the replay of each micro-step: the GPU's own time

        def timed(source):
            while True:
                t = time.perf_counter()
                try:
                    item = next(source)
                except StopIteration:
                    return
                spent["data"] += time.perf_counter() - t
                yield item

        try:
            for batch_idx, batch in enumerate(timed(iterator)):
                slot = batch_idx % ga
                if slot == 0:
                    client.call_callbacks("on_train_step_start", batch, batch_idx)
                    has_template = False
                    started = time.perf_counter()
                    spent.update(data=0.0, copy=0.0, replay=0.0)
                    marks.clear()
                # The batch is in pinned host memory (pin_memory + pin_batch): both copies are asynchronous and queue behind the
                # previous replay on the stream, so the host never waits for the GPU and enqueues the next micro-step early.
                has_template = has_template or batch.template.mask.shape[1] > 0
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                t = time.perf_counter()
                copy_static(self.static, batch)
                self._fill_atom_weight(batch)
                if self.use_bond:
                    self._fill_bond_pairs(batch)
                if os.environ.get("MW_P2_ALIGN_DIAG") and batch_idx < 400 and getattr(self, "_diag_left", 4) > 0:
                    if float(self.atom_weight.max()) > 1.0:  # a batch with nucleic-acid / ligand atoms
                        self._diag_left = getattr(self, "_diag_left", 4) - 1
                        self._align_diagnostic(client.model, torch.cuda.get_rng_state(), self.recycles[-1])
                spent["copy"] += time.perf_counter() - t
                events[1].record()
                t = time.perf_counter()
                if os.environ.get("MW_P2_PROFILE") and batch_idx == 3 and client.is_global_zero and client.epoch == 0:
                    self._profile_replay()
                recycle = self._draw_recycle()
                self.graphs[recycle].replay()
                losses[slot].copy_(self.losses[recycle])
                spent["replay"] += time.perf_counter() - t
                events[2].record()
                marks.append(events)
                if slot != ga - 1:
                    continue
                enqueued = time.perf_counter()
                self._optimizer_step(has_template)
                steps_done += 1
                values = losses.tolist()  # one host sync per optimizer step
                tail = time.perf_counter() - enqueued  # the GPU work still queued when the host got here, plus the step itself
                result = {
                    "diffusion_loss": 0.0, "smooth_lddt_loss": 0.0, "bond_loss": 0.0, "total_loss": 0.0, "main_loss": 0.0,
                }
                client.call_callbacks(
                    "on_train_step_end", batch, batch_idx, {**result, "total_loss": values[-1][0], "main_loss": values[-1][0], "bond_loss": values[-1][2]},
                )
                if client.global_step % 50 == 0:
                    client.logger.info(
                        "[graph] step %d: %.0f ms per optimizer step (host: data wait %.0f, copy %.0f, replay enqueue %.0f; "
                        "GPU tail + optimizer %.0f; GPU time per micro-step: copy %.1f ms, replay %.1f ms)",
                        client.global_step, (time.perf_counter() - started) * 1000,
                        spent["data"] * 1000, spent["copy"] * 1000, spent["replay"] * 1000, tail * 1000,
                        sum(a.elapsed_time(b) for a, b, _ in marks) / len(marks),
                        sum(b.elapsed_time(c) for _, b, c in marks) / len(marks),
                    )
                for v in values:
                    yield {
                        "diffusion_loss": v[1], "smooth_lddt_loss": 0.0, "bond_loss": v[2], "total_loss": v[0], "main_loss": v[0],
                    }
        finally:
            if steps_done:
                # The aggregator's ``epoch_time`` starts when the first result arrives, i.e. after the first optimizer step here
                # (results are yielded once a step is done), so it misses one step: this is the whole epoch.
                client.logger.info(
                    "[graph] epoch %d: %d optimizer steps in %.1f s (%.0f ms each; the 'epoch_time' metric excludes the first step)",
                    client.epoch, steps_done, time.perf_counter() - epoch_started,
                    (time.perf_counter() - epoch_started) * 1000 / steps_done,
                )
            self._clear_grads()
            client._epoch += 1  # noqa: SLF001
            client.call_callbacks("on_train_epoch_end")
