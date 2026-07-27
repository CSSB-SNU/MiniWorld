"""No-recycle CUDA-graph trainer (manual DDP + torch.cuda.CUDAGraph).

Dispatched from ``run_miniworld_distogram_train.train`` when the model runs a
FIXED recycle count (``n_recycle_max == 1``). Unlike the Fabric + plain-compile
path (used for random recycle), this captures the whole fwd+loss+bwd as ONE
standard ``torch.cuda.CUDAGraph`` and replays it — eliminating per-microbatch
kernel-launch overhead (measured 8-GPU: 71% -> ~96-100% GPU util, ~1.8x faster).

Why a separate loop (not Fabric):
  * Manual CUDA-graph capture of fwd+loss+bwd needs a static, hook-free backward.
    Fabric/DDP inserts grad-allreduce hooks INTO the backward, which conflicts
    with graph capture/replay ("accessing tensor output of CUDAGraphs that has
    been overwritten"). So DDP is done manually: replay accumulates grads into
    persistent .grad buffers, then ONE eager cross-rank all_reduce per opt-step.
  * Capturing raw eager ops (no torch.compile) is deliberate: the trunk's custom
    ``@torch.compiler.disable()`` cute/quack kernels are captured fine by the
    standard CUDA-graph API but make inductor's cudagraph-trees (reduce-overhead)
    silently skip / crash.

Checkpoints are written in the SAME dict layout as the Fabric ``Client`` path
(config / model_state_dict / optimizer_state_dict / scheduler_state_dict / epoch
/ global_step / ema) so runs are interchangeable between the two trainers.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import wandb
from miniworld.configs import TemplateConfig
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.models.distogram_only import MiniSWAModel
from miniworld.utils import get_step_decay_scheduler_with_warmup

log = logging.getLogger("cudagraph_trainer")

_CONSUMED = ("sequence", "structure", "reference", "scheme", "msa", "template")


def _load_static(dst, src) -> None:
    """Copy a real batch into the static capture buffers (fixed bucket shape)."""
    for fname in _CONSUMED:
        d = getattr(dst, fname, None)
        s = getattr(src, fname, None)
        if d is None or s is None:
            continue
        for k, v in vars(s).items():
            if isinstance(v, torch.Tensor):
                dv = getattr(d, k, None)
                if isinstance(dv, torch.Tensor) and dv.shape == v.shape:
                    dv.copy_(v, non_blocking=True)


def train_cudagraph(cfg, job_name: str, run_sub_dir: Path, ckpt: Path | None) -> None:
    """Manual-DDP + CUDA-graph training loop for the fixed-recycle model."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world)
    is_zero = rank == 0

    def info(msg: str) -> None:
        if is_zero:
            log.info(msg)

    n_rec = cfg.model.trunk.n_recycle_max
    info(f"[cudagraph] FIXED recycle={n_rec}  world={world}  "
         f"grad_accum={cfg.train.grad_accum_steps}  eff_batch="
         f"{cfg.train.num_batch * cfg.train.grad_accum_steps * world}")

    torch.manual_seed(cfg.train.seed or 0)
    model = MiniSWAModel(cfg.model).to(dev)
    model.train()
    model._forced_n_recycle = n_rec  # noqa: SLF001
    if world > 1:  # identical init across ranks
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    opt = (
        torch.optim.Adam(model.parameters(), cfg.train.max_lr, betas=(0.9, 0.95))
        if cfg.train.optimizer == "Adam"
        else torch.optim.AdamW(model.parameters(), cfg.train.max_lr)
    )
    sched = get_step_decay_scheduler_with_warmup(
        optimizer=opt, warmup_steps=cfg.train.warmup_steps,
        decay_steps=cfg.train.decay_steps, decay_factor=cfg.train.decay_factor,
    )
    w = cfg.loss.distogram_loss

    # EMA (rank-0 maintains, broadcast so all ranks agree; saved in ckpt)
    use_ema = cfg.train.use_ema
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if use_ema else None

    epoch = 0
    global_step = 0
    if ckpt is not None and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state_dict"], strict=True)
        if "optimizer_state_dict" in sd:
            opt.load_state_dict(sd["optimizer_state_dict"])
        if "scheduler_state_dict" in sd and sched is not None:
            sched.load_state_dict(sd["scheduler_state_dict"])
        epoch = int(sd.get("epoch", 0))
        global_step = int(sd.get("global_step", 0))
        if use_ema and sd.get("ema") is not None:
            ema = {k: v.to(dev) for k, v in sd["ema"].items()}
        info(f"[cudagraph] resumed from {ckpt} (epoch={epoch}, step={global_step})")

    # ---- dataloader (real bucket shape; same as Fabric path) ----
    world_item = cfg.train.train_item // world
    ds = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
    dl = ds.create_ddp_dataloader(
        world_size=world, rank=rank, seed=cfg.train.seed, drop_last=True,
        batch_size=cfg.train.num_batch, num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor, num_samples_per_rank=world_item,
        persistent_workers=cfg.train.num_workers > 0, shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
        bucket_template_multiple=TemplateConfig().n_templates)
    it = iter(dl)
    static = next(it).to(device=dev)

    # CB/pseudo-beta distogram target (cfg.loss.distogram_cb_target). Default off keeps the
    # legacy shortest-inter-atom-distance target.
    _use_cb = cfg.loss.distogram_cb_target
    if is_zero:
        log.info("[cudagraph] distogram target = %s",
                 "CB/pseudo-beta (rep atom)" if _use_cb else "shortest inter-atom")

    def step():
        rep = static.structure.atom_is_rep if _use_cb else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model.forward(
                msa=static.msa, reference=static.reference, scheme=static.scheme,
                sequence=static.sequence, structure=static.structure, template=static.template)
            loss = w * cal_atom_distogram_loss(
                logit, static.structure.atom_pos, static.structure.atom_pos_mask,
                static.scheme.atom_to_token_idx_map, rep_atom_mask=rep)
        loss.backward()
        return loss

    # ---- warmup on side stream (JIT kernels, allocate grads), then capture ----
    info("[cudagraph] warmup + capture (torch.cuda.CUDAGraph, fwd+loss+bwd)")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad(set_to_none=True); step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    opt.zero_grad(set_to_none=False)  # static grad buffers for capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss = step()
    info(f"[cudagraph] CAPTURE OK  peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

    ckpt_dir = run_sub_dir / "checkpoints"
    if is_zero:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path: Path) -> None:
        if not is_zero:
            return
        torch.save({
            "config": cfg.model_dump(mode="json"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched else None,
            "epoch": epoch, "global_step": global_step,
            "ema": ema if use_ema else None,
        }, path)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    ga = cfg.train.grad_accum_steps
    steps_per_epoch = world_item // ga
    ema_decay = cfg.train.ema_decay
    info(f"[cudagraph] Start training (steps/epoch={steps_per_epoch})")

    while epoch < cfg.train.num_epoch:
        dl.sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        ep_t0 = time.perf_counter()
        losses = []
        for _ in range(steps_per_epoch):
            micro_acc = None
            for _ in range(ga):
                _load_static(static, next_batch().to(device=dev))
                graph.replay()  # fwd+loss+bwd -> accumulate into .grad
                # Accumulate the per-microbatch loss on-GPU (read BEFORE the next
                # replay overwrites static_loss). Logging only the last microbatch made
                # the per-step curve a 1/ga-sample estimate (32x noisier than the true
                # eff-batch loss); averaging all ga gives the loss over the full
                # effective batch, so train/distogram_loss and its epoch mean are the
                # same quantity at two resolutions.
                _l = static_loss.detach().float()
                micro_acc = _l.clone() if micro_acc is None else micro_acc + _l
            if world > 1:  # eager cross-rank grad average (once per opt-step)
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad); p.grad /= world
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_max_norm)
            opt.step()
            if sched:
                sched.step()
            if use_ema:  # rank-0 EMA update (params identical across ranks)
                with torch.no_grad():
                    sd = model.state_dict()
                    for k in ema:
                        ema[k].mul_(ema_decay).add_(sd[k], alpha=1 - ema_decay)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            global_step += 1
            step_loss = (micro_acc / ga).item()  # mean over the full effective batch
            losses.append(step_loss)
            if is_zero:  # per-step (step-wise) loss, like the Fabric/Client path
                log.info(
                    "Step %8d (Epoch %5d) │ train/distogram_loss_step=%.4f",
                    global_step, epoch + 1, step_loss,
                )
                if cfg.train.use_wandb:
                    # Two explicit curves: *_step (per opt-step, noisy) on the global_step
                    # x-axis, *_epoch (per-epoch mean, smooth) on the epoch x-axis — bound
                    # via define_metric so wandb renders them as two clean panels instead
                    # of one overlaid jagged curve. Log global_step as a key so the step
                    # curve has its own x-axis.
                    wandb.log(
                        {"train/distogram_loss_step": step_loss,
                         "global_step": global_step},
                        step=global_step,
                    )
        torch.cuda.synchronize()
        epoch += 1
        mean_loss = sum(losses) / max(len(losses), 1)
        ep_time = time.perf_counter() - ep_t0
        info(f"Epoch {epoch:5d} │ train/distogram_loss={mean_loss:.3f}  "
             f"train/epoch_time={ep_time:.0f}  step={global_step}")
        if is_zero and cfg.train.use_wandb:
            # per-step loop already logged the loss at each step; only add the
            # epoch-level metrics here (same step -> wandb merges).
            wandb.log({"train/epoch_time": ep_time, "epoch": epoch,
                       "train/distogram_loss_epoch": mean_loss}, step=global_step)
        save_ckpt(ckpt_dir / "last.pt")
        if epoch % cfg.train.save_freq == 0:
            save_ckpt(ckpt_dir / f"epoch={epoch:04d}.pt")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    info("[cudagraph] DONE")
