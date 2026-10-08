"""Phase1 CUDA graphs with random recycle, stable gradients and full-state resume.

Independent graph pools permit arbitrary recycle order. NCCL runs outside graphs,
once per optimizer step in buckets. This module never changes effective batch.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import logging
import os
import random
import time
import uuid
from dataclasses import fields, is_dataclass, replace
from types import MethodType
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger("random_recycle_graph")
CONSUMED = ("sequence", "structure", "reference", "scheme", "msa", "template")


def copy_static(dst, src):
    """Fail on an incompatible batch instead of replaying stale input fields."""
    for group in CONSUMED:
        a, b = getattr(dst, group), getattr(src, group)
        for name, value in vars(a).items():
            if group == "template" and name == "_graph_present":
                value.fill_(b.mask.shape[1] > 0)
                continue
            if (
                group == "template"
                and hasattr(a, "_graph_present")
                and b.mask.shape[1] == 0
            ):
                if isinstance(value, torch.Tensor):
                    value.zero_()
                continue
            if group == "structure" and name in {
                "token_bond",
                "token_contacts",
                "atom_bond",
                "bond_atom_pairs",
            }:
                # Phase1 reads fixed-size token_bond_feat, not sparse bond/contact
                # lists or diffusion-only bond targets. Require the dense path.
                if (
                    getattr(a, "token_bond_feat", None) is None
                    or getattr(b, "token_bond_feat", None) is None
                ):
                    raise ValueError("Phase1 graph requires dense token_bond_feat")
                continue
            other = getattr(b, name)
            if isinstance(value, torch.Tensor):
                if (
                    not isinstance(other, torch.Tensor)
                    or value.shape != other.shape
                    or value.dtype != other.dtype
                ):
                    raise ValueError(
                        f"Graph input mismatch: {group}.{name}: {value.shape}/{value.dtype} vs {getattr(other, 'shape', None)}/{getattr(other, 'dtype', None)}"
                    )
                value.copy_(other, non_blocking=True)
            elif value is None and other is not None:
                raise ValueError(f"Graph optional input changed: {group}.{name}")
    weight_fn = getattr(dst, "_pair_weight_fn", None)
    if weight_fn is not None:
        # outside the graph, on the caller's stream, into the buffer the captured loss reads (chain counts vary per batch)
        dst.pair_weight.copy_(weight_fn(src))


def template_graph_forward(self, pair, template, token_asym_id, token_mask):
    # Original T=0 returns proj_out(0), exactly zero because bias=False.
    # A padded slot is NOT semantically empty in this model: explicitly gate
    # the full update instead of assuming template.mask gates it (it does not).
    update = self._graph_original_forward(pair, template, token_asym_id, token_mask)
    return update * template._graph_present.to(update.dtype)


def prepare_template_graph(model, static, n_templates):
    if not model.use_template:
        return []
    module = model.temp_embedder
    assert module.proj_out.bias is None
    original_count = static.template.mask.shape[1]
    for name, value in vars(static.template).copy().items():
        if isinstance(value, torch.Tensor) and value.shape[1] == 0:
            shape = list(value.shape)
            shape[1] = n_templates
            setattr(static.template, name, value.new_zeros(shape))
    static.template._graph_present = torch.tensor(
        original_count > 0, device=static.device
    )
    module._graph_original_forward = module.forward
    module.forward = MethodType(template_graph_forward, module)
    # T=0 does not connect these parameters to the loss. Preserve Adam's
    # grad=None behavior when an entire global optimizer batch has no templates.
    return [
        p for name, p in module.named_parameters() if not name.startswith("proj_out.")
    ]


def empty_template_batch(batch):
    template = replace(
        batch.template,
        **{
            f.name: getattr(batch.template, f.name)[:, :0]
            for f in fields(batch.template)
        },
    )
    return replace(batch, template=template)


def validate_empty_template(model, static, original):
    if not model.use_template:
        return
    module = model.temp_embedder
    mode = module.training
    module.eval()
    pair = torch.randn(
        (
            *static.structure.token_mask.shape,
            static.token_length,
            model.config.shared.d_pair,
        ),
        device=static.device,
        dtype=torch.bfloat16,
    )
    from miniworld.training.precision import model_autocast

    empty = empty_template_batch(original).template.to(device=static.device)
    params = tuple(module.parameters())
    # under the model's own precision: a bf16-mixed model has fp32 parameters and its forward runs under bf16 autocast
    with model_autocast(model):
        expected = module._graph_original_forward(
            pair, empty, static.scheme.token_asym_id, static.structure.token_mask
        )
    ref = torch.autograd.grad(expected.sum(), params, allow_unused=True)
    static.template._graph_present.fill_(False)
    with model_autocast(model):
        actual = module(
            pair, static.template, static.scheme.token_asym_id, static.structure.token_mask
        )
    grads = torch.autograd.grad(actual.sum(), params, allow_unused=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(grads, ref):
        if a is not None:
            if b is None:
                assert torch.count_nonzero(a) == 0
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    module.train(mode)
    copy_static(static, original)


def pin_batch(value):
    """Called by DataLoader's pinning thread, never by the training main thread."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if is_dataclass(value):
        return type(value)(
            **{f.name: pin_batch(getattr(value, f.name)) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: pin_batch(v) for k, v in value.items()}
    if isinstance(value, list):
        return [pin_batch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(pin_batch(v) for v in value)
    return value


def average_gradients(params, world):
    """Bucket NCCL transfers while retaining captured .grad storage and strides."""
    if world == 1:
        return
    bucket = []
    size = 0
    dtype = None

    def flush(items):
        grads = [p.grad for p in items]
        flat = torch._utils._flatten_dense_tensors(grads)
        flat.div_(world)
        dist.all_reduce(flat)
        views = torch._utils._unflatten_dense_tensors(flat, grads)
        torch._foreach_copy_(grads, views)

    for p in params:
        if bucket and p.grad.dtype != dtype:
            flush(bucket)
            bucket = []
            size = 0
        dtype = p.grad.dtype
        bucket.append(p)
        size += p.grad.numel() * p.grad.element_size()
        if size >= 25 * 1024**2:
            flush(bucket)
            bucket = []
            size = 0
    if bucket:
        flush(bucket)


class RecycleGraphs:
    def __init__(self, model, batch, loss_fn, ga, max_recycle, *, forward_kwargs=None):
        self.model, self.batch, self.loss_fn, self.ga = model, batch, loss_fn, ga
        self.forward_kwargs = dict(forward_kwargs or {})
        self.graphs = {}
        self.losses = {}
        self.metrics = {}
        self.last_metrics = {}
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        rng = torch.cuda.get_rng_state()
        with torch.cuda.stream(self.stream):
            for r in range(1, max_recycle + 1):
                model._forced_n_recycle = r
                for _ in range(3):
                    self.clear()
                    self.compute()
                if any(p.grad is None for p in self.params):
                    raise RuntimeError(
                        "Graph capture requires gradients for every trainable parameter"
                    )
                self.clear()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream):
                    self.losses[r] = self.compute()
                self.graphs[r] = g
                self.metrics[r] = self.last_metrics
                print(
                    f"[graph] captured R{r}; allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )
            self.clear()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng)
        self.pointers = [p.grad.data_ptr() for p in self.params]

    def clear(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

    def compute(self):
        # Use the capture stream for the ordinary reference too: AccumulateGrad
        # nodes retained by captured losses belong to that stream.
        caller = torch.cuda.current_stream()
        if caller == self.stream:
            return self._compute_on_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            loss = self._compute_on_stream()
        caller.wait_stream(self.stream)
        return loss

    def _compute_on_stream(self):
        b = self.batch
        from miniworld.training.precision import model_autocast

        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if os.environ.get("MW_GRAPH_AMP") == "1"
            else model_autocast(self.model)    # bf16 autocast for an fp32-master (bf16-mixed) model
        ):
            y = self.model(
                msa=b.msa,
                reference=b.reference,
                scheme=b.scheme,
                sequence=b.sequence,
                structure=b.structure,
                template=b.template,
                **self.forward_kwargs,
            )
            if isinstance(y, tuple) and len(y) == 2 and isinstance(y[1], dict):
                self.last_metrics = y[1]
            loss = self.loss_fn(y, b)
        (loss / self.ga).backward()
        return loss

    def replay(self, r):
        self.graphs[int(r)].replay()
        return self.losses[int(r)]

    def assert_storage(self):
        assert self.pointers == [p.grad.data_ptr() for p in self.params]


def configure_graph_cublas():
    """Fix cuBLAS workspace selection before creating any CUDA handles.

    Capture uses a side stream; the ordinary reference uses the caller stream.
    Default workspace selection can change GEMM rounding across those streams.
    See NVIDIA cuBLAS Results Reproducibility documentation.
    """
    setting = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if setting is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA "
                "for random-recycle graph training"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif setting not in (":4096:8", ":16:8"):
        raise ValueError(
            "Random-recycle graphs require CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8 for reproducible multi-stream cuBLAS execution"
        )


def graph_objective(cfg):
    """Keep the captured objective identical to the ordinary training client."""
    if getattr(cfg.model.trunk, "diffusion", None) is not None:
        if cfg.model.trunk.n_recycle_max != 1 or not cfg.loss.distogram_cb_target:
            raise ValueError("Diffusion graphs require one trunk pass and pseudo-beta targets")
        return (
            lambda output, batch: cfg.loss.distogram_loss * output[0],
            {"interchain_weight": cfg.loss.distogram_interchain_weight},
        )
    from miniworld.loss.auxiliary import cal_atom_distogram_loss

    def loss_fn(y, b):
        return cfg.loss.distogram_loss * cal_atom_distogram_loss(
            y, b.structure.atom_pos, b.structure.atom_pos_mask,
            b.scheme.atom_to_token_idx_map,
            rep_atom_mask=b.structure.atom_is_rep if cfg.loss.distogram_cb_target else None,
            token_asym_id=b.scheme.token_asym_id,
            interchain_weight=cfg.loss.distogram_interchain_weight,
            pair_weight=getattr(b, "pair_weight", None),
        )

    return loss_fn, {}


def attach_pair_weight(static, loss_cfg):
    """Give the static batch the chemistry pair weight buffer of the distogram CE (no-op when every weight is 1).

    ``copy_static`` keeps it current: the weight depends on chain features whose size changes from batch to batch, so it is
    computed eagerly per batch and copied into one fixed [B, L, L] buffer that the captured loss reads."""
    from miniworld.models.distogram_only.client import _pair_weight, _pair_weights_enabled

    if not _pair_weights_enabled(loss_cfg):
        return False
    static._pair_weight_fn = lambda batch: _pair_weight(loss_cfg, batch)
    static.pair_weight = static._pair_weight_fn(static).clone()
    return True


def resolve_run_directory(cfg, ckpt, run_dir, state):
    if run_dir is not None:
        return Path(run_dir).resolve()
    if ckpt is not None:
        if state.get("run_dir"):
            return Path(state["run_dir"]).resolve()
        checkpoint = Path(ckpt).resolve()
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent
        raise ValueError("A copied checkpoint requires MW_RESUME_RUN_SUBDIR")
    root = Path(cfg.train.run_dir).resolve()
    if (root / "wandb_run_id.txt").exists() or (root / "latest_run.txt").exists():
        raise ValueError(f"Existing training run in {root}; pass --ckpt to resume it")
    return root / time.strftime("%Y-%m-%d") / f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"


def prepare_wandb_id(root, run_dir, state, fresh, generate_id):
    """Persist one identity and reject accidentally mixing two training runs."""
    paths = [Path(root) / "wandb_run_id.txt", Path(run_dir) / "wandb_run_id.txt"]
    ids = {p.read_text().strip() for p in paths if p.exists()}
    if state.get("wandb_run_id"):
        ids.add(state["wandb_run_id"])
    if "" in ids or len(ids) > 1 or (fresh and ids):
        raise ValueError("Conflicting/existing W&B run identity; resume its checkpoint")
    if not fresh and not ids:
        raise ValueError("Resuming with W&B requires the original run identity")
    run_id = next(iter(ids)) if ids else generate_id()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run_id + "\n")
    return run_id


def capture_rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng_state(state):
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def gather_rng_states(world):
    local = capture_rng_state()
    if world == 1:
        return [local]
    states = [None] * world
    dist.all_gather_object(states, local)
    return states


def save_checkpoint(run_dir, payload, archive=False):
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(exist_ok=True)
    temporary = directory / "last.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, directory / "last.pt")
    if archive:
        torch.save(payload, directory / f"epoch={payload['epoch']:04d}.pt")


def restore_training_state(model, optimizer, scheduler, state):
    model.load_state_dict(state["model_state_dict"], strict=True)
    # Adam's CPU step tensors otherwise alias the loaded checkpoint. A temporary
    # validation update would then change the state we intend to restore.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer_state_dict"]))
    if scheduler:
        scheduler.load_state_dict(copy.deepcopy(state["scheduler_state_dict"]))


def train(cfg, ckpt, run_dir=None, *, diagnostic_steps=0, validate=True, job_name=None):
    configure_graph_cublas()
    if getattr(cfg.train, "forbid_triton", False):
        from miniworld.training import no_triton

        no_triton.install()
    from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_

    from miniworld.configs import TemplateConfig
    from miniworld.data.dataloader.dataloader import BioMolData
    from miniworld.models.distogram_only import MiniSWAModel
    from miniworld.training.engine_backend import configure_engine_backend
    from miniworld.training.precision import apply_precision
    from miniworld.utils import get_step_decay_scheduler_with_warmup

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    configure_engine_backend(cfg.train.engine_backend)
    os.environ["MINIWORLD_OPM_TRAIN"] = os.environ["MINIWORLD_PWA_TRAIN"] = (
        "1"
        if getattr(
            cfg.train, "fused_msa_train", os.environ.get("MINIWORLD_PWA_TRAIN") == "1"
        )
        else "0"
    )
    torch.manual_seed(cfg.train.seed or 0)
    torch.set_float32_matmul_precision("medium")
    # Keep BF16 operands/output with FP32 GEMM partial reductions. This policy
    # alone does not fix the D64 Transition input-buffer race; that requires
    # the engine's input_barrier_v1 kernel (verified in the resumed snapshot).
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = apply_precision(MiniSWAModel(cfg.model), cfg.train.precision).to(dev).train()
    is_diffusion = getattr(cfg.model.trunk, "diffusion", None) is not None
    fresh = ckpt is None
    opt = (
        torch.optim.Adam(model.parameters(), lr=cfg.train.max_lr, betas=(0.9, 0.95))
        if cfg.train.optimizer == "Adam"
        else torch.optim.AdamW(model.parameters(), lr=cfg.train.max_lr)
    )
    sched = get_step_decay_scheduler_with_warmup(
        optimizer=opt,
        warmup_steps=cfg.train.warmup_steps,
        decay_steps=cfg.train.decay_steps,
        decay_factor=cfg.train.decay_factor,
    )
    if ckpt is None:
        state = {
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched else None,
            "ema_state_dict": {k: v.detach().float().cpu().clone() for k, v in model.named_parameters()}
            if cfg.train.use_ema else {},
            "epoch": 0, "global_step": 0,
        }
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    selection = [str(resolve_run_directory(cfg, ckpt, run_dir, state)) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(selection, src=0)
    run_dir = Path(selection[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not diagnostic_steps:
        (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2) + "\n")

    def restore(saved_state):
        restore_training_state(model, opt, sched, saved_state)
        align_optimizer_state_layout_(opt)

    restore(state)
    if is_diffusion:
        # Weights start identically on all ranks; diffusion draws are independent.
        torch.cuda.manual_seed((cfg.train.seed or 0) + rank)
    epoch = int(state["epoch"])
    global_step = int(state["global_step"])
    saved_rng = state.get("rng_states")
    if saved_rng and len(saved_rng) == world:
        restore_rng_state(saved_rng[rank])
    elif saved_rng:
        torch.manual_seed((cfg.train.seed or 0) + global_step * world + rank)
        if rank == 0:
            log.warning("World size changed; starting independent RNG streams for %d ranks", world)
    startup_rng = capture_rng_state()
    if rank == 0:
        print(
            f"[graph] restored epoch={epoch} step={global_step}; building loader",
            flush=True,
        )
    ema = (
        {
            k: v.to(dev, dtype=torch.float32).clone()
            for k, v in state["ema_state_dict"].items()
        }
        if cfg.train.use_ema
        else {}
    )
    live = dict(model.named_parameters())
    assert set(ema) <= live.keys()
    if cfg.train.compile:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 512
        model.compile(dynamic=False)
    from miniworld.data.features.batch import Batch

    use_pin = os.environ.get("MW_GRAPH_PIN", "1") == "1"
    if use_pin:
        Batch.pin_memory = pin_batch
    if os.environ.get("MW_GRAPH_CACHED_INPUTS") == "1":
        assert diagnostic_steps, "Cached-input mode is diagnostic only"
        cached = torch.load(
            Path(run_dir) / f"input-rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if use_pin:
            cached = [pin_batch(x) for x in cached]
        first, second = cached
        it = itertools.cycle(cached)
    else:
        ds = BioMolData(
            BioMolData.BioMolConfig(
                crop_config=cfg.data.crop,
                msa_config=cfg.data.msa,
                DB_config=cfg.data.train_db,
                sampler_config=cfg.data.sampler,
                tokenizer_config=cfg.data.tokenizer,
            )
        )
        dl = ds.create_ddp_dataloader(
            world_size=world,
            rank=rank,
            seed=cfg.train.seed,
            drop_last=True,
            batch_size=cfg.train.num_batch,
            num_workers=cfg.train.num_workers,
            pin_memory=use_pin,
            prefetch_factor=cfg.train.prefetch_factor,
            num_samples_per_rank=cfg.train.train_item // world,
            persistent_workers=cfg.train.num_workers > 0,
            shuffle=True,
            bucket_msa_multiple=cfg.train.bucket_msa_multiple,
            bucket_token_multiple=cfg.train.bucket_token_multiple,
            bucket_atom_multiple=cfg.train.bucket_atom_multiple,
            bucket_template_multiple=TemplateConfig().n_templates,
        )
        dl.sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        it = iter(dl)
        first = next(it)
        second = next(it)
        if rank == 0:
            print("[graph] real batches ready; compile/capture begins", flush=True)
    if diagnostic_steps:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        torch.save([first, second], Path(run_dir) / f"input-rank{rank}.pt")
        if os.environ.get("MW_GRAPH_REPEAT_DIAGNOSTIC_INPUTS") == "1":
            # Short numerical/compute check; deliberately excludes loader throughput.
            it = itertools.cycle([first, second])
    static = first.to(device=dev)
    if attach_pair_weight(static, cfg.loss) and rank == 0:
        print("[graph] distogram chemistry pair weights ride in the static batch", flush=True)
    unused_if_empty = prepare_template_graph(model, static, TemplateConfig().n_templates)
    validate_empty_template(model, static, first)

    loss_fn, forward_kwargs = graph_objective(cfg)
    graphs = RecycleGraphs(
        model, static, loss_fn, cfg.train.grad_accum_steps, cfg.model.trunk.n_recycle_max,
        forward_kwargs=forward_kwargs,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not diagnostic_steps:
        log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][rank=0][GraphTrainer][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in (
            logging.FileHandler(run_dir / "train.log"),
            logging.StreamHandler(),
        ):
            handler.setFormatter(formatter)
            log.addHandler(handler)
        log.info(
            "Graph training epoch=%d step=%d world=%d accumulation=%d",
            epoch,
            global_step,
            world,
            cfg.train.grad_accum_steps,
        )
    if validate:
        checks = []
        for stage in ("initial", "after_optimizer"):
            if stage == "after_optimizer":
                torch.nn.utils.clip_grad_norm_(
                    graphs.params, cfg.train.grad_clip_max_norm
                )
                rates = [group["lr"] for group in opt.param_groups]
                # Fresh warmup starts at lr=0; make a temporary nonzero decoder
                # update so validation exercises upstream gradients. Restore all
                # weights/Adam/scheduler state before the first training step.
                if fresh:
                    for group in opt.param_groups:
                        if group["lr"] == 0:
                            group["lr"] = min(cfg.train.max_lr, 1e-4)
                opt.step()
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate
            rng = torch.cuda.get_rng_state()
            graphs.clear()
            expected = []
            expected_sigma = []
            reference_times = []
            base_seq = (
                [1, 3, 2, 4]
                if cfg.model.trunk.n_recycle_max == 4
                else list(graphs.graphs)
            )
            seq = [
                base_seq[i % len(base_seq)] for i in range(cfg.train.grad_accum_steps)
            ]
            # Warm every real/empty-template input and recycle combination before
            # establishing the ordinary-execution reference. The first traversal
            # may still specialize compiled backward paths after graph capture.
            for i, r in enumerate(seq):
                copy_static(static, [first, second, empty_template_batch(first)][i % 3])
                model._forced_n_recycle = r
                graphs.compute()
            torch.cuda.synchronize()
            graphs.clear()
            torch.cuda.set_rng_state(rng)
            for i, r in enumerate(seq):
                copy_static(static, [first, second, empty_template_batch(first)][i % 3])
                model._forced_n_recycle = r
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                loss = graphs.compute().detach()
                b.record()
                torch.cuda.synchronize()
                expected.append(float(loss))
                if "sigma_mean" in graphs.last_metrics:
                    expected_sigma.append(float(graphs.last_metrics["sigma_mean"]))
                reference_times.append(a.elapsed_time(b))
            average_gradients(graphs.params, world)
            reference = [p.grad.clone() for p in graphs.params]
            graphs.clear()
            torch.cuda.set_rng_state(rng)
            actual = []
            actual_sigma = []
            graph_times = []
            for i, r in enumerate(seq):
                copy_static(static, [first, second, empty_template_batch(first)][i % 3])
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                loss = graphs.replay(r).detach()
                b.record()
                torch.cuda.synchronize()
                actual.append(float(loss))
                if "sigma_mean" in graphs.metrics[r]:
                    actual_sigma.append(float(graphs.metrics[r]["sigma_mean"]))
                graph_times.append(a.elapsed_time(b))
            average_gradients(graphs.params, world)
            errors = [
                float(
                    (p.grad.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-8)
                )
                for p, ref in zip(graphs.params, reference)
            ]
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            worst = sorted(zip(names, errors), key=lambda x: x[1], reverse=True)[:10]
            details = []
            for name, p, ref, rel in zip(names, graphs.params, reference, errors):
                delta = p.grad.float() - ref.float()
                details.append(
                    dict(
                        name=name,
                        relative_l2=rel,
                        reference_norm=float(ref.float().norm()),
                        difference_norm=float(delta.norm()),
                        max_abs=float(delta.abs().max()),
                        reference_max=float(ref.float().abs().max()),
                    )
                )
            details.sort(key=lambda x: x["relative_l2"], reverse=True)
            if fresh and is_diffusion and stage == "after_optimizer":
                assert any(d["name"] == "diffusion.encoder.weight" and d["reference_norm"] > 0 for d in details)
                assert any(d["name"].startswith("pairformer_blocks.") and d["reference_norm"] > 0 for d in details)
            failed_gradients = [
                d
                for d in details
                if not np.isfinite(d["relative_l2"])
                or (d["relative_l2"] >= 1e-3
                    and not (d["max_abs"] <= 1e-8 and d["reference_max"] <= 1e-6))
            ]
            if failed_gradients:
                graph_grads = [p.grad.clone() for p in graphs.params]
                graphs.clear()
                torch.cuda.set_rng_state(rng)
                repeated_losses = []
                for i, r in enumerate(seq):
                    copy_static(
                        static, [first, second, empty_template_batch(first)][i % 3]
                    )
                    model._forced_n_recycle = r
                    repeated_losses.append(float(graphs.compute().detach()))
                average_gradients(graphs.params, world)
                repeated = []
                for name, p, ref in zip(names, graphs.params, reference):
                    delta = p.grad.float() - ref.float()
                    repeated.append(
                        dict(
                            name=name,
                            relative_l2=float(
                                delta.norm() / ref.float().norm().clamp_min(1e-8)
                            ),
                            max_abs=float(delta.abs().max()),
                        )
                    )
                (run_dir / f"gradient-diagnostics-{stage}-rank{rank}.json").write_text(
                    json.dumps(
                        dict(
                            graph=details,
                            ordinary_repeat=repeated,
                            expected_losses=expected,
                            repeated_losses=repeated_losses,
                        ),
                        indent=2,
                    )
                )
                for p, g in zip(graphs.params, graph_grads):
                    p.grad.copy_(g)
                del graph_grads
            sigma_ok = (
                not expected_sigma
                or (np.allclose(actual_sigma, expected_sigma, rtol=0, atol=0)
                    and len(set(actual_sigma)) > 1)
            )
            if not np.allclose(actual, expected, rtol=0, atol=1e-6) or failed_gradients or not sigma_ok:
                (run_dir / f"validation-failure-rank{rank}.json").write_text(
                    json.dumps(
                        dict(
                            stage=stage,
                            expected_losses=expected,
                            actual_losses=actual,
                            expected_sigma=expected_sigma,
                            actual_sigma=actual_sigma,
                            worst_gradients=worst,
                        ),
                        indent=2,
                    )
                )
                raise AssertionError((stage, worst, expected, actual))
            graphs.assert_storage()
            checks.append(
                dict(
                    stage=stage,
                    max_gradient_relative_l2=max(errors),
                    gradient_details=details,
                    tolerance="relative L2 < 1e-3; for near-zero gradients (max <= 1e-6), max absolute error <= 1e-8",
                    expected_losses=expected,
                    actual_losses=actual,
                    expected_sigma=expected_sigma,
                    actual_sigma=actual_sigma,
                    reference_compute_ms=reference_times,
                    graph_compute_ms=graph_times,
                    recycles=seq,
                    precision=cfg.train.precision,
                )
            )
            del reference
        (run_dir / f"validation-rank{rank}.json").write_text(
            json.dumps(checks, indent=2)
        )
        restore(state)
        graphs.clear()
    restore_rng_state(startup_rng)
    if world > 1:
        dist.barrier()
    if rank == 0:
        print("[graph] validation passed; full model/Adam restored", flush=True)
    import wandb
    from miniworld.utils import new_wandb_run_id

    wandb_id = state.get("wandb_run_id")
    if rank == 0 and cfg.train.use_wandb and not diagnostic_steps:
        wandb_id = prepare_wandb_id(cfg.train.run_dir, run_dir, state, fresh, new_wandb_run_id)

    def checkpoint_payload(rng_states):
        return {
            "config": cfg.model_dump(mode="json"), "run_dir": str(run_dir),
            "wandb_run_id": wandb_id, "rng_states": rng_states,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched else None,
            "ema_state_dict": ema, "epoch": epoch, "global_step": global_step,
        }

    if not diagnostic_steps:
        initial_rng = gather_rng_states(world)
        if rank == 0:
            if fresh:
                save_checkpoint(run_dir, checkpoint_payload(initial_rng))
            root = Path(cfg.train.run_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "latest_run.txt").write_text(str(run_dir) + "\n")
            if cfg.train.use_wandb:
                wandb.init(project=cfg.train.wandb_project, id=wandb_id,
                           name=job_name or cfg.train.comment,
                           resume="allow" if (is_diffusion or fresh) else "must",
                           config=cfg.model_dump(mode="json"))
        if world > 1:
            dist.barrier()
    del state
    ga = cfg.train.grad_accum_steps
    assert cfg.train.train_item % (world * ga * cfg.train.num_batch) == 0
    nsteps = cfg.train.train_item // (world * ga * cfg.train.num_batch)
    records = []
    done = 0
    while epoch < cfg.train.num_epoch:
        recycles = np.random.default_rng((cfg.train.seed or 0) + epoch)
        if done == 0:
            batches = itertools.chain([first, second], it)
        else:
            dl.sampler.set_epoch(epoch)
            ds.set_epoch(epoch)
            batches = iter(dl)
        ep_start = time.perf_counter()
        epoch_loss = 0.0
        for step in range(nsteps):
            graphs.clear()
            acc = torch.zeros((), device=dev)
            timing = {"data_next_ms": 0.0, "copy_cpu_ms": 0.0, "replay_cpu_ms": 0.0}
            events = []
            begin = time.perf_counter()
            has_template = False
            for micro in range(ga):
                t = time.perf_counter()
                batch = next(batches)
                has_template = has_template or batch.template.mask.shape[1] > 0
                timing["data_next_ms"] += (time.perf_counter() - t) * 1000
                a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                a.record()
                t = time.perf_counter()
                copy_static(static, batch)
                timing["copy_cpu_ms"] += (time.perf_counter() - t) * 1000
                b.record()
                r = int(recycles.integers(1, cfg.model.trunk.n_recycle_max + 1))
                t = time.perf_counter()
                loss = graphs.replay(r)
                acc.add_(loss.detach().reshape(()))
                timing["replay_cpu_ms"] += (time.perf_counter() - t) * 1000
                c.record()
                events.append((a, b, c, r))
            start_comm = torch.cuda.Event(enable_timing=True)
            end_comm = torch.cuda.Event(enable_timing=True)
            end_opt = torch.cuda.Event(enable_timing=True)
            start_comm.record()
            average_gradients(graphs.params, world)
            end_comm.record()
            norm = torch.nn.utils.clip_grad_norm_(
                graphs.params, cfg.train.grad_clip_max_norm
            )
            # Only one host validation per optimizer step; abort before corrupt updates.
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradient norm")
            present = torch.tensor(int(has_template), device=dev)
            if world > 1:
                dist.all_reduce(present, op=dist.ReduceOp.MAX)
            held_grads = []
            if not bool(present):
                for parameter in unused_if_empty:
                    held_grads.append((parameter, parameter.grad))
                    parameter.grad = None
            opt.step()
            for parameter, grad in held_grads:
                parameter.grad = grad
            if sched:
                sched.step()
            with torch.no_grad():
                if ema:
                    torch._foreach_mul_(list(ema.values()), cfg.train.ema_decay)
                    torch._foreach_add_(
                        list(ema.values()),
                        [live[k].detach() for k in ema],
                        alpha=1 - cfg.train.ema_decay,
                    )
            end_opt.record()
            if world > 1:
                dist.all_reduce(acc)
            value = float(acc / (ga * world))
            torch.cuda.synchronize()
            global_step += 1
            done += 1
            epoch_loss += value
            timing.update(
                step=global_step,
                wall_ms=(time.perf_counter() - begin) * 1000,
                copy_gpu_ms=sum(a.elapsed_time(b) for a, b, c, r in events),
                replay_gpu_ms=sum(b.elapsed_time(c) for a, b, c, r in events),
                comm_ms=start_comm.elapsed_time(end_comm),
                optimizer_ema_ms=end_comm.elapsed_time(end_opt),
                recycles=[r for a, b, c, r in events],
            )
            if diagnostic_steps:
                records.append(timing)
            metrics_dir = os.environ.get("MW_GRAPH_METRICS_DIR")
            if metrics_dir and done <= int(os.environ.get("MW_GRAPH_METRICS_STEPS", "100")):
                destination = Path(metrics_dir)
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / f"rank{rank}.jsonl").open("a") as handle:
                    handle.write(json.dumps(timing) + "\n")
            if rank == 0:
                message = f"Step {global_step:8d} (Epoch {epoch:5d}) | train/distogram_loss={value:.6f} step_time_ms={timing['wall_ms']:.1f}"
                if diagnostic_steps:
                    print(message, flush=True)
                else:
                    log.info(message)
                if cfg.train.use_wandb and not diagnostic_steps:
                    wandb.log(
                        {
                            "train/distogram_loss_step": value,
                            "train/total_loss_step": value,
                            "train/main_loss_step": value,
                        },
                        step=global_step,
                    )
            if diagnostic_steps and done >= diagnostic_steps:
                (run_dir / f"timing-rank{rank}.json").write_text(
                    json.dumps(records, indent=2)
                )
                if world > 1:
                    dist.barrier()
                    dist.destroy_process_group()
                return
        epoch += 1
        epoch_rng = gather_rng_states(world)
        if rank == 0:
            elapsed = time.perf_counter() - ep_start
            log.info("Epoch %d time=%.3fs", epoch, elapsed)
            if cfg.train.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/distogram_loss": epoch_loss / nsteps,
                        "train/total_loss": epoch_loss / nsteps,
                        "train/main_loss": epoch_loss / nsteps,
                        "train/epoch_time": elapsed,
                    },
                    step=global_step,
                )
            save_checkpoint(run_dir, checkpoint_payload(epoch_rng), archive=epoch % cfg.train.save_freq == 0)
        if world > 1:
            dist.barrier()
    if rank == 0 and cfg.train.use_wandb and not diagnostic_steps:
        wandb.finish()
    if world > 1:
        dist.destroy_process_group()
