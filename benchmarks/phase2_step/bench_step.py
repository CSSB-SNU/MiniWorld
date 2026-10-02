"""Phase-2 micro-step time per recycle count and under the real random-recycle draw (compile path as the real script:
Fabric + whole-model compile; optional frozen-trunk CUDA graph = the repo's ``enable_trunk_cudagraph``).

Every recycle count goes through the SAME code path the real training takes: with ``model.train_recycle == "random"`` the
count is drawn from ``raw.rng`` (a stub Generator whose first draw is the wanted count), so the compiled variants are shared
with the real random draw (no second compile, no ``_forced_n_recycle`` guard variant)."""

import argparse
import collections
import os
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts")
)  # run_miniworld_*_train
from hydra import compose, initialize_config_dir
from lightning import Fabric

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--recycles", default="4")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--random-steps", type=int, default=0)
ap.add_argument(
    "--forced-then-real",
    action="store_true",
    help="compile a forced count first, then time the real path's first step",
)
ap.add_argument("--profile-r", type=int, default=0)
ap.add_argument(
    "--stall-probe",
    type=int,
    default=0,
    help="N random-draw steps with per-step wall/CPU time and python-GC events",
)
ap.add_argument(
    "--script-warmup",
    action="store_true",
    help="run the (patched) R._warmup_bucket_shapes first, as the real script does",
)
ap.add_argument(
    "--forced",
    action="store_true",
    help="select the count through _forced_n_recycle instead of the stub rng",
)
ap.add_argument("--trunk-graph", default="")
ap.add_argument(
    "--engine-setting",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="miniworld_engine.settings field, e.g. b200_engine_triton=True (repeatable; an unknown field is an error)",
)
ap.add_argument("--tag", default="p2")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()

from benchmarks.common import apply_engine_settings, header  # noqa: E402

apply_engine_settings(a.engine_setting)

import run_miniworld_distogram_train  # noqa: F401
import run_miniworld_diffusion_train as R
from miniworld.configs import TemplateConfig
from miniworld.models.diffusion import Client
from miniworld.training import trainable_parameters
from miniworld.utils import get_step_decay_scheduler_with_warmup

cfgp = Path(a.config).resolve()
with initialize_config_dir(str(cfgp.parent), version_base=None):
    cfg = R.Config.model_validate(
        compose(config_name=cfgp.name, overrides=["train.use_wandb=False", *a.overrides])
    )
fabric = Fabric(devices=1)
fabric.launch()
fabric.seed_everything(cfg.train.seed or 0)
client = Client(
    Client.Config(train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss)
)
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 512
raw = R._find_recycle_model(client.model)
for name in raw._TRUNK_MODULE_NAMES:
    getattr(raw, name).requires_grad_(False)
raw._set_trunk_eval()
dev = torch.device("cuda", 0)
n_tok, n_atom = cfg.data.crop.max_tokens, cfg.data.crop.max_atoms
batch = R._build_precompile_batch(
    device=dev,
    msa_depth=cfg.data.msa.max_msa_depth,
    n_tokens=n_tok,
    n_atoms=n_atom,
    n_templates=TemplateConfig().n_templates,
    num_res_class=cfg.model.shared.num_res_class,
).to(device=dev)
if cfg.train.compile and a.trunk_graph:
    client.model.diffusion_module.compile(dynamic=False)
    client.model.enable_trunk_cudagraph(a.trunk_graph)
elif cfg.train.compile:
    client.model.compile(dynamic=False)
optimizer = torch.optim.Adam(
    trainable_parameters(client.model), cfg.train.max_lr, betas=(0.9, 0.95)
)
scheduler = get_step_decay_scheduler_with_warmup(
    optimizer=optimizer,
    warmup_steps=cfg.train.warmup_steps,
    decay_steps=cfg.train.decay_steps,
    decay_factor=cfg.train.decay_factor,
)
client.setup(
    fabric=fabric,
    optimizer=optimizer,
    scheduler=scheduler,
    gradient_accumulation_steps=cfg.train.grad_accum_steps,
    gradient_clip_norm=cfg.train.grad_clip_max_norm,
)
client.model.train()
hi = raw.n_recycle_max + 1
random_mode = raw.config.train_recycle == "random"
print(header(a.tag), flush=True)
print(
    f"BENCH [{a.tag}] tokens {n_tok} atoms {n_atom} A {cfg.train.num_augment} train_recycle={raw.config.train_recycle} n_recycle_max={raw.n_recycle_max} "
    f"trunk_graph='{a.trunk_graph}' ckpt token/atom {cfg.model.diffusion.token_dit.n_checkpoint_segments}/{cfg.model.diffusion.atom_swa.n_checkpoint_segments}",
    flush=True,
)


def seed_for(target):
    for s in range(100000):
        if int(np.random.default_rng(s).integers(1, hi)) == target:
            return s
    raise RuntimeError(target)


SEEDS = {r: seed_for(r) for r in range(1, hi)}


def set_count(r):
    """The next step's recycle count is ``r`` through the real path (random mode: stub rng; otherwise forced)."""
    if random_mode and not a.forced:
        raw._forced_n_recycle = None
        raw.rng = np.random.default_rng(SEEDS[r])
    else:
        raw._forced_n_recycle = r


def step():
    with fabric.no_backward_sync(client.model, enabled=True):
        return client.training_step(batch)


def timed(n, r):
    ts = []
    for _ in range(n):
        set_count(r)
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return ts


def ngraphs():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


if a.script_warmup:
    t0 = time.time()
    g0 = ngraphs()
    R._warmup_bucket_shapes(client, cfg)
    torch.cuda.synchronize()
    print(
        f"BENCH [{a.tag}] script warm-up (R._warmup_bucket_shapes) took {time.time() - t0:.0f}s | new dynamo graphs {ngraphs() - g0} | "
        f"rng restored: {raw.rng is not None} forced={raw._forced_n_recycle}",
        flush=True,
    )
    client.model.train()
    client.optimizer.zero_grad(set_to_none=True)

if a.forced_then_real:
    # the ORIGINAL script warmed up with _forced_n_recycle=2 and then trained on the unforced path: is that compile reused?
    raw._forced_n_recycle = 2
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    print(
        f"BENCH [{a.tag}] forced _forced_n_recycle=2 steps: {', '.join(f'{x:.0f}' for x in ts)} ms (first = compile) | unique dynamo graphs {ngraphs()}",
        flush=True,
    )
    g0 = ngraphs()
    ts = []
    for _ in range(3):
        set_count(2)  # same count, but through the unforced path the training takes
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    print(
        f"BENCH [{a.tag}] SAME count 2 via the unforced real path: {', '.join(f'{x:.0f}' for x in ts)} ms | new dynamo graphs {ngraphs() - g0}",
        flush=True,
    )
    client.optimizer.zero_grad(set_to_none=True)

res = {}
for r in [int(x) for x in a.recycles.split(",") if x]:
    t0 = time.time()
    g0 = ngraphs()
    first = timed(1, r)[0]
    timed(2, r)
    torch.cuda.synchronize()
    client.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    ts = timed(a.steps, r)
    res[r] = statistics.median(ts)
    print(
        f"BENCH [{a.tag}] recycle={r}: micro-step median {res[r]:.1f} ms (min {min(ts):.1f}, max {max(ts):.1f}, n={len(ts)}) | first call {first:.0f} ms | "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | new dynamo graphs {ngraphs() - g0} | {time.time() - t0:.0f}s",
        flush=True,
    )
    if a.profile_r == r:
        from torch.profiler import profile, ProfilerActivity

        set_count(r)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            step()
            torch.cuda.synchronize()
        evs = sorted(
            [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA],
            key=lambda e: e.time_range.start,
        )
        busy = sum(e.device_time for e in evs)
        span = evs[-1].time_range.end - evs[0].time_range.start
        cnt = collections.Counter(
            e.name
            for e in p.events()
            if e.name
            in (
                "cudaGraphLaunch",
                "cudaLaunchKernel",
                "cuLaunchKernel",
                "cuLaunchKernelEx",
                "cudaLaunchKernelExC",
            )
        )
        print(
            f"BENCH [{a.tag}] recycle={r} PROFILE kernels {len(evs)} GPU busy {busy / 1000:.1f} ms span {span / 1000:.1f} ms idle {(span - busy) / 1000:.1f} ms | launches {dict(cnt)}",
            flush=True,
        )
if len(res) > 1:
    print(
        f"BENCH [{a.tag}] per count: "
        + ", ".join(f"R{r} {v:.1f}" for r, v in sorted(res.items()))
        + f" | mean over counts = uniform-draw expectation {sum(res.values()) / len(res):.1f} ms",
        flush=True,
    )

if a.random_steps:
    seed = cfg.train.seed or 0
    n = a.random_steps
    raw._forced_n_recycle = None
    ref = np.random.default_rng(seed)
    seq = [int(ref.integers(1, hi)) for _ in range(n)]
    raw.rng = np.random.default_rng(
        seed
    )  # the real Generator: the model draws one count per step
    g0 = ngraphs()
    ts = []
    for k in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    chk = np.random.default_rng(seed)
    for _ in range(n):
        chk.integers(1, hi)
    one_draw_per_step = raw.rng.bit_generator.state == chk.bit_generator.state
    by = collections.defaultdict(list)
    for t, c in zip(ts, seq):
        by[c].append(t)
    print(
        f"BENCH [{a.tag}] RANDOM draw, real rng, {n} steps: mean {sum(ts) / n:.1f} ms, median {statistics.median(ts):.1f} ms, max {max(ts):.0f} ms | "
        f"exactly one draw per step: {one_draw_per_step} | new dynamo graphs {ngraphs() - g0} | counts {dict(sorted(collections.Counter(seq).items()))}",
        flush=True,
    )
    for c in sorted(by):
        print(
            f"BENCH [{a.tag}]   drawn {c}: n={len(by[c])} median {statistics.median(by[c]):.1f} ms",
            flush=True,
        )
    print(
        f"BENCH [{a.tag}]   first 12 steps (count:ms) "
        + " ".join(f"{c}:{t:.0f}" for c, t in list(zip(seq, ts))[:12]),
        flush=True,
    )

if a.stall_probe:
    import gc

    gc_events = []

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_cb.t = time.perf_counter()
        else:
            gc_events.append(
                (
                    info["generation"],
                    (time.perf_counter() - _gc_cb.t) * 1000,
                    info["collected"],
                )
            )

    gc.callbacks.append(_gc_cb)
    n = a.stall_probe
    raw._forced_n_recycle = None
    raw.rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        g0 = len(gc_events)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c0 = time.process_time()
        step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        c1 = time.process_time()
        rows.append(
            (
                k,
                (t1 - t0) * 1000,
                (c1 - c0) * 1000,
                [(g, round(d)) for g, d, _ in gc_events[g0:]],
            )
        )
    walls = sorted(r[1] for r in rows)
    med = walls[len(walls) // 2]
    slow = [r for r in rows if r[1] > 3 * med]
    by_gen = collections.Counter(g for g, _, _ in gc_events)
    tot_gc = sum(d for _, d, _ in gc_events)
    print(
        f"BENCH [{a.tag}] STALL probe {n} random steps: median {med:.1f} ms, mean {sum(walls) / n:.1f} ms, max {walls[-1]:.0f} ms | steps > 3x median: {len(slow)} | "
        f"python GC events {dict(by_gen)} total {tot_gc:.0f} ms, longest gen2 {max([d for g, d, _ in gc_events if g == 2] or [0]):.0f} ms",
        flush=True,
    )
    for k, w, c, gcs in slow[:8]:
        print(
            f"BENCH [{a.tag}]   slow step {k}: wall {w:.0f} ms, process CPU {c:.0f} ms, GC inside {gcs}",
            flush=True,
        )
    cpu_med = sorted(r[2] for r in rows)[len(rows) // 2]
    print(
        f"BENCH [{a.tag}]   typical step: wall {med:.1f} ms vs process CPU {cpu_med:.1f} ms (CPU-bound launch loop if close)",
        flush=True,
    )
