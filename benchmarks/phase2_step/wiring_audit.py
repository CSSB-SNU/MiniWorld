"""Kernel wiring audit of the phase-2 training micro-step (current config), three layers of evidence:

  1. static: every engine-selectable module's ``implementation`` (+ plain torch norm/linear counts) per region
  2. eager run (no compile): per-CPU-op GPU time and call counts at recycle 1 and recycle 4 -> per-recycle vs fixed counts, each op
     marked engine (non-aten) or PyTorch (aten::*), with the module forward-call counts that explain them
  3. the real compiled run: GPU time by kernel family (engine / cuBLAS+cutlass / FA4 / inductor / aten / memcpy) and the largest
     non-engine kernels
Nothing here times a step; the profiler only inventories what executes."""

import argparse
import collections
import os
import re
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--tag", default="wire")
ap.add_argument("--no-compiled", action="store_true")
ap.add_argument("--skip-eager", action="store_true")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()


def log(msg):
    print(f"WIRE [{a.tag}] {msg}", flush=True)


from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts")
)  # run_miniworld_*_train
from hydra import compose, initialize_config_dir
from lightning import Fabric
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
    if getattr(raw, name) is not None:  # add_pair_recycle / distogram_head are None on a distogram-diffusion trunk
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
log(
    f"tokens {n_tok} atoms {n_atom} A {cfg.train.num_augment} diffusion dtype {cfg.model.diffusion.dtype} token_dit_kind {cfg.model.diffusion.token_dit_kind} "
    f"ckpt token/atom {cfg.model.diffusion.token_dit.n_checkpoint_segments}/{cfg.model.diffusion.atom_swa.n_checkpoint_segments} engine_backend {cfg.train.engine_backend}"
)

if not a.skip_eager:
    # ------------------------------------------------------------------ 1. static module inventory
    def region_of(name):
        p = name.split(".")
        return ".".join(p[:2]) if p[0] == "diffusion_module" and len(p) > 1 else p[0]

    inv = collections.Counter()
    plain = collections.Counter()
    PLAIN = (
        torch.nn.LayerNorm,
        torch.nn.RMSNorm,
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.GroupNorm,
    )
    for name, m in raw.named_modules():
        impl, backend = getattr(m, "implementation", None), getattr(m, "_backend", None)
        cls = f"{type(m).__module__.split('.')[0]}.{type(m).__name__}"
        if impl is not None or backend is not None:
            inv[
                (region_of(name), cls, str(getattr(impl, "value", impl)), str(backend))
            ] += 1
        elif isinstance(m, PLAIN):
            plain[(region_of(name), type(m).__name__)] += 1
    log(
        "trunk modules: "
        + ", ".join(
            f"{n}={type(getattr(raw, n)).__name__}" for n in raw._TRUNK_MODULE_NAMES
        )
    )
    log(
        "STATIC engine-selectable modules (region | class | implementation | backend) x count:"
    )
    for (reg, cls, impl, be), n in sorted(inv.items()):
        log(f"  {reg:34s} {cls:58s} {impl:12s} {be:10s} x{n}")
    log(
        "STATIC plain torch modules (region | class) x count (these run PyTorch kernels unless fused by the engine op that owns them):"
    )
    for (reg, cls), n in sorted(plain.items()):
        log(f"  {reg:34s} {cls:14s} x{n}")


# ------------------------------------------------------------------ helpers
from torch.profiler import profile, ProfilerActivity

ENGINE_PAT = re.compile(
    r"_sm100|k3g|k1w|bo_pv|adaln|res_c|res_adaln|pair_bias|ln_vg|swiglu|opm_|pwa_|swa_|transition_|trimul|conditioned|dpb|token_pair_init|\(anonymous namespace\)",
    re.I,
)


def kernel_family(n):
    l = n.lower()
    if l.startswith("triton_"):
        return "inductor"
    if "flash" in l:
        return "FA4"
    if ENGINE_PAT.search(n):
        return "engine(custom)"
    if any(k in l for k in ("nvjet", "cutlass", "gemm", "cublas", "splitk", "gemv")):
        return "GEMM(cuBLAS/cutlass)"
    if "memcpy" in l or "memset" in l:
        return "memcpy/memset"
    return "aten/other"


def step(r):
    raw._forced_n_recycle = r
    with fabric.no_backward_sync(client.model, enabled=True):
        return client.training_step(batch)


def prof_step(r):
    torch.cuda.synchronize()
    client.optimizer.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        step(r)
        torch.cuda.synchronize()
    return p


def kernels_of(p):
    return [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]


if not a.skip_eager:
    # ------------------------------------------------------------------ 2. eager run: per-op counts at recycle 1 and 4
    call_counts = {}
    hooks = []
    cnt = collections.Counter()
    for name, m in raw.named_modules():
        if (
            getattr(m, "implementation", None) is not None
            or getattr(m, "_backend", None) is not None
            or isinstance(m, PLAIN)
        ):
            key = (
                region_of(name),
                f"{type(m).__module__.split('.')[0]}.{type(m).__name__}",
            )
            hooks.append(
                m.register_forward_pre_hook(lambda mod, args, key=key: cnt.update([key]))
            )
    for _ in range(2):
        step(2)
    res = {}
    for r in (1, 4):
        cnt.clear()
        p = prof_step(r)
        ops = collections.defaultdict(lambda: [0, 0.0])
        for e in p.key_averages():
            if (
                e.device_type == torch.autograd.DeviceType.CPU
                and e.self_device_time_total > 0
            ):
                ops[e.key][0] += e.count
                ops[e.key][1] += e.self_device_time_total
        ks = kernels_of(p)
        res[r] = (ops, dict(cnt), sum(e.device_time for e in ks), len(ks))
    for h in hooks:
        h.remove()
    tot1, tot4 = res[1][2], res[4][2]
    log(
        f"EAGER profile: GPU busy recycle1 {tot1 / 1000:.1f} ms ({res[1][3]} kernels), recycle4 {tot4 / 1000:.1f} ms ({res[4][3]} kernels) -> per recycle {(tot4 - tot1) / 3000:.1f} ms, fixed {(tot1 - (tot4 - tot1) / 3) / 1000:.1f} ms"
    )
    log(
        "EAGER GPU time by launching CPU op (engine = non-aten op names; PyTorch = aten::*), per recycle = (r4 - r1)/3, fixed = r1 - per recycle:"
    )
    names = set(res[1][0]) | set(res[4][0])
    rows = []
    for n in names:
        c1, t1 = res[1][0].get(n, [0, 0.0])
        c4, t4 = res[4][0].get(n, [0, 0.0])
        per_c, per_t = (c4 - c1) / 3, (t4 - t1) / 3
        rows.append((t4, n, c4, per_c, c1 - per_c, per_t, t1 - per_t))
    rows.sort(reverse=True)
    engine_t = sum(r[0] for r in rows if not r[1].startswith("aten::"))
    aten_t = sum(r[0] for r in rows if r[1].startswith("aten::"))
    log(
        f"EAGER recycle-4 GPU time: launched by non-aten (engine/custom) ops {engine_t / 1000:.1f} ms, by aten:: ops {aten_t / 1000:.1f} ms"
    )
    for t4, n, c4, per_c, fix_c, per_t, fix_t in rows[:48]:
        kind = "PyTorch" if n.startswith("aten::") else "ENGINE "
        log(
            f"  {kind} {t4 / 1000:8.2f} ms  calls r4 {c4:5.0f} = {per_c:6.1f}/recycle + {fix_c:6.1f} fixed | GPU/recycle {per_t / 1000:6.2f} ms, fixed {fix_t / 1000:6.2f} ms  {n[:90]}"
        )
    log("EAGER module forward calls at recycle 4 (region | class: calls | per recycle):")
    for key in sorted(set(res[4][1]) | set(res[1][1])):
        c4, c1 = res[4][1].get(key, 0), res[1][1].get(key, 0)
        log(
            f"  {key[0]:34s} {key[1]:56s} r4 {c4:5d} = {(c4 - c1) / 3:6.1f}/recycle + {c1 - (c4 - c1) / 3:6.1f} fixed"
        )


# ------------------------------------------------------------------ 3. the real compiled run
if not a.no_compiled:
    raw.compile(
        dynamic=False
    )  # the raw module (compiling the Fabric wrapper after setup recurses); same graph the script compiles before setup
    for _ in range(3):
        step(4)
    p = prof_step(4)
    ks = kernels_of(p)
    busy = sum(e.device_time for e in ks)
    fam = collections.defaultdict(lambda: [0, 0.0])
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in ks:
        f = kernel_family(e.name)
        fam[f][0] += 1
        fam[f][1] += e.device_time
        agg[(f, e.name[:110])][0] += 1
        agg[(f, e.name[:110])][1] += e.device_time
    log(
        f"COMPILED recycle-4 micro-step: {len(ks)} kernels, GPU busy {busy / 1000:.1f} ms"
    )
    for f, (c, t) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
        log(f"  {f:22s} {t / 1000:8.2f} ms ({100 * t / busy:5.1f}%)  {c:5d} kernels")
    log(
        "COMPILED largest kernels outside the engine family (family | ms | count | name):"
    )
    for (f, n), (c, t) in sorted(
        ((k, v) for k, v in agg.items() if k[0] not in ("engine(custom)",)),
        key=lambda kv: -kv[1][1],
    )[:22]:
        log(f"  {f:22s} {t / 1000:7.3f} ms x{c:<4d} {n}")
    log("COMPILED largest engine kernels (ms | count | name):")
    for (f, n), (c, t) in sorted(
        ((k, v) for k, v in agg.items() if k[0] == "engine(custom)"),
        key=lambda kv: -kv[1][1],
    )[:30]:
        log(f"  {t / 1000:7.3f} ms x{c:<4d} {n}")
