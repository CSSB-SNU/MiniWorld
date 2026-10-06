"""Phase-2 full-step CUDA graphs, one graph per recycle count: GPU noise sampling + frozen trunk + diffusion head forward +
EDM loss + backward, gradients accumulating into the static ``.grad``. Timed in ONE process against the same step eager (graph-safe
pieces) and the step as the trainer runs it (``client.training_step``); correctness is checked against eager, including after an
in-place weight update. ``--share-pool`` captures all recycle graphs into one memory pool (replays are sequential).

The capture-safe stand-ins live in ``src/miniworld/training/phase2_graph_safe.py``. The recycle count is baked into each graph at capture time
(``_forced_n_recycle``); a trainer draws the count on the host and replays that graph.

    python -m benchmarks.phase2_step.graph_full --config configs/miniworld/phase2a_diffusion_v200.yaml --steps 10 [--share-pool]
"""

import argparse
import os
import statistics
import sys
import time
import traceback
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts")
)  # run_miniworld_*_train

from miniworld.training.phase2_graph_safe import (
    cal_loss_gs,
    clear_pack_caches,
    graph_safe_sampling,
    quat_to_rotmat,
    rot_trans_gs,
    sample_noise_gs,
    weighted_align_gs,
)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--recycles", default="1,2,3,4")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--tag", default="p2g")
ap.add_argument("--selfcheck-only", action="store_true")
ap.add_argument("--cpu-selfcheck", action="store_true")
ap.add_argument("--no-correctness", action="store_true")
ap.add_argument("--only-correctness", action="store_true")
ap.add_argument(
    "--share-pool",
    action="store_true",
    help="capture every recycle graph into ONE memory pool (replays are sequential)",
)
ap.add_argument("--profile-r", type=int, default=0)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
TAG = a.tag


def log(msg):
    print(f"GRAPH [{TAG}] {msg}", flush=True)


def rel(a_, b_):
    a_, b_ = a_.float().flatten(), b_.float().flatten()
    return float((a_ - b_).norm() / b_.norm().clamp_min(1e-30))


def selfcheck(device):
    from team_gm.utils.align import weighted_align

    g = torch.Generator(device=device)
    g.manual_seed(7)
    n, L = 48, 4096
    x = torch.randn(n, L, 3, device=device, generator=g) * 20
    q = torch.randn(n, 4, device=device, generator=g)
    Rt = quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    w = (torch.rand(n, L, device=device, generator=g) > 0.1).float()
    cases = {
        "rotation+noise": torch.bmm(x, Rt.transpose(-1, -2))
        + 0.5 * torch.randn(n, L, 3, device=device, generator=g),
        "mirror (reflection branch)": torch.bmm(
            x * torch.tensor([1.0, 1.0, -1.0], device=device), Rt.transpose(-1, -2)
        )
        + 0.5 * torch.randn(n, L, 3, device=device, generator=g),
        "noise only (y unrelated)": torch.randn(n, L, 3, device=device, generator=g)
        * 20,
    }
    for name, y in cases.items():
        ref = weighted_align(x, y, w)
        got = weighted_align_gs(x, y, w)
        e_ref = ((ref - y).pow(2).sum(-1) * w).sum(-1) / w.sum(-1)
        e_got = ((got - y).pow(2).sum(-1) * w).sum(-1) / w.sum(-1)
        log(
            f"selfcheck align [{name}]: max|aligned_gs - aligned_svd| {float((got - ref).abs().max()):.3e} | weighted MSE gs {float(e_got.mean()):.5f} vs svd {float(e_ref.mean()):.5f} "
            f"(gs <= svd*(1+1e-4): {bool((e_got <= e_ref * (1 + 1e-4) + 1e-6).all())})"
        )
    Rm = quat_to_rotmat(
        torch.nn.functional.normalize(
            torch.randn(100000, 4, device=device, generator=g), dim=-1
        )
    )
    I = torch.eye(3, device=device)
    log(
        f"selfcheck rotations: max|RR^T-I| {float((Rm @ Rm.transpose(-1, -2) - I).abs().max()):.2e}, det min {float(torch.linalg.det(Rm).min()):.6f}, "
        f"E[R00^2] {float((Rm[:, 0, 0] ** 2).mean()):.4f} (uniform: 0.3333)"
    )
    if device.type != "cuda":
        return

    # capture-ability evidence for the blockers / the replacements
    def try_capture(label, fn):
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        try:
            fn()  # warm
            torch.cuda.synchronize()
            with torch.cuda.stream(s), torch.cuda.graph(gr, stream=s):
                out = fn()
            gr.replay()
            torch.cuda.synchronize()
            return out, "ok"
        except Exception as e:
            torch.cuda.synchronize()
            return None, f"FAILS: {str(e).strip().splitlines()[0][:150]}"

    cov = torch.randn(48, 3, 3, device=device)
    log(
        f"selfcheck capture torch.linalg.svd(48x3x3): {try_capture('svd', lambda: torch.linalg.svd(cov))[1]}"
    )
    xx = torch.randn(1000, device=device)
    log(
        f"selfcheck capture `if torch.isnan(x).any()`: {try_capture('isnan', lambda: bool(torch.isnan(xx).any()))[1]}"
    )
    log(
        f"selfcheck capture `.item()`: {try_capture('item', lambda: xx.sum().item())[1]}"
    )
    log(
        f"selfcheck capture torch.tensor(-1.0, device=cuda): {try_capture('tensor', lambda: torch.tensor(-1.0, device=device))[1]}"
    )
    out, st = try_capture(
        "align_gs", lambda: weighted_align_gs(x[:8], cases["rotation+noise"][:8], w[:8])
    )
    log(f"selfcheck capture weighted_align_gs: {st}")
    holder = {}
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    sch = types.SimpleNamespace(
        config=types.SimpleNamespace(
            sigma_data=16.0, P_mean=-1.2, P_std=1.5, sigma_min=4e-4, sigma_max=160.0
        ),
    )

    def sampler():
        return sample_noise_gs(sch, 48), rot_trans_gs(
            types.SimpleNamespace(config=types.SimpleNamespace(translation_noise=1.0)),
            x[:48, :16],
        )

    sampler()
    torch.cuda.synchronize()
    with torch.cuda.stream(s), torch.cuda.graph(gr, stream=s):
        holder["o"] = sampler()
    gr.replay()
    torch.cuda.synchronize()
    s1 = holder["o"][0].clone()
    r1 = holder["o"][1].clone()
    gr.replay()
    torch.cuda.synchronize()
    log(
        f"selfcheck captured GPU sampler draws fresh numbers on each replay: sigma changed {not torch.equal(s1, holder['o'][0])}, rotation changed {not torch.equal(r1, holder['o'][1])}"
    )


if a.cpu_selfcheck:
    selfcheck(torch.device("cpu"))
    sys.exit(0)
selfcheck(torch.device("cuda", 0))
if a.selfcheck_only:
    sys.exit(0)

# ---------------------------------------------------------------- model / batch (same construction as p2_bench2)
from hydra import compose, initialize_config_dir
from lightning import Fabric
import run_miniworld_distogram_train  # noqa: F401
import run_miniworld_diffusion_train as R
from miniworld.configs import TemplateConfig
from miniworld.models.diffusion import Client
from miniworld.training import trainable_parameters
from miniworld.training.precision import model_autocast
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
if cfg.train.compile:
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
params = [p for p in client.model.parameters() if p.requires_grad]
A = cfg.train.num_augment
lc = client.config.loss
assert lc.smooth_lddt_loss == 0 and lc.bond_loss == 0, (
    "the graph-safe loss covers the EDM term only (phase 2a/2b weights)"
)
log(
    f"tokens {n_tok} atoms {n_atom} A {A} compile {cfg.train.compile} trainable tensors {len(params)} train_recycle={raw.config.train_recycle} "
    f"ckpt token/atom {cfg.model.diffusion.token_dit.n_checkpoint_segments}/{cfg.model.diffusion.atom_swa.n_checkpoint_segments}"
)


def loss_fn_gs(x0, x_input, t_emb, sigma, x_mask):
    b = batch
    with model_autocast(client.model):     # as Client.loss_fn: the forward under the training precision, the loss fp32
        upd = client.model.forward(
            msa=b.msa,
            template=b.template,
            reference=b.reference,
            scheme=b.scheme,
            sequence=b.sequence,
            structure=b.structure,
            x_t=x_input,
            x_mask=x_mask,
            t_emb=t_emb,
        ).float()
    et = b.chain.entity_type
    w_chain = (
        1.0
        + lc.alpha_dna * (et == 4).to(upd.dtype)
        + lc.alpha_rna * ((et == 3) | (et == 5)).to(upd.dtype)
        + lc.alpha_ligand * ((et == 6) | (et == 7)).to(upd.dtype)
    )
    atom_weight = torch.gather(w_chain, dim=1, index=b.scheme.atom_to_chain_id)
    return lc.diffusion_loss * cal_loss_gs(
        client.diffuser, x0, x_input, upd, sigma, x_mask, atom_weight
    )


def gs_step(r):
    raw._forced_n_recycle = r
    with graph_safe_sampling(client.diffuser):
        x0, x_input, x_mask, t_emb, sigma = client.diffuser.sample(
            batch.structure.atom_pos, num_augment=A, mask=batch.structure.atom_pos_mask
        )
    loss = loss_fn_gs(x0, x_input, t_emb, sigma, x_mask)
    loss.backward()
    return loss.detach()


def orig_step(r):
    raw._forced_n_recycle = r
    with fabric.no_backward_sync(client.model, enabled=True):
        return client.training_step(batch)


class StepGraphs:
    def __init__(self, body):
        self.body, self.graphs, self.out = body, {}, {}
        self.stream = torch.cuda.Stream()
        self.pool = torch.cuda.graph_pool_handle() if a.share_pool else None

    def clear(self):
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    def warm(self, r, n=3):
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            for _ in range(n):
                self.clear()
                self.body(r)
        cur.wait_stream(self.stream)
        torch.cuda.synchronize()
        if any(p.grad is None for p in params):
            raise RuntimeError(
                "graph capture needs a .grad on every trainable parameter"
            )

    def capture(self, r):
        clear_pack_caches()
        self.clear()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(self.stream),
            torch.cuda.graph(g, stream=self.stream, pool=self.pool),
        ):
            self.out[r] = self.body(r)
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graphs[r] = g
        clear_pack_caches()
        self.clear()

    def replay(self, r):
        self.graphs[r].replay()


def t_sync(fn, n):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts)


def t_pipe(fn, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / n


def profile_once(fn, label):
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    evs = sorted(
        [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA],
        key=lambda e: e.time_range.start,
    )
    busy = sum(e.device_time for e in evs)
    span = evs[-1].time_range.end - evs[0].time_range.start
    log(
        f"PROFILE {label}: kernels {len(evs)} GPU busy {busy / 1000:.1f} ms span {span / 1000:.1f} ms idle {(span - busy) / 1000:.1f} ms"
    )


G = StepGraphs(gs_step)
recs = [int(x) for x in a.recycles.split(",") if x]
table = {}
for r in [] if a.only_correctness else recs:
    t0 = time.time()
    try:
        G.warm(r)  # eager (compile) on the side stream
        orig = t_sync(
            lambda: orig_step(r), a.steps
        )  # the step as the repo runs it (per-step host syncs)
        client.optimizer.zero_grad(set_to_none=True) if False else None
        G.clear()
        gs_s = t_sync(lambda: gs_step(r), a.steps)
        gs_p = t_pipe(lambda: gs_step(r), a.steps)
        torch.cuda.reset_peak_memory_stats()
        G.capture(r)
        pool = torch.cuda.memory_allocated() / 2**30
        gr_s = t_sync(lambda: G.replay(r), a.steps)
        gr_p = t_pipe(lambda: G.replay(r), a.steps)
        table[r] = (orig[0], gs_s[0], gs_p, gr_s[0], gr_p)
        log(
            f"recycle={r}: as-is {orig[0]:.1f} ms | eager graph-safe (sync/step) {gs_s[0]:.1f}, (pipelined) {gs_p:.1f} | "
            f"CUDA graph (sync/step) {gr_s[0]:.1f} [min {gr_s[1]:.1f} max {gr_s[2]:.1f}], (pipelined) {gr_p:.1f} | allocated after capture {pool:.1f} GiB "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB | {time.time() - t0:.0f}s"
        )
        if a.profile_r == r:
            profile_once(lambda: gs_step(r), f"eager R{r}")
            profile_once(lambda: G.replay(r), f"graph replay R{r}")
    except Exception:
        log(
            f"recycle={r}: FAILED\n"
            + "".join(traceback.format_exc().splitlines(True)[-14:])
        )
        break
if table:
    k = len(table)
    cols = list(zip(*[table[r] for r in sorted(table)]))
    log(
        "mean over counts (uniform draw expectation): as-is {:.1f} | eager gs sync {:.1f} / pipelined {:.1f} | graph sync {:.1f} / pipelined {:.1f} ms".format(
            *[sum(c) / k for c in cols]
        )
    )


# ---------------------------------------------------------------- input staging cost (what a graph needs per step)
def collect(o, out, depth=0):
    if isinstance(o, torch.Tensor):
        out.append(o)
    elif depth < 5 and isinstance(o, (list, tuple)):
        for v in o:
            collect(v, out, depth + 1)
    elif depth < 5 and isinstance(o, dict):
        for v in o.values():
            collect(v, out, depth + 1)
    elif depth < 5 and hasattr(o, "__dict__"):
        for v in vars(o).values():
            collect(v, out, depth + 1)


try:
    if a.only_correctness:
        raise RuntimeError("skipped (--only-correctness)")
    ts_ = []
    collect(batch, ts_)
    seen = set()
    uniq = []
    for t in ts_:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append(t)
    pinned = [t.detach().cpu().pin_memory() for t in uniq]
    tot = sum(t.numel() * t.element_size() for t in uniq) / 2**20
    lat = []
    for _ in range(5):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for d_, s_ in zip(uniq, pinned):
            d_.copy_(s_, non_blocking=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    log(
        f"input staging: {len(uniq)} batch tensors, {tot:.0f} MiB, pinned H2D copy into the static buffers {statistics.median(lat):.1f} ms (overlappable with the previous replay)"
    )
    del pinned
except Exception as e:
    log(f"input staging measurement failed: {type(e).__name__}: {str(e)[:160]}")

# ---------------------------------------------------------------- correctness: replay uses the CURRENT weights
if not a.no_correctness and recs:
    r = recs[0] if 2 not in recs else 2
    with torch.no_grad(), graph_safe_sampling(client.diffuser):
        s0 = client.diffuser.sample(
            batch.structure.atom_pos, num_augment=A, mask=batch.structure.atom_pos_mask
        )
    names = ("x0", "x_input", "x_mask", "t_emb", "sigma")
    ST = {k: v.clone() for k, v in zip(names, s0)}

    def body_static(r_):
        raw._forced_n_recycle = r_
        loss = loss_fn_gs(
            ST["x0"], ST["x_input"], ST["t_emb"], ST["sigma"], ST["x_mask"]
        )
        loss.backward()
        return loss.detach()

    def flat_grads():
        return torch.cat([p.grad.detach().float().flatten() for p in params])

    Gs = StepGraphs(body_static)
    # (0) the alignment replacement in isolation: ONE forward, then the as-is cal_loss (SVD) vs cal_loss_gs (Horn) on its output,
    #     under the training scripts' matmul precision ('medium': fp32 matmuls in bf16) and under 'highest'
    raw._forced_n_recycle = r
    b_ = batch
    with model_autocast(client.model):
        upd = client.model.forward(
            msa=b_.msa,
            template=b_.template,
            reference=b_.reference,
            scheme=b_.scheme,
            sequence=b_.sequence,
            structure=b_.structure,
            x_t=ST["x_input"],
            x_mask=ST["x_mask"],
            t_emb=ST["t_emb"],
        ).float().detach()
    et_ = b_.chain.entity_type
    w_chain_ = (
        1.0
        + lc.alpha_dna * (et_ == 4).to(upd.dtype)
        + lc.alpha_rna * ((et_ == 3) | (et_ == 5)).to(upd.dtype)
        + lc.alpha_ligand * ((et_ == 6) | (et_ == 7)).to(upd.dtype)
    )
    aw_ = torch.gather(w_chain_, dim=1, index=b_.scheme.atom_to_chain_id)
    for prec in ("medium", "highest"):
        torch.set_float32_matmul_precision(prec)
        s_orig = float(
            client.diffuser.cal_loss(
                x0=ST["x0"],
                x_input=ST["x_input"],
                x_update=upd,
                sigma=ST["sigma"],
                mask=ST["x_mask"],
                atom_weight=aw_,
            )
        )
        s_gs = float(
            cal_loss_gs(
                client.diffuser,
                ST["x0"],
                ST["x_input"],
                upd,
                ST["sigma"],
                ST["x_mask"],
                aw_,
            )
        )
        log(
            f"correctness R{r}: structure loss, matmul precision '{prec}': as-is (SVD) {s_orig:.6f} vs graph-safe (Horn, exact fp32) {s_gs:.6f} (rel {abs(s_orig - s_gs) / abs(s_orig):.2e})"
        )
    torch.set_float32_matmul_precision("medium")

    def eager_run():
        Gs.clear()
        l = body_static(r)
        torch.cuda.synchronize()
        return float(l), flat_grads().clone()

    Gs.warm(r)
    l1, g1 = eager_run()
    l2, g2 = eager_run()
    Gs.capture(r)
    Gs.clear()
    Gs.replay(r)
    torch.cuda.synchronize()
    lg, gg = float(Gs.out[r]), flat_grads().clone()
    log(
        f"correctness R{r}: eager vs eager (noise floor): loss rel {abs(l1 - l2) / abs(l1):.2e}, grad rel-L2 {rel(g1, g2):.2e} | "
        f"graph vs eager: loss rel {abs(lg - l1) / abs(l1):.2e}, grad rel-L2 {rel(gg, g1):.2e}"
    )
    if (
        a.share_pool
    ):  # graphs of two recycle counts in ONE pool, replayed in an interleaved order
        r2 = 4 if r != 4 else 3

        def eager_run_r(rr):
            Gs.clear()
            l_ = body_static(rr)
            torch.cuda.synchronize()
            return float(l_), flat_grads().clone()

        Gs.warm(r2)
        e1, e2 = (l1, g1), eager_run_r(r2)
        Gs.capture(r2)
        worst = 0.0
        for rr in (r2, r, r2, r):
            Gs.clear()
            Gs.replay(rr)
            torch.cuda.synchronize()
            lg_, gg_ = float(Gs.out[rr]), flat_grads().clone()
            ref = e2 if rr == r2 else e1
            worst = max(worst, rel(gg_, ref[1]), abs(lg_ - ref[0]) / abs(ref[0]))
        log(
            f"correctness SHARED POOL: graphs r={r} and r={r2} captured into one pool, replayed in the order {r2},{r},{r2},{r}: worst rel diff (loss or grads) vs eager {worst:.2e}"
        )
        raw._forced_n_recycle = r
    with torch.no_grad():  # an in-place weight change (what an optimizer step does)
        for p in params:
            p.add_(torch.randn_like(p) * p.detach().abs().mean().clamp_min(1e-6) * 0.05)
    l3, g3 = eager_run()
    Gs.clear()
    Gs.replay(r)
    torch.cuda.synchronize()
    lg2, gg2 = float(Gs.out[r]), flat_grads().clone()
    log(
        f"correctness R{r} AFTER in-place weight update: eager old->new loss change {abs(l3 - l1) / abs(l1):.2e}, grad rel-L2 {rel(g3, g1):.2e} | "
        f"graph(new) vs eager(new): loss rel {abs(lg2 - l3) / abs(l3):.2e}, grad rel-L2 {rel(gg2, g3):.2e} | graph(new) bit-identical to graph(old): {bool(torch.equal(gg2, gg)) and lg2 == lg}"
    )
