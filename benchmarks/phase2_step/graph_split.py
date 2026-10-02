"""Phase-2 SPLIT CUDA graphs, independent of the recycle count.

The per-recycle full-step graphs (p2_graph.py) differ only in the length of the trunk's recycle loop; sampling + diffusion head +
EDM loss + backward are identical, and the frozen trunk runs under no_grad, so autograd never crosses the trunk/head boundary
(token_single_input, token_pair_trunk).  Cut there:

  E  : input embedder (once per micro-step)                    -> writes the static boundary buffers
  T  : ONE trunk recycle step, replayed r times (PT <- step(PT))
  H  : GPU sampling + diffusion head fwd + EDM loss + backward  (reads the static boundary buffers)

Timed per recycle count r, all in this process:
  (a) split, compiled pieces, host loop, NO graph        (baseline of this structure)
  (c) trunk as (a), head as ONE graph H                  (1 graph)
  (b) E + T x r + H graphs                               (3 graphs: embed, trunk step, head; r-independent)
and memory (reserved) after each capture, plus replay-vs-eager correctness incl. an in-place weight update.
The capture-safe stand-ins (GPU sampling, Horn alignment, no host reads, engine weight-pack caches cleared around each capture)
live in ``graph_safe.py``; ``graph_full.py`` is the per-recycle-count variant.

    python -m benchmarks.phase2_step.graph_split --config configs/miniworld/phase2a_diffusion_v200.yaml --steps 10"""

import argparse
import os
import statistics
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts")
)  # run_miniworld_*_train

from benchmarks.phase2_step.graph_safe import (
    cal_loss_gs,
    clear_pack_caches,
    graph_safe_sampling,
)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--recycles", default="1,2,3,4")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--tag", default="p2g2")
ap.add_argument("--no-correctness", action="store_true")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
TAG = a.tag


def log(msg):
    print(f"SPLIT [{TAG}] {msg}", flush=True)


def rel(a_, b_):
    a_, b_ = a_.float().flatten(), b_.float().flatten()
    return float((a_ - b_).norm() / b_.norm().clamp_min(1e-30))


# ---------------------------------------------------------------- model / batch (same construction as p2_graph)
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
# the three pieces compiled separately (the trunk step ONCE: no recycle count in any compile key)
raw.diffusion_module.compile(dynamic=False)
embed_c = torch.compile(raw._embed, dynamic=False)
step_c = torch.compile(raw._trunk_step, dynamic=False)
params = [p for p in client.model.parameters() if p.requires_grad]
A = cfg.train.num_augment
lc = client.config.loss
assert lc.smooth_lddt_loss == 0 and lc.bond_loss == 0
dit = raw.dit_dtype
asym, templ = batch.scheme.token_asym_id, batch.template
log(
    f"tokens {n_tok} atoms {n_atom} A {A} trainable tensors {len(params)} ckpt token/atom {cfg.model.diffusion.token_dit.n_checkpoint_segments}/{cfg.model.diffusion.atom_swa.n_checkpoint_segments}"
)


def t_sync(fn, n):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def t_pipe(fn, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / n


def clear_grads():
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def gib():
    return torch.cuda.memory_reserved() / 2**30


# ---------------------------------------------------------------- the pieces
with torch.no_grad():
    E0 = embed_c(
        batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure
    )  # (pair_init, single_in, msa_feat, msa_mask, token_mask)
SH = E0[1].to(dit).clone()  # static boundary buffers: the head reads ONLY these
PT = torch.zeros_like(
    E0[0]
)  # trunk-dtype pair buffer: the trunk step reads and writes it in place
del E0


def embed_body():
    with torch.no_grad():
        outs = embed_c(
            batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure
        )
        SH.copy_(outs[1].to(dit))
        PT.zero_()
    return outs


E_OUT = {}


def trunk_step_body():
    with torch.no_grad():
        init, single, msa_feat, msa_mask, token_mask = E_OUT["o"]
        out = step_c(PT, init, single, msa_feat, msa_mask, token_mask, asym, templ)
        PT.copy_(out)


def trunk_eager(r):
    """The trunk exactly as DiffusionModel._condition_impl runs it (compiled pieces, host loop), into SH / PT."""
    with torch.no_grad():
        init, single, msa_feat, msa_mask, token_mask = embed_c(
            batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure
        )
        SH.copy_(single.to(dit))
        PT.zero_()
        for _ in range(r):
            PT.copy_(
                step_c(PT, init, single, msa_feat, msa_mask, token_mask, asym, templ)
            )


def atom_weight():
    et = batch.chain.entity_type
    w = (
        1.0
        + lc.alpha_dna * (et == 4).float()
        + lc.alpha_rna * ((et == 3) | (et == 5)).float()
        + lc.alpha_ligand * ((et == 6) | (et == 7)).float()
    )
    return torch.gather(w, dim=1, index=batch.scheme.atom_to_chain_id)


def head_loss(x0, x_input, x_mask, t_emb, sigma):
    upd = raw.diffusion_forward(
        batch.reference,
        batch.scheme,
        batch.structure,
        x_input,
        x_mask,
        t_emb,
        SH.to(dit),
        PT.to(dit),
    )
    return lc.diffusion_loss * cal_loss_gs(
        client.diffuser, x0, x_input, upd, sigma, x_mask, atom_weight()
    )


def head_body():
    with graph_safe_sampling(client.diffuser):
        x0, x_input, x_mask, t_emb, sigma = client.diffuser.sample(
            batch.structure.atom_pos, num_augment=A, mask=batch.structure.atom_pos_mask
        )
    loss = head_loss(x0, x_input, x_mask, t_emb, sigma)
    loss.backward()
    return loss.detach()


class Graph:
    def __init__(self, body, name):
        self.body, self.name, self.g, self.out = body, name, None, None
        self.stream = torch.cuda.Stream()

    def warm(self, n=3):
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            for _ in range(n):
                clear_grads()
                self.body()
        cur.wait_stream(self.stream)
        torch.cuda.synchronize()

    def capture(self):
        clear_pack_caches()
        clear_grads()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.cuda.graph(g, stream=self.stream):
            self.out = self.body()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.g = g
        clear_pack_caches()
        clear_grads()

    def replay(self):
        self.g.replay()


recs = [int(x) for x in a.recycles.split(",") if x]
res = {}
mem = {}
try:
    # ---- (a) split eager baseline
    H = Graph(head_body, "H")
    for r in recs:
        trunk_eager(r)
    H.warm()  # compiles the head, creates every .grad
    clear_grads()
    base = {}
    for r in recs:
        trunk_eager(r)
        base[r] = (
            t_sync(lambda: (trunk_eager(r), head_body()), a.steps),
            t_pipe(lambda: (trunk_eager(r), head_body()), a.steps),
        )
    log(
        "split eager (compiled pieces, host loop, no graph) per count: "
        + ", ".join(f"R{r} {base[r][0]:.1f}/{base[r][1]:.1f}" for r in recs)
        + " ms (sync/pipelined)"
    )

    # ---- head graph H
    H.capture()  # capture starts with empty_cache(): reserved afterwards = live tensors + the graph pool
    r_h = gib()
    mem["H"] = r_h - torch.cuda.memory_allocated() / 2**30
    log(
        f"captured H (sampling + head + loss + backward): reserved {r_h:.1f} GiB, live tensors {torch.cuda.memory_allocated() / 2**30:.1f} GiB -> H pool ~{mem['H']:.1f} GiB"
    )
    one = {}
    for r in recs:
        trunk_eager(r)
        one[r] = (
            t_sync(lambda: (trunk_eager(r), H.replay()), a.steps),
            t_pipe(lambda: (trunk_eager(r), H.replay()), a.steps),
        )
    log(
        "(c) trunk eager + head graph H per count: "
        + ", ".join(f"R{r} {one[r][0]:.1f}/{one[r][1]:.1f}" for r in recs)
        + " ms (sync/pipelined)"
    )

    # ---- embed graph E and trunk-step graph T
    E = Graph(embed_body, "E")
    cur = torch.cuda.current_stream()
    E.stream.wait_stream(cur)
    with torch.cuda.stream(E.stream):
        for _ in range(2):
            embed_body()
    cur.wait_stream(E.stream)
    torch.cuda.synchronize()
    E.capture()
    r_e = gib()
    mem["E"] = r_e - r_h
    E_OUT["o"] = E.out
    E.replay()
    torch.cuda.synchronize()  # populate the static embed outputs the trunk step reads
    T = Graph(trunk_step_body, "T")
    T.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(T.stream):
        for _ in range(2):
            trunk_step_body()
    torch.cuda.current_stream().wait_stream(T.stream)
    torch.cuda.synchronize()
    T.capture()
    mem["T"] = gib() - r_e
    log(
        f"captured E (embed) +{mem['E']:.2f} GiB, T (one trunk recycle step) +{mem['T']:.2f} GiB | total pools: H {mem['H']:.1f} + E {mem['E']:.2f} + T {mem['T']:.2f} = {sum(mem.values()):.1f} GiB | reserved now {gib():.1f} GiB"
    )

    def split_graph_step(r):
        E.replay()
        for _ in range(r):
            T.replay()
        H.replay()

    for r in recs:
        split_graph_step(r)
        res[r] = (
            t_sync(lambda: split_graph_step(r), a.steps),
            t_pipe(lambda: split_graph_step(r), a.steps),
        )
    log(
        "(b) E + T x r + H graphs per count: "
        + ", ".join(f"R{r} {res[r][0]:.1f}/{res[r][1]:.1f}" for r in recs)
        + " ms (sync/pipelined)"
    )
    k = len(recs)
    log(
        "mean over counts (uniform draw expectation), sync/pipelined: split-eager {:.1f}/{:.1f} | (c) head graph {:.1f}/{:.1f} | (b) E+T+H graphs {:.1f}/{:.1f} ms".format(
            sum(base[r][0] for r in recs) / k,
            sum(base[r][1] for r in recs) / k,
            sum(one[r][0] for r in recs) / k,
            sum(one[r][1] for r in recs) / k,
            sum(res[r][0] for r in recs) / k,
            sum(res[r][1] for r in recs) / k,
        )
    )
except Exception:
    log("FAILED\n" + "".join(traceback.format_exc().splitlines(True)[-16:]))
    sys.exit(1)

# ---------------------------------------------------------------- correctness
if not a.no_correctness:
    try:
        r = 2 if 2 in recs else recs[0]
        with torch.no_grad(), graph_safe_sampling(client.diffuser):
            s0 = client.diffuser.sample(
                batch.structure.atom_pos,
                num_augment=A,
                mask=batch.structure.atom_pos_mask,
            )
        ST = {
            k_: v.clone()
            for k_, v in zip(("x0", "x_input", "x_mask", "t_emb", "sigma"), s0)
        }

        def head_static():
            loss = head_loss(
                ST["x0"], ST["x_input"], ST["x_mask"], ST["t_emb"], ST["sigma"]
            )
            loss.backward()
            return loss.detach()

        Hs = Graph(head_static, "Hs")
        Hs.warm()
        flat = lambda: torch.cat([p.grad.detach().float().flatten() for p in params])

        def eager_run():
            trunk_eager(r)
            clear_grads()
            l = head_static()
            torch.cuda.synchronize()
            return float(l), flat().clone(), PT.clone()

        l1, g1, p1 = eager_run()
        l2, g2, _ = eager_run()
        Hs.capture()
        # trunk through graphs: E, then T r times, then the static-sample head graph
        E.replay()
        for _ in range(r):
            T.replay()
        torch.cuda.synchronize()
        p_graph = PT.clone()
        clear_grads()
        Hs.replay()
        torch.cuda.synchronize()
        lg, gg = float(Hs.out), flat().clone()
        log(
            f"correctness R{r}: eager vs eager (noise floor): loss rel {abs(l1 - l2) / abs(l1):.2e}, grad rel-L2 {rel(g1, g2):.2e} | "
            f"graphs (E + T x{r} + H) vs eager: pair rel-L2 {rel(p_graph, p1):.2e}, loss rel {abs(lg - l1) / abs(l1):.2e}, grad rel-L2 {rel(gg, g1):.2e}"
        )
        with torch.no_grad():
            for p in params:
                p.add_(
                    torch.randn_like(p) * p.detach().abs().mean().clamp_min(1e-6) * 0.05
                )
        l3, g3, _ = eager_run()
        E.replay()
        for _ in range(r):
            T.replay()
        clear_grads()
        Hs.replay()
        torch.cuda.synchronize()
        lg2, gg2 = float(Hs.out), flat().clone()
        log(
            f"correctness R{r} AFTER in-place weight update: eager old->new grad rel-L2 {rel(g3, g1):.2e} | graphs(new) vs eager(new): loss rel {abs(lg2 - l3) / abs(l3):.2e}, grad rel-L2 {rel(gg2, g3):.2e} | "
            f"graphs(new) bit-identical to graphs(old): {bool(torch.equal(gg2, gg)) and lg2 == lg}"
        )
    except Exception:
        log(
            "correctness FAILED\n"
            + "".join(traceback.format_exc().splitlines(True)[-16:])
        )
