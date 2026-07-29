"""Per-module fwd+bwd timing under compile=True + full-model CUDA graph capture
(recycle=1, real shape). CUDA events are inserted at module boundaries; recorded
during capture they become graph nodes that re-fire on every replay -> per-module
GPU time in the real (compiled+captured) execution. Event wraps also act as dynamo
graph-breaks, so torch.compile compiles WITHIN each module."""
import sys, statistics, torch
sys.path.insert(0, "scripts")
from pathlib import Path
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config, _build_precompile_batch
from miniworld.configs import TemplateConfig
from miniworld.models.distogram_only import MiniSWAModel
from miniworld.loss.auxiliary import cal_atom_distogram_loss

dev = "cuda"; torch.set_float32_matmul_precision("medium")
torch._dynamo.config.cache_size_limit = 256
with initialize_config_dir(str(Path("configs/miniworld").absolute()), version_base=None):
    raw = compose(config_name="config_distogram_swa_af3_mix_local_8gpu")
cfg = Config.model_validate(raw)
w = cfg.loss.distogram_loss
batch = _build_precompile_batch(
    device=torch.device(dev), msa_depth=cfg.train.bucket_msa_multiple,
    n_tokens=cfg.train.bucket_token_multiple, n_atoms=cfg.train.bucket_atom_multiple,
    n_templates=TemplateConfig().n_templates, num_res_class=cfg.model.shared.num_res_class)
print("device", torch.cuda.get_device_name(0), "| msa", cfg.train.bucket_msa_multiple, "| compile+cudagraph", flush=True)

model = MiniSWAModel(cfg.model).to(dev); model.train(); model._forced_n_recycle = 1
order = ["input_feature_embedder", "add_pair_recycle", "msa_module", "pairformer_blocks", "distogram_head"]
fev, bev = {}, {}
def wrap_fwd(name):
    mod = getattr(model, name); orig = mod.forward
    def timed(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = orig(*a, **k); e.record(); fev[name] = (s, e); return out
    mod.forward = timed
def wrap_bwd(name):
    mod = getattr(model, name)
    def pre(m, go):
        s = torch.cuda.Event(enable_timing=True); s.record(); bev[name] = [s]
    def post(m, gi, go):
        e = torch.cuda.Event(enable_timing=True); e.record()
        if name in bev: bev[name].append(e)
    mod.register_full_backward_pre_hook(pre); mod.register_full_backward_hook(post)
for n in order: wrap_fwd(n); wrap_bwd(n)

def cstep():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = model(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                      sequence=batch.sequence, structure=batch.structure)
        loss = w * cal_atom_distogram_loss(logit, batch.structure.atom_pos,
                                           batch.structure.atom_pos_mask,
                                           batch.scheme.atom_to_token_idx_map)
    loss.backward()

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        model.zero_grad(set_to_none=True); cstep()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
model.zero_grad(set_to_none=False)
try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cstep()
    print("[capture] compile+full-model cudagraph OK", flush=True)
except Exception as ex:
    print("[capture] FAILED:", repr(ex)[:300], flush=True); raise
cap_f, cap_b = dict(fev), dict(bev)

N = 15; step_ms = []
mf = {n: [] for n in order}; mb = {n: [] for n in order}
for _ in range(N):
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
    step_ms.append(t0.elapsed_time(t1))
    for n in order:
        mf[n].append(cap_f[n][0].elapsed_time(cap_f[n][1]))
        if len(cap_b.get(n, [])) == 2: mb[n].append(cap_b[n][0].elapsed_time(cap_b[n][1]))
med = lambda x: statistics.median(x) if x else float('nan')
print("\n=== per-module fwd+bwd under full-model CUDAGRAPH (compile≈no-op: kernels not compiled) (recycle=1, ms) ===", flush=True)
print(f"  {'module':26} {'fwd':>8} {'bwd':>8} {'fwd+bwd':>9}", flush=True)
for n in order:
    f, b = med(mf[n]), med(mb[n])
    bs = "  n/a" if b != b else f"{b:8.2f}"
    tt = "" if (f != f or b != b) else f"{f+b:9.2f}"
    print(f"  {n:26} {f:8.2f} {bs} {tt}", flush=True)
print(f"  {'-'*26}", flush=True)
print(f"  total captured STEP: {med(step_ms):.2f} ms/step", flush=True)
print("MODULE_TIMING_DONE", flush=True)
