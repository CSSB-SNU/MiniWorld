"""GPU test of AF3TemplateEmbedder: kernel correctness (batched vs per-template loop),
finiteness, and timing at realistic shapes (L=384, N_temp=4)."""
import torch, time
from miniworld.data.features import TemplateFeatures
from miniworld.modules.template_embedder_af3 import (
    AF3TemplateEmbedder, _dgram_from_positions, _backbone_unit_vectors)
import torch.nn.functional as F

dev = "cuda"
B, L, T, d_pair = 1, 384, 4, 128
torch.manual_seed(0)
print("device", torch.cuda.get_device_name(0))

emb = AF3TemplateEmbedder(d_pair=d_pair).to(dev)
tmpl = TemplateFeatures(
    mask=torch.tensor([[True, True, True, False]], device=dev),   # 3 valid, 1 pad
    ids=torch.zeros((B, T, L), dtype=torch.long, device=dev),
    res_type=torch.randint(0, 32, (B, T, L), device=dev),
    cb_xyz=torch.randn(B, T, L, 3, device=dev) * 10,
    cb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev),
    bb_xyz=torch.randn(B, T, L, 3, 3, device=dev) * 10,
    bb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev),
)
pair = torch.randn(B, L, L, d_pair, device=dev)
asym = torch.zeros(B, L, dtype=torch.long, device=dev); asym[:, L // 2:] = 1
tmask = torch.ones((B, L), dtype=torch.bool, device=dev)

def ref_loop(emb, pair, tmpl, asym, tmask):
    dt = pair.dtype
    query = emb.proj_query(emb.ln_query(pair))
    mc = (asym[:, :, None] == asym[:, None, :])[..., None].to(dt)
    summed = torch.zeros_like(query)
    for t in range(tmpl.mask.shape[1]):
        cb, cbm = tmpl.cb_xyz[:, t], tmpl.cb_mask[:, t]
        rt = tmpl.res_type[:, t].clamp(0, emb.num_res_class - 1)
        bb, bbm = tmpl.bb_xyz[:, t], tmpl.bb_mask[:, t]
        dgram = _dgram_from_positions(cb, emb.dgram_min, emb.dgram_max, emb.dgram_bins)
        pb2d = (cbm[:, :, None] & cbm[:, None, :]).to(dt)[..., None]
        aa = F.one_hot(rt, emb.num_res_class).to(dt)
        uv = _backbone_unit_vectors(bb)
        bb2d = (bbm[:, :, None] & bbm[:, None, :]).to(dt)[..., None]
        act = (query + emb.proj_dgram(dgram) + emb.proj_pb_mask(pb2d)
               + emb.proj_aatype_i(aa)[:, None, :, :] + emb.proj_aatype_j(aa)[:, :, None, :]
               + emb.proj_unit_vec(uv) + emb.proj_bb_mask(bb2d))
        act = act * mc
        act = emb.template_pairformer(act, mask=tmask)
        act = emb.ln_out(act)
        summed = summed + act * tmpl.mask[:, t].to(dt)[:, None, None, None]
    nv = tmpl.mask.to(dt).sum(1)[:, None, None, None]
    return emb.proj_out(F.relu(summed / (1e-7 + nv)))

emb.eval()
with torch.no_grad():
    out_vec = emb(pair, tmpl, asym, tmask)
    out_ref = ref_loop(emb, pair, tmpl, asym, tmask)
diff = (out_vec - out_ref).abs().max().item()
print(f"[correctness] batched vs loop max|diff| = {diff:.3e}  shape={tuple(out_vec.shape)}  finite={torch.isfinite(out_vec).all().item()}")
assert diff < 1e-4, "batched != loop!"

# ---- timing (train mode, fwd + fwd/bwd) ----
emb.train()
for _ in range(3):  # warmup / JIT kernels
    o = emb(pair, tmpl, asym, tmask); o.square().mean().backward(); emb.zero_grad()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        emb(pair, tmpl, asym, tmask)
torch.cuda.synchronize()
fwd = (time.perf_counter() - t0) / 10 * 1000
t0 = time.perf_counter()
for _ in range(10):
    o = emb(pair, tmpl, asym, tmask); o.square().mean().backward(); emb.zero_grad()
torch.cuda.synchronize()
fb = (time.perf_counter() - t0) / 10 * 1000
mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"[speed] L={L} N_temp={T} (BT={B*T})  fwd={fwd:.1f}ms  fwd+bwd={fb:.1f}ms  peak_mem={mem:.2f}GB")
print("AF3_TEMPLATE_GPU_DONE")
