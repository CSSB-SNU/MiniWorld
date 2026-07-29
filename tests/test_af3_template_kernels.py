"""Kernel correctness: AF3 template embedder with TRITON kernels vs pure-PyTorch
reference (same weights, same input, batched BT=4). Reports numeric agreement + speed."""
import torch, time
import torch.nn.functional as F
from miniworld.data.features import TemplateFeatures
from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder
from team_gm.modules.exceptions import ImplementationType

dev = "cuda"; B, L, T, d_pair = 1, 384, 4, 128
torch.manual_seed(0)
print("device", torch.cuda.get_device_name(0))

def mk_inputs():
    tmpl = TemplateFeatures(
        mask=torch.tensor([[True, True, True, False]], device=dev),
        ids=torch.zeros((B, T, L), dtype=torch.long, device=dev),
        res_type=torch.randint(0, 32, (B, T, L), device=dev),
        cb_xyz=torch.randn(B, T, L, 3, device=dev) * 10,
        cb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev),
        bb_xyz=torch.randn(B, T, L, 3, 3, device=dev) * 10,
        bb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev))
    pair = torch.randn(B, L, L, d_pair, device=dev)
    asym = torch.zeros(B, L, dtype=torch.long, device=dev); asym[:, L//2:] = 1
    tmask = torch.ones((B, L), dtype=torch.bool, device=dev)
    return pair, tmpl, asym, tmask

emb_pt = AF3TemplateEmbedder(d_pair=d_pair, implementation=ImplementationType.PYTORCH).to(dev).eval()
emb_tr = AF3TemplateEmbedder(d_pair=d_pair, implementation=ImplementationType.MINIWORLD_KERNELS).to(dev).eval()
emb_tr.load_state_dict(emb_pt.state_dict())   # identical weights
pair, tmpl, asym, tmask = mk_inputs()

def run(emb):
    with torch.no_grad():
        return emb(pair, tmpl, asym, tmask)

try:
    o_pt = run(emb_pt); o_tr = run(emb_tr)
    d = (o_pt - o_tr).abs()
    rel = d.max() / (o_pt.abs().max() + 1e-9)
    cos = F.cosine_similarity(o_pt.flatten().float(), o_tr.flatten().float(), dim=0)
    print(f"[kernel vs pytorch] max|diff|={d.max():.3e} mean|diff|={d.mean():.3e} "
          f"rel={rel:.3e} cos={cos:.6f}")
except Exception as ex:
    print("KERNEL PATH FAILED:", repr(ex)[:300]); raise

# timing both
for name, emb in [("pytorch", emb_pt), ("triton", emb_tr)]:
    for _ in range(3): run(emb)   # warmup/JIT
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): run(emb)
    torch.cuda.synchronize()
    print(f"[speed] {name:8} fwd={ (time.perf_counter()-t0)/10*1000:.1f}ms")
print("AF3_KERNEL_CMP_DONE")
