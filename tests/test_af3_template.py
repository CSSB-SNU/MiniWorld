"""AF3TemplateEmbedder: matches an independent AF3 reference, and the
engine kernel path agrees with the pure-PyTorch path (same weights)."""
import pytest
import torch
import torch.nn.functional as F

from miniworld.data.features import TemplateFeatures
from miniworld.modules.template_embedder_af3 import (
    AF3TemplateEmbedder, _backbone_unit_vectors)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

B, L, T, D_PAIR = 1, 384, 4, 128


def make_inputs(dev="cuda"):
    torch.manual_seed(0)
    tmpl = TemplateFeatures(
        mask=torch.tensor([[True, True, True, False]], device=dev),   # 3 valid, 1 pad
        ids=torch.zeros((B, T, L), dtype=torch.long, device=dev),
        res_type=torch.randint(0, 32, (B, T, L), device=dev),
        cb_xyz=torch.randn(B, T, L, 3, device=dev) * 10,
        cb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev),
        bb_xyz=torch.randn(B, T, L, 3, 3, device=dev) * 10,
        bb_mask=torch.ones((B, T, L), dtype=torch.bool, device=dev),
    )
    pair = torch.randn(B, L, L, D_PAIR, device=dev)
    asym = torch.zeros(B, L, dtype=torch.long, device=dev)
    asym[:, L // 2:] = 1
    tmask = torch.ones((B, L), dtype=torch.bool, device=dev)
    return pair, tmpl, asym, tmask


def ref_loop(emb, pair, tmpl, asym, tmask):
    """AF3 template embedding written out independently of the module: geometric features are
    masked by the pseudo-beta / backbone masks (intra-chain only) before projection, and every
    template slot is averaged over the FIXED slot count; no template at all gives zero."""
    dt = pair.dtype
    query = emb.proj_query(emb.ln_query(pair))
    mc = (asym[:, :, None] == asym[:, None, :])[..., None].to(dt)
    summed = torch.zeros_like(query)
    n_temp = tmpl.mask.shape[1]
    for t in range(n_temp):
        valid = tmpl.mask[:, t, None]
        cb, cbm = torch.nan_to_num(tmpl.cb_xyz[:, t]), tmpl.cb_mask[:, t] & valid
        bb, bbm = torch.nan_to_num(tmpl.bb_xyz[:, t]), tmpl.bb_mask[:, t] & valid
        rt = tmpl.res_type[:, t].clamp(0, emb.num_res_class - 1)
        dist2 = (cb[:, :, None, :] - cb[:, None, :, :]).pow(2).sum(-1)[..., None]
        edges = torch.linspace(emb.dgram_min, emb.dgram_max, emb.dgram_bins, device=pair.device) ** 2
        upper = torch.cat([edges[1:], edges.new_tensor([float("inf")])])
        dgram = ((dist2 > edges) & (dist2 < upper)).to(dt)
        pb2d = (cbm[:, :, None] & cbm[:, None, :]).to(dt)[..., None] * mc
        bb2d = (bbm[:, :, None] & bbm[:, None, :]).to(dt)[..., None] * mc
        aa = F.one_hot(rt, emb.num_res_class).to(dt)
        uv = _backbone_unit_vectors(bb)
        act = (query + emb.proj_dgram(dgram * pb2d) + emb.proj_pb_mask(pb2d)
               + emb.proj_aatype_i(aa)[:, None, :, :] + emb.proj_aatype_j(aa)[:, :, None, :]
               + emb.proj_unit_vec(uv * bb2d) + emb.proj_bb_mask(bb2d))
        act = emb.template_pairformer(act, mask=tmask)
        summed = summed + emb.ln_out(act)
    present = tmpl.mask.any(dim=1).to(dt)[:, None, None, None]
    return emb.proj_out(F.relu(summed / n_temp)) * present


def test_embedder_matches_independent_reference():
    emb = AF3TemplateEmbedder(d_pair=D_PAIR).cuda().eval()
    pair, tmpl, asym, tmask = make_inputs()
    with torch.no_grad():
        out_vec = emb(pair, tmpl, asym, tmask)
        out_ref = ref_loop(emb, pair, tmpl, asym, tmask)
    assert torch.isfinite(out_vec).all()
    assert (out_vec - out_ref).abs().max().item() < 1e-4


def test_engine_template_path_matches_pytorch():
    emb_pt = AF3TemplateEmbedder(d_pair=D_PAIR, implementation="PYTORCH").cuda().eval()
    emb_en = AF3TemplateEmbedder(d_pair=D_PAIR, implementation="MINIWORLD_ENGINE").cuda().eval()
    emb_en.load_state_dict(emb_pt.state_dict())
    pair, tmpl, asym, tmask = make_inputs()
    with torch.no_grad():
        o_pt = emb_pt(pair, tmpl, asym, tmask)
        o_en = emb_en(pair, tmpl, asym, tmask)
    assert torch.isfinite(o_en).all()
    rel = (o_pt - o_en).float().norm() / o_pt.float().norm()
    cos = F.cosine_similarity(o_pt.flatten().float(), o_en.flatten().float(), dim=0)
    assert rel < 1e-2 and cos > 0.9999, (rel.item(), cos.item())
