"""BiasOnlyTokenDiT(fp32_pair_bias=True) vs False on a small DiT with HUGE attention logits (to_bias weights scaled so |logit| ~ 1e3), under bf16 autocast, on the GPU.
Checks: (1) in plain fp32 the two switches agree (same math); (2) under bf16 autocast the fp32-bias model is much closer to the fp32 reference than the default;
(3) gradients reach to_bias.weight / ln_pair.weight and are finite; (4) the logits really are fp32 when handed to the blocks."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src")]
from miniworld.modules.bias_only_token_dit import BiasOnlyTokenDiT  # noqa: E402

torch.set_float32_matmul_precision("medium")  # what the training scripts set: the fp32 projection runs on TF32 tensor cores
torch.manual_seed(0)
dev = "cuda"
A, L, DS, DC, DP, H, NB = 6, 32, 64, 32, 128, 4, 3


def build(fp32_bias):
    torch.manual_seed(1)
    m = BiasOnlyTokenDiT(BiasOnlyTokenDiT.Config(d_single=DS, d_cond=DC, d_pair=DP, n_head=H, n_block=NB, fp32_pair_bias=fp32_bias)).to(dev).float()
    torch.manual_seed(2)
    for blk in m.blocks:
        a = blk.attention
        a.to_bias.weight.data.normal_(0, 100.0)      # |logit| ~ 1e3
        a.to_out.weight.data.normal_(0, 0.2)         # zero-init layers made live
        a.to_gate.weight.data.normal_(0, 0.2)
        blk.transition.squeeze.weight.data.normal_(0, 0.2)
    return m


torch.manual_seed(3)
single = torch.randn(A, 1, L, DS, device=dev)
cond = torch.randn(A, 1, L, DC, device=dev)
pair = torch.randn(1, L, L, DP, device=dev) * 3
mask = torch.ones(1, L, dtype=torch.bool, device=dev)
mask[:, -4:] = False

m_off, m_on = build(False), build(True)
m_on.load_state_dict(m_off.state_dict())  # identical weights (separately seeded inits are not guaranteed bit-equal)
assert all(torch.equal(a, b) for a, b in zip(m_off.state_dict().values(), m_on.state_dict().values()))

with torch.enable_grad():
    ref = m_off(single, cond, pair, mask)                 # fp32 reference, default path
    on32 = m_on(single, cond, pair, mask)                 # fp32, fp32-bias path
    print(f"(1) fp32, switch on vs off: max |diff| = {(on32 - ref).abs().max().item():.3e}   (|ref| max {ref.abs().max().item():.2f})")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        off_b = m_off(single, cond, pair, mask)
        on_b = m_on(single, cond, pair, mask)
    e_off = (off_b.float() - ref).abs()
    e_on = (on_b.float() - ref).abs()
    print(f"(2) bf16 autocast vs fp32 reference:  default  mean err {e_off.mean().item():.4f}  max {e_off.max().item():.3f}")
    print(f"                                      fp32-bias mean err {e_on.mean().item():.4f}  max {e_on.max().item():.3f}")
    # a rounded logit moves a softmax row rarely but a lot: the effect is in the tail (max), the mean is dominated by the rest of the net's bf16 noise
    assert e_on.max() < 0.3 * e_off.max(), "the fp32-bias model is not clearly closer to the fp32 reference in the tail"
    assert e_on.mean() <= e_off.mean(), "the fp32-bias model is worse on average"

    for p in m_on.parameters():
        p.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m_on(single, cond, pair, mask).float().pow(2).mean()
    loss.backward()
    g_tb = [b.attention.to_bias.weight.grad for b in m_on.blocks]
    g_ln = [b.attention.ln_pair.weight.grad for b in m_on.blocks]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in g_tb + g_ln), "to_bias / ln_pair gradients missing or not finite"
    print(f"(3) gradients OK: to_bias |grad| {[round(float(g.abs().mean()), 6) for g in g_tb]}, ln_pair |grad| {[round(float(g.abs().mean()), 6) for g in g_ln]}")

    seen = []
    orig = m_on.blocks[0].__class__.forward

    def spy(self, single, cond, pair, mask=None, *, bias=None):
        seen.append((None if bias is None else bias.dtype, None if bias is None else tuple(bias.shape)))
        return orig(self, single, cond, pair, mask, bias=bias)

    m_on.blocks[0].__class__.forward = spy
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m_on(single, cond, pair, mask)
    m_on.blocks[0].__class__.forward = orig
    print(f"(4) bias handed to the blocks: {sorted(set(seen), key=str)}")
    assert all(d is torch.float32 for d, _ in seen), seen
print("TEST_OK")
