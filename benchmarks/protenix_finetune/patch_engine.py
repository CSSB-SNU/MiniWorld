"""Swap Protenix v1 blocks for miniworld-engine modules (B200 CUDA kernels) in place.

The engine module owns the (mapped) weights while it is installed; ``to_protenix`` writes them back into the Protenix layout.
Blocks run in bf16 inside (the engine kernels are bf16) and return the caller's dtype.
"""
import os
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
PAD = 128
from miniworld_engine.modules import OuterProductMean as EOPM, MSAPairWeightedAveraging as EPWA
from miniworld_engine.modules import PairformerBlock as EPairformerBlock, PairformerConfig, AttentionPairBias as EAPB, Transition as ETransition
from miniworld_engine.modules.exceptions import ImplementationType

IMPL = ImplementationType.MINIWORLD
NO_REZERO = os.environ.get("PFX_NO_REZERO") == "1"     # diagnostic: no re-zeroing of padded positions between ops
BF = torch.bfloat16

# (engine name, protenix name) for one Pairformer block's pair track
def _pair_map(prefix_src=""):
    m = []
    for e, p in (("tri_multi_outgoing", "tri_mul_out"), ("tri_multi_incoming", "tri_mul_in")):
        m += [(f"{e}.ln_pair.weight", f"{p}.layer_norm_in.weight"), (f"{e}.ln_pair.bias", f"{p}.layer_norm_in.bias"),
              (f"{e}.to_left.weight", f"{p}.linear_a_p.weight"), (f"{e}.to_left_gate.weight", f"{p}.linear_a_g.weight"),
              (f"{e}.to_right.weight", f"{p}.linear_b_p.weight"), (f"{e}.to_right_gate.weight", f"{p}.linear_b_g.weight"),
              (f"{e}.ln_out.weight", f"{p}.layer_norm_out.weight"), (f"{e}.ln_out.bias", f"{p}.layer_norm_out.bias"),
              (f"{e}.to_gate.weight", f"{p}.linear_g.weight"), (f"{e}.to_out.weight", f"{p}.linear_z.weight")]
    for e, p in (("tri_atten_starting", "tri_att_start"), ("tri_atten_ending", "tri_att_end")):
        m += [(f"{e}.ln_pair.weight", f"{p}.layer_norm.weight"), (f"{e}.ln_pair.bias", f"{p}.layer_norm.bias"),
              (f"{e}.to_query.weight", f"{p}.mha.linear_q.weight"), (f"{e}.to_key.weight", f"{p}.mha.linear_k.weight"),
              (f"{e}.to_value.weight", f"{p}.mha.linear_v.weight"), (f"{e}.to_bias.weight", f"{p}.linear.weight"),
              (f"{e}.to_gate.weight", f"{p}.mha.linear_g.weight"), (f"{e}.to_out.weight", f"{p}.mha.linear_o.weight")]
    m += [("transition_pair.ln_in.weight", "pair_transition.layernorm1.weight"), ("transition_pair.ln_in.bias", "pair_transition.layernorm1.bias"),
          ("transition_pair.expand_a.weight", "pair_transition.linear_no_bias_a.weight"), ("transition_pair.expand_b.weight", "pair_transition.linear_no_bias_b.weight"),
          ("transition_pair.squeeze.weight", "pair_transition.linear_no_bias.weight")]
    return m

APB_MAP = [("ln_single.weight", "layernorm_a.weight"), ("ln_single.bias", "layernorm_a.bias"),
           ("to_query.weight", "attention.linear_q.weight"), ("to_query.bias", "attention.linear_q.bias"),
           ("to_key.weight", "attention.linear_k.weight"), ("to_value.weight", "attention.linear_v.weight"),
           ("ln_pair.weight", "layernorm_z.weight"), ("ln_pair.bias", "layernorm_z.bias"), ("to_bias.weight", "linear_nobias_z.weight"),
           ("to_gate.weight", "attention.linear_g.weight"), ("to_out.weight", "attention.linear_o.weight")]
TR_MAP = [("ln_in.weight", "layernorm1.weight"), ("ln_in.bias", "layernorm1.bias"), ("expand_a.weight", "linear_no_bias_a.weight"),
          ("expand_b.weight", "linear_no_bias_b.weight"), ("squeeze.weight", "linear_no_bias.weight")]


def _copy(dst: nn.Module, src: nn.Module, mapping, partial=False):
    dp, sp = dict(dst.named_parameters()), dict(src.named_parameters())
    assert partial or set(dp) == {e for e, _ in mapping}, sorted(set(dp) ^ {e for e, _ in mapping})
    with torch.no_grad():
        for e, p in mapping:
            assert dp[e].shape == sp[p].shape, (e, p, dp[e].shape, sp[p].shape)
            dp[e].copy_(sp[p])
            if sp[p].dtype == torch.float32:          # the original fp32 weight: the trainer builds its fp32 master from it
                dp[e]._pfx_src32 = sp[p].detach().clone()


class EnginePairformerBlock(nn.Module):
    """Protenix PairformerBlock's call signature over the engine PairformerBlock (+ single track)."""

    def __init__(self, src: nn.Module):
        super().__init__()
        c_z = src.tri_mul_out.layer_norm_in.weight.shape[0]
        self.pair = EPairformerBlock(PairformerConfig(d_pair=c_z, d_hidden_tri_multi=src.tri_mul_out.linear_a_p.weight.shape[0],
                                                      d_hidden_tri_attention=32, n_head_tri_attention=src.tri_att_start.linear.weight.shape[0],
                                                      p_drop=src.p_drop), implementation=IMPL)
        if src.pair_transition.n != 4:                 # the engine block builds n = 4; Protenix's template blocks use n = 2
            self.pair.transition_pair = ETransition(c_z, n=src.pair_transition.n, implementation=IMPL)
        _copy(self.pair, src, _pair_map())
        self.c_s = src.c_s
        if self.c_s > 0:
            apb = src.attention_pair_bias
            self.apb = EAPB(self.c_s, c_z, apb.n_heads, implementation=IMPL)
            _copy(self.apb, apb, APB_MAP)
            self.single_transition = ETransition(self.c_s, n=4, implementation=IMPL) if "implementation" in ETransition.__init__.__code__.co_varnames else ETransition(self.c_s, n=4)
            _copy(self.single_transition, src.single_transition, TR_MAP)
        self.to(device=src.tri_mul_out.linear_z.weight.device, dtype=BF)

    def core(self, sb, zb, keep, mask):
        """One block on the padded bf16 streams. ``keep`` [B, Lp, Lp, 1] (None: nothing padded). The padding is re-zeroed after
        every op: masked entries are don't-care for the outputs, but each weight gradient sums over them (0 * NaN = NaN) and
        TriAttn's training kernel can leave non-finite values in a padded column. where()'s backward never reads the values."""
        if keep is None or NO_REZERO:
            zb = self.pair(zb, mask)
        else:
            p = self.pair
            ops = ([lambda t: p.tri_multi(t, mask)] if p.tri_multi is not None else
                   [lambda t: p.tri_multi_outgoing(t, mask), lambda t: p.tri_multi_incoming(t, mask)])
            if p.tri_atten_starting is not None:
                ops += [lambda t: p.tri_atten_starting(t, mask), lambda t: p.tri_atten_ending(t, mask)]
            ops += [lambda t: p.transition_pair(t)]
            for op in ops:
                zb = torch.where(keep, op(zb), 0)
        if self.c_s > 0 and sb is not None:
            sb = self.apb(sb, zb, mask)                                 # the engine op owns the residual
            sb = self.single_transition(sb)
            if keep is not None and not NO_REZERO:
                sb = torch.where(mask[..., None], sb, 0)                # padded single rows stay zero across the stack
        return sb, zb

    def forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        sb, zb, keep, mask, back = _enter(s, z, pair_mask, self.c_s > 0)
        sb, zb = self.core(sb, zb, keep, mask)
        return back(sb, zb)


def _enter(s, z, pair_mask, with_s):
    """Pad tokens to a multiple of 128 (the B200 kernels' tile) with masked tokens and cast to bf16; the returned closure
    crops and casts back. The engine ops honour the token mask, so the real rows / columns are unchanged."""
    unb = z.dim() == 3
    zb = (z.unsqueeze(0) if unb else z).to(BF)
    B, L = zb.shape[0], zb.shape[1]
    Lp = (L + PAD - 1) // PAD * PAD
    if pair_mask is not None:
        pm = pair_mask.unsqueeze(0) if unb else pair_mask
        tok = pm.any(-1) > 0
    else:
        tok = torch.ones(B, L, dtype=torch.bool, device=z.device)
    sb = None
    if with_s and s is not None:
        sb = (s.unsqueeze(0) if unb else s).to(BF)
    if Lp != L:
        zb = F.pad(zb, (0, 0, 0, Lp - L, 0, Lp - L))
        tok = F.pad(tok, (0, Lp - L), value=False)
        if sb is not None:
            sb = F.pad(sb, (0, 0, 0, Lp - L))
    mask = tok if (pair_mask is not None or Lp != L) else None     # decided from shapes: no host sync
    keep = (tok[:, :, None] & tok[:, None, :])[..., None] if Lp != L else None
    zdt, sdt = z.dtype, (s.dtype if s is not None else None)

    def back(sb, zb):
        zo = zb[:, :L, :L]
        zo = (zo.squeeze(0) if unb else zo).to(zdt)
        so = s
        if sb is not None:
            so = sb[:, :L]
            so = (so.squeeze(0) if unb else so).to(sdt)
        return so, zo
    return sb, zb.contiguous(), keep, mask, back


class EnginePairformerStack(nn.Module):
    """Protenix PairformerStack over EnginePairformerBlocks: pad and cast once for the whole stack. The blocks' outputs are
    bf16 anyway, so keeping the stream in bf16 between blocks changes no value. Per-block activation checkpointing as
    Protenix's (blocks_per_ckpt) while grads are on."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(EnginePairformerBlock(b) for b in src.blocks)
        self.c_s = self.blocks[0].c_s
        self.ckpt = getattr(src, "blocks_per_ckpt", None) is not None

    def forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        sb, zb, keep, mask, back = _enter(s, z, pair_mask, self.c_s > 0)
        for blk in self.blocks:
            if self.ckpt and torch.is_grad_enabled():
                sb, zb = torch.utils.checkpoint.checkpoint(blk.core, sb, zb, keep, mask, use_reentrant=False)
            else:
                sb, zb = blk.core(sb, zb, keep, mask)
        return back(sb, zb)

DIT_MAP = [("attention.ada_ln_in.ln_cond.weight", "attention_pair_bias.layernorm_a.layernorm_s.weight"),
           ("attention.ada_ln_in.to_scale.weight", "attention_pair_bias.layernorm_a.linear_s.weight"),
           ("attention.ada_ln_in.to_scale.bias", "attention_pair_bias.layernorm_a.linear_s.bias"),
           ("attention.ada_ln_in.to_bias.weight", "attention_pair_bias.layernorm_a.linear_nobias_s.weight"),
           ("attention.to_query.weight", "attention_pair_bias.attention.linear_q.weight"),
           ("attention.to_query.bias", "attention_pair_bias.attention.linear_q.bias"),
           ("attention.to_key.weight", "attention_pair_bias.attention.linear_k.weight"),
           ("attention.to_value.weight", "attention_pair_bias.attention.linear_v.weight"),
           ("attention.ln_pair.weight", "attention_pair_bias.layernorm_z.weight"),
           ("attention.to_bias.weight", "attention_pair_bias.linear_nobias_z.weight"),
           ("attention.to_gate.weight", "attention_pair_bias.attention.linear_g.weight"),
           ("attention.to_out.weight", "attention_pair_bias.attention.linear_o.weight"),
           ("attention.to_scale.weight", "attention_pair_bias.linear_a_last.weight"),
           ("attention.to_scale.bias", "attention_pair_bias.linear_a_last.bias"),
           ("transition.ada_ln_in.ln_cond.weight", "conditioned_transition_block.adaln.layernorm_s.weight"),
           ("transition.ada_ln_in.to_scale.weight", "conditioned_transition_block.adaln.linear_s.weight"),
           ("transition.ada_ln_in.to_scale.bias", "conditioned_transition_block.adaln.linear_s.bias"),
           ("transition.ada_ln_in.to_bias.weight", "conditioned_transition_block.adaln.linear_nobias_s.weight"),
           ("transition.expand_a.weight", "conditioned_transition_block.linear_nobias_a1.weight"),
           ("transition.expand_b.weight", "conditioned_transition_block.linear_nobias_a2.weight"),
           ("transition.squeeze.weight", "conditioned_transition_block.linear_nobias_b.weight"),
           ("transition.to_scale.weight", "conditioned_transition_block.linear_s.weight"),
           ("transition.to_scale.bias", "conditioned_transition_block.linear_s.bias")]


class EngineTokenTransformer(nn.Module):
    """Protenix's token DiffusionTransformer (24 blocks, 16 x 48) over engine DiTBlocks. Keeps the caller's dtype: fp32 runs the
    engine's TF32 kernels (Protenix trains its diffusion module outside autocast), bf16 the bf16 ones. Tokens are padded to a
    multiple of 128 (masked keys) once for the whole stack."""

    def __init__(self, src: nn.Module):
        super().__init__()
        from miniworld_engine.modules.dit import DiTBlock
        b0 = src.blocks[0]
        self.blocks = nn.ModuleList()
        for b in src.blocks:
            assert isinstance(b.drop_path, nn.Identity)
            e = DiTBlock(b.c_a, b.c_s, b.c_z, b.n_heads, n=2, implementation=IMPL)
            _copy(e, b, DIT_MAP)
            self.blocks.append(e)
        self.to(device=b0.attention_pair_bias.linear_a_last.weight.device)
        if os.environ.get("PFX_DIT_BF16") == "1":
            # bf16 weights from the start (before the trainer builds its fp32 masters / pointer tables): converting lazily in the
            # first forward would swap the parameters' storage under the optimizer
            self.to(BF)
        self.ckpt = os.environ.get("PFX_DIT_CKPT", "0") == "1"

    def forward(self, a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False):
        assert not (n_queries and n_keys), "token transformer only"
        N, ca = a.shape[-2], a.shape[-1]
        lead = a.shape[:-2]
        A = 1
        for d_ in lead:
            A *= d_
        single = a.reshape(A, 1, N, ca)
        cond = s.expand(*lead, N, s.shape[-1]).reshape(A, 1, N, s.shape[-1])
        pair = z.reshape(1, N, N, z.shape[-1])
        Np = (N + PAD - 1) // PAD * PAD
        mask = None
        if Np != N:
            single = F.pad(single, (0, 0, 0, Np - N))
            cond = F.pad(cond, (0, 0, 0, Np - N))
            pair = F.pad(pair, (0, 0, 0, Np - N, 0, Np - N))
            mask = torch.zeros(1, Np, dtype=torch.bool, device=a.device)
            mask[:, :N] = True
        if os.environ.get("PFX_DIT_BF16") == "1":      # ablation: the token DiT in bf16 (MiniWorld v200 runs its diffusion in bf16)
            single, cond, pair = single.to(BF), cond.to(BF), pair.to(BF)
        if not torch.is_grad_enabled() and os.environ.get("PFX_DIT_FUSED_INFER", "1") == "1":
            out = self._infer_fused(single.contiguous(), cond.contiguous(), pair, z, mask)
            return out[:, :, :N].reshape(*lead, N, ca).to(a.dtype)
        dt = single.dtype
        for blk in self.blocks:
            for p_ in blk.parameters():
                if p_.dtype != dt:
                    blk.to(dt)
                break
            if self.ckpt and torch.is_grad_enabled():
                # Protenix checkpoints every diffusion block (use_fine_grained_checkpoint): same memory profile
                single = torch.utils.checkpoint.checkpoint(blk, single.contiguous(), cond.contiguous(), pair.contiguous(), mask, use_reentrant=False)
            else:
                single = blk(single.contiguous(), cond.contiguous(), pair.contiguous(), mask)
        return single[:, :, :N].reshape(*lead, N, ca).to(a.dtype)


    def _infer_fused(self, single, cond, pair, pair_base, mask):
        """No-grad path (the mini-rollout): ONE engine FusedTokenDiT runner over all 24 blocks -- every block's pair bias hoisted
        once per pair tensor, every block's AdaLN / gate tables from one GEMM per step, the blocks back to back in one call
        (the engine's per-module path builds a one-block runner per call). The runner is rebuilt when any weight changes
        (pointer / version); the bias is reused while the pair is the same tensor object at the same version."""
        import weakref
        from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import FusedTokenDiT
        import miniworld_engine.integrations.token_dit as TD
        dt = single.dtype
        for p_ in self.blocks.parameters():
            if p_.dtype != dt:
                self.blocks.to(dt)
            break
        wkey = (dt, tuple((p_.data_ptr(), p_._version) for p_ in self.blocks.parameters()))
        if getattr(self, "_frun_key", None) != wkey:
            self._frun = FusedTokenDiT(list(self.blocks), dtype=dt)
            self._frun_key = wkey
            self._fbias = None
        run = self._frun
        run._mm_cfg, run._gated_cfg = TD._TUNING.setdefault((single.device, dt), ({}, {}))
        hb = self._fbias
        if hb is not None and hb[0]() is pair_base and hb[1] == pair_base._version and hb[2] == tuple(pair.shape):
            bias = hb[3]
        else:
            bias = run.hoist(pair.contiguous(), None if mask is None else mask.reshape(-1))
            self._fbias = (weakref.ref(pair_base), pair_base._version, tuple(pair.shape), bias)
        return run.step(single, cond, bias)

def _swap_token_transformer(model: nn.Module, report):
    dm = model.diffusion_module
    dm.diffusion_transformer = EngineTokenTransformer(dm.diffusion_transformer)
    report["engine token DiT (DiTBlock)"] = ["diffusion_module.diffusion_transformer"]


class EngineTemplateBlock(nn.Module):
    """Protenix's template Pairformer block (c_z 64, TriMul hidden 128, TriAttn 4 x 32, transition n 2), every op on the engine:
    the TriMuls as the engine's TriangleMultiplication(64, d_hidden=128) (its B200 D64 kernels with a one-direction 128-channel
    contraction), TriAttn and the transition as engine modules. All own residual and dropout."""

    def __init__(self, src: nn.Module):
        super().__init__()
        c_z = src.tri_mul_out.layer_norm_in.weight.shape[0]
        # the engine's TriangleMultiplication(d_pair 64, d_hidden 128): served by its B200 path (hidden = 2 x width, one direction)
        from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication as ETM
        tm_map = [(e.split(".", 1)[1], q.split(".", 1)[1]) for e, q in _pair_map() if e.startswith("tri_multi_outgoing.")]
        hid = src.tri_mul_out.linear_a_p.weight.shape[0]
        self.tmo = ETM(c_z, d_hidden=hid, outgoing=True, implementation=IMPL, p_drop=src.p_drop)
        self.tmi = ETM(c_z, d_hidden=hid, outgoing=False, implementation=IMPL, p_drop=src.p_drop)
        _copy(self.tmo, src.tri_mul_out, tm_map)
        _copy(self.tmi, src.tri_mul_in, tm_map)
        for mod in (self.tmo, self.tmi):
            mod.to(device=src.tri_mul_out.linear_z.weight.device, dtype=BF)
        eng = EPairformerBlock(PairformerConfig(d_pair=c_z, d_hidden_tri_multi=c_z, d_hidden_tri_attention=32,
                                                n_head_tri_attention=src.tri_att_start.linear.weight.shape[0], p_drop=src.p_drop),
                               implementation=IMPL)
        eng.transition_pair = ETransition(c_z, n=src.pair_transition.n, implementation=IMPL)
        eng.tri_multi_outgoing = eng.tri_multi_incoming = eng.tri_multi = None
        _copy(eng, src, [(e, q) for e, q in _pair_map() if not e.startswith("tri_multi")])
        self.att_s, self.att_e, self.trans = eng.tri_atten_starting, eng.tri_atten_ending, eng.transition_pair
        for mod in (self.att_s, self.att_e, self.trans):
            mod.to(device=src.tri_mul_out.linear_z.weight.device, dtype=BF)

    def forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        _, zb, keep, mask, back = _enter(None, z, pair_mask, False)
        with torch.autocast("cuda", enabled=False):
            for op in (lambda t: self.tmo(t, mask), lambda t: self.tmi(t, mask), lambda t: self.att_s(t, mask),
                       lambda t: self.att_e(t, mask), lambda t: self.trans(t)):
                zb = op(zb) if keep is None or NO_REZERO else torch.where(keep, op(zb), 0)
        return s, back(None, zb)[1]


def batched_template_forward(self, input_feature_dict, z, pair_mask=None, triangle_attention="torch",
                             triangle_multiplicative="torch", inplace_safe=False, chunk_size=None):
    """Protenix TemplateEmbedder.forward with the templates stacked: one [T, L, L, 64] pass through the template Pairformer
    blocks instead of T passes (the engine template block takes B = T). Same per-template math; the template-independent
    linear_no_bias_z(z) is computed once."""
    import torch.nn.functional as F
    from protenix.model.utils import expand_at_dim
    from protenix.data.constants import STD_RESIDUES_WITH_GAP
    if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
        return 0
    f = input_feature_dict
    asym_id = f["asym_id"]
    multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)
    n = z.shape[0]
    T = f["template_aatype"].shape[0]
    if pair_mask is None:
        pair_mask = z.new_ones(z.shape[:-1])
    z = self.layernorm_z(z)
    vz = self.linear_no_bias_z(z)
    mm, pm = multichain_mask, pair_mask
    dgram = f["template_distogram"] * mm[..., None] * pm[..., None]                                  # [T, L, L, 39]
    pb = (f["template_pseudo_beta_mask"] * mm * pm)[..., None]
    aat = F.one_hot(f["template_aatype"], num_classes=len(STD_RESIDUES_WITH_GAP)).to(dgram.dtype)  # [T, L, 32]
    uv = f["template_unit_vector"] * mm[..., None] * pm[..., None]
    bb = (f["template_backbone_frame_mask"] * mm * pm)[..., None]
    at = torch.cat([dgram, pb, aat[:, None, :, :].expand(T, n, n, -1), aat[:, :, None, :].expand(T, n, n, -1), uv, bb], dim=-1)
    v = vz[None] + self.linear_no_bias_a(at)
    _, v = self.pairformer_stack(s=None, z=v, pair_mask=pm[None].expand(T, n, n), triangle_multiplicative=triangle_multiplicative,
                                 triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
    u = self.layernorm_v(v).sum(0) / (1e-7 + T)
    return self.linear_no_bias_u(self.relu(u))


def _swap_pairformer_blocks(model: nn.Module, report):
    from protenix.model.modules.pairformer import PairformerBlock as PBlock, PairformerStack as PStack
    ok = lambda b: b.tri_mul_out.layer_norm_in.weight.shape[0] == b.tri_mul_out.linear_a_p.weight.shape[0]
    stack_mode = os.environ.get("PFX_PAIR_STACK", "1") == "1"
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if stack_mode and isinstance(child, PStack) and len(child.blocks) and all(ok(b) for b in child.blocks):
                setattr(mod, cname, EnginePairformerStack(child))
                report.setdefault("engine PairformerStack", []).append(f"{full} ({len(child.blocks)} blocks)")
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, PBlock):
                full = f"{name}.{cname}" if name else cname
                if not ok(child):                   # Protenix templates: c_pair 64, c_hidden 128 -- no engine TriMul for it
                    if os.environ.get("PFX_TEMPLATE", "1") == "1":
                        setattr(mod, cname, EngineTemplateBlock(child))
                        report.setdefault("template block: engine TriMul (D64 kernels, 128-ch contraction) + TriAttn + transition", []).append(full)
                    else:
                        report.setdefault("kept (asymmetric TriMul)", []).append(full)
                    continue
                setattr(mod, cname, EnginePairformerBlock(child))
                report.setdefault("engine PairformerBlock", []).append(full)


def _pad_to(x, dim, mult):
    n = x.shape[dim]
    p = (-n) % mult
    if not p:
        return x
    shp = list(x.shape); shp[dim] = p
    return torch.cat([x, x.new_zeros(shp)], dim)


class EngineOPM(nn.Module):
    """Protenix OuterProductMean (no mask; (W sum + b) / (M + eps)) over the engine fused OPM.

    The engine kernel divides before the projection (AF3 order), so it runs bias-free and the bias is added as b / (M + eps).
    MSA rows pad to a multiple of 256 and tokens to 128, both masked out of the count.
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        c_m, c_h, c_z = src.c_m, src.c_hidden, src.c_z
        self.eps = src.eps
        self.opm = EOPM(c_m, c_z, c_h, implementation=IMPL)
        _copy(self.opm, src, [("ln_msa.weight", "layer_norm.weight"), ("ln_msa.bias", "layer_norm.bias"),
                              ("to_left.weight", "linear_1.weight"), ("to_right.weight", "linear_2.weight"),
                              ("to_out.weight", "linear_out.weight"), ("to_out.bias", "linear_out.bias")])
        self.bias = nn.Parameter(self.opm.to_out.bias.detach().clone().float())
        with torch.no_grad():
            self.opm.to_out.bias.zero_()
        self.opm.to_out.bias.requires_grad_(False)
        self.opm.to(BF)

    def forward(self, m, inplace_safe=False, chunk_size=None):
        import pfx_graphsafe as gs
        M, L = m.shape[-3], m.shape[-2]
        mb = _pad_to(_pad_to(m.to(BF), -3, 256), -2, PAD)[None]
        mask = torch.zeros(mb.shape[:3], dtype=torch.bool, device=m.device)
        mask[:, :M, :L] = True
        count = M
        if gs.MSA_ROWS is not None and gs.MSA_ROWS.shape[0] == M:     # random_masked sampling: only the first k rows are real
            mask[:, :M] &= gs.MSA_ROWS[None, :, None]
            count = gs.MSA_K
        with torch.autocast("cuda", enabled=False):
            u = self.opm(mb, mask)[0, :L, :L]
        return u.to(m.dtype) + (self.bias / (count + self.eps)).to(m.dtype)


class EngineMSAStack(nn.Module):
    """Protenix MSAStack (m + rowwise_dropout(PWA(m, z)); m + transition(m)) over the engine PWA + Transition.

    Both engine modules own their residual (and the PWA its row-shared dropout). Rows pad to 128 and tokens to 128 (masked keys).
    Protenix's PWA heads are 8 wide; the engine's B200 kernels take 32: the value / gate projections get zero rows and the
    output projection zero columns per head (an exact identity -- the extra channels are 0 and never reach the output); the
    real Protenix-shaped weights are the parameters and the padded ones are rebuilt from them every call.
    """
    HEAD = 32

    def __init__(self, src: nn.Module):
        super().__init__()
        pw, tr = src.msa_pair_weighted_averaging, src.transition_m
        c_m, c_z, H, c = pw.c_m, pw.c_z, pw.n_heads, pw.c
        self.H, self.c = H, c
        self.cp = max(c, self.HEAD)
        self.pwa = EPWA(c_m, c_z, n_head=H, d_hidden=self.cp, p_drop=src.p_drop, implementation=IMPL)
        _copy(self.pwa, pw, [("ln_msa.weight", "layernorm_m.weight"), ("ln_msa.bias", "layernorm_m.bias"),
                             ("to_value.weight", "linear_no_bias_mv.weight"), ("ln_pair.weight", "layernorm_z.weight"),
                             ("ln_pair.bias", "layernorm_z.bias"), ("to_bias.weight", "linear_no_bias_z.weight"),
                             ("to_gate.weight", "linear_no_bias_mg.weight"), ("to_out.weight", "linear_no_bias_out.weight")]
              if self.cp == c else
              [("ln_msa.weight", "layernorm_m.weight"), ("ln_msa.bias", "layernorm_m.bias"), ("ln_pair.weight", "layernorm_z.weight"),
               ("ln_pair.bias", "layernorm_z.bias"), ("to_bias.weight", "linear_no_bias_z.weight")], partial=self.cp != c)
        if self.cp != c:
            self.wv = nn.Parameter(pw.linear_no_bias_mv.weight.detach().clone())
            self.wg = nn.Parameter(pw.linear_no_bias_mg.weight.detach().clone())
            self.wo = nn.Parameter(pw.linear_no_bias_out.weight.detach().clone())
            for w_, src_ in ((self.wv, pw.linear_no_bias_mv.weight), (self.wg, pw.linear_no_bias_mg.weight), (self.wo, pw.linear_no_bias_out.weight)):
                w_._pfx_src32 = src_.detach().float().clone()
            for lin in (self.pwa.to_value, self.pwa.to_gate, self.pwa.to_out):
                del lin._parameters["weight"]
        self.transition = ETransition(c_m, n=tr.n, implementation=IMPL)
        _copy(self.transition, tr, TR_MAP)
        self.to(BF)

    def _padded_weights(self):
        H, c, cp = self.H, self.c, self.cp
        rows = lambda w: F.pad(w.view(H, c, w.shape[-1]), (0, 0, 0, cp - c)).reshape(H * cp, w.shape[-1])
        self.pwa.to_value.weight = rows(self.wv)
        self.pwa.to_gate.weight = rows(self.wg)
        self.pwa.to_out.weight = F.pad(self.wo.view(self.wo.shape[0], H, c), (0, cp - c)).reshape(self.wo.shape[0], H * cp)

    def forward(self, m, z):
        M, L = m.shape[-3], m.shape[-2]
        if self.cp != self.c:
            self._padded_weights()
        mb = _pad_to(_pad_to(m.to(BF), -3, 128), -2, PAD)[None]
        zb = _pad_to(_pad_to(z.to(BF), -3, PAD), -2, PAD)[None]
        mask = torch.zeros((1, mb.shape[2]), dtype=torch.bool, device=m.device)
        mask[:, :L] = True
        with torch.autocast("cuda", enabled=False):
            mb = self.pwa(mb, zb, mask)
            mb = self.transition(mb)
        return mb[0, :M, :L].to(m.dtype)


def _swap_msa(model, report):
    for name, blk in model.msa_module.blocks.named_children():
        blk.outer_product_mean_msa = EngineOPM(blk.outer_product_mean_msa).to(next(blk.parameters()).device)
        report.setdefault("msa.opm", []).append(name)
        if os.environ.get("PFX_MSA_STACK", "1") != "0" and getattr(blk, "msa_stack", None) is not None and not blk.is_last_block:
            blk.msa_stack = EngineMSAStack(blk.msa_stack).to(next(blk.parameters()).device)
            report.setdefault("msa.stack", []).append(name)



KV_MAP = [("attention.ada_ln_kv.ln_cond.weight", "attention_pair_bias.layernorm_kv.layernorm_s.weight"),
          ("attention.ada_ln_kv.to_scale.weight", "attention_pair_bias.layernorm_kv.linear_s.weight"),
          ("attention.ada_ln_kv.to_scale.bias", "attention_pair_bias.layernorm_kv.linear_s.bias"),
          ("attention.ada_ln_kv.to_bias.weight", "attention_pair_bias.layernorm_kv.linear_nobias_s.weight")]


class EngineAtomTransformer(nn.Module):
    """Protenix AtomTransformer (3 cross-attention blocks, 32 x 128 windows, 4 heads) over engine LocalDiTBlocks.

    The engine's sm_100a local kernels are bf16: the stream runs in bf16 inside and returns the caller's dtype (Protenix runs its
    diffusion module in fp32, so this is a precision change, unlike the token transformer's TF32 path).
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        from miniworld_engine.modules.local_dit import LocalDiTBlock
        self.blocks = nn.ModuleList()
        for b in src.diffusion_transformer.blocks:
            assert isinstance(b.drop_path, nn.Identity)
            e = LocalDiTBlock(b.c_a, b.c_s, b.c_z, b.n_heads, n=2, cross_attention=True, implementation=IMPL)
            _copy(e, b, DIT_MAP + KV_MAP)
            self.blocks.append(e)
        self.to(device=src.diffusion_transformer.blocks[0].attention_pair_bias.linear_a_last.weight.device, dtype=BF)
        self.ckpt = os.environ.get("PFX_DIT_CKPT", "0") == "1"

    def forward(self, q, c, p, inplace_safe=False, chunk_size=None):
        N, ca = q.shape[-2], q.shape[-1]
        lead = q.shape[:-2]
        A = 1
        for d_ in lead:
            A *= d_
        assert p.shape[-4:-1] == ((N + 31) // 32, 32, 128) and p[..., 0, 0, 0, 0].numel() == 1, p.shape
        single = q.reshape(A, 1, N, ca).to(BF).contiguous()
        cond = c.expand(*lead, N, c.shape[-1]).reshape(A, 1, N, c.shape[-1]).to(BF).contiguous()
        pair = p.reshape(1, *p.shape[-4:]).to(BF).contiguous()
        with torch.autocast("cuda", enabled=False):
            for blk in self.blocks:
                if self.ckpt and torch.is_grad_enabled():
                    single = torch.utils.checkpoint.checkpoint(blk, single, cond, pair, None, use_reentrant=False)
                else:
                    single = blk(single, cond, pair, None)
        return single.reshape(*lead, N, ca).to(q.dtype)


def _swap_atom_transformers(model, report):
    dm = model.diffusion_module
    for part in ("atom_attention_encoder", "atom_attention_decoder"):
        mod = getattr(dm, part)
        mod.atom_transformer = EngineAtomTransformer(mod.atom_transformer)
        report.setdefault("engine atom transformer (LocalDiTBlock, bf16)", []).append(f"diffusion_module.{part}.atom_transformer")
    if os.environ.get("PFX_EMB_ATOM") == "1":
        # the input embedder's atom transformer (same 3 cross blocks, 32 x 128 windows); it already runs under bf16 autocast
        enc = model.input_embedder.atom_attention_encoder
        enc.atom_transformer = EngineAtomTransformer(enc.atom_transformer)
        report.setdefault("engine atom transformer (LocalDiTBlock, bf16)", []).append("input_embedder.atom_attention_encoder.atom_transformer")



def install(model: nn.Module, cfg, which: str = "pair"):
    report = {}
    parts = set(which.split(","))
    if os.environ.get("PFX_FAST_T", "1") == "1":       # TriAttn ending direction: fast row-permuting transpose (pfx_pair_t)
        import pfx_pair_t
        pfx_pair_t.install()
        report["TriAttn ending transposes on pfx_pair_t"] = ["miniworld_engine.integrations.triattn_b200.rearrange"]
    if "pair" in parts:
        _swap_pairformer_blocks(model, report)
    if "msa" in parts:
        _swap_msa(model, report)
    if "pair" in parts and os.environ.get("PFX_TEMPLATE", "1") == "1" and os.environ.get("PFX_TEMPLATE_BATCH", "1") == "1":
        import types
        model.template_embedder.forward = types.MethodType(batched_template_forward, model.template_embedder)
        report["template embedder: templates batched (one [T, L, L, 64] pass)"] = ["template_embedder"]
    if "atom" in parts:
        _swap_atom_transformers(model, report)
    if "tokendit" in parts:
        _swap_token_transformer(model, report)
    for k, v in report.items():
        print(f"[patch_engine] {k}: {len(v)}  e.g. {v[:3]}")
    return report
