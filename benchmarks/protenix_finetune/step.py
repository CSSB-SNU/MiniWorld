"""One Protenix v1 fine-tuning step without the data pipeline: trunk (N_cycle recycles, grad on the last), mini-rollout (no grad),
confidence head, diffusion training (N_sample samples), distogram; a surrogate loss over every head, backward. Features from featurize.py,
ground-truth coordinates synthetic (they only feed the noise / augmentation)."""
import os, sys, time
from pfx_common import *
from protenix.model.generator import sample_diffusion_training
from protenix.utils.torch_utils import autocasting_disable_decorator
name = sys.argv[1] if len(sys.argv) > 1 else "7r6r"
NCYC = int(os.environ.get("NCYC", "4")); NSAMP = int(os.environ.get("NSAMP", "48"))
torch.manual_seed(0)
cfg, m = build_model("cuda", extra=os.environ.get("PFX_EXTRA", ""))
import pfx_graphsafe
pfx_graphsafe.install()
if os.environ.get("CAPTURE"):
    _cap = {}
    def _hook(name):
        def f(mod, args, kwargs):
            if torch.is_grad_enabled() and name not in _cap:
                _cap[name] = [t.detach().clone() if torch.is_tensor(t) else t for t in args[:3]]
                torch.save(_cap, os.environ["CAPTURE"])
        return f
    for _n in ("atom_attention_encoder", "atom_attention_decoder"):
        getattr(m.diffusion_module, _n).atom_transformer.register_forward_pre_hook(_hook(_n), with_kwargs=True)
if os.environ.get("PATCH"):
    import patch_engine
    patch_engine.install(m, cfg, os.environ["PATCH"])
m.train()
d = torch.load(FEATS / f"{name}.pt", weights_only=False)
from protenix.model.protenix import update_input_feature_dict
f = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d["input_feature_dict"].items()}
f = update_input_feature_dict(m.relative_position_encoding.generate_relp(f))
N_TOKEN = int(d['N_token'])
pfx_graphsafe.N_TOKEN = N_TOKEN
if os.environ.get('PFX_DIFF_CACHE') == '1':
    m.enable_diffusion_shared_vars_cache = True     # Protenix's own option (its inference config sets it): pair_z / p_lm once per step
f['distogram_rep_atom_idx'] = f['distogram_rep_atom_mask'].bool().nonzero().squeeze(-1)


def _sample_diffusion_gs(denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, noise_schedule, N_sample=1,
                         enable_efficient_fusion=False, **_):
    """protenix.model.generator.sample_diffusion (no guidance, no chunking) with the gamma branch taken from a host copy of
    the (deterministic) schedule instead of comparing a CUDA scalar on the host. Same ops and RNG order."""
    from protenix.model.utils import centre_random_augmentation
    sd = cfg.sample_diffusion
    host = NOISE_SCHED_HOST
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape, device, dtype = s_inputs.shape[:-2], s_inputs.device, s_inputs.dtype
    x_l = noise_schedule[0] * torch.randn(size=(*batch_shape, N_sample, N_atom, 3), device=device, dtype=dtype)
    for i in range(len(host) - 1):
        c_tau_last, c_tau = noise_schedule[i], noise_schedule[i + 1]
        x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1).squeeze(dim=-3).to(dtype)
        gamma = float(sd.gamma0) if host[i + 1] > sd.gamma_min else 0
        t_hat = c_tau_last * (gamma + 1)
        delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
        x_noisy = x_l + sd.noise_scale_lambda * delta_noise_level * torch.randn(size=x_l.shape, device=device, dtype=dtype)
        t_hat = t_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, N_sample).to(dtype)
        x_denoised = denoise_net(x_noisy=x_noisy, t_hat_noise_level=t_hat, input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                                 s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, chunk_size=None,
                                 inplace_safe=False, enable_efficient_fusion=enable_efficient_fusion)
        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
        dt = c_tau - t_hat
        x_l = x_noisy + sd.step_scale_eta * dt[..., None, None] * delta
    return x_l
# deterministic schedule, built once (its scalar fill is a host copy, not capturable)
NOISE_SCHED = m.inference_noise_scheduler(N_step=cfg.sample_diffusion['N_step_mini_rollout'], device='cuda', dtype=torch.float32)
NOISE_SCHED_HOST = NOISE_SCHED.tolist()
SAMPLE_DIFFUSION = _sample_diffusion_gs if os.environ.get('PFX_GRAPH_SAFE', '1') == '1' else m.sample_diffusion
n_atom = f["atom_to_token_idx"].shape[-1]
lab = {"coordinate": torch.randn(n_atom, 3, device="cuda") * 10, "coordinate_mask": torch.ones(n_atom, device="cuda")}
print(f"{name}: N_token {d['N_token'].item()} N_atom {n_atom} N_msa {d['N_msa'].item()}  N_cycle {NCYC} N_sample {NSAMP}  "
      f"tri_att {cfg.triangle_attention} tri_mul {cfg.triangle_multiplicative} skip_amp {dict(cfg.skip_amp)}", flush=True)

T = {}
def mark(k):
    if os.environ.get('STAGES'):
        torch.cuda.synchronize(); T[k] = time.time()


if os.environ.get("PFX_DIFF_ONLY") == "1":   # diffusion-only training (as MiniWorld phase 2): the heads are not trained
    for _n, _p in m.named_parameters():
        if _n.startswith(("confidence_head.", "distogram_head.")):
            _p.requires_grad_(False)
# ---- full training step (PFX_FULL=1): Protenix loss + grad clip + Adam + EMA, as runner/train.py's train_step ----
FULL = os.environ.get("PFX_FULL") == "1"
if os.environ.get("PFX_FREEZE_TRUNK") == "1":        # ablation: trunk params frozen -> no trunk backward (heads still train)
    FREEZE_KEEP = ("diffusion_module.", "confidence_head.", "distogram_head.")
    for _n, _p in m.named_parameters():
        if not _n.startswith(FREEZE_KEEP):
            _p.requires_grad_(False)
    m.train_confidence_only = True          # Protenix's own switch: every recycle cycle under no_grad
    for _t in ("input_embedder", "template_embedder", "msa_module", "pairformer_stack", "relative_position_encoding"):
        getattr(m, _t).eval()                # frozen trunk in eval mode (no dropout), as MiniWorld phase 2
TRAINER = os.environ.get("PFX_TRAINER", "protenix")      # protenix: its own loop (EMAWrapper, empty_cache, host NaN check); fast: fused / foreach, graph-capturable
if FULL:
    from protenix.model.loss import ProtenixLoss
    LOSS = ProtenixLoss(cfg)
    PARAMS = [p for p in m.parameters() if p.requires_grad]
    EMA_DECAY = 0.999
    kw = dict(lr=cfg.adam.lr, betas=(cfg.adam.beta1, cfg.adam.beta2), weight_decay=cfg.adam.weight_decay)
    if TRAINER == "protenix":
        OPT = torch.optim.Adam(PARAMS, **kw)
        from runner.ema import EMAWrapper
        EMA = EMAWrapper(m, EMA_DECAY); EMA.register()
    else:
        # fused Adam (one launch per dtype group) -- but it does not bump the parameters' _version, which the engine's weight-pack
        # caches key on, so the step below bumps it explicitly (no kernel); otherwise an eager call would reuse old weights' packs
        # Protenix trains fp32 parameters (autocast computes in bf16); the engine blocks hold bf16 weights for their kernels, so
        # (PFX_FP32_MASTER, default on) clip / Adam / EMA run on fp32 masters made from the original Protenix weights: each step
        # casts the bf16 grads up into the masters' grad buffers and the updated masters back down into the bf16 weights
        LOWP = [p for p in PARAMS if p.dtype != torch.float32] if os.environ.get("PFX_FP32_MASTER", "1") == "1" else []
        MASTERS = [getattr(p, "_pfx_src32", None) for p in LOWP]
        MASTERS = [(s32.to(p.device, torch.float32) if s32 is not None and s32.shape == p.shape else p.detach().float().clone())
                   for p, s32 in zip(LOWP, MASTERS)]
        _n_src32 = sum(getattr(p, "_pfx_src32", None) is not None for p in LOWP)
        for _p in LOWP:
            _p.__dict__.pop("_pfx_src32", None)
        for _mp in MASTERS:
            _mp.grad = torch.zeros_like(_mp)
        _M_OF = {id(p): mp for p, mp in zip(LOWP, MASTERS)}
        OPT_PARAMS = [_M_OF.get(id(p), p) for p in PARAMS]
        if LOWP:
            with torch.no_grad():
                torch._foreach_copy_(LOWP, MASTERS)          # the bf16 weights = round(fp32 master), as autocast would cast them
        # fused Adam (one launch per dtype group) -- but it does not bump the parameters' _version, which the engine's weight-pack
        # caches key on, so the step below bumps it explicitly (no kernel); otherwise an eager call would reuse old weights' packs
        OPT = torch.optim.Adam(OPT_PARAMS, fused=True, capturable=True, **kw)
        # per-dtype groups for the clip and the EMA: one mixed fp32 / bf16 list sends torch's foreach ops to their per-tensor
        # slow path (thousands of launches); split by dtype they take the multi-tensor kernels
        _GROUPS = {}
        for _p in OPT_PARAMS:
            _GROUPS.setdefault(_p.dtype, []).append(_p)
        EMA_SHADOW = [p.detach().clone() for p in OPT_PARAMS]
        _EMA_G = {}
        for _p, _s in zip(OPT_PARAMS, EMA_SHADOW):
            _EMA_G.setdefault(_p.dtype, ([], []))
            _EMA_G[_p.dtype][0].append(_s); _EMA_G[_p.dtype][1].append(_p.detach())
        FOPT = None
        if os.environ.get("PFX_FUSED_OPT") == "1":           # clip + Adam + master->weight cast + EMA in 3 launches (pfx_fused_opt)
            import pfx_fused_opt
            FOPT = pfx_fused_opt.FusedTrainerOpt(PARAMS, OPT_PARAMS, cfg.adam.lr, (cfg.adam.beta1, cfg.adam.beta2), 1e-8,
                                                 cfg.adam.weight_decay, EMA_DECAY, cfg.grad_clip_norm)
            EMA_SHADOW = FOPT.shadow_views
        print(f"fp32 masters for {len(LOWP)} low-precision tensors ({sum(p.numel() for p in LOWP) / 1e6:.1f}M, "
              f"{_n_src32} from the original fp32 weights)", flush=True)
    print(f"full step: trainer {TRAINER}, {sum(p.numel() for p in PARAMS) / 1e6:.1f}M params", flush=True)
for _p in m.parameters():
    _p.__dict__.pop("_pfx_src32", None)          # the adapters' copies of the original fp32 weights (fp32 masters made above)

def clip_grouped(max_norm):
    """clip_grad_norm_(PARAMS, max_norm) (L2, clip_coef = max_norm / (total + 1e-6) clamped at 1) with one foreach norm and
    one foreach mul per dtype group."""
    gs = {dt: [p.grad for p in ps if p.grad is not None] for dt, ps in _GROUPS.items()}
    norms = [torch.stack(torch._foreach_norm(g)).float() for g in gs.values() if g]     # one stack per group, then fp32
    total = torch.linalg.vector_norm(torch.cat(norms))
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for dt, g in gs.items():
        if g:
            torch._foreach_mul_(g, coef.to(dt))       # the scalar in the list's dtype keeps the multi-tensor path
    return total


_DO_OPT = [True]
def opt_core():
    """fast trainer: (bf16 grads -> fp32 master grads) clip, fused Adam, (masters -> bf16 weights), EMA; graph-capturable"""
    if FOPT is not None:
        FOPT.step([p.grad for p in PARAMS])
        return
    if LOWP:
        torch._foreach_copy_([mp.grad for mp in MASTERS], [p.grad for p in LOWP])
    clip_grouped(cfg.grad_clip_norm)
    OPT.step()
    if LOWP:
        with torch.no_grad():
            torch._foreach_copy_(LOWP, MASTERS)
    for _sh, _pd in _EMA_G.values():
        torch._foreach_lerp_(_sh, _pd, 1.0 - EMA_DECAY)


def opt_part():
    # Trainer.update (clip) + optimizer step + EMA update; Protenix's trainer also empties the cache every step
    if TRAINER == "protenix":
        torch.nn.utils.clip_grad_norm_(PARAMS, cfg.grad_clip_norm)
        OPT.step()
        EMA.update()
        torch.cuda.empty_cache()
    else:
        opt_core()
        torch.autograd.graph.increment_version(PARAMS)

SIDE_CONF = os.environ.get("PFX_SIDE_CONF") == "1"
NOJOIN = SIDE_CONF and os.environ.get("PFX_SIDE_NOJOIN") == "1"     # whole_graph.py only
LATE = os.environ.get("PFX_CONF_LATE") == "1"                       # whole_graph.py only: confidence head after the main backward
LATE_CTX = []
SIDE_LOSS = [None]
CONF_ST = torch.cuda.Stream() if SIDE_CONF else None
DBG = {}
def forward_preds():
    mark('t0')
    # cache_enabled=False as Protenix's Trainer.train_step (and required for CUDA-graphed callables)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        if os.environ.get("PFX_FREEZE_TRUNK") == "1":           # ablation: trunk forward only (no trunk backward)
            with torch.no_grad():
                s_inputs, s, z = m.get_pairformer_output(input_feature_dict=f, N_cycle=NCYC, inplace_safe=False, chunk_size=None)
        else:
            s_inputs, s, z = m.get_pairformer_output(input_feature_dict=f, N_cycle=NCYC, inplace_safe=False, chunk_size=None)
        mark('trunk')
        if s_inputs is not None: DBG['s_inputs'] = s_inputs.detach()
        if s is not None: DBG['s'] = s.detach()
        if z is not None: DBG['z'] = z.detach()
        cache = {"pair_z": None, "p_lm/c_l": [None, None]}
        if m.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion)(m.diffusion_module.diffusion_conditioning.prepare_cache)(f["relp"], z, False)
            cache["p_lm/c_l"] = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion)(m.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=f["ref_pos"], ref_charge=f["ref_charge"], ref_mask=f["ref_mask"], ref_element=f["ref_element"],
                ref_atom_name_chars=f["ref_atom_name_chars"], atom_to_token_idx=f["atom_to_token_idx"], d_lm=f["d_lm"], v_lm=f["v_lm"],
                pad_info=f["pad_info"], r_l=True, z=cache["pair_z"], inplace_safe=False)
        # PFX_SIDE_CONF=1: the mini-rollout (no grad) and the confidence head (stop_gradient: detached trunk inputs) form a branch
        # independent of the diffusion-training branch up to the loss sum -> issue it on a second stream so the two overlap (its
        # backward runs on that stream too: autograd replays each op's backward on its forward stream)
        main_st = torch.cuda.current_stream()
        side = SIDE_CONF
        if side:
            CONF_ST.wait_stream(main_st)
            for t_ in (s_inputs, s, z, cache["pair_z"], *cache["p_lm/c_l"]):
                if t_ is not None:
                    t_.record_stream(CONF_ST)
        with torch.cuda.stream(CONF_ST if side else main_st):
            if os.environ.get("PFX_NO_ROLLOUT") == "1":          # ablation: no mini-rollout; the confidence head gets the label coords
                cm = lab["coordinate"][None].detach()
            else:
              with torch.no_grad():
                m.diffusion_module.eval()
                cm = SAMPLE_DIFFUSION(denoise_net=m.diffusion_module, input_feature_dict=f, s_inputs=s_inputs.detach(), s_trunk=s.detach(),
                                        z_trunk=None if cache["pair_z"] is not None else z.detach(),
                                        pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                                        p_lm=None if cache["p_lm/c_l"][0] is None else cache["p_lm/c_l"][0].detach(),
                                        c_l=None if cache["p_lm/c_l"][1] is None else cache["p_lm/c_l"][1].detach(),
                                        N_sample=1, noise_schedule=NOISE_SCHED,
                                        enable_efficient_fusion=m.enable_efficient_fusion)
                m.diffusion_module.train()
            mark('mini_rollout')
            if cm is not None: DBG['cm'] = cm.detach()
            if LATE:
                # the confidence head runs after the diffusion / trunk backward, on the main stream (finish): the side stream then
                # only carries the rollout -- no TriMul B200 backward (persistent, inter-CTA spin waits) on two streams at once
                LATE_CTX[:] = [s_inputs.detach(), s.detach(), z.detach(), cm]
                plddt = pae = pde = resolved = None
            elif os.environ.get("PFX_DIFF_ONLY") == "1":
                plddt = pae = pde = resolved = None
            else:
              plddt, pae, pde, resolved = m.run_confidence_head(input_feature_dict=f, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None,
                                                              x_pred_coords=cm, use_embedding=True, triangle_multiplicative=cfg.triangle_multiplicative,
                                                              triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
            if side and NOJOIN and plddt is not None:     # the confidence loss on the side stream: no join before backward
                SIDE_LOSS[0] = autocasting_disable_decorator(cfg.skip_amp.loss)(GSL)(
                    {"plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved, "coordinate_mini": cm}, which="conf")[0]
        mark('confidence')
        side_out = (cm, plddt, pae, pde, resolved)
        if plddt is not None: DBG['plddt'] = plddt.detach()
        if pae is not None: DBG['pae'] = pae.detach()
        if pde is not None: DBG['pde'] = pde.detach()
        if resolved is not None: DBG['resolved'] = resolved.detach()
        _, xd, sig = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion_training)(sample_diffusion_training)(
            noise_sampler=m.train_noise_sampler, denoise_net=m.diffusion_module, label_dict=lab, input_feature_dict=f, s_inputs=s_inputs,
            s_trunk=s, z_trunk=None if cache["pair_z"] is not None else z, pair_z=cache["pair_z"], p_lm=cache["p_lm/c_l"][0], c_l=cache["p_lm/c_l"][1],
            N_sample=NSAMP, diffusion_chunk_size=cfg.diffusion_chunk_size, use_conditioning=True, enable_efficient_fusion=m.enable_efficient_fusion)
        mark('diffusion_train')
        if xd is not None: DBG['xd'] = xd.detach()
        if sig is not None: DBG['sig'] = sig.detach()
        dg = None if os.environ.get("PFX_DIFF_ONLY") == "1" else autocasting_disable_decorator(True)(m.distogram_head)(z)
        if side and not NOJOIN:                               # join the confidence branch
            main_st.wait_stream(CONF_ST)
            for t_ in side_out:
                if t_ is not None:
                    t_.record_stream(main_st)
    return plddt, pae, pde, resolved, dg, xd, sig, cm


GSL = None
if FULL and os.environ.get("PFX_LOSS_GS") == "1":
    # graph-safe ProtenixLoss: sample-only data precomputed here, the per-step part has no host syncs
    import pfx_loss_gs
    GSL = pfx_loss_gs.GraphSafeLoss(LOSS, cfg, f, lab)
    if os.environ.get("PFX_LOSS_COMPILE") == "1":
        GSL.forward = torch.compile(GSL.forward, dynamic=False)


def pred_dict(preds):
    plddt, pae, pde, resolved, dg, xd, sig, cm = preds
    pred = {"plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved, "distogram": dg, "coordinate": xd, "noise_level": sig, "coordinate_mini": cm}
    return {k: v for k, v in pred.items() if v is not None}


def compute_loss(preds):
    """the graph-safe loss (PFX_LOSS_GS=1) on the forward outputs (NOJOIN: the diffusion / distogram part; the confidence part
    is SIDE_LOSS[0], made on the side stream)"""
    return autocasting_disable_decorator(cfg.skip_amp.loss)(GSL)(pred_dict(preds), which="diff" if NOJOIN else "all")[0]


def finish(preds, loss=None):
    """backward (+ optimizer) of a step; loss given: it was computed by the caller"""
    plddt, pae, pde, resolved, dg, xd, sig, cm = preds if preds is not None else (None,) * 8
    if loss is not None:
        pass
    elif GSL is not None:
        loss = compute_loss(preds)
    elif FULL and os.environ.get("PFX_LOSS_SURROGATE") == "1":
        # measurement only: a trivial loss over the same outputs (the cost of Protenix's loss = the difference)
        loss = sum(t.float().mean() for t in (plddt, pae, pde, resolved, dg, xd) if t is not None)
    elif FULL:
        # Protenix's own loss (train mode), as Trainer.get_loss: outside autocast per skip_amp.loss
        loss, _ = autocasting_disable_decorator(cfg.skip_amp.loss)(LOSS)(feat_dict=f, pred_dict=pred_dict(preds), label_dict=dict(lab), mode="train")
    else:
        loss = (xd.float() - lab["coordinate"]).square().mean() * 1e-2 + sum(t.float().mean() for t in (plddt, pae, pde, resolved, dg))
    mark('heads')
    if FULL and TRAINER == "protenix" and bool(torch.isnan(loss) | torch.isinf(loss)):   # Trainer's is_loss_nan_check (host)
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
    if LATE and LATE_CTX:
        loss.backward()                                   # diffusion + distogram terms: the diffusion module and the trunk
        si_, s_, z_, cm_ = LATE_CTX
        main_st = torch.cuda.current_stream()
        if SIDE_CONF:
            main_st.wait_stream(CONF_ST)                  # the rollout (side stream) is done
            cm_.record_stream(main_st)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            pl_, pa_, pd_, rs_ = m.run_confidence_head(input_feature_dict=f, s_inputs=si_, s_trunk=s_, z_trunk=z_, pair_mask=None,
                                                       x_pred_coords=cm_, use_embedding=True, triangle_multiplicative=cfg.triangle_multiplicative,
                                                       triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
        lc = autocasting_disable_decorator(cfg.skip_amp.loss)(GSL)({"plddt": pl_, "pae": pa_, "pde": pd_, "resolved": rs_, "coordinate_mini": cm_},
                                                                   which="conf")[0]
        lc.backward()
        loss = loss.detach() + lc.detach()
        LATE_CTX.clear()
    elif NOJOIN and SIDE_LOSS[0] is not None:
        # two roots: the diffusion-side loss (this stream) and the confidence loss (side stream) -- no forward join
        torch.autograd.backward([loss, SIDE_LOSS[0]])
        loss = loss.detach() + SIDE_LOSS[0].detach()      # backward has synced the side stream into this one
        SIDE_LOSS[0] = None
    else:
        loss.backward()
    mark('bwd')
    if FULL and os.environ.get("PFX_NO_OPT") != "1" and _DO_OPT[0]:
        opt_part()
    mark('backward')
    return loss.detach()


def step():
    return finish(forward_preds())

for i in range(2):
    l = step(); m.zero_grad(set_to_none=True)
torch.cuda.synchronize()
times = []
for i in range(int(os.environ.get("NREP", "4"))):
    torch.cuda.synchronize(); t0 = time.time()
    l = step(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); times.append(time.time() - t0)
if T:
    ks = list(T); print('stages ms:', ' '.join(f'{ks[i]} {1e3 * (T[ks[i]] - T[ks[i - 1]]):.0f}' for i in range(1, len(ks))))
if times: print(f"step ms: {[round(t * 1e3, 1) for t in times]}  median {sorted(times)[len(times) // 2] * 1e3:.1f}  mean {sum(times) / len(times) * 1e3:.1f}  loss {l.item():.4f}  "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
