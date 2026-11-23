"""Which part of the diffusion module breaks the peptide bonds when it runs in bf16?

Start from the fp32 diffusion model (hydra: train.precision=native model.diffusion.dtype=fp32, exact fp32 matmuls) and, AFTER the checkpoint (with its EMA swap) is loaded,
turn the parts named in env MW_BF16_PARTS back into the original bf16 computation: parameters .to(bfloat16) (the engine norms keep fp32 affine params under that cast, as in
the original model), tensor inputs of the part cast to bf16 by a forward pre-hook, tensor outputs cast back to fp32 by a forward hook. Everything else stays fp32.
Parts: diffusion_conditioning, atom_attention_encoder, add_single_token_cond, diffusion_transformer, ln_token_single_rep, atom_attention_decoder (children of
diffusion_module) and to_token_single_trunk (child of the model). Prints the parameter dtype census so the variant is verified, not assumed. No repo code is modified."""
import collections
import fnmatch
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402

torch.set_float32_matmul_precision("highest")
PARTS = [p for p in os.environ.get("MW_BF16_PARTS", "").split(",") if p]


def cast_tree(x, dtype):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dtype)
    if isinstance(x, tuple):
        return tuple(cast_tree(v, dtype) for v in x)
    if isinstance(x, list):
        return [cast_tree(v, dtype) for v in x]
    if isinstance(x, dict):
        return {k: cast_tree(v, dtype) for k, v in x.items()}
    return x


def cast_top(x, dtype):
    """Floating tensors only, top level: tuples / lists / dicts (the SWA attention parameters `carry` with its fp32 RoPE cos / sin) are left as they are."""
    return x.to(dtype) if torch.is_tensor(x) and x.is_floating_point() else x


def make_bf16(mod):
    mod.to(torch.bfloat16)
    rel = getattr(mod, "relative_position_embedder", None)
    if rel is not None:
        rel.float()  # the model keeps this tiny module fp32 on purpose (its one-hot is fp32; only its OUTPUT is cast): restore that
    mod.register_forward_pre_hook(
        lambda m, a, kw: (tuple(cast_top(v, torch.bfloat16) for v in a), {k: cast_top(v, torch.bfloat16) for k, v in kw.items()}), with_kwargs=True
    )
    mod.register_forward_hook(lambda m, a, o: cast_tree(o, torch.float32))
    if hasattr(mod, "pair_conditioning"):  # DiffusionConditioning.pair_conditioning is called as a METHOD (ModelWrapper._prepare_condition): hooks never see it
        orig = mod.pair_conditioning

        def wrapped(*a, **k):
            out = orig(*[cast_top(v, torch.bfloat16) for v in a], **{kk: cast_top(v, torch.bfloat16) for kk, v in k.items()})
            return cast_tree(out, torch.float32)

        mod.pair_conditioning = wrapped


_Client = inf.Client


class Client(_Client):
    def load_state_dict(self, *a, **k):
        out = super().load_state_dict(*a, **k)
        raw = getattr(self.model, "module", self.model)
        dm = raw.diffusion_module
        print("diffusion_module children:", [n for n, _ in dm.named_children()], flush=True)
        if os.environ.get("MW_FULL_REDUCE") == "1":  # bf16 GEMMs must reduce in fp32 (no split-K partial sums rounded to bf16)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            print("  allow_bf16_reduced_precision_reduction = False", flush=True)
        print("  allow_bf16_reduced_precision_reduction is", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, flush=True)
        fns = [s for s in os.environ.get("MW_ROUND_FN", "").split(",") if s]
        if fns:  # round the results of the attention's functional ops (the ones inside BiasOnlyAttention.forward) to bf16 values; everything stays fp32 otherwise
            import miniworld.modules.bias_only_token_dit as bd

            if not getattr(bd, "_fn_rounded", False):
                rb = lambda t: t.to(torch.bfloat16).to(torch.float32) if torch.is_tensor(t) and t.is_floating_point() else t  # noqa: E731
                if "gate" in fns:  # sigmoid_gate(gate, x) = sigmoid(gate) * x : both the sigmoid and the product rounded
                    bd.sigmoid_gate = lambda gate, x: rb(rb(torch.sigmoid(gate)) * x)

                if "einsum" in fns:  # the attention-weighted sum of the values

                    class _T:
                        def __getattr__(self, n):
                            return (lambda *a, **k: rb(torch.einsum(*a, **k))) if n == "einsum" else getattr(torch, n)

                    bd.torch = _T()
                bd._fn_rounded = True
                print(f"  functional-op results rounded to bf16: {fns}", flush=True)
        if os.environ.get("MW_ROUND_P") == "1":  # the attention probabilities p = softmax(pair bias) rounded to bf16 (what `p.to(value.dtype)` does in the bf16 DiT), nothing else
            from miniworld.modules.bias_only_token_dit import BiasOnlyAttention

            if not getattr(BiasOnlyAttention, "_p_rounded", False):
                _orig = BiasOnlyAttention.attention_pattern

                def _rounded(self, *a, **k):
                    return _orig(self, *a, **k).to(torch.bfloat16).to(torch.float32)

                BiasOnlyAttention.attention_pattern = _rounded
                BiasOnlyAttention._p_rounded = True
                print("  attention pattern p rounded to bf16 (BiasOnlyAttention.attention_pattern)", flush=True)
        def resolve(token):
            if "*" in token or "." in token:  # a glob over the dotted names of diffusion_module's modules
                found = [m for n, m in dm.named_modules() if fnmatch.fnmatchcase(n, token)]
                assert found, f"no module matches {token}"
                return found
            mod = getattr(dm, token, None)
            return [mod if mod is not None else getattr(raw, token)]

        for pat in [p for p in os.environ.get("MW_ROUND_W", "").split(",") if p]:  # weights (ndim >= 2) of these modules rounded to bf16 values, kept as fp32 tensors
            n_round = 0
            with torch.no_grad():
                for mod in resolve(pat):
                    for p in mod.parameters():
                        if p.ndim >= 2 and p.dtype == torch.float32:
                            p.copy_(p.to(torch.bfloat16).to(torch.float32))
                            n_round += 1
            print(f"  weights rounded to bf16 values (fp32 storage / fp32 compute): {n_round} matrices under {pat}", flush=True)
        for name in PARTS:
            if name.startswith("round:"):  # keep the module fp32 but round its OUTPUT to bf16 (storage precision only)
                mods = resolve(name[6:])
                for mod in mods:
                    mod.register_forward_hook(lambda m, a, o: cast_tree(cast_tree(o, torch.bfloat16), torch.float32))
                print(f"  round-to-bf16 output hook on {len(mods)} modules ({name[6:]})", flush=True)
            else:
                mods = resolve(name)
                for mod in mods:
                    make_bf16(mod)
                print(f"  bf16 compute on {len(mods)} modules ({name})", flush=True)
        for label, root in (("diffusion_module", dm), ("to_token_single_trunk", raw.to_token_single_trunk)):
            census = collections.Counter()
            for n, p in root.named_parameters():
                part = n.split(".")[0] if root is dm else "to_token_single_trunk"
                census[(part, str(p.dtype).replace("torch.", ""))] += 1
            for (part, dt), c in sorted(census.items()):
                print(f"  param dtype  {part:<28}{dt:<10}{c}", flush=True)
        for pat in [p for p in os.environ.get("MW_FP32_CHILDREN", "").split(",") if p]:
            # inside an already-bf16 module: run THIS child in fp32 (inputs cast up by a pre-hook, output cast back to bf16 by a post-hook);
            # a "keep:" prefix keeps the OUTPUT in fp32 as well (no post-hook), for a consumer that wants the precision (e.g. the attention logits)
            keep = pat.startswith("keep:")
            pat = pat[5:] if keep else pat
            for mod in resolve(pat):
                mod.float()
                mod.register_forward_pre_hook(
                    lambda m, a, kw: (tuple(cast_top(v, torch.float32) for v in a), {k: cast_top(v, torch.float32) for k, v in kw.items()}), with_kwargs=True
                )
                if not keep:
                    mod.register_forward_hook(lambda m, a, o: cast_tree(o, torch.bfloat16))
            print(f"  fp32 child inside bf16 module: {pat}{' (output kept fp32)' if keep else ''}", flush=True)
        ffns = [s for s in os.environ.get("MW_FP32_FN", "").split(",") if s]
        if ffns:  # inside the bf16 attention: compute these functional ops in fp32, hand the result back as bf16
            import miniworld.modules.bias_only_token_dit as bd

            if not getattr(bd, "_fp32fn", False):
                if "gate" in ffns:
                    bd.sigmoid_gate = lambda gate, x: (torch.sigmoid(gate.float()) * x.float()).to(torch.bfloat16)
                if "einsum" in ffns:

                    class _T2:
                        def __getattr__(self, n):
                            if n == "einsum":
                                return lambda eq, *ops, **k: torch.einsum(eq, *[o.float() for o in ops], **k).to(torch.bfloat16)
                            return getattr(torch, n)

                    bd.torch = _T2()
                bd._fp32fn = True
                print(f"  functional ops computed in fp32 inside the bf16 attention: {ffns}", flush=True)
        if os.environ.get("MW_DIAG_LINEAR"):  # accuracy of the bf16 GEMM of the named Linear(s): module output vs the fp32 GEMM on the same operands (correctly rounded to bf16)
            import atexit

            import torch.nn.functional as F

            stats = collections.defaultdict(lambda: collections.Counter())

            def _diag(mod, args, out, key=None):
                x = args[0]
                w = mod.weight
                if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
                    return
                ref = F.linear(x.float(), w.float())  # fp32 GEMM on the same (bf16-valued) operands, exact fp32 matmul
                refb = ref.to(torch.bfloat16)
                d = (out.float() - ref).abs()
                s = stats[key]
                s["calls"] += 1
                s["sum_abs_err"] += float(d.sum())
                s["sum_abs_ref"] += float(ref.abs().sum())
                s["n"] += d.numel()
                s["n_not_correctly_rounded"] += int((out != refb).sum())
                s["max_abs_err"] = max(s["max_abs_err"], float(d.max()))
                s["max_abs_ref"] = max(s["max_abs_ref"], float(ref.abs().max()))

            for pat in os.environ["MW_DIAG_LINEAR"].split(","):
                for n, mod in dm.named_modules():
                    if fnmatch.fnmatchcase(n, pat):
                        mod.register_forward_hook(lambda m, a, o, key=pat: _diag(m, a, o, key))

            def _rep():
                for k, s in stats.items():
                    print(f"\n=== DIAG {k}: {s['calls']} calls, {s['n']} outputs ===")
                    print(f"  mean |out - fp32 GEMM| / mean |fp32 GEMM| = {s['sum_abs_err'] / max(s['sum_abs_ref'], 1e-30):.5f}   (bf16 half-ulp is about 0.002-0.004)")
                    print(f"  fraction of outputs that are NOT the correctly rounded bf16 value: {s['n_not_correctly_rounded'] / max(s['n'], 1):.4f}")
                    print(f"  max |err| = {s['max_abs_err']:.4f}   max |ref| = {s['max_abs_ref']:.2f}")

            atexit.register(_rep)
        if os.environ.get("MW_DIAG_OUT"):  # what the LAST hook of the named module(s) hands to the caller: dtype / contiguity / magnitude (registered after every other hook)
            import atexit

            seen_out = collections.Counter()
            mags = collections.defaultdict(float)

            def _o(mod, args, out, key=None):
                seen_out[(key, str(out.dtype), bool(out.is_contiguous()))] += 1
                mags[key + ":sum_abs"] += float(out.float().abs().sum())
                mags[key + ":n"] += out.numel()
                mags[key + ":max_abs"] = max(mags[key + ":max_abs"], float(out.float().abs().max()))
                # per-row spread of the logits over the key axis (what the softmax sees): max - min over the last-but-one axis
                if out.ndim == 4:
                    spread = (out.float().amax(dim=-2) - out.float().amin(dim=-2)).mean()  # to_bias output is [B, m, n, H]: the softmax runs over the key axis n
                    mags[key + ":spread_sum"] += float(spread)
                    mags[key + ":spread_n"] += 1

            for pat in os.environ["MW_DIAG_OUT"].split(","):
                for n, mod in dm.named_modules():
                    if fnmatch.fnmatchcase(n, pat):
                        mod.register_forward_hook(lambda m, a, o, key=pat: _o(m, a, o, key))

            def _rep_out():
                print("\n=== DIAG_OUT (module, output dtype, contiguous) -> calls")
                for k, c in sorted(seen_out.items()):
                    print(f"  {k}: {c}")
                for k in [k for k in mags if k.endswith(":n")]:
                    base = k[:-2]
                    print(f"  mean |out| of {base}: {mags[base + ':sum_abs'] / max(mags[k], 1):.4f}   max |out|: {mags[base + ':max_abs']:.2f}"
                          f"   mean (max-min over keys) of 4-D outputs: {mags[base + ':spread_sum'] / max(mags[base + ':spread_n'], 1):.2f}")

            atexit.register(_rep_out)
        print(f"BF16 PARTS = {PARTS or 'none (all fp32)'}", flush=True)
        if os.environ.get("MW_PROFILE") == "1" and not globals().get("_PROF"):
            import atexit
            import re

            from torch.profiler import ProfilerActivity, profile

            allow = re.compile(r"^(memcpy|memset|void at::|at::|void \(anonymous namespace\)::|void cunn_|void cub::|void at_cuda_detail::cub|void thrust::|ampere_|sm\d+_xmma|nvjet_|cutlass_\d+|"
                               r"fmha_cutlassF|void gemv|void splitK|void dot_kernel|void reduce_1Block|cublas|void cublasLt|std::enable_if|void scal_kernel|void sgemm|void flash::|void pytorch_flash)", re.I)
            globals()["_PROF"] = prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()

            def _report():
                prof.__exit__(None, None, None)
                kern = collections.Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
                print(f"\n=== CUDA kernels not on the PyTorch/cuBLAS allow-list ({sum(kern.values())} launches in total) ===")
                for n, c in sorted(kern.items(), key=lambda x: -x[1]):
                    if not allow.match(n.strip()):
                        print(f"{c:8d}  {n[:150]}")
                print("=== end ===")

            atexit.register(_report)
        return out


inf.Client = Client
inf.cli()
