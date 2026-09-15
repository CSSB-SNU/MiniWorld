"""Record engine module calls and run a real diffusion forward/backward."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_phase2_cudagraph import difference, load_model_batch
from team_gm.diffusion import EDMScheduler, EuclideanDiffuser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--catalog-snapshot", type=Path, required=True)
    ap.add_argument("--assert-engine", action="store_true")
    ap.add_argument("--compare-before-wiring", action="store_true")
    args = ap.parse_args()
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    model, batch, cfg, ck = load_model_batch(
        args.config, args.ckpt, catalog_snapshot=args.catalog_snapshot
    )
    del ck
    model._forced_n_recycle = 2
    record = {"modules": {}, "calls": {}}
    stage = "frozen"
    hooks = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "_backend")
            or hasattr(module, "implementation")
            or isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm))
            or module.__class__.__name__ == "SWA3DRoPEAttention"
        ):
            implementation = getattr(module, "implementation", None)
            backend = getattr(module, "_backend", None)
            if backend is None:
                backend = (
                    getattr(implementation, "value", implementation)
                    if implementation is not None
                    else "pytorch"
                )
            record["modules"][name] = {
                "class": type(module).__module__ + "." + type(module).__name__,
                "backend": str(backend),
                "implementation": str(implementation),
            }

            def hook(m, args, name=name):
                key = stage + ":" + name
                record["calls"].setdefault(
                    key,
                    {
                        "count": 0,
                        "dtype": str(args[0].dtype)
                        if args and isinstance(args[0], torch.Tensor)
                        else None,
                    },
                )["count"] += 1

            hooks.append(module.register_forward_pre_hook(hook))
    diffuser = EuclideanDiffuser(
        config=EuclideanDiffuser.EuclideanConfig(seed=0),
        scheduler=EDMScheduler(cfg.diffuser.scheduler),
    )
    _, x, mask, t, _ = diffuser.sample(
        batch.structure.atom_pos, num_augment=1, mask=batch.structure.atom_pos_mask
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = None
        if args.compare_before_wiring:
            from miniworld_engine.modules.dispatch import KernelBackend
            from team_gm.modules.exceptions import ImplementationType

            changes = []
            stage = "before_wiring"
            for name, module in model.named_modules():
                kind = type(module).__name__
                if (kind == "RMSNorm" and hasattr(module, "_backend")) or (
                    kind == "LayerNorm" and hasattr(module, "_backend") and (
                        name.startswith("diffusion_module.")
                        or name in {"add_pair_recycle.0", "temp_embedder.ln_query", "temp_embedder.ln_out"}
                    )
                ):
                    changes.append((module, "_backend", module._backend))
                    module._backend = KernelBackend.PYTORCH
                if kind in {"SWAAtomBlock", "SwiGLUFFN"}:
                    changes.append((module, "implementation", module.implementation))
                    module.implementation = ImplementationType.PYTORCH
            try:
                old_single, old_pair = model.condition_forward(
                    batch.msa, batch.reference, batch.scheme, batch.sequence,
                    batch.structure, batch.template,
                )
                old_output = model.diffusion_forward(
                    batch.reference, batch.scheme, batch.structure,
                    x, mask, t, old_single, old_pair,
                )
                baseline = old_single, old_pair, old_output
            finally:
                for module, attribute, value in changes:
                    setattr(module, attribute, value)
            stage = "frozen"
        tsi, pair = model.condition_forward(
            batch.msa,
            batch.reference,
            batch.scheme,
            batch.sequence,
            batch.structure,
            batch.template,
        )
        stage = "diffusion_inference"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            y = model.diffusion_forward(
                batch.reference, batch.scheme, batch.structure, x, mask, t, tsi, pair
            )
            torch.cuda.synchronize()
        record["output_finite"] = bool(y.isfinite().all())
        if baseline is not None:
            valid = mask.bool().unsqueeze(-1).expand_as(y)
            record["before_wiring_difference"] = {
                "single": difference(tsi, baseline[0]),
                "pair": difference(pair, baseline[1]),
                "valid_atom_output": difference(y[valid], baseline[2][valid]),
            }
            del baseline, old_single, old_pair, old_output
        prof.export_chrome_trace(str(out.with_suffix(".trace.json")))
        # Count the exported events directly. Building the profiler's CPU event
        # tree for a first-call trace can dominate the entire GPU diagnostic.
        events = json.loads(out.with_suffix(".trace.json").read_text())["traceEvents"]
        record["engine_ops"] = dict(
            Counter(
                e["name"]
                for e in events
                if e.get("cat") == "cpu_op" and "miniworld_engine::" in e.get("name", "")
            )
        )
        del y, events, prof
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("Inference recorded; starting diffusion backward", flush=True)
    stage = "diffusion_training"
    model.diffusion_module.requires_grad_(True)
    model.to_token_single_trunk.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model.diffusion_forward(
            batch.reference, batch.scheme, batch.structure, x, mask, t, tsi, pair
        )
        loss = y.float().square().mean()
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    record["backward"] = {
        "loss": float(loss.detach()),
        "parameters": len(grads),
        "grad_present": sum(g is not None for g in grads.values()),
        "all_present_finite": all(
            g is None or bool(g.isfinite().all()) for g in grads.values()
        ),
        "missing": [n for n, g in grads.items() if g is None],
    }
    record["pytorch_module_calls"] = {
        key: call
        for key, call in record["calls"].items()
        if record["modules"][key.split(":", 1)[1]]["backend"].lower()
        in {"pytorch", "kernelbackend.pytorch"}
        and not key.startswith("before_wiring:")
    }
    for hook in hooks:
        hook.remove()
    out.write_text(json.dumps(record, indent=2) + "\n")
    if args.assert_engine:
        assert not record["pytorch_module_calls"], record["pytorch_module_calls"]
        assert record["output_finite"] and record["backward"]["all_present_finite"]
        assert not record["backward"]["missing"], record["backward"]["missing"]
    print(
        "RESULT",
        json.dumps({k: v for k, v in record.items() if k not in ("modules", "calls")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
