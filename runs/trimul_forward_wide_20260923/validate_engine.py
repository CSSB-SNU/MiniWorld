"""Final packaged forward validation, paired timings, and sanitizer entry."""

from pathlib import Path
import sys, os, json, argparse, statistics, hashlib

R = Path(__file__).resolve().parent
ROOT = R.parent.parent
sys.path.insert(0, str(ROOT / ".engine-release-2.0.0/src"))
sys.path.insert(0, str(R.parent / "trimul_cuda_widths_opt_20260923"))
import torch
from fixture import setup
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as W
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as OLD
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def paired(fns):
    graphs = {k: capture(fn)[0] for k, fn in fns.items()}
    results = {k: [] for k in graphs}
    for rep in range(3):
        for g in graphs.values():
            for _ in range(10):
                g.replay()
        events = {k: [] for k in graphs}
        for i in range(50):
            for k in list(graphs) if (i + rep) % 2 else list(graphs)[::-1]:
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                graphs[k].replay()
                b.record()
                events[k].append((a, b))
        torch.cuda.synchronize()
        for k, pairs in events.items():
            results[k].extend(a.elapsed_time(b) * 1000 for a, b in pairs)
    return {
        k: dict(median_us=statistics.median(v), samples_us=v) for k, v in results.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--sanitize", action="store_true")
    a = p.parse_args()
    D, N = a.width, a.length
    torch.manual_seed(777)
    leaves, dy, mask, ds, ref, triton, _ = setup(D, N)
    record = dict(
        D=D, L=N, job=os.environ.get("SLURM_JOB_ID"), checks={}, complete=False
    )
    path = R / f"final-D{D}-L{N}.json"

    def save():
        path.write_text(json.dumps(record, indent=2))

    with torch.no_grad(), T.native_context(leaves[0].device):
        model = W.Forward(leaves, mask, ds)
        model()
        torch.cuda.synchronize()
        if a.sanitize:
            leaves[0].mul_(0.9)
            mask.zero_()
            model.mask.copy_(mask)
            model()
            torch.cuda.synchronize()
            print("SANITIZER_DONE", D, N, flush=True)
            return
        old = OLD.Training(*leaves, mask, ds, dy, forward_only=True)
        previous = old.forward().clone()
        new = model().clone()
        torch.cuda.synchronize()
        checks = dict(
            previous=error(new, previous),
            xn=error(model.front.xn, old.xn),
            ab=error(model.front.ab, old.front.ab),
            tri=error(model.tri, old.tri),
        )
        assert all(v < 0.005 for v in checks.values()), checks
        record["checks"]["previous"] = checks
        save()
        print("PREVIOUS", checks, flush=True)
        opts = dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        reference = torch.compile(ref, **opts)
        # The same engine Triton forward, fixed dropout mask and live weights.
        baseline = torch.compile(triton, **opts)
        for mode in ("dropout25", "dropout0"):
            if mode == "dropout0":
                ds.fill_(1)
            y = model().clone()
            yr = reference(*leaves, mask, ds)
            yt = baseline(*leaves, mask, ds)
            checks = dict(
                pytorch=error(y, yr),
                triton=error(y, yt),
                finite=bool(y.isfinite().all()),
            )
            assert (
                checks["finite"]
                and checks["pytorch"] < 0.005
                and checks["triton"] < 0.005
            ), checks
            record["checks"][mode] = checks
            record[mode] = paired(
                {
                    "previous": old.forward,
                    "new": model,
                    "triton": lambda: baseline(*leaves, mask, ds),
                }
            )
            save()
            print(
                "TIMES",
                mode,
                {k: v["median_us"] for k, v in record[mode].items()},
                flush=True,
            )
        graph, gy = capture(model)
        leaves[0].mul_(0.97)
        leaves[1].add_(0.003)
        leaves[5].mul_(1.03)
        leaves[9][0] = 0
        leaves[10].add_(0.017)
        ds[:, :, ::2].zero_()
        expected = model().clone()
        graph.replay()
        torch.cuda.synchronize()
        record["checks"]["graph_bitexact"] = bool(torch.equal(gy, expected))
        assert record["checks"]["graph_bitexact"]
        er = error(expected, reference(*leaves, mask, ds))
        record["checks"]["mutated_reference"] = er
        assert er < 0.005
        saved = [v.clone() for v in model.saved]
        second = W.Forward(leaves, mask, ds)
        second()
        torch.cuda.synchronize()
        record["checks"]["independent_saves"] = all(
            torch.equal(x, y) for x, y in zip(saved, model.saved)
        )
        assert record["checks"]["independent_saves"]
        # Sparse/all-zero masks exercise the gate and normalization endpoints.
        model.mask.zero_()
        mask.zero_()
        leaves[7].zero_()
        y = model()
        yr = reference(*leaves, mask, ds)
        torch.cuda.synchronize()
        er = error(y, yr)
        record["checks"]["zero_mask_gamma"] = er
        assert er < 0.005
        record["selection"] = T.read_config("wide_forward/selection.json")[f"{D}-{N}"]
        record["cubins"] = [
            dict(
                path=k.unit.cubin_path,
                sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest(),
            )
            for k in (model.front.k, model.output.k)
        ]
        record["complete"] = True
        save()
        print("DONE", D, N, flush=True)


if __name__ == "__main__":
    main()
