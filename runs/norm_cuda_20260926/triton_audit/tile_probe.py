"""Analyze existing Triton bodies with whole-row tiles; no kernel/source edits."""
import json
import math
from pathlib import Path
import torch
from audit import measure
from miniworld_engine.kernels.rmsnorm.triton.main import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel

root = Path(__file__).parent
torch.set_num_threads(4)
torch.manual_seed(823)
results = []
for idx in (1, 2, 3, 4):
    baseline = json.loads((root / f"bench-1-{idx}.json").read_text())
    m, d = baseline["M"], baseline["D"]
    x = torch.randn(m, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    w = torch.randn(d, device="cuda")
    y, dx = torch.empty_like(x), torch.empty_like(x)
    inv = torch.empty(m, device="cuda")
    dw = torch.zeros(d, device="cuda")

    def fwd(c):
        return rmsnorm_fwd_kernel.fn[(math.ceil(m / c["bm"]),)](
            x, y, w, inv, d, 1, m, d, 1e-5, c["bm"], c["bk"], 0, True,
            num_warps=c["warps"], num_stages=c["stages"])

    def bwd(c):
        dw.zero_()
        return rmsnorm_bwd_kernel.fn[(math.ceil(m / c["bm"]),)](
            dx, dy, dw, x, w, inv, d, 1, m, d, c["bm"], c["bk"], 0, True,
            num_warps=c["warps"], num_stages=c["stages"])

    trials = {}
    winners = {}
    old_configs = {}
    for name, fn in (("fwd", fwd), ("bwd", bwd)):
        old = baseline["configs"][f"rmsnorm_{name}_kernel"]
        old_config = dict(bm=old["kwargs"]["BLOCK_M1"], bk=old["kwargs"]["BLOCK_K"], warps=old["warps"], stages=old["stages"])
        old_configs[name] = old_config
        options = [old_config] + [dict(bm=bm, bk=1 << (d - 1).bit_length(), warps=warps, stages=1) for bm in (4, 8, 16, 32, 64) for warps in (4, 8)]
        trials[name] = []
        for c in options:
            try:
                compiled = fn(c)
                t = measure(lambda: fn(c))
                trials[name].append(dict(config=c, timing=t, registers=compiled.n_regs, spills=compiled.n_spills))
            except Exception as e:
                trials[name].append(dict(config=c, error=str(e)))
        winner = min((r for r in trials[name] if "timing" in r), key=lambda r: r["timing"]["median_ms"])
        winners[name] = winner
        compiled = fn(winner["config"])
        (root / f"rms-{idx}-{name}-winner.ptx").write_text(compiled.asm["ptx"])
    def full():
        fwd(winners["fwd"]["config"])
        bwd(winners["bwd"]["config"])
    def baseline_full():
        fwd(old_configs["fwd"])
        bwd(old_configs["bwd"])
    baseline_timing = measure(baseline_full)
    timing = measure(full)
    full()
    xf = x.float()
    rh = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    xh = xf * rh
    wd = dy.float() * w
    expected_dx = ((wd - xh * (xh * wd).mean(-1, keepdim=True)) * rh).to(x.dtype)
    expected_dw = (dy.float() * xh).sum(0)
    errors = [float((a.float() - b.float()).norm() / b.float().norm()) for a, b in ((y, (xh * w).to(x.dtype)), (dx, expected_dx), (dw, expected_dw))]
    row = dict(M=m, D=d, case=idx, trials=trials, winners=winners, baseline_train=baseline_timing, train=timing, errors=errors)
    results.append(row)
    (root / "tile-probe.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k:v for k,v in row.items() if k != "trials"}), flush=True)
