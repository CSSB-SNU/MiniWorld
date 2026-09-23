"""A/B experiment: split the front's serial channel loop into grouped grid tiles."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def tiled_front(X, W, LEFT, RIGHT, PRE, MASK, M: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), 2 * tl.cdiv(H, BN)
    group = pid // (GROUP_M * nn)
    first = group * GROUP_M
    size = tl.minimum(nm-first, GROUP_M)
    pm = first + (pid % (GROUP_M * nn)) % size
    pn = (pid % (GROUP_M * nn)) // size
    side = pn // tl.cdiv(H, BN)
    channel = (pn % tl.cdiv(H, BN)) * BN
    rows = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    kk = tl.arange(0, BK)
    cols = 2 * channel + tl.arange(0, 2 * BN)
    acc = tl.full((BM, 2 * BN), 0, tl.float32)
    for k in range(tl.cdiv(K, BK)):
        rk = k * BK + kk
        x = tl.load(X + rows[:, None] * K + rk[None, :],
                    (rows[:, None] < M) & (rk[None, :] < K), 0)
        w = tl.load(W + rk[:, None] * (4 * H) + side * (2 * H) + cols[None, :],
                    (rk[:, None] < K) & (cols[None, :] < 2 * H), 0)
        acc = tl.dot(x, w, acc)
    gate, proj = tl.split(tl.reshape(acc, (BM, BN, 2)))
    channels = (channel + tl.arange(0, BN)).to(tl.int64)
    valid = (channels[:, None] < H) & (rows[None, :] < M)
    tl.store(PRE + (side * 2 * H + 2 * channels[:, None]) * M + rows[None, :],
             tl.trans(gate), valid)
    tl.store(PRE + (side * 2 * H + 2 * channels[:, None] + 1) * M + rows[None, :],
             tl.trans(proj), valid)
    scale = tl.load(MASK + rows, rows < M, 0).to(tl.float32)
    result = (tl.sigmoid(gate) * proj).to(LEFT.dtype.element_ty).to(tl.float32) * scale[:, None]
    if side == 0:
        tl.store(LEFT + channels[:, None] * M + rows[None, :], tl.trans(result), valid)
    else:
        tl.store(RIGHT + channels[:, None] * M + rows[None, :], tl.trans(result), valid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import _bidir_front_launch
    from miniworld_engine.autotune.shape_key import token_key
    m, k, h = 384**2, 128, 256
    torch.manual_seed(65)
    opts = dict(device='cuda', dtype=torch.bfloat16)
    x, w = torch.randn(m, k, **opts), torch.randn(k, 4*h, **opts) / k**.5
    mask = (torch.rand(m, device='cuda') > .2).to(torch.bfloat16)
    baseline = lambda: _bidir_front_launch(x, w, h, 384, True, token_key(384), mask)
    ref = baseline()
    left, right, pre = [torch.empty_like(v) for v in ref]
    baseline_ms = triton.testing.do_bench_cudagraph(baseline, rep=80)
    rows = []
    for bm, bn, bk, group, warps, stages in itertools.product([32, 64, 128], [32, 64], [32, 64], [1, 4], [4, 8], [3]):
        cfg = dict(BM=bm, BN=bn, BK=bk, GROUP_M=group, num_warps=warps, num_stages=stages)
        call = lambda: tiled_front[(triton.cdiv(m,bm)*2*triton.cdiv(h,bn),)](x,w,left,right,pre,mask,m,k,h,**cfg)
        try:
            call()
            errors = [float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip((left,right,pre),ref)]
            assert max(errors)<.002, errors
            ms = triton.testing.do_bench_cudagraph(call, rep=12)
            rows.append(dict(config=cfg, ms=ms, errors=errors))
        except triton.OutOfResources as error:
            rows.append(dict(config=cfg, error=str(error)))
        if len(rows)%24==0:
            print('CHECKED',len(rows),flush=True)
    report = dict(baseline_ms=baseline_ms, rows=rows,
                  best=min((r for r in rows if 'ms' in r), key=lambda r:r['ms']))
    args.output.write_text(json.dumps(report, indent=2))
    print('RESULT',report['baseline_ms'],report['best'],flush=True)


if __name__ == '__main__':
    main()
