"""The machine's ACHIEVABLE bandwidth under the access patterns these kernels actually use, so "% of Speed of Light" is
measured rather than read off a spec sheet.

Each row moves a known number of bytes with no arithmetic; the rate it reaches is the ceiling any kernel with that
pattern can reach on this card.  Reported against the 3.35 TB/s HBM3 peak.
"""
import argparse, json, statistics, torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--output", default="")
a = p.parse_args()
PEAK = 3.35e12
N, C, H = a.length, a.width, a.n_head
D = C // H
dev = "cuda"


def timed(fn, bytes_moved, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(3):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record(); torch.cuda.synchronize()
        o.append(st.elapsed_time(en) * 1e3 / iters)
    us = statistics.median(o)
    return us, bytes_moved / (us * 1e-6)


z = torch.randn(N, N, C, device=dev, dtype=torch.bfloat16)
dst = torch.empty_like(z)
qkv = torch.empty(N, H, N, D, device=dev, dtype=torch.bfloat16)
g = torch.empty(N, N, H * D, device=dev, dtype=torch.bfloat16)
o5 = torch.randn(N, H, N, D, device=dev, dtype=torch.bfloat16)
Z = z.numel() * 2
rows = []


def add(name, fn, nbytes, note=""):
    us, bw = timed(fn, nbytes)
    rows.append(dict(name=name, us=us, MB=nbytes / 1e6, TBs=bw / 1e12, pct_peak=100 * bw / PEAK, note=note))
    print("%-38s %8.1f us  %8.1f MB  %5.2f TB/s  %5.1f %% of 3.35 TB/s  %s"
          % (name, us, nbytes / 1e6, bw / 1e12, 100 * bw / PEAK, note), flush=True)


add("copy [N,N,C] -> [N,N,C]", lambda: dst.copy_(z), 2 * Z, "pure stream, 1 read + 1 write")
add("scatter [N,N,C] -> [N,H,N,D]", lambda: qkv.copy_(z.view(N, N, H, D).permute(0, 2, 1, 3)), 2 * Z, "the q/k/v store layout")
add("gather  [N,H,N,D] -> [N,N,C]", lambda: g.copy_(o5.permute(0, 2, 1, 3).reshape(N, N, H * D)), 2 * Z, "the epilogue's o read")
add("prologue byte pattern", lambda: (qkv.copy_(z.view(N, N, H, D).permute(0, 2, 1, 3)), g.copy_(z)), 4 * Z, "read z twice, write q-shape + g")
add("epilogue byte pattern", lambda: (g.copy_(o5.permute(0, 2, 1, 3).reshape(N, N, H * D)), dst.copy_(z)), 4 * Z, "gather o + read/write z")
print("RESULT " + json.dumps({"length": N, "width": C, "rows": rows}), flush=True)
if a.output:
    json.dump({"length": N, "width": C, "H": H, "D": D, "rows": rows}, open(a.output, "w"), indent=1)
