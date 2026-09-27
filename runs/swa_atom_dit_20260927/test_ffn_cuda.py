import os, sys, torch, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_swa as TS
from torch.utils.cpp_extension import load
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"); os.makedirs(os.path.join(d, "build_ffnb"), exist_ok=True)
ext = load(name="swa_ffn_bwd_ext", sources=[os.path.join(d, "swa_ffn_bwd.cu")], build_directory=os.path.join(d, "build_ffnb"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xptxas=-v"], verbose=False)
C, NH = 128, 256
def pack(wu, wd):
    wab = wu.view(2, NH // 32, 32, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
    return wab, wd.t().contiguous(), wab.t().contiguous()
def triton_ref(dq2, q1, y, ffn, mod, wu, wd, A, B, S, eps):
    M = A * B * S; dev = dq2.device
    dmod = torch.zeros(B * S, 6 * C, device=dev); dq1 = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dab = torch.empty(M, 2 * NH, device=dev, dtype=torch.bfloat16); hh = torch.empty(M, NH, device=dev, dtype=torch.bfloat16); dffn = torch.empty_like(dq1)
    grid = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    TS._ffn_bwd[grid](dq2, q1, mod, wu, wd, y, ffn, dq1, dab, hh, dffn, dmod, S, A, B, eps, TS._bucket(M), C=C, NHID=NH, MODW=6 * C, DWOPS=True)
    return dq1, dffn, hh, dab, dmod
def cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp=0):
    dmod = torch.zeros(B * S, 6 * C, device=dq2.device)
    dq1, dffn, hh, dab = ext.ffn_bwd(dq2, q1, y, ffn, mod, dmod, *packed, A, B, S, eps, sp)
    return dq1, dffn, hh, dab, dmod
def make(A, B, S):
    g = torch.Generator(device="cuda").manual_seed(0); M = A * B * S
    r = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=g) * sc)
    dq2 = r(M, C).bfloat16(); q1 = r(M, C).bfloat16(); y = r(M, C).bfloat16(); ffn = r(M, C, sc=0.5).bfloat16()
    mod = r(B * S, 6 * C, sc=0.3); wu = r(2 * NH, C, sc=C ** -0.5).bfloat16(); wd = r(C, NH, sc=NH ** -0.5).bfloat16()
    return dq2, q1, y, ffn, mod, wu, wd
eps = torch.finfo(torch.float32).eps
names = ["dq1", "dffn", "h", "dab", "dmod"]
for (A, B, S) in [(4, 1, 300), (6, 2, 517), (48, 1, 4096)]:
    dq2, q1, y, ffn, mod, wu, wd = make(A, B, S)
    ref = triton_ref(dq2, q1, y, ffn, mod, wu, wd, A, B, S, eps)
    packed = pack(wu, wd)
    for sp in (1, 2, 4, 8):
        out = cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp)
        torch.cuda.synchronize()
        errs = []
        for n, a, b in zip(names, out, ref):
            a = a.float(); b = b.float()
            errs.append("%s %.1e" % (n, ((a - b).norm() / b.norm().clamp_min(1e-30)).item()))
        print("A=%d B=%d S=%d sp=%d | " % (A, B, S, sp) + " ".join(errs), flush=True)
A, B, S = 48, 1, 4096
dq2, q1, y, ffn, mod, wu, wd = make(A, B, S); packed = pack(wu, wd)
def bench(f, n=50):
    for _ in range(5): f()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / n
print("triton _ffn_bwd  %.3f ms" % bench(lambda: triton_ref(dq2, q1, y, ffn, mod, wu, wd, A, B, S, eps)))
for sp in (1, 2, 4, 8):
    print("cuda sp=%d       %.3f ms" % (sp, bench(lambda: cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp))))
M = A * B * S
print("HBM floor (dq2,q1,y,ffn reads + dq1,dffn,h,dab writes = %d B/row): %.3f ms @3.35TB/s" % (2 * (4 * C + 2 * C + 3 * NH), M * 2 * (6 * C + 3 * NH) / 3.35e12 * 1e3))
