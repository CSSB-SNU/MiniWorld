import os, sys, torch, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_swa as TS
from torch.utils.cpp_extension import load
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"); os.makedirs(os.path.join(d, "build_oprojb"), exist_ok=True)
ext = load(name="swa_oproj_bwd_ext", sources=[os.path.join(d, "swa_oproj_bwd.cu")], build_directory=os.path.join(d, "build_oprojb"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"], verbose=False)
C, H, D = 128, 4, 32
def triton_ref(dq1, o, g, att, mod, wo, A, B, S):
    N = A * B; M = N * S; dev = dq1.device
    dmod = torch.zeros(B * S, 6 * C, device=dev); dO = torch.empty_like(dq1); dG = torch.empty_like(dq1); Dv = torch.empty(N, H, S, device=dev)
    datt = torch.empty_like(dq1); gated = torch.empty_like(dq1)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    TS._oproj_bwd[grid_t](dq1, o, g, mod, wo, att, dO, dG, Dv, datt, gated, dmod, S, A, B, TS._bucket(M), C=C, H=H, D=D, MODW=6 * C)
    return dO, dG, Dv, datt, gated, dmod
def cuda(dq1, o, g, att, mod, wot, A, B, S):
    dmod = torch.zeros(B * S, 6 * C, device=dq1.device)
    dO, dG, Dv, datt, gated = ext.oproj_bwd(dq1, o, g, att, mod, dmod, wot, A, B, S)
    return dO, dG, Dv, datt, gated, dmod
def make(A, B, S):
    gen = torch.Generator(device="cuda").manual_seed(0); M = A * B * S
    r = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=gen) * sc)
    return (r(M, C).bfloat16(), r(M, C).bfloat16(), r(M, C).bfloat16(), r(M, C).bfloat16(), r(B * S, 6 * C, sc=0.3), r(C, C, sc=C ** -0.5).bfloat16())
names = ["dO", "dG", "Dv", "datt", "gated", "dmod"]
for (A, B, S) in [(4, 1, 300), (6, 2, 517), (48, 1, 4096), (5, 1, 4096)]:
    dq1, o, g, att, mod, wo = make(A, B, S); wot = wo.t().contiguous()
    ref = triton_ref(dq1, o, g, att, mod, wo, A, B, S); out = cuda(dq1, o, g, att, mod, wot, A, B, S)
    torch.cuda.synchronize()
    print("A=%d B=%d S=%d | " % (A, B, S) + " ".join("%s %.1e" % (n, ((a.float() - b.float()).norm() / b.float().norm()).item()) for n, a, b in zip(names, out, ref)), flush=True)
def bench(f, n=50):
    for _ in range(5): f()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / n
for (A, B, S) in [(48, 1, 4096), (5, 1, 4096)]:
    dq1, o, g, att, mod, wo = make(A, B, S); wot = wo.t().contiguous(); M = A * B * S
    t0 = bench(lambda: triton_ref(dq1, o, g, att, mod, wo, A, B, S)); t1 = bench(lambda: cuda(dq1, o, g, att, mod, wot, A, B, S))
    fl = M * (8 * C * 2 + 16) / 3.35e12 * 1e3
    print("A=%d S=%d  triton %.3f ms  cuda %.3f ms  (%.2fx)  HBM floor %.3f ms" % (A, S, t0, t1, t0 / t1, fl))
