import os, sys, torch, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_swa as TS
from torch.utils.cpp_extension import load
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"); os.makedirs(os.path.join(d, "build_qkvgf"), exist_ok=True)
ext = load(name="swa_qkvg_fwd_ext", sources=[os.path.join(d, "swa_qkvg_fwd.cu")], build_directory=os.path.join(d, "build_qkvgf"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"], verbose=False)
C, H, D = 128, 4, 32
FE = torch.finfo(torch.float32).eps
def triton_ref(q, mod, cs, sn, wqkv, wg, A, B, S, save):
    N = A * B; M = N * S; dev = q.device
    Qh = torch.empty(N, H, S, D, device=dev, dtype=q.dtype); Kh = torch.empty_like(Qh); Vh = torch.empty_like(Qh); G = torch.empty_like(q)
    r1 = torch.empty(M, device=dev)
    Xs = torch.empty_like(q) if save else G; PQs = torch.empty_like(q) if save else G; PKs = torch.empty_like(q) if save else G
    TS._qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](q, mod, cs, sn, wqkv, wg, Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, FE, FE, TS._bucket(M),
                                                       C=C, H=H, D=D, MODW=6 * C, SAVE=save)
    return (Qh, Kh, Vh, G, Xs, PQs, PKs) if save else (Qh, Kh, Vh, G)
def make(A, B, S):
    gen = torch.Generator(device="cuda").manual_seed(0); M = A * B * S
    r = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=gen) * sc)
    q = r(M, C).bfloat16(); mod = r(B * S, 6 * C, sc=0.3)
    ang = r(B * S, D // 2); cs = torch.cos(ang).contiguous(); sn = torch.sin(ang).contiguous()
    wqkv = r(3 * C, C, sc=C ** -0.5).bfloat16(); wg = r(C, C, sc=C ** -0.5).bfloat16()
    return q, mod, cs, sn, wqkv, wg
names = ["Q", "K", "V", "G", "X", "PQ", "PK"]
for (A, B, S) in [(4, 1, 300), (6, 2, 517), (48, 1, 4096), (5, 1, 4096)]:
    q, mod, cs, sn, wqkv, wg = make(A, B, S); w = torch.cat([wqkv, wg]).contiguous()
    for save in (True, False):
        ref = triton_ref(q, mod, cs, sn, wqkv, wg, A, B, S, save)
        out = ext.qkvg_fwd(q, mod, cs, sn, w, A, B, S, FE, FE, save)
        torch.cuda.synchronize()
        errs = []
        for n, a, b in zip(names, out, ref):
            a = a.float(); b = b.float()
            errs.append("%s %.1e" % (n, ((a - b).norm() / b.norm()).item()))
        print("A=%d B=%d S=%d save=%d | " % (A, B, S, save) + " ".join(errs), flush=True)
def bench(f, n=50):
    for _ in range(5): f()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / n
for (A, B, S) in [(48, 1, 4096), (5, 1, 4096)]:
    q, mod, cs, sn, wqkv, wg = make(A, B, S); w = torch.cat([wqkv, wg]).contiguous(); M = A * B * S
    for save in (True, False):
        t0 = bench(lambda: triton_ref(q, mod, cs, sn, wqkv, wg, A, B, S, save)); t1 = bench(lambda: ext.qkvg_fwd(q, mod, cs, sn, w, A, B, S, FE, FE, save))
        fl = M * (5 + (3 if save else 0)) * C * 2 / 3.35e12 * 1e3
        print("A=%d S=%d save=%d  triton %.3f ms  cuda %.3f ms  (%.2fx)  HBM floor %.3f ms" % (A, S, save, t0, t1, t0 / t1, fl))
