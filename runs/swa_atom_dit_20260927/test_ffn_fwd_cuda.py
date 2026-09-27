import os, sys, torch, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_swa as TS
from torch.utils.cpp_extension import load
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"); os.makedirs(os.path.join(d, "build_ffnf"), exist_ok=True)
ext = load(name="swa_ffn_fwd_ext", sources=[os.path.join(d, "swa_ffn_fwd.cu")], build_directory=os.path.join(d, "build_ffnf"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"], verbose=False)
C, NH = 128, 256
def pack64(wu): return wu.view(2, NH // 64, 64, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
def triton_ref(q, g, o, mod, wo, wu, wd, A, B, S, eps, save):
    M = A * B * S; dev = q.device
    q2 = torch.empty_like(q); q1 = torch.empty_like(q) if save else q2; r2 = torch.empty(M, device=dev)
    Att = torch.empty_like(q) if save else q2; Ys = torch.empty_like(q) if save else q2; FFs = torch.empty_like(q) if save else q2
    TS._oproj_ffn_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](q, o, g, mod, wo, wu, wd, q1, q2, r2, Att, Ys, Ys, FFs, M, S, B, eps, TS._bucket(M),
                                                          C=C, NHID=NH, MODW=6 * C, SAVE=save)
    return (q2, q1, Att, Ys, FFs) if save else (q2,)
def make(A, B, S):
    gen = torch.Generator(device="cuda").manual_seed(0); M = A * B * S
    r = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=gen) * sc)
    q = r(M, C).bfloat16(); g = r(M, C).bfloat16(); o = r(M, C).bfloat16(); mod = r(B * S, 6 * C, sc=0.3)
    wo = r(C, C, sc=C ** -0.5).bfloat16(); wu = r(2 * NH, C, sc=C ** -0.5).bfloat16(); wd = r(C, NH, sc=NH ** -0.5).bfloat16()
    return q, g, o, mod, wo, wu, wd
eps = torch.finfo(torch.float32).eps
names = ["out", "q1", "att", "y", "ffn"]
for (A, B, S) in [(4, 1, 300), (6, 2, 517), (48, 1, 4096), (5, 1, 4096)]:
    q, g, o, mod, wo, wu, wd = make(A, B, S)
    for save in (True, False):
        ref = triton_ref(q, g, o, mod, wo, wu, wd, A, B, S, eps, save)
        out = ext.ffn_fwd(q, g, o, mod, wo, pack64(wu), wd, A, B, S, eps, save)
        torch.cuda.synchronize()
        errs = []
        for n, a, b in zip(names, out, ref):
            a = a.float(); b = b.float()
            errs.append("%s %.1e (max %.1e)" % (n, ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item()))
        print("A=%d B=%d S=%d save=%d | " % (A, B, S, save) + " ".join(errs), flush=True)
def bench(f, n=50):
    for _ in range(5): f()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / n
for (A, B, S) in [(48, 1, 4096), (5, 1, 4096)]:
    q, g, o, mod, wo, wu, wd = make(A, B, S); wab = pack64(wu); M = A * B * S
    for save in (True, False):
        t0 = bench(lambda: triton_ref(q, g, o, mod, wo, wu, wd, A, B, S, eps, save)); t1 = bench(lambda: ext.ffn_fwd(q, g, o, mod, wo, wab, wd, A, B, S, eps, save))
        fl = M * (4 + (4 if save else 0)) * C * 2 / 3.35e12 * 1e3
        print("A=%d S=%d save=%d  triton %.3f ms  cuda %.3f ms  (%.2fx)  HBM floor %.3f ms" % (A, S, save, t0, t1, t0 / t1, fl))
