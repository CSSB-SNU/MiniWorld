import os, sys
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ffn_cuda.py")).read().split("eps = torch.finfo")[0])
eps = torch.finfo(torch.float32).eps
A, B, S = 48, 1, 4096
dq2, q1, y, ffn, mod, wu, wd = make(A, B, S); packed = pack(wu, wd)
def bench(f, n=50):
    for _ in range(5): f()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / n
for ab in (0, 1, 2, 4, 6):
    os.environ["SWA_FFN_ABL"] = str(ab)
    print("ablate %2d  %.3f ms" % (ab, bench(lambda: cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, 4))), flush=True)
