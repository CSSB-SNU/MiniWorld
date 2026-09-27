import os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ffn_cuda.py")).read().split("eps = torch.finfo")[0])
eps = torch.finfo(torch.float32).eps
A, B, S = 48, 1, 4096
dq2, q1, y, ffn, mod, wu, wd = make(A, B, S); packed = pack(wu, wd)
for sp, abl in ((8, 0), (8, 16), (8, 32), (8, 64), (8, 112)):
    os.environ["SWA_FFN_ABL"] = str(abl); print("ablate", abl)
    os.environ["SWA_FFN_PROF"] = "0"
    for _ in range(3): cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp)
    os.environ["SWA_FFN_PROF"] = "1"
    cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp); torch.cuda.synchronize()
    p = ext.prof().cpu().tolist(); ncta = 132
    names = ["gemm: gap", "gemm: wait afull", "gemm: chunks", "gemm: wait pfree", "gemm: last chunk", "gemm: tail (store wait+dy)",
             "row: gap", "row: wait cfull", "row: reduce gate (rest)", "passA: gap", "passA: wait xfull", "passA: work", "row: reduce shift", "row: dq1 rows", "row: reduce scale", "row: gate rows"]
    print("sp=%d  (us per CTA at 1.755 GHz; row = sum of both row WGs)" % sp)
    for i, n in enumerate(names):
        if i in (2, 3, 13, 15): print("   %-28s %8.1f" % (n, p[i] / ncta / 1755.0))
    os.environ["SWA_FFN_PROF"] = "0"
    for _ in range(3): cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp)
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); st.record()
    for _ in range(20): cuda(dq2, q1, y, ffn, mod, packed, A, B, S, eps, sp)
    en.record(); torch.cuda.synchronize(); print("   time %.3f ms" % (st.elapsed_time(en) / 20))
