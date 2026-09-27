"""Refresh the triattn payload's integrity records in place after a local rebuild (no snapshot copy).

build_prebuilt.py writes the .so and its record, but not the two SHA256SUMS files the face checks before loading,
so a freshly built extension is refused ("digest:triattn_m1_ext:SHA256SUMS") and the block silently uses flash.

    python refresh_sums.py <dir containing opt_core> [...]
"""
import hashlib, os, sys

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for root in sys.argv[1:]:
    NAT = os.path.join(root, "opt_core/kernels/triattn/triattn_native")
    PB = os.path.join(NAT, "pkg/v11/triattn_pkg/prebuilt")
    n = 0
    for stack in sorted(os.listdir(PB)):
        f = os.path.join(PB, stack, "SHA256SUMS")
        if not os.path.isfile(f):
            continue
        out = []
        for ln in open(f):
            q = ln.split()
            p = os.path.join(PB, stack, q[1]) if len(q) == 2 else None
            if p and os.path.exists(p):
                h = sha(p)
                n += h != q[0]
                out.append("%s  %s\n" % (h, q[1]))
            else:
                out.append(ln)
        open(f, "w").writelines(out)
    top = os.path.join(NAT, "SHA256SUMS")
    out, m = [], 0
    for ln in open(top):
        q = ln.split()
        p = os.path.join(NAT, q[1]) if len(q) == 2 else None
        if p and os.path.exists(p):
            h = sha(p)
            m += h != q[0]
            out.append("%s  %s\n" % (h, q[1]))
        else:
            out.append(ln)
    open(top, "w").writelines(out)
    live = os.path.join(PB, "torch2.10.0+cu128-cpython-310-x86_64-linux-gnu/triattn_m1_ext.so")
    print("%s: %d stack-dir entries, %d payload entries refreshed; live .so %s"
          % (root, n, m, sha(live)[:16] if os.path.exists(live) else "-"))
