"""Install a locally built prebuilt snapshot and refresh the package's integrity records.

The face verifies, in order: source_sha256 (csrc) -> so_sha256 (the record) -> the payload's SHA256SUMS -> loadcheck.
A locally rebuilt .so therefore needs the last two refreshed too, or it is refused and the block silently falls back
to flash_triattn (which is what made three earlier A/B runs meaningless).

    python install_variant.py pb-base2      # or pb-f16x2b
"""
import hashlib, os, sys, shutil

R = os.path.dirname(os.path.abspath(__file__))
NAT = os.path.join(R, "oc/opt_core/kernels/triattn/triattn_native")
PB = os.path.join(NAT, "pkg/v11/triattn_pkg/prebuilt")
src = os.path.join(R, sys.argv[1])
if not any(n.startswith("torch") for n in os.listdir(src)) and os.path.isdir(os.path.join(src, "prebuilt")):
    src = os.path.join(src, "prebuilt")          # only when the snapshot itself is a wrapper: a nested copy can be stale

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if os.path.exists(PB):
    shutil.rmtree(PB)
shutil.copytree(src, PB)

# 1. the stack directory's own SHA256SUMS (file name -> hash, one per .so/.json beside it)
for stack in sorted(os.listdir(PB)):
    d = os.path.join(PB, stack)
    if not os.path.isdir(d):
        continue
    f = os.path.join(d, "SHA256SUMS")
    if not os.path.exists(f):
        continue
    lines = []
    for ln in open(f):
        parts = ln.split()
        if len(parts) == 2 and os.path.exists(os.path.join(d, parts[1])):
            lines.append("%s  %s\n" % (sha(os.path.join(d, parts[1])), parts[1]))
        else:
            lines.append(ln)
    open(f, "w").writelines(lines)

# 2. the payload-level SHA256SUMS (paths relative to triattn_native/)
top = os.path.join(NAT, "SHA256SUMS")
out, fixed = [], 0
for ln in open(top):
    parts = ln.split()
    if len(parts) == 2:
        p = os.path.join(NAT, parts[1])
        if os.path.exists(p):
            h = sha(p)
            if h != parts[0]:
                fixed += 1
            out.append("%s  %s\n" % (h, parts[1]))
            continue
    out.append(ln)
open(top, "w").writelines(out)

# 3. the test vectors this variant is byte-gated against.  A kernel that changes numerics (e.g. the f16x2 exp2) cannot
#    reproduce the stock bytes, so it carries its own generated vectors in tv/<variant>/; anything else gets the pristine set.
TV = os.path.join(NAT, "pkg/v11/testvectors")
tvsrc = os.path.join(R, "tv", os.path.basename(sys.argv[1]))
if not os.path.isdir(tvsrc):
    tvsrc = os.path.join(R, "tv", "pristine")
nv = 0
for n in sorted(os.listdir(tvsrc)):
    d = os.path.join(TV, n); s_ = os.path.join(tvsrc, n)
    if not os.path.exists(d) or sha(d) != sha(s_):
        shutil.copyfile(s_, d); nv += 1

stack = os.path.join(PB, "torch2.10.0+cu128-cpython-310-x86_64-linux-gnu")
print("installed %s; refreshed %d payload entries; vectors from %s (%d file(s) replaced); live .so %s"
      % (os.path.basename(sys.argv[1]), fixed, os.path.basename(tvsrc), nv, sha(os.path.join(stack, "triattn_m1_ext.so"))[:16]))
