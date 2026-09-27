from pathlib import Path
import sys, json, argparse, os
R = Path(__file__).resolve().parent
ROOT = R.parent.parent
sys.path.insert(0, str(ROOT / '.engine-release-2.0.0/src'))
sys.path.insert(0, str(R.parent / 'trimul_cuda_widths_opt_20260923'))
sys.path.insert(0, str(R.parent / 'trimul_forward_wide_20260923'))
from fixture import setup
from validate_engine import paired, error
sys.path.insert(0, str(R))
from plan import B1, tail
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

a = argparse.ArgumentParser()
a.add_argument('--width', type=int, required=True)
a.add_argument('--length', type=int, default=384)
a.add_argument('--sanitize', action='store_true')
args = a.parse_args()
D, N = args.width, args.length
leaves, dy, mask, ds, ref, triton, names = setup(D,N)
record = dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path = R / f'result-D{D}-L{N}.json'

with torch.no_grad(), T.native_context(leaves[0].device):
    f = F.Forward(leaves, mask, ds)
    f()
    p = W.Training(*leaves, mask, ds, dy, saved=(f.front.ab,f.tri,f.front.xn))
    b1 = B1(p)
    def old_b1(): W.launch(p.ks['b1'],p.params,p.grid,D=D)
    def new_bwd():
        b1()
        return tail(p)
    def old_full():
        f()
        return p.backward()
    def new_full():
        f()
        return new_bwd()
    if args.sanitize:
        new_full()
        torch.cuda.synchronize()
        print('SANITIZER_DONE', D, N, flush=True)
        raise SystemExit
    old = [t.clone() for t in p.backward()]
    new = [t.clone() for t in new_bwd()]
    record['vs_previous'] = dict(zip(names[1:], [error(x,y) for x,y in zip(new,old)]))
    print('previous', record['vs_previous'], flush=True)
    path.write_text(json.dumps(record,indent=2))

# Compiled PyTorch reference follows the established BF16 training fixture.
reference = torch.compile(ref)
y = reference(*leaves,mask,ds)
grads = torch.autograd.grad(y,leaves,dy)
record['vs_pytorch'] = dict(zip(names[1:], [error(x,y) for x,y in zip(new,grads)]))
print('pytorch', record['vs_pytorch'], flush=True)
# Existing width-validation bound for all eleven gradients.
assert all(v < .01 for v in record['vs_pytorch'].values()), record
assert all(v < .005 for v in record['vs_previous'].values()), record
del y, grads, old, new
with torch.no_grad(), T.native_context(leaves[0].device):
    record['times'] = paired(dict(old_b1=old_b1,new_b1=b1,old_backward=p.backward,new_backward=new_bwd,old_full=old_full,new_full=new_full))
record['complete'] = True
path.write_text(json.dumps(record,indent=2))
print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
