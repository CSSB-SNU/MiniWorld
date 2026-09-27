"""Paired measurements against the validated warp-specialized checkpoint."""
from pathlib import Path
import os, sys, json
THIS = Path(__file__).resolve().parent
PRE = THIS.parent / 'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0, str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0, str(THIS))
from selected_current import Training
from saved_norm import enable
N = int(os.environ.get('LENGTH', '384'))
assert os.environ.get('SAVE_NORM', '0') == '0'
leaves, dy, mask, ds, ref, triton, names = setup(256, N)
record = dict(L=N, job=os.environ.get('SLURM_JOB_ID'), complete=False)
with torch.no_grad(), T.native_context(leaves[0].device):
    old = Training(leaves, mask, ds, dy)
    new = Training(leaves, mask, ds, dy)
    enable(new)
    yo, go = old()
    expected = [t.clone() for t in go]
    yn, gn = new()
    record['forward_equal'] = torch.equal(yo, yn)
    record['save_equal'] = {name:torch.equal(a,b) for name,a,b in (
        ('norm',old.p.tensors[6],new.p.tensors[6]),
        ('mu',old.p.floats[5],new.p.floats[5]), ('rs',old.p.floats[6],new.p.floats[6]))}
    record['errors'] = {name:error(a,b) for name,a,b in zip(names[1:],gn,expected)}
    record['strict'] = all(v < (2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK', record, flush=True)
    assert record['forward_equal'] and all(record['save_equal'].values()) and record['strict']
    record['times'] = paired(dict(old_fwd=old.forward,new_fwd=new.forward,
                                 old_bwd=old.backward,new_bwd=new.backward,old_full=old,new_full=new))
    print({k:v['median_us'] for k,v in record['times'].items()}, flush=True)
record['complete'] = True
record['config'] = {k:v for k,v in os.environ.items() if k.startswith(('GP_', 'DX_', 'LN_', 'CUTE_', 'SAVE_'))}
(THIS/f'result-saved-norm-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
