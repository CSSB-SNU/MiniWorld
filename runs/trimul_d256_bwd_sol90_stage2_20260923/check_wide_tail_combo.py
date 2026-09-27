"""Measure the complete combination of individually strict long-shape pilots."""
from pathlib import Path
import sys,os,json,gc,ctypes as C
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from prefix_gate_pipeline import PipelinedGateEpi
from wide_specialized_output_epi import SpecializedOutputEpi
from lt_config_search import ConfigSearch
from lt_contract import Algorithm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=(384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=768
search_path=THIS/f'result-lt-config-search-D{D}-L768-{18336 if D==384 else 18339}.json'
search_result=json.loads(search_path.read_text());assert search_result['complete']
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,lt_source=str(search_path),candidates=[])
path=THIS/f'result-wide-tail-combo-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    ops={'dx':plan.schedule.kernels['dx'],'input_dw':plan.input_dw}
    searches={n:ConfigSearch(op) for n,op in ops.items()}
    original={n:s.copy(ops[n].heuristics[ops[n].index].algo) for n,s in searches.items()}
    choices={n:r['selected'] for n,r in search_result['operations'].items() if 'selected' in r}
    selected={n:Algorithm((C.c_uint64*8)(*r['algo'])) for n,r in choices.items()}
    old_gate=plan.b1.epi;out=plan.product_output;old_out=out.epi;new_out=SpecializedOutputEpi(plan,4)
    for slots in (1,2):
        gate=PipelinedGateEpi(plan,slots)
        item=dict(slots=slots,vec=4,algorithms=choices,cubins=[str(gate.cubin),str(new_out.cubin)])
        record['candidates'].append(item)
        def install(new):
            for n,s in searches.items():s.install(selected.get(n,original[n]) if new else original[n])
            plan.b1.epi=gate if new else old_gate;out.epi=new_out if new else old_out
        def old():install(False);return plan()
        def new():install(True);return plan()
        def old_bwd():install(False);return plan.backward()
        def new_bwd():install(True);return plan.backward()
        yn,gn=new();torch.cuda.synchronize()
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['backward_times']=paired(dict(old_bwd=old_bwd,new_bwd=new_bwd));gc.collect()
            item['full_times']=paired(dict(old_full=old,new_full=new));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in item[key].items()},flush=True)
        old()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
