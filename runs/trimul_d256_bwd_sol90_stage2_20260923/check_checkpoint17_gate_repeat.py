from pathlib import Path
import sys,os,json,gc,ctypes
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint17 import Training
from prefix_gate_pipeline import PipelinedGateEpi
from wide_specialized_output_epi import SpecializedOutputEpi
from lt_config_search import ConfigSearch
from lt_contract import Algorithm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-checkpoint17-gate-repeat-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    oldgate=plan.b1.epi;oldout=plan.product_output.epi
    newgate=PipelinedGateEpi(plan,2);newout=SpecializedOutputEpi(plan,4)
    choices={};controls={};searches={}
    source={ (384,768):'result-lt-config-search-D384-L768-18336.json',
             (512,384):'result-lt-config-search-D512-L384-18345.json' }.get((D,N))
    if source:
        report=json.loads((THIS/source).read_text());assert report['complete'] and report['strict']
        for name,row in report['operations'].items():
            if 'selected' not in row:continue
            choice=row['selected'];assert choice['strict'] and choice['bitwise']
            op=plan.input_dw if name=='input_dw' else plan.schedule.kernels[name]
            searches[name]=ConfigSearch(op);controls[name]=searches[name].copy(op.heuristics[op.index].algo)
            choices[name]=Algorithm((ctypes.c_uint64*8)(*choice['algo']))
    def install(mode):
        plan.b1.epi=oldgate if mode=='old' else newgate
        plan.product_output.epi=newout if mode=='combined' else oldout
        for name,search in searches.items():search.install(choices[name] if mode=='combined' else controls[name])
    def before():install('old');return plan()
    for mode in ('gate','gate','gate'):
        def after():install(mode);return plan()
        yn,gn=after();torch.cuda.synchronize()
        item=dict(mode=mode,source=source,algorithms={k:list(v.data) for k,v in choices.items()},cubins=[str(newgate.cubin),str(newout.cubin)])
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        record['candidates'].append(item);print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            def oldback():install('old');return plan.backward()
            def newback():install(mode);return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldback,new_backward=newback));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('backward_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
