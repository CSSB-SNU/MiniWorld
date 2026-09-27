from pathlib import Path
import sys,os,json,time,gc,statistics
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from lt_config_search import ConfigSearch
from lt_contract import Algorithm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
if D==256:from d256_gate_checkpoint import Training
else:from wide_checkpoint14 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,operations={},screening_version=2)
path=THIS/f'result-lt-config-search-D{D}-L{N}-{record["job"]}.json'
def save():path.write_text(json.dumps(record,indent=2))
def quick(op,ceiling=None):
    if ceiling is None:op()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    repeats=4 if ceiling is None else 1
    for _ in range(repeats):op()
    b.record();b.synchronize()
    us=a.elapsed_time(b)*1000/repeats
    if ceiling is not None and us<=ceiling:
        a.record()
        for _ in range(4):op()
        b.record();b.synchronize();us=a.elapsed_time(b)*250
    return us
def strict(es):return all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    ops={'dx':plan.dx_lt if D==256 else plan.schedule.kernels['dx']}
    if D>256:ops['input_dw']=plan.input_dw
    ops['dn']=plan.schedule.kernels['dn']
    for name in ('fw0','fw1','bc0','bc1','bc2','bc3','proj','gate'):
        if name in plan.schedule.kernels and (D==256 or not name.startswith('bc')):
            ops[name]=plan.schedule.kernels[name]
    if D>256 and plan.split_dwp is not None:ops['dwp']=plan.split_dwp.matmul
    else:ops['dwp']=plan.schedule.kernels['dwp']
    if D==256 or not plan.joint_choice:ops['dwg']=plan.prefix_gate_dw
    wanted=os.environ.get('SEARCH_OPS','dx,input_dw').replace(':',',').split(',');ops={k:v for k,v in ops.items() if k in wanted}
    searches={k:ConfigSearch(v) for k,v in ops.items()}
    original={k:searches[k].copy(v.heuristics[v.index].algo) for k,v in ops.items()}
    selected={}
    for name,op in ops.items():
        search=searches[name];search.install(original[name]);op();torch.cuda.synchronize();target=op.out.clone()
        baseline=quick(op);candidates=list(search.candidates())
        report={'supported':len(candidates),'baseline_us':baseline,'candidates':[],'launch_failures':0,'accuracy_failures':0}
        record['operations'][name]=report;save()
        print('SEARCH',D,N,name,'supported',len(candidates),'baseline_us',baseline,flush=True)
        start=time.monotonic()
        for number,algo in enumerate(candidates):
            search.install(algo)
            try:op();torch.cuda.synchronize()
            except RuntimeError as exc:
                report['launch_failures']+=1
                print('LAUNCH_FAIL',name,number,str(exc),flush=True)
                continue
            same=torch.equal(op.out,target)
            e=0.0 if same else error(op.out,target)
            if not e<(2e-5 if op.out.dtype==torch.bfloat16 else 5e-4):
                report['accuracy_failures']+=1;continue
            us=quick(op,baseline*1.5)
            report['candidates'].append({'algo':list(algo.data),'error':e,'bitwise':same,'quick_us':us})
            if number%100==0:
                print('PROGRESS',name,number,'seconds',round(time.monotonic()-start,1),'best',min(v['quick_us'] for v in report['candidates']),flush=True);save()
        search.install(original[name]);del target;gc.collect()
        top=sorted(report['candidates'],key=lambda v:v['quick_us'])[:12]
        functions={'baseline':op}
        def invoke(algo):
            def run():search.install(algo);op()
            return run
        functions['baseline']=invoke(original[name])
        for i,row in enumerate(top):functions[str(i)]=invoke(Algorithm((__import__('ctypes').c_uint64*8)(*row['algo'])))
        report['paired']=paired(functions)
        for i,row in enumerate(top):row['paired_us']=report['paired'][str(i)]['median_us']
        report['top']=sorted(top,key=lambda v:v['paired_us'])
        search.install(original[name])
        for row in report['top']:
            if row['paired_us']>=report['paired']['baseline']['median_us']*.995:continue
            algo=Algorithm((__import__('ctypes').c_uint64*8)(*row['algo']));search.install(algo)
            yn,gn=plan();es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
            row['full_errors']=es;row['strict']=strict(es)
            if row['strict']:
                selected[name]=algo;report['selected']=row;break
        search.install(original[name]);save()
        print('RESULT',name,'baseline',report['paired']['baseline']['median_us'],'selected',report.get('selected'),flush=True)
    def old():
        for name,search in searches.items():search.install(original[name])
        return plan()
    def new():
        for name,search in searches.items():search.install(selected.get(name,original[name]))
        return plan()
    yn,gn=new();record['combined_errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=strict(record['combined_errors'])
    if selected and record['strict']:
        record['times']=paired({'old_full':old,'new_full':new})
        print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
    record['complete']=True;save()
