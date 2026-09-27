"""Diagnostic single-tile clocks, never an end-to-end performance authority."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from initial_register_pool import initial_pool
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];old=plan.pool_source
    stamps=torch.zeros((32*old.splits,8),device=p.x.device,dtype=torch.int64)
    body=old.source_text
    marker='float* part;int M;';assert body.count(marker)==1
    body=body.replace(marker,'float* part;unsigned long long* clocks;int M;')
    def stamp(i):return f'\n  if(tid==0 && tile==begin+4)p.clocks[blockIdx.x*8+{i}]=clock64();\n'
    marker='  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);';assert body.count(marker)==1
    body=body.replace(marker,stamp(0)+marker+stamp(1))
    marker='wgmma_wait<0>();fence_regs(pre);';assert body.count(marker)==1
    body=body.replace(marker,marker+stamp(2))
    marker='  packed_glu(pre,xn,sm+DERIV,ma,mb);';assert body.count(marker)==1
    body=body.replace(marker,marker+stamp(3))
    marker='  if(tid==0){store_gp(';assert body.count(marker)==1
    body=body.replace(marker,stamp(4)+marker)
    marker='});wgmma_commit();wgmma_wait<0>();fence_regs(dw);';assert body.count(marker)==1
    body=body.replace(marker,'});wgmma_commit();'+stamp(5)+'wgmma_wait<0>();fence_regs(dw);'+stamp(6))
    marker='if(tid==0)mbar_arrive(bar+3+slot);';assert body.count(marker)==1
    body=body.replace(marker,marker+stamp(7))
    body=body.replace('mw_d256_register_budget_source','mw_d256_source_clock')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={old.splits}']
    cubin=T.compile_text(body,flags);cubin,metadata=initial_pool(cubin,'mw_d256_source_clock',120,208)
    k=T.load_unit(str(cubin),'mw_d256_source_clock').kernel('mw_d256_source_clock');k.set_max_dynamic_smem(old.smem)
    fields=old.params.fields.copy();fields.insert(-1,stamps);params=T._launch_module().Struct(fields)
    def op():k.launch((32*old.splits,1,1),(256,1,1),[params],old.smem)
    plan.late_output_dw.oldsource=op;y,g=plan();torch.cuda.synchronize()
    errors={n:error(a,b) for n,a,b in zip(names,[y,*g],expected)}
    strict=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in errors.items());assert strict,errors
    labels=['input_wait','projection','glu','glu_sync','dw_issue','dw_wait','store_wait']
    samples=[]
    for i in range(5):
        stamps.zero_();op();torch.cuda.synchronize();raw=stamps.cpu();assert bool((raw>0).all());delta=raw[:,1:]-raw[:,:-1];assert bool((delta>=0).all())
        samples.append({n:dict(median_cycles=float(delta[:,j].float().median()),mean_cycles=float(delta[:,j].float().mean()),max_cycles=int(delta[:,j].max())) for j,n in enumerate(labels)})
    record=dict(job=os.environ.get('SLURM_JOB_ID'),strict=strict,errors=errors,local_bytes=k.local_bytes,samples=samples,cubin=str(cubin),diagnostic_only=True)
    (THIS/f'result-d256-source-clock-{record["job"]}.json').write_text(json.dumps(record,indent=2));print('CLOCK',record,flush=True)
