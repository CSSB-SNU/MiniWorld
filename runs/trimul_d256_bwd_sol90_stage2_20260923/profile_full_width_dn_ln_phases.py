"""Per-CTA cycle diagnostics; these are not end-to-end performance evidence."""
from pathlib import Path
import sys,os,json,statistics
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from wide_full_width_dn_ln import FullWidthDnLN
from d256_async_full_width_dn_ln import AsyncFullWidthDnLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),diagnostic_only=True,candidates=[])
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();dt=p.dt.clone()
    for label,cls in [('synchronous',FullWidthDnLN),('independent_producer',AsyncFullWidthDnLN)]:
        op=cls(plan,False,64);body=op.source_text
        marker='bf* dn;int M;';assert body.count(marker)==1
        body=body.replace(marker,'bf* dn;unsigned long long* clocks;int M;')
        stamp=lambda phase:f' if(threadIdx.x==0)p.clocks[size_t(blockIdx.x)*6+{phase}]=clock64();\n'
        marker=' for(int col=0;col<H;col+=H){';assert body.count(marker)==1;body=body.replace(marker,stamp(0)+marker)
        marker=' // GEMM accumulators are dead';assert body.count(marker)==1;body=body.replace(marker,stamp(1)+marker)
        marker='transpose32<false>(sm);';assert body.count(marker)==1;body=body.replace(marker,marker+'\n'+stamp(2))
        barrier='__syncthreads();' if label=='synchronous' else 'named_bar_sync(1,NT);'
        marker='  '+barrier+'\n  for(int r=warp;';assert body.count(marker)==1;body=body.replace(marker,'  '+barrier+'\n'+stamp(3)+'  for(int r=warp;')
        marker='transpose32<true>(sm);fence_proxy_async();'+barrier;assert body.count(marker)==1;body=body.replace(marker,marker+'\n'+stamp(4))
        marker='tma_store_commit();tma_store_wait_read<0>();}'+barrier;assert body.count(marker)==1;body=body.replace(marker,marker+'\n'+stamp(5))
        oldname='mw_wide_full_width_dn_ln' if label=='synchronous' else 'mw_d256_async_full_width_dn_ln'
        name='mw_profile_full_width_dn_ln';body=body.replace(oldname,name)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256','-DEMIT_DN=0']
        cubin=T.compile_text(body,flags);k=T.load_unit(str(cubin),name).kernel(name);k.set_max_dynamic_smem(op.smem)
        clocks=torch.empty((op.rows,6),device=p.x.device,dtype=torch.int64)
        fields=op.params.fields.copy();fields.insert(len(fields)-1,clocks);params=T._launch_module().Struct(fields)
        drv=k.unit.drv;fn=drv.d.CUfunction(int(k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        item=dict(label=label,cubin=str(cubin),registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS'),local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES'))
        if label=='independent_producer':assert item['registers']>=128
        for _ in range(4):k.launch((op.rows,1,1),(op.threads,1,1),[params],op.smem)
        torch.cuda.synchronize();item['dt_error']=error(p.dt,dt)
        cycles=(clocks[:,1:]-clocks[:,:-1]).cpu().tolist()
        labels=['gemm','triangle_load_transpose','affine','ln_transpose','store_read_wait']
        item['median_cycles']={n:statistics.median(x[i] for x in cycles) for i,n in enumerate(labels)}
        item['p90_cycles']={n:sorted(x[i] for x in cycles)[int(len(cycles)*.9)] for i,n in enumerate(labels)}
        record['candidates'].append(item);print('PHASES',item,flush=True)
record['complete']=True
(THIS/f'profile-full-width-dn-ln-phases-{record["job"]}.json').write_text(json.dumps(record,indent=2))
