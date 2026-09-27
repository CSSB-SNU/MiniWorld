from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from cluster_bulk_dn_ln import ClusterBulkDnLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,*_=setup(512,384)
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();p=plan.p;op=ClusterBulkDnLN(plan)
    body=(THIS/'cluster_bulk_dn_ln.cu').read_text().replace('// MMA_HELPERS',(PRE/'mma.cuh').read_text())
    body=body.replace('int M;};','int M;uint64_t* clocks;};\nTMN_DEVI void mark(const Params& p,int point){if(threadIdx.x==0)p.clocks[size_t(blockIdx.x)*10+point]=clock64();}')
    body=body.replace('}__syncthreads();cluster.sync();','}__syncthreads();mark(p,0);cluster.sync();mark(p,1);')
    body=body.replace('gemm(p,sm,bar,row,rank*128);normalize','gemm(p,sm,bar,row,rank*128);mark(p,2);normalize')
    body=body.replace('__syncthreads();if(rank)cluster_wait(bar+4);__syncthreads();','__syncthreads();mark(p,3);if(rank)cluster_wait(bar+4);__syncthreads();mark(p,4);')
    body=body.replace(' fence_proxy_async();__syncthreads();\n if(tid==0){',' mark(p,5);fence_proxy_async();__syncthreads();\n if(tid==0){')
    body=body.replace('cluster_wait(bar+5);__syncthreads();','cluster_wait(bar+5);__syncthreads();mark(p,6);')
    body=body.replace('__syncthreads();transpose<true>(sm);','__syncthreads();mark(p,7);transpose<true>(sm);')
    body=body.replace(' cluster.sync();\n}', ' mark(p,8);cluster.sync();mark(p,9);\n}')
    body=body.replace('mw_cluster_bulk_dn_ln','mw_trace_cluster_bulk_dn_ln')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=512','-DEMIT_DN=0']
    cubin=T.compile_text(body,flags);unit=T.load_unit(str(cubin),'mw_trace_cluster_bulk_dn_ln')
    op.k=unit.kernel('mw_trace_cluster_bulk_dn_ln');op.finish=unit.kernel('mw_trace_cluster_bulk_dn_ln_finish');op.k.set_max_dynamic_smem(op.smem)
    clocks=torch.zeros((p.M//64,8,10),device=p.x.device,dtype=torch.int64)
    L=T._launch_module();op.params=L.Struct([*op.params.fields,clocks]);op.args=L._Packed([op.params])
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();op();b.record();b.synchronize()
    stamps=clocks.cpu();dur=stamps[:,:,1:]-stamps[:,:,:-1]
    record=dict(D=512,L=384,job=os.environ.get('SLURM_JOB_ID'),cubin=str(cubin),instrumented_us=a.elapsed_time(b)*1000,
                labels=['initial_sync','gemm','precompute','chain_wait','chain_update','final_wait','normalize','transpose_store_affine','final_sync'],
                cycles_mean=dur.double().mean(0).tolist(),cycles_median=dur.double().median(0).values.tolist(),
                first_wave=stamps[:30].tolist())
    out=THIS/f'trace-cluster-bulk-dn-ln-{record["job"]}.json';out.write_text(json.dumps(record,indent=2))
    print({k:v for k,v in record.items() if k not in ('first_wave','cycles_median')},flush=True)
