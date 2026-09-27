"""Per-CTA diagnostic cycle counts; whole-workload timings remain separate."""
from pathlib import Path
import sys,os,json,statistics
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from d256_full_width_prefix_dx import FullWidthPrefixDX
from d256_full_prefix_dx_ln import FullPrefixDxLN
from d256_saved_stats_prefix_dx_ln import SavedStatsPrefixDxLN
from d256_full_rows_prefix_dx_ln import FullRowsPrefixDxLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,*_=setup(256,384)
record=dict(job=os.environ.get('SLURM_JOB_ID'),diagnostic_only=True,complete=False,candidates=[])
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();expected=plan.p.dx.clone()
    for name,op in (
        ('bare',FullWidthPrefixDX(plan,4,64,0,'256B')),
        ('rows16',FullPrefixDxLN(plan,False,4,64,0)),
        ('rows16_stats',SavedStatsPrefixDxLN(plan,False,4,64,0)),
        ('rows32',FullRowsPrefixDxLN(plan,False,4,64,0,32)),
    ):
        clocks=torch.empty((op.grid,5),device='cuda',dtype=torch.int64)
        body=op.source_text
        marker='struct Params{';start=body.index(marker);end=body.index('};',start)
        body=body[:end]+'unsigned long long* clocks;'+body[end:]
        stamp=lambda i:f'if constexpr(WG==0)if(tid==0)p.clocks[blockIdx.x*5+{i}]=clock64();'
        marker=' float acc[128]={};';assert body.count(marker)==1;body=body.replace(marker,stamp(0)+'\n'+marker)
        marker=' named_bar_sync(3,256);';pos=body.index(marker);body=body[:pos]+stamp(1)+'\n'+body[pos:]
        marker=' fence_proxy_async();named_bar_sync(WG+1,128);';assert body.count(marker)==1;body=body.replace(marker,stamp(2)+'\n'+marker)
        marker='  tma_store_commit();tma_store_wait_all();';assert body.count(marker)==1;body=body.replace(marker,marker+stamp(3))
        old_name=('mw_d256_full_width_prefix_dx' if name=='bare' else 'mw_d256_saved_stats_prefix_dx_ln' if name.endswith('stats') else 'mw_d256_full_rows_prefix_dx_ln' if name=='rows32' else 'mw_d256_full_prefix_dx_ln')
        new_name='mw_profile_prefix_dx_ln';body=body.replace(old_name,new_name)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=4','-DDX_FULL_K=64','-DDX_FULL_DEPTH=0','-DEMIT_DXN=0']
        cubin=T.compile_text(body,flags);k=T.load_unit(str(cubin),new_name).kernel(new_name);k.set_max_dynamic_smem(op.smem)
        params=T._launch_module().Struct([*op.params.fields,clocks])
        for _ in range(3):k.launch((op.grid,1,1),(384,1,1),[params],op.smem)
        torch.cuda.synchronize();values=clocks.cpu().tolist()
        medians={label:statistics.median(row[b]-row[a] for row in values) for label,a,b in [('gemm',0,1),('epilogue',1,2),('store',2,3)]}
        errors=None
        if name!='bare':errors=error(plan.p.dx,expected)
        item=dict(name=name,median_cycles=medians,dx_error=errors,cubin=str(cubin))
        if name!='bare':
            kernel=lambda:op.k.launch((op.grid,1,1),(384,1,1),[op.params],op.smem)
            item['stage_times']=paired(dict(gemm_ln=kernel,complete_finish=op))
        print('PHASE',name,medians,'dx',errors,{k:v['median_us'] for k,v in item.get('stage_times',{}).items()},flush=True)
        record['candidates'].append(item)
record['complete']=True
(THIS/f'profile-prefix-dx-ln-phases-{record["job"]}.json').write_text(json.dumps(record,indent=2))
