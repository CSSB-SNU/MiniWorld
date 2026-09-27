"""No full-workload claim: verify and time the direct compact GP producer."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint23 import Training
from wide_tiled_compact_mask_gp import CompactPrefixMap as CompactMaskMap,TiledCompactMaskGP as CompactMaskGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=(384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),producer_only=True,complete=False)
path=THIS/f'result-wide-tiled-compact-mask-gp-D{D}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;plan();expected=p.gp_all.clone();old=plan.contract_gp
    mapping=CompactMaskMap(plan);mapping();torch.cuda.synchronize()
    op=CompactMaskGP(plan,mapping)
    record.update(registers=op.registers,local_bytes=op.local_bytes,occupancy=op.occupancy,cubin=str(op.cubin),metadata=mapping.metadata.tolist())
    p.gp_all.fill_(float('nan'));mapping(pad=True);op();torch.cuda.synchronize()
    restored=p.gp_all.reshape(8*D,p.M).index_select(1,mapping.rows[:-1].clamp_max(p.M-1).long())
    if record["metadata"][1]:restored[:,plan.f.mask.reshape(-1)==0]=0
    record['gp_error']=error(restored,expected.reshape_as(restored));record['bitwise']=torch.equal(restored,expected.reshape_as(restored))
    record['padding_zero']=bool((p.gp_all.reshape(8*D,p.M)[:,record['metadata'][0]:mapping.capacity]==0).all().item())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['bitwise'] and record['padding_zero'] and not op.local_bytes:
        def combined():mapping(pad=True);op()
        record['stage_times']=paired(dict(old_gp=old,new_gp=op,new_map_gp=combined));gc.collect()
        print('TIMES',{k:v['median_us'] for k,v in record['stage_times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
