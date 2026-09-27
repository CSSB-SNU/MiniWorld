"""Isolated lossless consumer prototype; forward encoder is not integrated."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from wide_bf12_contract_gp import PackedPre,PackedContractGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),complete=False,forward_integrated=False)
path=THIS/f'result-bf12-contract-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    # Every BF16 bit pattern, including signed zero and exponent exceptions.
    basis=torch.arange(65536,device='cuda',dtype=torch.int32)
    tiny=PackedPre(basis.to(torch.int16).view(torch.bfloat16).reshape(1,-1));tiny()
    mb=tiny.mant.reshape(-1).int();eb=tiny.exps.reshape(-1).int()
    codes=torch.stack((eb&15,eb>>4),dim=1).reshape(-1)
    restored=(mb&127)|((mb&128)<<8)|(torch.where(codes==0,0,codes+115)<<7)
    restored=torch.where(codes==15,basis,restored)
    record['all_bf16_patterns']=torch.equal(restored,basis)
    assert record['all_bf16_patterns']
    del basis,tiny,mb,eb,codes,restored
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    packed=PackedPre(plan.pre);packed();op=PackedContractGP(plan,packed);original=plan.contract_gp
    record.update(escapes=int(packed.escapes.item()),values=plan.pre.numel(),compressed_bytes=packed.mant.numel()+packed.exps.numel(),cubin=str(op.cubin))
    # Retain a complete GP comparison only when device memory has sufficient headroom.
    wanted=p.gp_all.clone() if torch.cuda.mem_get_info()[0]>p.gp_all.numel()*2+6*1024**3 else None
    op();torch.cuda.synchronize()
    if wanted is not None:
        record['gp_bitwise']=torch.equal(p.gp_all,wanted)
        del wanted;gc.collect()
    def old():plan.contract_gp=original;return plan.backward()
    def new():plan.contract_gp=op;return plan.backward()
    gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[y,*gn],expected)}
    record['strict']=record.get('gp_bitwise',True) and all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        record['stage_times']=paired(dict(old_gp=original,new_gp=op))
        record['backward_times']=paired(dict(old_bwd=old,new_bwd=new))
        print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
