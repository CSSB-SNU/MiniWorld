"""Component timings for the qualified combined wide path; no dispatch change."""
from pathlib import Path
import sys,os,json,ctypes
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,diagnostic_only=True)
path=THIS/f'components-wide-combined-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);plan();p=plan.p;b=plan.b1
 def ln():
  L=T._launch_module();drv=b.ln.unit.drv;args=L._Packed([b.lp])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(b.ln.handle)),b.grid,1,1,b.threads,1,1,b.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
 def forward_contract():
  ab=plan.f.front.ab;d=p.D;h=2*d
  torch.bmm(ab[:d],ab[h:h+d].transpose(-1,-2),out=plan.f.tri[:d])
  torch.bmm(ab[d:h].transpose(-1,-2),ab[h+d:],out=plan.f.tri[d:])
 functions=dict(front=plan.f.front,forward_contract=forward_contract,output=plan.f.output,
  b1=b,proj=lambda:torch.mm(p.tensors[6],b.wp.t(),out=b.proj),
  gate=lambda:torch.mm(p.xn.reshape(p.M,p.D),b.wg.t(),out=b.gate),
  epi=lambda:b.epi.launch((1056,1,1),(256,1,1),[b.ep],0),
  dn=lambda:torch.mm(p.tensors[7],b.wp,out=p.tensors[9]),
  dwp=b.exact_dwp if b.exact_dwp is not None else lambda:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp),
  dwg=lambda:torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg),
  output_ln=ln,backward_contract=plan.contract,source=plan.b7.source_only,
  prefix_copy=plan.dx.copy_prefix,weight_pack=plan.dx.pack_weights,
  dx_gemm=plan.dx.gemm_only,input_ln_reduce=plan.dx.reduce_only)
 if D==512:functions['norm']=lambda:b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem)
 record['times']=paired(functions)
 print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
