"""Qualify every declared F567 config and record source-sensitive graph timings."""
import argparse,json,time
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.autotune import native,native_compile,capture,cache
from miniworld_engine.autotune.cute_config import f567_candidates
from miniworld_engine.kernels.trimul_inproj.cute.output_f567 import _launch

p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--compile-jobs',type=int,default=8);a=p.parse_args()
guard=a.output.parent/"tasks/15_final_tests.done"
if guard.exists():assert json.loads(guard.read_text())["exit_code"]==0,"correctness test failed"
settings.configure(compile_jobs=a.compile_jobs)
L=a.length;m=L*L;n=128;kp=256;kg=128
kw=dict(device='cuda',dtype=torch.bfloat16);torch.manual_seed(567)
x=torch.randn(m,kg,**kw);norm=torch.randn(m,kp,**kw)
wp=torch.randn(n,kp,**kw)/kp**.5;wg=torch.randn(n,kg,**kw)/kg**.5
res=torch.randn(m,n,**kw);ds=(torch.rand(L,n,device='cuda')>.25).to(torch.bfloat16)/.75
bias=torch.randn(n,**kw)*.1;outs=tuple(torch.empty_like(res) for _ in range(3))
limit=torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
configs=f567_candidates(kp,kg,limit);op='trimul_output_f567_sm90_cute'
bucket=native.tensor_key(norm,x,wp,wg,res,ds,bias,extra=(L,limit));identity=native.source_identity()
assert native.candidates_for(op,bucket)==configs
result=dict(length=L,source_identity=identity,bucket=bucket,configs=[],grid=len(configs),complete=False)
def save():a.output.write_text(json.dumps(result,indent=2))
save()
# Native isolated compiler contract uses exact extents/strides; CUDA stays hidden.
compiled=native_compile.precompile(op,configs,bucket)
result['precompile']=compiled;save()
pr=(norm.float()@wp.float().t()+bias.float()).bfloat16().float()
ga=torch.sigmoid((x.float()@wg.float().t()).bfloat16().float())
refs=(res.float()+pr*ga*ds.float()[torch.arange(m,device=x.device)%L],pr,ga)
for i,c in enumerate(configs):
 start=time.time();fn=lambda:_launch(norm,x,wp,wg,res,ds,L,bias,outs,c)
 row=dict(index=i,config=c)
 try:
  assert not compiled or compiled[i]['status']=='ok',compiled.get(i)
  for t in outs:t.fill_(float('nan'))
  fn();torch.cuda.synchronize()
  errs=[float((v.float()-r).norm()/r.norm().clamp_min(1e-8)) for v,r in zip(outs,refs)]
  assert all(torch.isfinite(v).all() for v in outs) and max(errs)<.004,errs
  row.update(ms=triton.testing.do_bench_cudagraph(fn,rep=10),relative_l2=errs)
 except Exception as e:
  row['error']=str(e)
  if 'illegal' in str(e).lower() or 'misaligned' in str(e).lower():
   result['configs'].append(row);save();raise
 row['seconds']=time.time()-start;result['configs'].append(row);save()
 if i%12==0:print('MEASURED',i+1,'/',len(configs),flush=True)
assert all('ms' in r for r in result['configs']), 'failed configs: inspect report; no cache published'
measurement=dict(scheme=1,kind='native',implementation=identity,timing='cuda_graph',bench_clear_mb=0,bench_rep_ms=10,harness='trimul-f567-cute-v1')
grid=[{'kwargs':c} for c in configs];capture.reset()
for row in result['configs']:
 capture.record_native(op,grid,'bfloat16',bucket,row['config'],row['ms'],identity,measurement=measurement)
shard=a.output.with_suffix('.shard.json');capture.dump_shard(str(shard),unit_complete=True)
merged=capture.merge_shards([str(shard)],gpu=cache.gpu_key(),only_ops={op})
assert merged and not capture._MERGE_SKIPPED,(merged,capture._MERGE_SKIPPED)
result.update(complete=True,best=min(result['configs'],key=lambda r:r['ms']),published=merged);save();print('BEST',result['best'],flush=True)
