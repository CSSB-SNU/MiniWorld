from pathlib import Path
import sys,os,json,statistics,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
def select(cuda):
 os.environ['FPF_TRIATT_BACKEND']='cuda_tma' if cuda else 'triton'

def rms(t,r):return float((t.float()-r.float()).square().mean().sqrt()/r.float().square().mean().sqrt().clamp_min(1e-30))

results={n:{'rounds_us':[],'pro_rounds':[],'epi_rounds':[]} for n in ['baseline','cuda']};checks=[]
with torch.no_grad():
 assert float((ref-x.float()).norm())>0
 select(False);expected=call().clone();baseline_err=rms(expected,ref)
 qp,kp,vp,gp,bp=pf.prologue(x,W,impl='fpf',ending=ending)
 op=pf.core_attention(qp,kp,vp,bp,m5,core='tier:triattn_native')
 def pro():return pf.prologue(x,W,impl='fpf',ending=ending)
 def epi():return pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf,ending=ending)
 e0=epi().clone()
 for name in results:
  select(name=='cuda');y=call().clone();err=rms(y,ref)
  assert torch.isfinite(y).all() and err<=baseline_err*1.01+1e-8,(name,err,baseline_err)
  e1=epi().clone();assert torch.equal(e0,e1),(ending,a.length,'epilogue mismatch',float((e0.float()-e1.float()).abs().max()))
  kernels=cupti(call,10)
  assert not any('flash_triattn' in k for k in kernels),kernels
  assert any(('triattn_m1_kernel' in k if a.length>=512 else k=='_fwd') for k in kernels),kernels
  if name=='cuda':
   assert any('triattn_tma_pro' in k for k in kernels) and any('triattn_tma_epi' in k for k in kernels),kernels
  results[name].update(rel_rms=err,baseline_delta_rms=rms(y,expected),bitwise_equal=torch.equal(y,expected),epi_equal=True,kernels=kernels)
 for round_ in range(3):
  for name in (['baseline','cuda'] if round_%2==0 else ['cuda','baseline']):
   select(name=='cuda')
   for key,fn in [('rounds_us',call),('pro_rounds',pro),('epi_rounds',epi)]:results[name][key].append(graph_us(fn))
   print('ROUND',ending,a.length,round_,name,results[name]['rounds_us'][-1],flush=True)
 for seed,mode in [(881,'dense'),(882,'empty'),(883,'irregular')]:
  torch.manual_seed(seed);x.normal_()
  if mode=='dense':mask.fill_(True)
  elif mode=='empty':mask.fill_(False)
  else:mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.3)
  select(False);y0=call().clone();p0=pro()
  select(True);y1=call().clone();p1=pro()
  assert torch.isfinite(y1).all() and rms(y1,y0)<1e-4,(ending,seed,mode,rms(y1,y0))
  pieces={n:rms(t,r) for n,t,r in zip(['q','k','v','g','bias'],p1,p0)}
  assert max(pieces.values())<1e-4,pieces
  # In-place residual is a separate aliasing path.
  zcopy=x.clone();ein=pf.epilogue(op,gp,W,zcopy,impl='fpf',ending=ending,residual=True)
  assert ein.data_ptr()==zcopy.data_ptr()
  select(False);eexpected=pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf,ending=ending).clone()
  assert torch.equal(zcopy,eexpected)
  checks.append(dict(seed=seed,mask=mode,delta_rms=rms(y1,y0),pieces=pieces,inplace_equal=True))
 # Exercise every finite BF16 gate value, independently of the random block inputs.
 if a.length==384:
  bits=torch.arange(gp.numel(),device='cuda',dtype=torch.int32).bitwise_and(65535)
  bits.masked_fill_(bits.bitwise_and(0x7f80)==0x7f80,0)
  all_gates=bits.to(torch.int16).view(torch.bfloat16).reshape_as(gp)
  select(False);eg0=pf.epilogue(op,all_gates,W,x,impl='fpf',ending=ending,residual=True,out=buf).clone()
  select(True);eg1=pf.epilogue(op,all_gates,W,x,impl='fpf',ending=ending,residual=True,out=buf).clone()
  assert torch.equal(eg0,eg1),('all finite BF16 gates',ending)
  checks.append(dict(all_finite_bf16_gate_patterns=True,epilogue_equal=True))
 # The installed default must select CUDA without an environment override.
 os.environ.pop('FPF_TRIATT_BACKEND',None)
 default_kernels=cupti(call,3)
 assert any('triattn_tma_pro' in k for k in default_kernels) and any('triattn_tma_epi' in k for k in default_kernels),default_kernels
 for r in results.values():
  r.update(block_us=statistics.median(r['rounds_us']),pro_us=statistics.median(r['pro_rounds']),epi_us=statistics.median(r['epi_rounds']))
so=list((R/'oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt').glob('torch2.10*/triattn_m1_ext.so'))[0]
res=dict(length=a.length,ending=ending,results=results,checks=checks,core_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),fallback=False)
Path(a.output).write_text(json.dumps(res,indent=2));print('RESULT',json.dumps(res),flush=True)
