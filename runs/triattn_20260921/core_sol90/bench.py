from pathlib import Path
import importlib.util,sys,os,json,statistics
R=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}').replace('e.name[:40]','e.name')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
mode=os.environ.get('SOL_VARIANT','r6n32')
if mode in ('m256ss2pref','m256qr2pref','persistkv2','persistkv2b4','persistkv2gb','persistkv2cp','m128s2p3qrn64'):
 import subprocess
 subprocess.run([sys.executable,str(HERE/'screen_m256_build.py'),str(HERE/('build-'+mode+'.log'))],check=True)
if mode.startswith('m128s2p3') and mode!='m128s2p3qrn64':
 import subprocess
 subprocess.run([sys.executable,str(HERE/'screen_r3_build.py'),str(HERE/('build-'+mode+'.log'))],check=True)
if mode=='m64s2p3qrr4':
 import subprocess
 subprocess.run([sys.executable,str(HERE/'screen_r4_build.py'),str(HERE/('build-'+mode+'.log'))],check=True)
name='triattn_sol_'+mode
direct=mode.startswith(('direct','base'))
spec=importlib.util.spec_from_file_location(name,(HERE/'build'/name if direct else HERE/('build_'+mode))/(name+'.so'))
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
if direct:
 spec=importlib.util.spec_from_file_location('direct_wrapper',HERE/mode/'triattn_m1.py')
 direct_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct_module)
 direct_module._EXT=ext
def candidate(q,k,v,b,mask,scale):
 if direct:return direct_module.triangle_attention_m1(q,k,v,b.reshape(1,1,4,a.length,a.length),mask=mask,scale=scale).reshape_as(q)
 return ext.forward(q,k,v,b.reshape(1,1,4,a.length,a.length),mask,scale).reshape_as(q)
variants={'serving':'tier:triattn_native',mode:candidate}
def rms(t,r):return float((t.double()-r.double()).square().mean().sqrt())
results={n:dict(core_rounds=[],block_rounds=[],hot_rounds=[]) for n in variants}
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 KW['core']=variants['serving'];expected=call().clone();expected_c=core_call().clone()
 qq=q.reshape(1,a.length,4,a.length,32)[:,:1].double();kk=k.reshape_as(q).reshape(1,a.length,4,a.length,32)[:,:1].double();vv=v.reshape_as(q).reshape(1,a.length,4,a.length,32)[:,:1].double()
 logits=qq@kk.transpose(-1,-2)/32**.5+b.reshape(1,1,4,a.length,a.length).double()
 logits.masked_fill_(~m5[:,:1],-torch.inf);reference=logits.softmax(-1)@vv
 base_err=rms(expected_c.reshape(1,a.length,4,a.length,32)[:,:1],reference)
 for n,fn in variants.items():
  print('CHECK',n,flush=True);KW['core']=fn
  y=call().clone()
  count_fixes=n!='serving' and direct and hasattr(direct_module,'FALLBACKS')
  fixes_before=direct_module.FALLBACKS['fix_tiles'] if count_fixes else 0
  c=core_call().clone().reshape_as(expected_c);torch.cuda.synchronize()
  if count_fixes:
   results[n]['core_fix_tiles']=direct_module.FALLBACKS['fix_tiles']-fixes_before
   print('FIX_TILES',n,results[n]['core_fix_tiles'],flush=True)
  err=rms(c.reshape(1,a.length,4,a.length,32)[:,:1],reference)
  assert torch.isfinite(c).all() and err<=base_err*1.05+1e-8,(n,err,base_err)
  kernels=cupti(call,5)
  expected_kernel='ta_core_broadcast::triattn_m1_kernel' if n=='serving' else 'ta_sol_'+mode+'::attention<false>'
  if n!='serving' and mode in ('split4','split4f'):expected_kernel='ta_sol_'+mode+'::specialized::attention'
  if n!='serving' and direct:expected_kernel='ta_sol_'+mode+'::triattn_m1_kernel'
  assert any(expected_kernel in key for key in kernels) and not any('flash_triattn' in key for key in kernels),kernels
  results[n].update(core_rms_fp64=err,baseline_core_rms_fp64=base_err,core_delta_rms=rms(c,expected_c),block_delta_rms=rms(y,expected),core_equal=torch.equal(c,expected_c),kernels=kernels)
 for rd in range(3):
  for n in (list(variants) if rd%2==0 else list(reversed(variants))):
   KW['core']=variants[n]
   results[n]['core_rounds'].append(graph_us(core_call));results[n]['block_rounds'].append(graph_us(call))
   print('ROUND',rd,n,results[n]['core_rounds'][-1],results[n]['block_rounds'][-1],flush=True)
 for d in results.values():
  d.update(core_us=statistics.median(d['core_rounds']),block_us=statistics.median(d['block_rounds']))
 out=dict(length=a.length,ending=ending,results=results)
 Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
