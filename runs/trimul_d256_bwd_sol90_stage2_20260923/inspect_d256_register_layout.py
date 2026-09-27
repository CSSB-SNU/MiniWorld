from pathlib import Path
import os,sys,json,subprocess,struct,re
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_spatial_checkpoint import Training
from d256_register_layout_source import RegisterLayoutSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,*_=setup(256,384)
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy)
 op=RegisterLayoutSource(plan,True)
 cubin=Path(op.cubin);job=os.environ['SLURM_JOB_ID']
 sass=subprocess.check_output([os.environ['CUDA_HOME']+'/bin/cuobjdump','--dump-sass',str(cubin)],text=True)
 out=THIS/f'register-layout-{job}.sass';out.write_text(sass)
 data=cubin.read_bytes();header=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
 shoff,entsize,n,strings=header[6],header[11],header[12],header[13]
 sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*entsize) for i in range(n)]
 st=sections[strings];names=data[st[4]:st[4]+st[5]]
 meta=[]
 for i,s in enumerate(sections):
  name=names[s[0]:].split(b'\0',1)[0].decode()
  if name.startswith(('.text.','.nv.info')):
   meta.append(dict(index=i,name=name,offset=s[4],size=s[5],info=s[7],hex=data[s[4]:s[4]+s[5]].hex() if name.startswith('.nv.info') else None))
 rec=dict(cubin=str(cubin),sass=str(out),sections=meta,resources={k:getattr(op,k,None) for k in ('registers','local_bytes','occupancy')})
 (THIS/f'register-layout-{job}.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
 print('ENTRY', '\n'.join(sass.splitlines()[:85]))
 print('SETMAX', '\n'.join(x for x in sass.splitlines() if 'SETMAX' in x))
