from pathlib import Path
import json,os,sys,torch,hashlib
from hydra import compose,initialize_config_dir
from run_miniworld_distogram_train import Config
from omegaconf import OmegaConf
import miniworld_engine
R=Path(__file__).resolve().parent;L=R.parent
meta=json.loads((R/'provenance.json').read_text())
assert str(R/'engine') in miniworld_engine.__file__,miniworld_engine.__file__
assert (L/'training/wandb_run_id.txt').read_text().strip()==meta['wandb_id']=='tapgki9e'
with initialize_config_dir(str(R/'configs/miniworld'),version_base=None):
 cfg=Config.model_validate(compose(config_name='phase1b_distogram_medium_v120.yaml',overrides=['train.engine_backend=auto','+train.force_trainer=fabric','train.compile=true','train.grad_accum_steps=32',f'train.run_dir={L}/training','data.train_db.catalog_cache_path=/home/psk6950/MiniWorld/runs/v1.1.0/phase1a/launch_20260916_233206/catalog.arrow']))
old=Config.model_validate(OmegaConf.to_container(OmegaConf.load(Path(meta['run_subdir'])/'config.yaml'),resolve=True))
a=old.model_dump(mode='json');b=cfg.model_dump(mode='json')
def diff(a,b,path=''):
 out=[]
 for k in a.keys()|b.keys():
  key=f'{path}.{k}' if path else k
  if isinstance(a.get(k),dict) and isinstance(b.get(k),dict):out+=diff(a[k],b[k],key)
  elif a.get(k)!=b.get(k):out.append(dict(key=key,old=a.get(k),new=b.get(k)))
 return out
changes=diff(a,b);assert cfg.model.trunk.pairformer.n_block==16
assert cfg.model.trunk.pairformer.n_checkpoint_segments==16
assert cfg.train.num_epoch==1000
assert cfg.data.crop.max_tokens==768 and cfg.data.crop.max_atoms==8192
assert cfg.train.grad_accum_steps*8==256
(R/'phase1b-resolved.json').write_text(json.dumps(b,indent=2))
source=Path(meta['checkpoint'])
c=torch.load(source,map_location='cpu',weights_only=False)
for key in ['model_state_dict','optimizer_state_dict','scheduler_state_dict','ema_state_dict','epoch','global_step']:assert key in c,key
assert c['optimizer_state_dict']['state'] and c['ema_state_dict']
assert c['global_step']==c['epoch']*100,(c['epoch'],c['global_step'])
assert all(torch.isfinite(v).all() for v in c['model_state_dict'].values() if torch.is_floating_point(v))
if '--prepare' not in sys.argv:
 assert torch.cuda.device_count()==8
 assert c['epoch']==800 and c['global_step']==80000,(c['epoch'],c['global_step'])
print(json.dumps(dict(preflight='PASS',engine=miniworld_engine.__file__,epoch=c['epoch'],step=c['global_step'],optimizer_states=len(c['optimizer_state_dict']['state']),ema_tensors=len(c['ema_state_dict']),config_changes=changes,wandb_id=meta['wandb_id']),indent=2),flush=True)
