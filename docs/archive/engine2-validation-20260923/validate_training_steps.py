"""Isolated profile of the running Phase1a snapshot; never writes training state."""
from pathlib import Path
import argparse, os, sys, json, time, statistics, hashlib, collections

P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--arm',choices=['baseline','engine2'],required=True)
ap.add_argument('--recycles',type=int,default=1)
ap.add_argument('--rounds',type=int,default=5)
ap.add_argument('--diagnostics',action='store_true')
ap.add_argument('--engine-root',type=Path)
ap.add_argument('--tag',default='')
ap.add_argument('--blocks',type=int,default=16)
args=ap.parse_args()
meta=json.loads((P/'source.json').read_text()); snapshot=Path(meta['snapshot'])
engine=args.engine_root or P/'engine2-cache'
sys.path[:0]=[str(engine),str(snapshot/'src'),str(snapshot/'libs/team-gm/src'),str(snapshot/'scripts')]
# Separate future process builds by arm to avoid rebuilding each other's extensions.
os.environ['TORCH_EXTENSIONS_DIR']=str(P/'extension-cache'/args.arm)
os.environ['WANDB_MODE']='disabled'
os.environ['MINIWORLD_PWA_TRAIN']='0' if args.arm=='baseline' else '1';os.environ['MINIWORLD_OPM_TRAIN']=os.environ['MINIWORLD_PWA_TRAIN']
import torch, numpy as np
from omegaconf import OmegaConf
from run_miniworld_distogram_train import Config
from miniworld.models.distogram_only import MiniSWAModel
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.training.engine_backend import configure_engine_backend
import miniworld_engine

def log(x):print(time.strftime('%H:%M:%S'),x,flush=True)

torch.manual_seed(17);np.random.seed(17)
cfg=Config.model_validate(OmegaConf.to_container(OmegaConf.load(P/'config.yaml'),resolve=True))
cfg.model.trunk.pairformer.n_block=args.blocks
backend='triton' if args.arm=='baseline' else 'auto'
configure_engine_backend(backend)
log({'arm':args.arm,'engine':miniworld_engine.__file__,'backend':backend,'recycles':args.recycles})
batch_path=P.parent/'phase1_v120_speed/real_batches_phase1a_distogram_v120.pt'
# The frozen real v1.2 sample pool is read-only and shared by both arms.
batches=torch.load(batch_path,map_location='cpu',weights_only=False)
batch=batches[0].to(device='cuda')
del batches
model=MiniSWAModel(cfg.model).cuda().train()
ckpt=torch.load(P/'checkpoint.pt',map_location='cpu',weights_only=False)
log({'checkpoint_epoch':ckpt['epoch'],'checkpoint_keys':list(ckpt),'first_parameter':next(iter(ckpt['model_state_dict']))})
state=ckpt['model_state_dict']
if args.blocks != 16:
 import re
 state={k:state[re.sub(r'^(pairformer_blocks.blocks\.)(\d+)(\.)',lambda m:m[1]+str(int(m[2])%16)+m[3],k)] for k in model.state_dict()}
model.load_state_dict(state,strict=True)
del state
model.set_seed(17);model._forced_n_recycle=args.recycles
optimizer=torch.optim.Adam(model.parameters(),lr=cfg.train.max_lr,betas=(0.9,0.95))
# Published run checkpoints may be model-only. Profile fresh optimizer buffers if absent.
optimizer_restored=False
if 'optimizer_state_dict' in ckpt and args.blocks==16:
 optimizer.load_state_dict(ckpt['optimizer_state_dict']);optimizer_restored=True
 if args.arm=='engine2':
  from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_
  log({'optimizer_tensors_realigned':align_optimizer_state_layout_(optimizer)})
del ckpt
shape={'tokens':int(batch.token_length),'atoms':int(batch.atom_length),'msa_pool':int(batch.msa_depth),'sample':batch.name}
log(shape)
assert batch.msa_depth==8192,shape
assert batch.structure.token_mask.shape[-1]==384,shape
kwargs=dict(msa=batch.msa,reference=batch.reference,scheme=batch.scheme,sequence=batch.sequence,structure=batch.structure,template=batch.template)

def loss_fn(y):
 return cfg.loss.distogram_loss*cal_atom_distogram_loss(y,batch.structure.atom_pos,batch.structure.atom_pos_mask,batch.scheme.atom_to_token_idx_map,rep_atom_mask=batch.structure.atom_is_rep,token_asym_id=batch.scheme.token_asym_id,interchain_weight=cfg.loss.distogram_interchain_weight)


# Compare arithmetic with dropout disabled to avoid comparing different random masks.
# Dropout ON is separately verified by module parity and full-model graph tests.
for m in model.modules():
 if hasattr(m,'p_drop'):m.p_drop=0.0
 if isinstance(m,torch.nn.Dropout):m.p=0.0
model.compile(dynamic=False)
records=[]
for step in range(3):
 model.set_seed(1700+step);torch.manual_seed(1700+step);np.random.seed(1700+step)
 optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):y=model(**kwargs);loss=loss_fn(y)
 (loss/cfg.train.grad_accum_steps).backward()
 gradients={n:v.grad.detach().cpu().float() for n,v in model.named_parameters() if v.grad is not None}
 assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients.values())
 norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip_max_norm))
 before={n:v.detach().clone() for n,v in model.named_parameters()}
 optimizer.step()
 updates={n:(v.detach()-before[n]).cpu().float() for n,v in model.named_parameters()}
 record=dict(step=step,loss=float(loss),grad_norm=norm,gradient_tensors=len(gradients))
 records.append(record);log(record)
 torch.save(dict(record=record,gradients=gradients,updates=updates,logits=y.detach().cpu().float()),P/f'accuracy-pf{args.blocks}-r{args.recycles}-{args.arm}-step{step}.pt')
 del before,gradients,updates,y,loss
(P/f'accuracy-pf{args.blocks}-r{args.recycles}-{args.arm}.json').write_text(json.dumps(dict(records=records,dropout=False,optimizer_restored=optimizer_restored),indent=2))
