import sys, collections
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path("scripts").resolve()))
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0); dev="cuda"
cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True); m["trunk"]["pairformer"]["n_block"]=48
model = MiniSWAModel(MiniSWAModel.Config(shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000), trunk=m["trunk"])).to(dev)
model.train(); model._forced_n_recycle=2
b=_build_precompile_batch(device=dev,msa_depth=16,n_tokens=64,n_atoms=256,n_templates=1,num_res_class=int(m["shared"]["num_res_class"]))
with torch.autocast("cuda",dtype=torch.bfloat16):
    lg=model.forward(msa=b.msa,reference=b.reference,scheme=b.scheme,sequence=b.sequence,structure=b.structure)
    loss=cal_atom_distogram_loss(lg,b.structure.atom_pos,b.structure.atom_pos_mask,b.scheme.atom_to_token_idx_map)
loss.backward()
none=collections.Counter(); has=collections.Counter()
for n,p in model.named_parameters():
    # collapse block index: pairformer_blocks.blocks.7.tri_multi.X -> ...tri_multi.X
    key=".".join(w for w in n.split(".") if not w.isdigit())
    (none if p.grad is None else has)[key]+=1
print("=== params with NO grad (name -> count) ===")
for k,v in none.most_common(): print(f"  {v:4d}  {k}")
print("=== sample full names (no grad) ===")
c=0
for n,p in model.named_parameters():
    if p.grad is None and "tri_multi" in n:
        print("   ", n, tuple(p.shape)); c+=1
        if c>=8: break
