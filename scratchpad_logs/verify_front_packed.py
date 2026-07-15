"""Load real training batches and check whether structure.atom_mask is FRONT-PACKED
per row (valid atoms first, padding at the end) -- the precondition for the
seqused_k SWA attention. CPU-only; num_workers=0 so it doesn't fight the running job.
"""
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path("scripts").absolute()))
import run_miniworld_no_single_edm_train as T  # noqa: E402
from miniworld.data.dataloader.dataloader import BioMolData  # noqa: E402

CFG = "config_exp_msa3_24_3_no_single_ropeswa_af3_mpfull_b200_8gpu_edm_lr1e4"
CFG_DIR = str(Path("configs/miniworld").absolute())
N_BATCHES = 6

with initialize_config_dir(CFG_DIR, version_base=None):
    cfg = compose(config_name=CFG)
cfg = T.Config.model_validate(cfg)

ds = BioMolData(BioMolData.BioMolConfig(
    crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
    sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer,
))
loader = ds.create_ddp_dataloader(
    world_size=1, rank=0, seed=cfg.train.seed, drop_last=True,
    batch_size=cfg.train.num_batch, num_workers=0, prefetch_factor=None, shuffle=True,
    bucket_msa_multiple=cfg.train.bucket_msa_multiple,
    bucket_token_multiple=cfg.train.bucket_token_multiple,
    bucket_atom_multiple=cfg.train.bucket_atom_multiple,
)
loader.sampler.set_epoch(0)
ds.set_epoch(0)

all_ok = True
rows = 0
for i, batch in enumerate(loader):
    am = batch.structure.atom_mask  # [B, L_atom] bool
    B, S = am.shape
    cnt = am.sum(-1)  # [B]
    expected = torch.arange(S).unsqueeze(0) < cnt.unsqueeze(1)  # front-packed pattern
    ok_rows = (am == expected).all(-1)  # [B]
    rows += B
    n_bad = int((~ok_rows).sum())
    # interior-gap count: False positions that occur before the row's last True
    last_true = torch.where(am, torch.arange(S).unsqueeze(0), -1).max(-1).values  # [B]
    interior_gaps = ((~am) & (torch.arange(S).unsqueeze(0) <= last_true.unsqueeze(1))).sum(-1)
    print(f"batch {i}: B={B} L_atom={S}  valid/row={cnt.tolist()}  "
          f"front_packed_rows={int(ok_rows.sum())}/{B}  interior_gaps/row={interior_gaps.tolist()}")
    if n_bad:
        all_ok = False
        b0 = int(torch.where(~ok_rows)[0][0])
        bad = am[b0]
        gpos = torch.where((~bad) & (torch.arange(S) <= last_true[b0]))[0][:10]
        print(f"  !! row {b0} NOT front-packed; first interior-gap positions: {gpos.tolist()}")
    if i + 1 >= N_BATCHES:
        break

print(f"\n=== {rows} rows over {min(N_BATCHES, i+1)} batches: "
      f"{'ALL FRONT-PACKED ✓ (seqused_k valid)' if all_ok else 'HAS INTERIOR GAPS ✗ (seqused_k UNSAFE)'} ===")
