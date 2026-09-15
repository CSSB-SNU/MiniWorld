"""Does an antibody-antigen item actually put the Ab-Ag INTERFACE in the crop?

An `antibody_protein` row names (antibody chain, antigen chain), but that pair only
biases the crop -- it does not guarantee the interface survives it:

* `Preprocessor._select_focus_atoms` takes the real interface only with probability
  ``1 - crop_config.chain_crop_prob``. The other half of the time it picks ONE of the two
  chains and draws the focus atom uniformly from ANYWHERE in it -- an antibody's constant
  domain is as likely as its CDR.
* `crop_spatial_segment_token` then grows outward from that focus until the token/atom
  budget runs out. The full assembly is loaded, so the partner antibody chain (H next to L)
  is right there, while the antigen may sit outside the ball.

So the crop can legitimately contain only an Ab-Ab interface. This measures how often,
by drawing `antibody_protein` items through the REAL sampler and crop pipeline and
testing, inside each crop, which inter-entity contacts actually exist.

CPU only, but heavy (each item reads a CIF and its MSA): run it as a batch job.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_diffusion_train import Config  # noqa: E402

from miniworld.data.dataloader.dataloader import BioMolData  # noqa: E402

ANTIBODY, PROTEIN = 0, 1
CONTACT_A = 5.0  # AF3's interface definition: min heavy-atom separation < 5 A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=cfg_path.name))

    # Draw ONLY antibody-antigen items: every other pdb bucket to zero, and the
    # distillation sources off, so the sampler can return nothing else.
    s = cfg.data.sampler
    for field in type(s).model_fields:
        setattr(s, field, 0.0)
    s.antibody_protein = 1.0
    cfg.data.train_db.source_weights = {"pdb": 1.0}

    print(f"config       : {cfg_path.name}")
    print(f"crop         : {cfg.data.crop.max_tokens} tokens / {cfg.data.crop.max_atoms} atoms")
    print(f"chain_crop_prob: {cfg.data.crop.chain_crop_prob}  "
          f"(prob the focus IGNORES the interface and sits anywhere in one chain)")

    data = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db, sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer))
    loader = data.create_ddp_dataloader(
        world_size=1, rank=0, seed=0, drop_last=False, batch_size=1,
        num_workers=args.workers, num_samples_per_rank=args.n, shuffle=True,
        bucket_msa_multiple=None, bucket_token_multiple=None,
        bucket_atom_multiple=None, bucket_template_multiple=4)
    data.set_epoch(0)

    def contact(pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
        if len(pos_a) == 0 or len(pos_b) == 0:
            return False
        return len(cKDTree(pos_a).query_ball_point(pos_b, CONTACT_A, return_sorted=False)) > 0 \
            and any(len(x) for x in cKDTree(pos_a).query_ball_point(pos_b, CONTACT_A))

    tally = collections.Counter()
    tok = collections.Counter()
    n = 0
    for batch in loader:
        et = batch.chain.entity_type[0]                  # [L_chain]
        asym = batch.scheme.token_asym_id[0]             # [L_token] -> chain idx
        tmask = batch.structure.token_mask[0].bool()
        a2t = batch.scheme.atom_to_token_idx_map[0]      # [L_atom] -> token
        apos = batch.structure.atom_pos[0].numpy()
        amask = batch.structure.atom_pos_mask[0].bool().numpy()

        tok_ent = et[asym]                               # [L_token]
        atom_ent = tok_ent[a2t].numpy()
        atom_chain = asym[a2t].numpy()
        valid = amask & np.isfinite(apos).all(-1)

        ab = valid & (atom_ent == ANTIBODY)
        ag = valid & (atom_ent == PROTEIN)
        n += 1
        tok[int(tmask.sum())] += 0
        has_ab, has_ag = bool(ab.any()), bool(ag.any())
        tally["has_antibody"] += has_ab
        tally["has_antigen"] += has_ag

        ab_ag = has_ab and has_ag and contact(apos[ab], apos[ag])
        tally["Ab-Ag contact in crop"] += ab_ag

        # Ab-Ab: two DIFFERENT antibody chains touching (H-L, or a second Fab)
        ab_chains = np.unique(atom_chain[ab])
        ab_ab = False
        for i in range(len(ab_chains)):
            for j in range(i + 1, len(ab_chains)):
                m1 = ab & (atom_chain == ab_chains[i])
                m2 = ab & (atom_chain == ab_chains[j])
                if contact(apos[m1], apos[m2]):
                    ab_ab = True
                    break
            if ab_ab:
                break
        tally["Ab-Ab contact in crop"] += ab_ab
        tally["Ab-Ab ONLY (no Ab-Ag)"] += (ab_ab and not ab_ag)
        tally["NO interface at all"] += (not ab_ab and not ab_ag)
        tally["antigen tokens"] += int((tok_ent[tmask] == PROTEIN).sum())
        tally["antibody tokens"] += int((tok_ent[tmask] == ANTIBODY).sum())
        tally["tokens"] += int(tmask.sum())
        if n >= args.n:
            break

    print(f"\nantibody_protein items drawn: {n}\n")
    print(f"{'outcome':<26}{'items':>8}{'% of items':>13}")
    print("-" * 47)
    for k in ("has_antibody", "has_antigen", "Ab-Ag contact in crop",
              "Ab-Ab contact in crop", "Ab-Ab ONLY (no Ab-Ag)", "NO interface at all"):
        print(f"{k:<26}{tally[k]:>8}{100 * tally[k] / n:>12.1f}%")
    t = tally["tokens"]
    print(f"\ntoken composition: antibody {100*tally['antibody tokens']/t:.1f}%  "
          f"antigen {100*tally['antigen tokens']/t:.1f}%  "
          f"other {100*(t-tally['antibody tokens']-tally['antigen tokens'])/t:.1f}%")


if __name__ == "__main__":
    main()
