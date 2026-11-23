"""Inference-side x0-hat trajectory: run_miniworld_diffusion_inference.py (foldbench) with Client.inference wrapped so that, for the first sample, the
x0-hat of EVERY solver step (and the noisy iterate) is saved and scored for bond geometry. Same settings as the normal FoldBench run otherwise.
env X0HAT_OUT = output dir. One npz per target: sigma[step], metrics of x0-hat and of the iterate per step, the bond list, the x0-hat trajectory."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(Path(__file__).resolve().parent)]
import run_miniworld_diffusion_inference as inf  # noqa: E402
from x0hat_common import bond_metrics, ideal_bonds  # noqa: E402

OUT = Path(os.environ["X0HAT_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ("mean_abs_dev", "frac_gt_0.1", "frac_gt_0.3", "max_dev", "n_clash")
_Client = inf.Client


class Client(_Client):
    def inference(self, batch, *a, **kw):
        out = super().inference(batch, *a, **kw)
        steps_arg = kw.get("timesteps", a[0] if a else 100)
        name = str(batch.name[0])
        if (OUT / f"{name}.npz").exists():
            return out
        dev = batch.structure.atom_pos.device
        ref = batch.reference
        valid = (batch.structure.atom_mask[0] > 0) & (ref.mask[0] > 0)
        uid = ref.space_uid[0].long()
        i, j, ideal = ideal_bonds(ref.pos[0], uid, valid)
        times = self.solver.scheduler.sampling_time_steps(steps_arg)
        sigmas = np.array([float(self.solver.scheduler.sampling_schedule(t)) for t in times.cpu()])
        x0hat = torch.from_numpy(np.asarray(out.model_traj))[0].to(dev)  # [steps, L, 3]
        itr = torch.from_numpy(np.asarray(out.inter_traj))[0].to(dev)
        res = {"x0hat": [], "iter": []}
        for s in range(x0hat.shape[0]):
            res["x0hat"].append(bond_metrics(x0hat[s], i, j, ideal, uid, valid))
            res["iter"].append(bond_metrics(itr[s], i, j, ideal, uid, valid))
        arrs = {f"{w}_{k}": np.array([r[k] for r in res[w]]) for w in res for k in KEYS}
        np.savez_compressed(
            OUT / f"{name}.npz", sigma=sigmas[: x0hat.shape[0]], n_bonds=len(i), bond_i=i.cpu().numpy(), bond_j=j.cpu().numpy(), ideal=ideal.cpu().numpy(),
            x0hat=x0hat.cpu().numpy().astype(np.float32), **arrs,
        )
        print(f"\n=== x0-hat bond trajectory: {name}  atoms={int(valid.sum())} bonds={len(i)} (ideal length mean {float(ideal.mean()):.3f} A) ===", flush=True)
        print(f"{'step':>5}{'sigma':>10}{'x0hat mean|d|':>15}{'>0.1A':>8}{'>0.3A':>8}{'clash':>7}  |{'iter mean|d|':>14}{'>0.1A':>8}{'clash':>7}", flush=True)
        n = x0hat.shape[0]
        for s in sorted(set(list(range(0, n, 10)) + [n - 1])):
            a, b = res["x0hat"][s], res["iter"][s]
            print(f"{s:5d}{sigmas[s]:10.3f}{a['mean_abs_dev']:15.3f}{a['frac_gt_0.1']:8.2f}{a['frac_gt_0.3']:8.2f}{a['n_clash']:7d}  |{b['mean_abs_dev']:14.3f}{b['frac_gt_0.1']:8.2f}{b['n_clash']:7d}", flush=True)
        return out


inf.Client = Client
inf.cli()
