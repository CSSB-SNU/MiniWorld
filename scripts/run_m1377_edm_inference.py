"""Predict M1377 (VEGF165 dimer + bivalent RNA aptamer) with the no-single EDM model.

Same build+sample path as ``run_h1356_edm_inference.py`` but for M1377, with one
extra capability: **synthetic Ca2+ injection**. The model's training CCD
(``preprocessed_CCD.lmdb``) excludes monatomic ions, so ``CCDLookup["CA"]`` would
KeyError. Since a Ca2+ ion at residue-level resolution is a single 1-atom token
(``CCDLookup.fragments`` is never called), we only need ``__getitem__`` to return a
synthetic single-atom ``CCDResidue``; we monkeypatch exactly that and delegate every
other CCD id to the real lookup. The element vocab already maps ``CA -> 19``.

NOTE: calcium is OUT-OF-DISTRIBUTION for this model (it never saw ions in training);
the Ca2+ entry only supplies a +2, element-19 atom with an origin reference position.

Usage:
    EDM_CENTER_PER_STEP=1 pixi run python scripts/run_m1377_edm_inference.py \
        --data-config inference/M1377/configs/data_A2B1_noHis_Ca2.yaml \
        --ckpt <epoch.pt> --output-dir inference/M1377/structures/edm_a2b1_noHis_Ca2 \
        --n-samples 5 --timesteps 200
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from lightning import Fabric

from miniworld.data.inference import build_inference_batch
from miniworld.data.inference import ccd as ccd_mod
from miniworld.data.inference.ccd import CCDResidue
from miniworld.data.inference.spec import InferenceSpec
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.models.miniworld_no_single_at_trunk import Client

# CCD ids the training CCD lacks -> synthetic single-atom ion residues.
# element symbol must be an uppercase key in constants.mapping._atom_mapping.
_SYNTH_IONS = {
    "CA": ("CA", "CA", 2.0),  # chemcomp_id -> (atom_id, element, formal_charge)
}


def _install_synthetic_ion_lookup() -> None:
    """Patch ``CCDLookup.__getitem__`` to mint single-atom ion residues on demand."""
    orig = ccd_mod.CCDLookup.__getitem__

    def patched(self, chemcomp_id: str) -> CCDResidue:  # noqa: ANN001
        if chemcomp_id in _SYNTH_IONS:
            if chemcomp_id in self._residue_cache:
                return self._residue_cache[chemcomp_id]
            atom_id, element, charge = _SYNTH_IONS[chemcomp_id]
            res = CCDResidue(
                chemcomp_id=chemcomp_id,
                atom_ids=np.array([atom_id], dtype=object),
                atom_elements=np.array([element], dtype=object),
                atom_charges=np.array([charge], dtype=np.float32),
                atom_xyz=np.zeros((1, 3), dtype=np.float32),
            )
            self._residue_cache[chemcomp_id] = res
            return res
        return orig(self, chemcomp_id)

    ccd_mod.CCDLookup.__getitem__ = patched


@click.command()
@click.option("--data-config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--n-samples", type=int, default=5, show_default=True)
@click.option("--timesteps", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-msa-depth", type=int, default=384, show_default=True)
@click.option("--ema/--no-ema", "use_ema", default=True, show_default=True)
def main(data_config, ckpt, output_dir, n_samples, timesteps, seed, max_msa_depth, use_ema):
    _install_synthetic_ion_lookup()

    spec = InferenceSpec.from_yaml(data_config)
    batch = build_inference_batch(
        spec, max_msa_depth=max_msa_depth, missing_policy="gap", seed=seed,
    )

    fabric = Fabric(devices=1, num_nodes=1)
    fabric.launch()
    fabric.seed_everything(seed)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Client.Config.model_validate(state["config"])
    cfg.train.use_ema = use_ema
    client = Client(cfg)
    client.setup(fabric=fabric)
    client.load_state_dict(state, model_only=True)
    client.model.eval()
    client.logger.info(
        "M1377 EDM infer: name=%s ckpt epoch=%s | n_tokens=%d n_atoms=%d n_msa=%d "
        "n_samples=%d timesteps=%d ema=%s",
        spec.name, state.get("epoch"),
        int(batch.token_length), int(batch.atom_length), int(batch.msa_count),
        n_samples, timesteps, use_ema,
    )

    batch = batch.to(device=client.device)
    wrapper, batch = client.prepare(batch)
    output = client.sample(wrapper, batch, n_samples=n_samples, timesteps=timesteps)

    output_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_samples):
        batch_to_cif(batch, output.atom_pos_pred[k:k + 1],
                     output_dir / f"{spec.name}_pred_{k}.cif")
    client.logger.info("Saved %d structures -> %s", n_samples, output_dir)


if __name__ == "__main__":
    main()
