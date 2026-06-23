"""Predict a de-novo target (H1356) with the no-single EDM model.

Mixes the inference-spec build pipeline (``miniworld.data.inference``) — which
turns a fasta/a3m/tokenization YAML into a ``Batch`` with zero GT coords — with
the no-single EDM ``Client`` sampler (same prepare+sample path as
``run_miniworld_no_single_edm_inference.py``). No LMDB / DB needed.

Per-step CoM removal (the AF3Solver fix) is on via EDM_CENTER_PER_STEP=1.

Usage:
    EDM_CENTER_PER_STEP=1 pixi run python scripts/run_h1356_edm_inference.py \
        --data-config inference/H1356/configs/data_A1.yaml \
        --ckpt <epoch.pt> --n-samples 5 --timesteps 200
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from lightning import Fabric

from miniworld.data.inference import build_inference_batch
from miniworld.data.inference.spec import InferenceSpec
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.models.miniworld_no_single_at_trunk import Client

# The H1356 configs were written with `targets/H1356/...` paths and a CCD LMDB
# that isn't on this box; remap to the real locations.
PATH_REMAP = ("targets/H1356", "inference/H1356")
CCD_DB = "/NHNHOME/WORKSPACE/0226010152_A/data/CCD/preprocessed_CCD.lmdb"


def _fix(p: str) -> str:
    return str(p).replace(*PATH_REMAP)


@click.command()
@click.option("--data-config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path),
              default=Path("inference/H1356/structures/edm"), show_default=True)
@click.option("--n-samples", type=int, default=5, show_default=True)
@click.option("--timesteps", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-msa-depth", type=int, default=384, show_default=True)
@click.option("--ema/--no-ema", "use_ema", default=True, show_default=True)
def main(data_config, ckpt, output_dir, n_samples, timesteps, seed, max_msa_depth, use_ema):
    # --- build the inference Batch from the spec (paths remapped) -------------
    spec = InferenceSpec.from_yaml(data_config)
    spec.fasta = {k: _fix(v) for k, v in spec.fasta.items()}
    if spec.a3m:
        spec.a3m = {k: _fix(v) for k, v in spec.a3m.items()}
    if spec.tokenization:
        spec.tokenization = _fix(spec.tokenization)
    spec.ccd_db = CCD_DB
    batch = build_inference_batch(
        spec, max_msa_depth=max_msa_depth, missing_policy="gap", seed=seed,
    )

    # --- load the no-single EDM client from the checkpoint --------------------
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
        "H1356 EDM infer: name=%s ckpt epoch=%s | n_tokens=%d n_atoms=%d n_msa=%d "
        "n_samples=%d timesteps=%d ema=%s",
        spec.name, state.get("epoch"),
        int(batch.token_length), int(batch.atom_length), int(batch.msa_count),
        n_samples, timesteps, use_ema,
    )

    # --- sample --------------------------------------------------------------
    batch = batch.to(device=client.device)
    wrapper, batch = client.prepare(batch)
    output = client.sample(wrapper, batch, n_samples=n_samples, timesteps=timesteps)

    output_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_samples):
        batch_to_cif(batch, output.atom_pos_pred[k:k + 1],
                     output_dir / f"{spec.name}_pred_{k}.cif")
    client.logger.info("Saved %d structures -> %s", n_samples, output_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
