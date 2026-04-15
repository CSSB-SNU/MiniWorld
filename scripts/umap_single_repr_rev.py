"""Extract pre- and post-pairformer single representations and visualize with UMAP.

Run once per model variant, then re-plot from saved .npz files.

Usage (extract + plot for one variant):
    pixi run python scripts/umap_single_repr_rev.py extract \
        --config configs/atom_token/config.yaml \
        --ckpt  path/to/checkpoint.pt \
        --variant onehot \
        --output-dir outputs/umap_analysis \
        --num-items 50

    pixi run python scripts/umap_single_repr_rev.py extract \
        --config configs/atom_token_fingerprint/config.yaml \
        --ckpt  path/to/checkpoint.pt \
        --variant fp \
        --output-dir outputs/umap_analysis

    pixi run python scripts/umap_single_repr_rev.py extract \
        --config configs/atom_token_explicit/config.yaml \
        --ckpt  path/to/checkpoint.pt \
        --variant explicit \
        --output-dir outputs/umap_analysis

Plot from saved npz files (no GPU needed):
    pixi run python scripts/umap_single_repr_rev.py plot \
        --output-dir outputs/umap_analysis

Per-item analysis (extract with boundaries, then plot):
    pixi run python scripts/umap_single_repr_rev.py extract-items \
        --config configs/atom_token/config.yaml \
        --ckpt  path/to/checkpoint.pt \
        --variant onehot \
        --output-dir outputs/umap_analysis \
        --num-items 50

    pixi run python scripts/umap_single_repr_rev.py plot-items \
        --output-dir outputs/umap_analysis
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torch.jit.script_method.*")

import click
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("umap_repr")

# -- Residue grouping -----------------------------------------------------

CHEM_GROUP = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic",
    "ILE": "hydrophobic", "MET": "hydrophobic", "PRO": "hydrophobic",
    "PHE": "hydrophobic", "TRP": "hydrophobic", "TYR": "hydrophobic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar", "CYS": "polar",
    "LYS": "positive", "ARG": "positive",
    "ASP": "negative", "GLU": "negative",
    "GLY": "special", "HIS": "special",
    "DA": "DNA", "DT": "DNA", "DC": "DNA", "DG": "DNA",
    "A": "RNA", "U": "RNA", "C": "RNA", "G": "RNA",
}

AROMATIC_RESIDUES = {"PHE", "TRP", "TYR"}

ENTITY_TYPE_NAMES = {
    0: "antibody", 1: "protein", 2: "d-protein",
    3: "RNA", 4: "DNA", 5: "NA", 6: "ligand", 7: "branched",
}

GROUP_COLORS = {
    "hydrophobic": "#1f77b4", "polar": "#2ca02c",
    "positive": "#d62728", "negative": "#ff7f0e", "special": "#9467bd",
    "DNA": "#8c564b", "RNA": "#e377c2", "ligand": "#7f7f7f", "other": "#bcbd22",
}


# -- Data structures ------------------------------------------------------

@dataclass
class RepresentationData:
    pre_pairformer: list[np.ndarray] = field(default_factory=list)
    post_pairformer: list[np.ndarray] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)
    entity_types: list[int] = field(default_factory=list)
    token_masks: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def stack(self):
        mask = np.concatenate(self.token_masks)
        return {
            "pre": np.concatenate(self.pre_pairformer)[mask],
            "post": np.concatenate(self.post_pairformer)[mask],
            "residue_names": np.array(self.residue_names)[mask],
            "entity_types": np.array(self.entity_types)[mask],
        }


# -- Hook to capture pre-pairformer single --------------------------------

def _patched_condition_forward(model, original_fn):
    captured = {}

    def hook_fn(module, input, output):
        captured["pre_pairformer_single"] = output[1].detach().cpu().float()

    def wrapper(*args, **kwargs):
        handle = model.input_feature_embedder.register_forward_hook(hook_fn)
        result = original_fn(*args, **kwargs)
        handle.remove()
        return result

    wrapper.captured = captured
    return wrapper


# -- Extract representations ----------------------------------------------

def extract_representations(
    config_path: Path, ckpt_path: Path, variant: str,
    num_items: int, seed: int, overrides: tuple[str, ...] = (),
) -> RepresentationData:
    import torch
    from hydra import compose, initialize_config_dir
    from lightning import Fabric
    from pydantic import BaseModel

    if variant == "onehot":
        from miniworld.data.dataloader.dataloader import BioMolData
        from miniworld.models.default_client import Client
        from miniworld.models.af3_like import Model
    elif variant == "fp":
        from miniworld.data.dataloader.dataloader import BioMolData
        from miniworld.models.embedding_client import Client
        from miniworld.models.af3_like_embedding import Model
    elif variant == "explicit":
        from miniworld.data.dataloader.dataloader_explicit import BioMolData
        from miniworld.models.explicit_client import Client
        from miniworld.models.af3_like_explicit import Model
    else:
        raise ValueError(f"Unknown variant: {variant}")

    from miniworld.configs import (
        BioMolDBConfig, CropConfig, EDMDiffuserConfig,
        MSAConfig, SamplerConfig, TemplateConfig, TokenizerConfig,
    )

    class DataConfig(BaseModel):
        train_db: BioMolDBConfig
        valid_db: BioMolDBConfig
        crop: CropConfig
        msa: MSAConfig
        tokenizer: TokenizerConfig
        sampler: SamplerConfig
        template: TemplateConfig = TemplateConfig()

    class Config(BaseModel):
        data: DataConfig
        train: Client.TrainConfig
        model: Model.Config
        diffuser: EDMDiffuserConfig
        loss: Client.LossConfig

    with initialize_config_dir(str(config_path.parent.absolute()), version_base=None):
        cfg = compose(config_name=config_path.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)

    fabric = Fabric(devices=1)
    fabric.launch()
    fabric.seed_everything(seed)

    client = Client(
        Client.Config(train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss),
    )
    client.setup(fabric=fabric)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True)
    client.model.eval()

    raw_model = getattr(client.model, "module", client.model)
    original_fn = raw_model.condition_forward
    patched_fn = _patched_condition_forward(raw_model, original_fn)
    raw_model.condition_forward = patched_fn

    data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db, sampler_config=None,
        tokenizer_config=cfg.data.tokenizer,
    )
    dataset = BioMolData(data_config)
    dataloader = dataset.create_ddp_dataloader(
        world_size=1, rank=0, seed=seed, drop_last=False, batch_size=1, num_workers=0,
    )

    repr_data = RepresentationData()

    for idx, _batch in enumerate(dataloader):
        if idx >= num_items:
            break
        batch = _batch.to(device=client.device)
        name = batch.name[0]
        if isinstance(name, list):
            name = name[0]
        logger.info("[%d/%d] %s (tokens=%d)", idx + 1, num_items, name, batch.token_length)

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                (_, token_single_trunk, _, _) = raw_model.condition_forward(
                    msa=batch.msa, reference=batch.reference,
                    scheme=batch.scheme, sequence=batch.sequence,
                    structure=batch.structure,
                )

            pre_single = patched_fn.captured["pre_pairformer_single"]
            post_single = token_single_trunk.detach().cpu().float()
            token_mask = batch.structure.token_mask[0].bool().cpu().numpy()

            repr_data.pre_pairformer.append(pre_single[0].numpy())
            repr_data.post_pairformer.append(post_single[0].numpy())
            repr_data.token_masks.append(token_mask)

            chem_comp_ids = batch.chem_comp_ids[0]
            if isinstance(chem_comp_ids, list) and len(chem_comp_ids) > 0 and isinstance(chem_comp_ids[0], list):
                chem_comp_ids = chem_comp_ids[0]
            token_res_idx = batch.scheme.token_residue_idx[0].cpu().numpy()
            chain_entity_types = batch.chain.entity_type[0].cpu().numpy()
            atom_to_chain = batch.scheme.atom_to_chain_id[0].cpu().numpy()
            token_to_atom = batch.scheme.atom_to_token_idx_map[0].cpu().numpy()

            n_tokens = token_mask.shape[0]
            for t in range(n_tokens):
                res_idx = token_res_idx[t]
                comp_id = chem_comp_ids[res_idx] if res_idx < len(chem_comp_ids) else "UNK"
                repr_data.residue_names.append(str(comp_id))
                atoms_for_token = np.where(token_to_atom == t)[0]
                if len(atoms_for_token) > 0:
                    chain_idx = atom_to_chain[atoms_for_token[0]]
                    etype = chain_entity_types[chain_idx] if chain_idx < len(chain_entity_types) else 6
                else:
                    etype = 6
                repr_data.entity_types.append(int(etype))

            repr_data.names.append(name)

        except Exception:
            logger.exception("[%d] Failed on %s, skipping", idx + 1, name)

    raw_model.condition_forward = original_fn
    return repr_data


# -- UMAP plotting --------------------------------------------------------

def plot_umap_grid(
    datasets: dict[str, dict],
    output_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Rows = models, cols = pre/post pairformer.

    Aromatic (PHE, TRP, TYR): square markers, same color as hydrophobic.
    All others: dot markers.
    """
    from umap import UMAP

    model_names = list(datasets.keys())

    for color_mode in ["chem_group", "entity_type"]:
        fig, axes = plt.subplots(
            len(model_names), 2,
            figsize=(16, 6 * len(model_names)),
            constrained_layout=True,
        )
        if len(model_names) == 1:
            axes = axes[np.newaxis, :]

        for row, model_name in enumerate(model_names):
            data = datasets[model_name]
            res_names = data["residue_names"]
            is_aromatic = np.array([r in AROMATIC_RESIDUES for r in res_names])

            for col, (stage_name, stage_key) in enumerate([
                ("Pre-Pairformer", "pre"), ("Post-Pairformer", "post"),
            ]):
                ax = axes[row, col]
                X = data[stage_key]
                logger.info("UMAP: %s / %s (%d tokens)", model_name, stage_name, X.shape[0])
                embedding = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(X)

                if color_mode == "chem_group":
                    groups = np.array([CHEM_GROUP.get(r, "other") for r in res_names])
                    for g in sorted(set(groups)):
                        color = GROUP_COLORS.get(g, "#333333")
                        gmask = groups == g
                        dot = gmask & ~is_aromatic
                        sq = gmask & is_aromatic
                        if dot.any():
                            ax.scatter(embedding[dot, 0], embedding[dot, 1],
                                       c=color, label=g, s=4, alpha=0.5, marker="o", rasterized=True)
                        if sq.any():
                            ax.scatter(embedding[sq, 0], embedding[sq, 1],
                                       c=color, s=14, alpha=0.7, marker="s", rasterized=True,
                                       label=f"{g} (aromatic)" if not dot.any() else None)
                else:
                    etypes = data["entity_types"]
                    cmap = plt.cm.tab10
                    for i, et in enumerate(sorted(set(etypes))):
                        emask = etypes == et
                        color = [cmap(i % 10)]
                        label = ENTITY_TYPE_NAMES.get(et, f"type_{et}")
                        dot = emask & ~is_aromatic
                        sq = emask & is_aromatic
                        if dot.any():
                            ax.scatter(embedding[dot, 0], embedding[dot, 1],
                                       c=color, label=label, s=4, alpha=0.5, marker="o", rasterized=True)
                        if sq.any():
                            ax.scatter(embedding[sq, 0], embedding[sq, 1],
                                       c=color, s=14, alpha=0.7, marker="s", rasterized=True)

                ax.set_title(f"{model_name} \u2014 {stage_name}")
                ax.set_xticks([]); ax.set_yticks([])
                ax.legend(markerscale=3, fontsize=7, loc="upper right")

        title_label = "chemical group" if color_mode == "chem_group" else "entity type"
        fig.suptitle(f"Single Representation UMAP (colored by {title_label})\n"
                     "\u25a0 = aromatic (PHE, TRP, TYR)  \u25cf = others", fontsize=14)
        out_path = output_dir / f"umap_{color_mode}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved %s", out_path)


# -- CLI ------------------------------------------------------------------

@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--variant", type=click.Choice(["onehot", "fp", "explicit"]), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/umap_analysis"))
@click.option("--num-items", type=int, default=640)
@click.option("--seed", type=int, default=0)
@click.argument("overrides", type=str, nargs=-1)
def extract(config: Path, ckpt: Path, variant: str, output_dir: Path, num_items: int, seed: int, overrides: tuple[str, ...]):
    """Extract representations for a single model variant and save to .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)

    repr_data = extract_representations(config, ckpt, variant, num_items, seed, overrides)
    stacked = repr_data.stack()

    npz_path = output_dir / f"repr_{variant}.npz"
    np.savez_compressed(npz_path, pre=stacked["pre"], post=stacked["post"],
                        residue_names=stacked["residue_names"], entity_types=stacked["entity_types"])
    logger.info("Saved %s (%d tokens)", npz_path, stacked["pre"].shape[0])


@cli.command()
@click.option("--output-dir", type=click.Path(exists=True, path_type=Path), default=Path("outputs/umap_analysis"))
def plot(output_dir: Path):
    """Plot UMAP from previously extracted .npz files (no GPU needed)."""
    variant_labels = {"onehot": "one-hot", "fp": "fingerprint", "explicit": "explicit"}
    all_data = {}

    for variant, label in variant_labels.items():
        npz_path = output_dir / f"repr_{variant}.npz"
        if not npz_path.exists():
            logger.warning("Skipping %s: %s not found", variant, npz_path)
            continue
        d = np.load(npz_path, allow_pickle=True)
        all_data[label] = {
            "pre": d["pre"], "post": d["post"],
            "residue_names": d["residue_names"], "entity_types": d["entity_types"],
        }
        logger.info("Loaded %s (%d tokens)", npz_path, d["pre"].shape[0])

    if not all_data:
        logger.error("No .npz files found in %s. Run 'extract' first.", output_dir)
        return

    plot_umap_grid(all_data, output_dir)


@cli.command()
@click.option("--output-dir", type=click.Path(exists=True, path_type=Path), default=Path("outputs/umap_analysis"))
@click.option("--n-neighbors", type=int, default=15)
@click.option("--min-dist", type=float, default=0.1)
def distance(output_dir: Path, n_neighbors: int, min_dist: float):
    """Compute per-token UMAP displacement between pre- and post-pairformer.

    Fits a single UMAP on the combined (pre + post) data per model so that
    distances are comparable.  Reports L2 distance in UMAP space and cosine
    distance in the original 384-dim space, broken down by chemical group.
    """
    import json
    from umap import UMAP
    from numpy.linalg import norm

    variant_labels = {"onehot": "one-hot", "fp": "fingerprint", "explicit": "explicit"}
    all_results = {}

    for variant, label in variant_labels.items():
        npz_path = output_dir / f"repr_{variant}.npz"
        if not npz_path.exists():
            logger.warning("Skipping %s: %s not found", variant, npz_path)
            continue
        d = np.load(npz_path, allow_pickle=True)
        pre, post = d["pre"], d["post"]
        res_names = d["residue_names"]
        n_tokens = pre.shape[0]
        logger.info("Processing %s (%d tokens)", label, n_tokens)

        # -- cosine distance in original 384-dim space -----------------
        pre_norm = pre / (norm(pre, axis=1, keepdims=True) + 1e-8)
        post_norm = post / (norm(post, axis=1, keepdims=True) + 1e-8)
        cosine_dist = 1.0 - np.sum(pre_norm * post_norm, axis=1)  # (N,)

        # -- L2 distance in original space -----------------------------
        l2_dist_hd = norm(post - pre, axis=1)  # (N,)

        # -- single UMAP on combined pre+post --------------------------
        combined = np.concatenate([pre, post], axis=0)  # (2N, 384)
        logger.info("  Fitting joint UMAP on %d points ...", combined.shape[0])
        emb = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(combined)
        emb_pre = emb[:n_tokens]
        emb_post = emb[n_tokens:]
        umap_dist = norm(emb_post - emb_pre, axis=1)  # (N,)

        # -- aggregate by chemical group -------------------------------
        groups = np.array([CHEM_GROUP.get(r, "other") for r in res_names])
        summary = {}
        for g in sorted(set(groups)):
            gmask = groups == g
            summary[g] = {
                "count": int(gmask.sum()),
                "umap_dist_mean": float(np.mean(umap_dist[gmask])),
                "umap_dist_std": float(np.std(umap_dist[gmask])),
                "cosine_dist_mean": float(np.mean(cosine_dist[gmask])),
                "cosine_dist_std": float(np.std(cosine_dist[gmask])),
                "l2_dist_mean": float(np.mean(l2_dist_hd[gmask])),
                "l2_dist_std": float(np.std(l2_dist_hd[gmask])),
            }
        summary["_overall"] = {
            "count": n_tokens,
            "umap_dist_mean": float(np.mean(umap_dist)),
            "umap_dist_std": float(np.std(umap_dist)),
            "cosine_dist_mean": float(np.mean(cosine_dist)),
            "cosine_dist_std": float(np.std(cosine_dist)),
            "l2_dist_mean": float(np.mean(l2_dist_hd)),
            "l2_dist_std": float(np.std(l2_dist_hd)),
        }
        all_results[label] = summary

        # -- save per-token distances ----------------------------------
        np.savez_compressed(
            output_dir / f"distance_{variant}.npz",
            umap_dist=umap_dist, cosine_dist=cosine_dist, l2_dist=l2_dist_hd,
            umap_pre=emb_pre, umap_post=emb_post,
            residue_names=res_names, groups=groups,
        )

    # -- save summary json ---------------------------------------------
    json_path = output_dir / "distance_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %s", json_path)

    # -- print table ---------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'model':<14} {'group':<14} {'count':>6}  {'UMAP dist':>12}  {'cosine dist':>12}  {'L2 dist':>12}")
    print("-" * 90)
    for label, summary in all_results.items():
        for g in sorted(summary):
            s = summary[g]
            print(f"{label:<14} {g:<14} {s['count']:>6}  "
                  f"{s['umap_dist_mean']:>5.2f} +/- {s['umap_dist_std']:<5.2f}  "
                  f"{s['cosine_dist_mean']:>5.3f} +/- {s['cosine_dist_std']:<5.3f}  "
                  f"{s['l2_dist_mean']:>5.2f} +/- {s['l2_dist_std']:<5.2f}")
        print("-" * 90)

    # -- bar plot: mean UMAP displacement by group, per model ----------
    if len(all_results) > 0:
        model_names = list(all_results.keys())
        # get all groups except _overall
        all_groups = sorted({g for s in all_results.values() for g in s if g != "_overall"})
        x = np.arange(len(all_groups))
        width = 0.8 / len(model_names)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        for ax, (metric, metric_label) in zip(axes, [
            ("umap_dist_mean", "UMAP displacement"),
            ("cosine_dist_mean", "Cosine distance"),
            ("l2_dist_mean", "L2 distance (384-dim)"),
        ]):
            for i, m in enumerate(model_names):
                vals = [all_results[m].get(g, {}).get(metric, 0) for g in all_groups]
                ax.bar(x + i * width, vals, width, label=m, alpha=0.8)
            ax.set_xticks(x + width * (len(model_names) - 1) / 2)
            ax.set_xticklabels(all_groups, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric_label)
            ax.legend(fontsize=8)

        fig.suptitle("Pre -> Post Pairformer Displacement by Chemical Group", fontsize=14)
        out_path = output_dir / "distance_by_group.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved %s", out_path)

        # -- arrow plot: pre->post displacement on joint UMAP ----------
        fig, axes = plt.subplots(1, len(all_results), figsize=(8 * len(all_results), 7), constrained_layout=True)
        if len(all_results) == 1:
            axes = [axes]
        for ax, (label, _) in zip(axes, all_results.items()):
            variant_key = {"one-hot": "onehot", "fingerprint": "fp", "explicit": "explicit"}[label]
            dd = np.load(output_dir / f"distance_{variant_key}.npz", allow_pickle=True)
            emb_pre_v = dd["umap_pre"]
            emb_post_v = dd["umap_post"]
            groups_v = dd["groups"]
            res_v = dd["residue_names"]
            is_arom = np.array([r in AROMATIC_RESIDUES for r in res_v])

            # subsample for readability
            n = len(emb_pre_v)
            step = max(1, n // 2000)
            idx = np.arange(0, n, step)

            for g in sorted(set(groups_v)):
                gmask_full = groups_v == g
                gmask = gmask_full[idx]
                color = GROUP_COLORS.get(g, "#333333")
                sel = idx[gmask]
                dot = sel[~is_arom[sel]]
                sq = sel[is_arom[sel]]
                if len(dot) > 0:
                    ax.scatter(emb_pre_v[dot, 0], emb_pre_v[dot, 1],
                               c=color, s=3, alpha=0.3, marker="o", rasterized=True)
                    ax.quiver(emb_pre_v[dot, 0], emb_pre_v[dot, 1],
                              emb_post_v[dot, 0] - emb_pre_v[dot, 0],
                              emb_post_v[dot, 1] - emb_pre_v[dot, 1],
                              color=color, alpha=0.3, scale_units="xy", angles="xy", scale=1,
                              width=0.002, headwidth=3, headlength=3, label=g)
                if len(sq) > 0:
                    ax.scatter(emb_pre_v[sq, 0], emb_pre_v[sq, 1],
                               c=color, s=10, alpha=0.4, marker="s", rasterized=True)
                    ax.quiver(emb_pre_v[sq, 0], emb_pre_v[sq, 1],
                              emb_post_v[sq, 0] - emb_pre_v[sq, 0],
                              emb_post_v[sq, 1] - emb_pre_v[sq, 1],
                              color=color, alpha=0.4, scale_units="xy", angles="xy", scale=1,
                              width=0.002, headwidth=3, headlength=3)

            ax.set_title(f"{label}")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(markerscale=3, fontsize=7, loc="upper right")

        fig.suptitle("Pre -> Post Pairformer displacement (arrows)\n"
                     "square = aromatic  circle = others", fontsize=13)
        out_path = output_dir / "distance_arrows.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved %s", out_path)

    logger.info("Done.")


# =========================================================================
# Per-item analysis (new commands)
# =========================================================================

@cli.command("extract-items")
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--variant", type=click.Choice(["onehot", "fp", "explicit"]), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/umap_analysis"))
@click.option("--num-items", type=int, default=50)
@click.option("--seed", type=int, default=0)
@click.argument("overrides", type=str, nargs=-1)
def extract_items(config: Path, ckpt: Path, variant: str, output_dir: Path,
                  num_items: int, seed: int, overrides: tuple[str, ...]):
    """Extract per-item representations preserving item boundaries.

    Saves to repr_{variant}_items.npz with keys:
      item_names, boundaries, all_pre, all_post, all_residue_names,
      all_entity_types, mean_pre, mean_post.

    Use 'plot-items' afterwards to generate per-item UMAP visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    repr_data = extract_representations(config, ckpt, variant, num_items, seed, overrides)

    n_items = len(repr_data.names)
    if n_items == 0:
        logger.error("No items extracted successfully.")
        return

    boundaries = [0]
    all_pre: list[np.ndarray] = []
    all_post: list[np.ndarray] = []
    all_res: list[np.ndarray] = []
    all_et: list[np.ndarray] = []
    mean_pre: list[np.ndarray] = []
    mean_post: list[np.ndarray] = []

    res_offset = 0
    for i in range(n_items):
        n_all = repr_data.token_masks[i].shape[0]
        mask_i = repr_data.token_masks[i].astype(bool)

        pre_i = repr_data.pre_pairformer[i][mask_i]
        post_i = repr_data.post_pairformer[i][mask_i]
        n_valid = pre_i.shape[0]

        res_i = np.array(repr_data.residue_names[res_offset:res_offset + n_all])[mask_i]
        et_i = np.array(repr_data.entity_types[res_offset:res_offset + n_all])[mask_i]
        res_offset += n_all

        all_pre.append(pre_i)
        all_post.append(post_i)
        all_res.append(res_i)
        all_et.append(et_i)
        boundaries.append(boundaries[-1] + n_valid)

        mean_pre.append(pre_i.mean(axis=0))
        mean_post.append(post_i.mean(axis=0))

    npz_path = output_dir / f"repr_{variant}_items.npz"
    np.savez_compressed(
        npz_path,
        item_names=np.array(repr_data.names),
        boundaries=np.array(boundaries),
        all_pre=np.vstack(all_pre),
        all_post=np.vstack(all_post),
        all_residue_names=np.concatenate(all_res),
        all_entity_types=np.concatenate(all_et),
        mean_pre=np.vstack(mean_pre),
        mean_post=np.vstack(mean_post),
    )
    logger.info("Saved %s (%d items, %d total tokens)", npz_path, n_items, boundaries[-1])


@cli.command("plot-items")
@click.option("--output-dir", type=click.Path(exists=True, path_type=Path),
              default=Path("outputs/umap_analysis"))
@click.option("--n-neighbors", type=int, default=15)
@click.option("--min-dist", type=float, default=0.1)
def plot_items(output_dir: Path, n_neighbors: int, min_dist: float):
    """Generate per-item UMAP analysis from repr_{variant}_items.npz files.

    Produces four outputs per variant found:

    1. item_mean_umap_joint.png  -- one UMAP with item-level mean
       representations. Pre (circle) and post (triangle) dots with arrows
       showing displacement per item.
    2. item_mean_umap_split.png  -- same UMAP coordinates split into two
       panels (pre-only and post-only), each with N dots.
    3. item_distances.csv        -- per-item UMAP / cosine / L2 distance
       of mean representation before vs after pairformer.
    4. residue_umaps/<item>.png  -- per-item UMAP of all token (residue)
       representations, colored by chemical group, comparing pre vs post.
    """
    from umap import UMAP
    from numpy.linalg import norm

    variant_labels = {"onehot": "one-hot", "fp": "fingerprint", "explicit": "explicit"}

    for variant, label in variant_labels.items():
        npz_path = output_dir / f"repr_{variant}_items.npz"
        if not npz_path.exists():
            logger.warning("Skipping %s: %s not found", variant, npz_path)
            continue

        d = np.load(npz_path, allow_pickle=True)
        item_names = d["item_names"]
        boundaries = d["boundaries"]
        all_pre = d["all_pre"]
        all_post = d["all_post"]
        all_res = d["all_residue_names"]
        all_et = d["all_entity_types"]
        mean_pre = d["mean_pre"]
        mean_post = d["mean_post"]
        n_items = len(item_names)

        logger.info("Loaded %s: %d items, %d total tokens",
                     label, n_items, all_pre.shape[0])

        variant_dir = output_dir / f"per_item_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        # =============================================================
        # 1 & 2.  Item-level UMAP on mean representations
        # =============================================================
        if n_items >= 2:
            combined_mean = np.concatenate([mean_pre, mean_post], axis=0)  # (2N, D)
            umap_nn = min(n_neighbors, 2 * n_items - 1)
            logger.info("Fitting item-level UMAP (%d points, n_neighbors=%d)",
                        combined_mean.shape[0], umap_nn)
            emb_mean = UMAP(
                n_neighbors=umap_nn, min_dist=min_dist, random_state=42,
            ).fit_transform(combined_mean)
            emb_pre = emb_mean[:n_items]
            emb_post = emb_mean[n_items:]

            # -- Plot 1: joint pre/post with arrows --------------------
            fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
            ax.scatter(
                emb_pre[:, 0], emb_pre[:, 1],
                c="#1f77b4", s=40, alpha=0.8, marker="o",
                label="Pre-Pairformer", zorder=3,
            )
            ax.scatter(
                emb_post[:, 0], emb_post[:, 1],
                c="#d62728", s=40, alpha=0.8, marker="^",
                label="Post-Pairformer", zorder=3,
            )
            ax.quiver(
                emb_pre[:, 0], emb_pre[:, 1],
                emb_post[:, 0] - emb_pre[:, 0],
                emb_post[:, 1] - emb_pre[:, 1],
                color="#888888", alpha=0.4,
                scale_units="xy", angles="xy", scale=1,
                width=0.003, headwidth=3, headlength=3, zorder=2,
            )
            for i in range(n_items):
                ax.annotate(
                    str(item_names[i]),
                    (emb_post[i, 0], emb_post[i, 1]),
                    fontsize=5, alpha=0.6, ha="left", va="bottom",
                )
            ax.set_title(f"{label} -- Item-level Mean Representation\n"
                         "circle = Pre-Pairformer   triangle = Post-Pairformer")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=9)
            out = variant_dir / "item_mean_umap_joint.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info("Saved %s", out)

            # -- Plot 2: split into two panels -------------------------
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(18, 8), constrained_layout=True,
            )
            # shared axis limits so panels are visually comparable
            margin = 0.05
            x_all, y_all = emb_mean[:, 0], emb_mean[:, 1]
            dx = (x_all.max() - x_all.min()) * margin
            dy = (y_all.max() - y_all.min()) * margin
            xlim = (x_all.min() - dx, x_all.max() + dx)
            ylim = (y_all.min() - dy, y_all.max() + dy)

            for ax_i, emb_i, color, marker, stage in [
                (ax1, emb_pre,  "#1f77b4", "o", "Pre-Pairformer"),
                (ax2, emb_post, "#d62728", "^", "Post-Pairformer"),
            ]:
                ax_i.scatter(emb_i[:, 0], emb_i[:, 1],
                             c=color, s=40, alpha=0.8, marker=marker)
                for i in range(n_items):
                    ax_i.annotate(
                        str(item_names[i]),
                        (emb_i[i, 0], emb_i[i, 1]),
                        fontsize=5, alpha=0.6, ha="left", va="bottom",
                    )
                ax_i.set_xlim(xlim)
                ax_i.set_ylim(ylim)
                ax_i.set_title(f"{label} -- {stage}")
                ax_i.set_xticks([]); ax_i.set_yticks([])

            out = variant_dir / "item_mean_umap_split.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logger.info("Saved %s", out)

            # UMAP displacement per item
            umap_dist = norm(emb_post - emb_pre, axis=1)
        else:
            logger.warning("Need >= 2 items for item-level UMAP, skipping")
            umap_dist = np.full(n_items, np.nan)

        # =============================================================
        # 3.  Per-item distances (always computable in original space)
        # =============================================================
        pre_n = mean_pre / (norm(mean_pre, axis=1, keepdims=True) + 1e-8)
        post_n = mean_post / (norm(mean_post, axis=1, keepdims=True) + 1e-8)
        cos_dist = 1.0 - np.sum(pre_n * post_n, axis=1)
        l2_dist = norm(mean_post - mean_pre, axis=1)

        csv_path = variant_dir / "item_distances.csv"
        with open(csv_path, "w") as f:
            f.write("item_name,umap_distance,cosine_distance,l2_distance\n")
            for i in range(n_items):
                f.write(f"{item_names[i]},{umap_dist[i]:.6f},"
                        f"{cos_dist[i]:.6f},{l2_dist[i]:.6f}\n")
        logger.info("Saved %s", csv_path)

        # print table
        print(f"\n{'=' * 80}")
        print(f"  {label} -- Per-item mean-representation distances")
        print(f"{'=' * 80}")
        print(f"{'item':<30} {'UMAP':>10} {'cosine':>10} {'L2':>10}")
        print(f"{'-' * 80}")
        for i in range(n_items):
            print(f"{str(item_names[i]):<30} "
                  f"{umap_dist[i]:>10.4f} "
                  f"{cos_dist[i]:>10.6f} "
                  f"{l2_dist[i]:>10.4f}")
        print(f"{'-' * 80}")
        print(f"{'MEAN':<30} "
              f"{np.nanmean(umap_dist):>10.4f} "
              f"{np.mean(cos_dist):>10.6f} "
              f"{np.mean(l2_dist):>10.4f}")
        print(f"{'STD':<30} "
              f"{np.nanstd(umap_dist):>10.4f} "
              f"{np.std(cos_dist):>10.6f} "
              f"{np.std(l2_dist):>10.4f}")

        # =============================================================
        # 4.  Per-item residue-level UMAPs
        # =============================================================
        residue_dir = variant_dir / "residue_umaps"
        residue_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_items):
            lo, hi = int(boundaries[idx]), int(boundaries[idx + 1])
            pre_i = all_pre[lo:hi]
            post_i = all_post[lo:hi]
            res_i = all_res[lo:hi]
            n_tok = pre_i.shape[0]

            if n_tok < 3:
                logger.warning("Item %s: only %d tokens, skipping residue UMAP",
                               item_names[idx], n_tok)
                continue

            logger.info("  Residue UMAP [%d/%d] %s (%d tokens)",
                        idx + 1, n_items, item_names[idx], n_tok)

            # fit one UMAP on pre+post combined so coordinates are comparable
            combined_i = np.concatenate([pre_i, post_i], axis=0)  # (2T, D)
            nn_i = max(2, min(n_neighbors, n_tok - 1))
            emb_i = UMAP(
                n_neighbors=nn_i, min_dist=min_dist, random_state=42,
            ).fit_transform(combined_i)
            emb_pre_i = emb_i[:n_tok]
            emb_post_i = emb_i[n_tok:]

            groups_i = np.array([CHEM_GROUP.get(str(r), "other") for r in res_i])
            is_arom = np.array([str(r) in AROMATIC_RESIDUES for r in res_i])

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(16, 7), constrained_layout=True,
            )

            # Left panel: pre (circle/square) + post (triangle/diamond)
            for g in sorted(set(groups_i)):
                gmask = groups_i == g
                color = GROUP_COLORS.get(g, "#333333")
                dot = gmask & ~is_arom
                sq = gmask & is_arom
                if dot.any():
                    ax1.scatter(emb_pre_i[dot, 0], emb_pre_i[dot, 1],
                                c=color, s=8, alpha=0.5, marker="o",
                                label=f"{g}", rasterized=True)
                    ax1.scatter(emb_post_i[dot, 0], emb_post_i[dot, 1],
                                c=color, s=8, alpha=0.5, marker="^",
                                rasterized=True)
                if sq.any():
                    ax1.scatter(emb_pre_i[sq, 0], emb_pre_i[sq, 1],
                                c=color, s=14, alpha=0.6, marker="s",
                                label=f"{g} (aromatic)" if not dot.any() else None,
                                rasterized=True)
                    ax1.scatter(emb_post_i[sq, 0], emb_post_i[sq, 1],
                                c=color, s=14, alpha=0.6, marker="D",
                                rasterized=True)
            ax1.set_title("Chemical Group\n"
                          "circle/square = pre   triangle/diamond = post")
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.legend(markerscale=2, fontsize=6, loc="upper right")

            # Right panel: arrows from pre to post
            for g in sorted(set(groups_i)):
                gmask = groups_i == g
                color = GROUP_COLORS.get(g, "#333333")
                if gmask.any():
                    ax2.scatter(emb_pre_i[gmask, 0], emb_pre_i[gmask, 1],
                                c=color, s=6, alpha=0.4, marker="o",
                                label=g, rasterized=True)
                    ax2.quiver(
                        emb_pre_i[gmask, 0], emb_pre_i[gmask, 1],
                        emb_post_i[gmask, 0] - emb_pre_i[gmask, 0],
                        emb_post_i[gmask, 1] - emb_pre_i[gmask, 1],
                        color=color, alpha=0.3,
                        scale_units="xy", angles="xy", scale=1,
                        width=0.003, headwidth=3, headlength=3,
                    )
            ax2.set_title("Pre -> Post Displacement")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.legend(markerscale=2, fontsize=6, loc="upper right")

            name_safe = "".join(
                c if c.isalnum() or c in "._-" else "_"
                for c in str(item_names[idx])
            )
            fig.suptitle(f"{label} -- {item_names[idx]} ({n_tok} tokens)",
                         fontsize=12)
            out = residue_dir / f"{name_safe}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)

        logger.info("Saved %d per-item residue UMAPs in %s", n_items, residue_dir)

    logger.info("Done.")


if __name__ == "__main__":
    cli()
