#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one horizontal contact-map row per sample: "
            "target(root-a) | pred(root-a) | pred(root-b) | pred(root-c)."
        )
    )
    parser.add_argument(
        "--root-a",
        type=Path,
        default=Path("/home/bsoohyuncd/software/MiniWorld/ckpt_refpos_onehot/contact_map_prediction"),
        help="First experiment root. Provides target-only and pred-only images.",
    )
    parser.add_argument(
        "--root-b",
        type=Path,
        default=Path(
            "/home/bsoohyuncd/software/MiniWorld/ckpt_noRefpos_onehot/contact_map_prediction"
        ),
        help="Second experiment root. Provides pred-only images.",
    )
    parser.add_argument(
        "--root-c",
        type=Path,
        default=Path(
            "/home/bsoohyuncd/software/MiniWorld/ckpt_noRefpos_fp/contact_map_prediction"
        ),
        help="Third experiment root. Provides pred-only images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/home/bsoohyuncd/software/MiniWorld/figure/refpos_onehot(ep287)noRefpos_onehot(ep290)noRefpos_fp(ep265)"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--target-suffix",
        default="_target_only.png",
        help="Filename suffix for target-only images.",
    )
    parser.add_argument(
        "--pred-suffix",
        default="_pred_only.png",
        help="Filename suffix for predicted-only images.",
    )
    return parser.parse_args()


def normalize_root(root: Path) -> Path:
    # Allow passing either ".../ckpt_xxx" or ".../ckpt_xxx/contact_map_prediction".
    if root.name == "contact_map_prediction":
        return root
    candidate = root / "contact_map_prediction"
    return candidate if candidate.exists() else root


def key_from_path(path: Path, suffix: str) -> str:
    # Key format: "<target_type>/<sample_id>", where sample_id excludes suffix.
    if not path.name.endswith(suffix):
        msg = f"File does not match suffix '{suffix}': {path}"
        raise ValueError(msg)
    if 'noRefpos' in path.stem:
        sample_id = '_'.join(path.name[: -len(suffix)].split('_')[:-1])
    else:
        sample_id = path.name[: -len(suffix)]
    return f"{path.parent.name}/{sample_id}"


def collect_images(root: Path, suffix: str) -> dict[str, Path]:
    root = normalize_root(root)
    images = {}
    for p in root.rglob(f"*{suffix}"):
        if p.is_file():
            images[key_from_path(p, suffix)] = p
    return images


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad_h = height - img.shape[0]
    pad = np.ones((pad_h, img.shape[1], img.shape[2]), dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)


def concat_row(img_paths: list[Path]) -> np.ndarray:
    imgs = [to_rgb(mpimg.imread(str(p))) for p in img_paths]
    max_h = max(im.shape[0] for im in imgs)
    imgs = [pad_to_height(im, max_h) for im in imgs]
    return np.concatenate(imgs, axis=1)


def save_row_with_titles(img_paths: list[Path], out_path: Path) -> None:
    imgs = [to_rgb(mpimg.imread(str(p))) for p in img_paths]
    titles = []
    for p in img_paths:
        if p.name.endswith("_target_only.png"):
            titles.append('target')
        elif 'noRefposFp' in p.stem:
            titles.append('noRefpos-fingerprint')
        elif 'noRefpos' in p.stem:
            titles.append('noRefpos-onehot')
        else:
            titles.append('default')
    # titles = [p.stem.split('_')[-3] if 'noRefpos' in p.stem else '' for p in img_paths]

    ncol = len(imgs)
    fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4), constrained_layout=True)
    if ncol == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle(out_path.stem, fontsize=10)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    target_a = collect_images(args.root_a, args.target_suffix)
    pred_a = collect_images(args.root_a, args.pred_suffix)
    pred_b = collect_images(args.root_b, args.pred_suffix)
    pred_c = collect_images(args.root_c, args.pred_suffix)

    shared_keys = sorted(set(target_a) & set(pred_a) & set(pred_b) & set(pred_c))
    if not shared_keys:
        raise SystemExit(
            "No shared samples found across: target(root-a), pred(root-a), pred(root-b), pred(root-c)."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for key in shared_keys:
        target_type, sample_id = key.split("/", 1)
        out_subdir = args.out_dir / target_type
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_path = out_subdir / f"{sample_id}.png"
        save_row_with_titles(
            [target_a[key], pred_a[key], pred_b[key], pred_c[key]],
            out_path,
        )
        n_written += 1

    print(f"wrote {n_written} concatenated rows to: {args.out_dir}")


if __name__ == "__main__":
    main()
