"""Compatibility test for a new CIF LMDB against the MiniWorld dataloader.

Checks whether a freshly built CIF LMDB (e.g. the OpenFold distillation set)
can be consumed by the MiniWorld dataloader pipeline. It exercises every stage
that ``BioMolData.get_item_by_id`` relies on and reports, per stage, whether the
new dataset is compatible.

Run directly:

    pixi run python tests/test_new_dataset_compat.py \
        --db /home/psk6950/data/openfold_distillation/cif_short.lmdb

Or via pytest (uses the default path below):

    pixi run pytest tests/test_new_dataset_compat.py -s
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
from biomol.core.utils import load_bytes

from miniworld.configs.data import (
    DynamicTokenizationConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader.dataloader import FragmentedCCDMolCache
from miniworld.data.features import make_batch
from miniworld.data.io import extract_lmdb_keys, load_msa, load_raw_data, load_templates
from miniworld.data.mols.cifmol_attached import CIFMolAttached
from miniworld.data.pipeline import Tokenizer
from miniworld.data.pipeline.utils import remove_terminal_oxygen
from miniworld.utils.crop import crop_spatial_segment_token

DEFAULT_DB = Path("/home/psk6950/data/openfold_distillation/cif_short.lmdb")
CCD_PATH = Path("/home/psk6950/data/CCD/preprocessed_CCD.lmdb")
A3M_PATH = Path("/home/psk6950/data/BioMolDB_20260224/a3m_16k.lmdb")
TEMPLATE_PATH = Path("/home/psk6950/data/BioMolDB_20260224/template.lmdb")

# How load_cifmol() expects the per-pdb value to be structured.
EXPECTED_OUTER = "{assembly_id}_{model_id}_{alt_id}"
EXPECTED_INNER = "cifmol_attached_dict"


class Reporter:
    """Collects PASS/FAIL results for each compatibility stage."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f"\n        {detail}"
        print(line)
        return ok

    def summary(self) -> bool:
        passed = sum(1 for _, ok, _ in self.rows if ok)
        print("\n" + "=" * 64)
        print(f"SUMMARY: {passed}/{len(self.rows)} checks passed")
        print("=" * 64)
        return passed == len(self.rows)


def _first_key_and_value(db_path: Path) -> tuple[str, dict]:
    keys = extract_lmdb_keys(db_path)
    key = keys[0]
    raw = load_raw_data(key, db_path)
    return key, load_bytes(raw)


def run_compat(db_path: Path) -> bool:  # noqa: C901, PLR0915
    """Run all compatibility checks against ``db_path`` and return overall pass."""
    print(f"\nTesting dataset: {db_path}\n" + "-" * 64)
    r = Reporter()
    rng = np.random.default_rng(0)

    # ---- 1. LMDB opens and has entries -------------------------------------
    try:
        keys = extract_lmdb_keys(db_path)
        r.check("lmdb opens & has keys", len(keys) > 0, f"{len(keys)} keys, e.g. {keys[:3]}")
    except Exception as exc:  # noqa: BLE001
        r.check("lmdb opens & has keys", False, repr(exc))
        return r.summary()

    key = keys[0]
    value = load_bytes(load_raw_data(key, db_path))
    outer_keys = list(value.keys())

    # ---- 2. Value layout matches load_cifmol() expectations ----------------
    # load_cifmol does: value.get(f"{assembly_id}_{model_id}_{alt_id}")["cifmol_attached_dict"]
    has_inner = any(
        isinstance(v, dict) and EXPECTED_INNER in v for v in value.values()
    )
    r.check(
        "value wrapped as {assembly}_{model}_{alt} -> 'cifmol_attached_dict'",
        has_inner,
        f"expected outer like '{EXPECTED_OUTER}' with inner '{EXPECTED_INNER}', "
        f"got outer keys {outer_keys}",
    )

    # ---- 3. CIFMolAttached parses ------------------------------------------
    # Locate the cifmol dict regardless of wrapping so later stages can run.
    cifmol_dict = None
    if has_inner:
        cifmol_dict = next(
            v[EXPECTED_INNER] for v in value.values() if isinstance(v, dict) and EXPECTED_INNER in v
        )
    elif "cifmol_dict" in value:
        cifmol_dict = value["cifmol_dict"]
    elif "cifmol_attached_dict" in value:
        cifmol_dict = value["cifmol_attached_dict"]

    try:
        cifmol = CIFMolAttached.from_dict(cifmol_dict)
        r.check(
            "CIFMolAttached.from_dict parses",
            True,
            f"{len(cifmol.chains)} chains, {len(cifmol.residues)} residues, "
            f"{len(cifmol.atoms)} atoms",
        )
    except Exception as exc:  # noqa: BLE001
        r.check("CIFMolAttached.from_dict parses", False, repr(exc))
        return r.summary()

    # ---- 4. Required chain features present --------------------------------
    chain_feats = list(cifmol.chains.mol.get_container("chain").keys())
    for feat in ("chain_id", "entity_type"):
        r.check(f"chain feature '{feat}' present", feat in chain_feats)
    # seq_id is required by load_msa / load_templates / get_query_sequence.
    r.check(
        "chain feature 'seq_id' present (needed for MSA & templates)",
        "seq_id" in chain_feats,
        f"chain features: {chain_feats}",
    )

    # ---- 5. Metadata present (assembly/model/alt ids) ----------------------
    md = cifmol_dict.get("metadata", {}) if isinstance(cifmol_dict, dict) else {}
    r.check(
        "metadata carries assembly/model/alt ids",
        all(k in (md or {}) for k in ("assembly_id", "model_id", "alt_id")),
        f"metadata = {md}",
    )

    # ---- 6. Tokenization ---------------------------------------------------
    ccd_keys = extract_lmdb_keys(CCD_PATH)
    ccd_cache = FragmentedCCDMolCache(CCD_PATH, ccd_keys)
    tokenizer = Tokenizer(config=TokenizerConfig(level="atom"))
    dyn_cfg = DynamicTokenizationConfig()

    valid = np.all(np.isfinite(cifmol.atoms.xyz.value), axis=-1)
    focus = cifmol.atoms.xyz.value[valid][0]

    atom_to_token = token_to_res = None
    try:
        atom_to_token, token_to_res = tokenizer.tokenize(
            cifmol, focus=focus, fragmented_ccd_mols=ccd_cache, config=dyn_cfg,
        )
        r.check(
            "tokenize() succeeds",
            True,
            f"{len(np.unique(token_to_res))} residues -> {token_to_res.max() + 1} tokens, "
            f"{len(atom_to_token)} atoms",
        )
    except Exception as exc:  # noqa: BLE001
        r.check("tokenize() succeeds", False, repr(exc))

    # ---- 7. Cropping -------------------------------------------------------
    if atom_to_token is not None:
        try:
            crop_indices = crop_spatial_segment_token(
                cifmol, focus, tokens_to_res=token_to_res,
                segment_size=21, max_tokens=384, max_atoms=4096,
            )
            r.check("crop_spatial_segment_token() succeeds", crop_indices.shape[0] > 0,
                    f"{crop_indices.shape[0]} cropped residues")
        except Exception as exc:  # noqa: BLE001
            r.check("crop_spatial_segment_token() succeeds", False, repr(exc))

    # ---- 8. MSA loading (needs seq_id) -------------------------------------
    chain_ids = list(cifmol.chains.chain_id.value)
    chain_id_to_crop = {c: np.arange(0) for c in chain_ids}
    # give the first chain its full residue range
    first = chain_ids[0]
    n_res_first = int((cifmol.residues.chain_id.value == first).sum()) if "chain_id" in \
        list(cifmol.residues.mol.get_container("residue").keys()) else len(cifmol.residues)
    chain_id_to_crop[first] = np.arange(n_res_first)
    try:
        load_msa(cifmol=cifmol, chain_id_to_crop_indices=chain_id_to_crop, env_path=A3M_PATH)
        r.check("load_msa() succeeds", True)
    except Exception as exc:  # noqa: BLE001
        r.check("load_msa() succeeds", False, f"{type(exc).__name__}: {exc}")

    # ---- 9. Template loading (needs seq_id) --------------------------------
    try:
        load_templates(cifmol=cifmol, chain_id_to_crop_indices=chain_id_to_crop,
                       env_path=TEMPLATE_PATH, n_templates=4, rng=rng)
        r.check("load_templates() succeeds", True)
    except Exception as exc:  # noqa: BLE001
        r.check("load_templates() succeeds", False, f"{type(exc).__name__}: {exc}")

    return r.summary()


def test_new_dataset_compat() -> None:
    """Pytest entry point against the default distillation LMDB."""
    if not DEFAULT_DB.exists():
        import pytest

        pytest.skip(f"dataset not found: {DEFAULT_DB}")
    run_compat(DEFAULT_DB)  # report-only; does not assert so all stages are shown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        ok = run_compat(args.db)
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
    raise SystemExit(0 if ok else 1)
