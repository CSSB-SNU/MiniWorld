# ruff: noqa: S101
from pathlib import Path

import pytest

from miniworld.data.dataloader.dataloader_v2 import (
    BioMolDBV2Config,
    DataRecord,
    DistillationSourceConfig,
    _configured_source_weights,
    _source_balanced_weights,
)


def _record(source: str, item_id: str) -> DataRecord:
    return DataRecord(
        item_id=item_id,
        source=source,
        record_id=item_id,
        cif_db_path=Path(f"/{source}.lmdb"),
        assembly_id="1",
        model_id="1",
        alt_id=".",
        chain_ids=("1",),
        feature_keys=(item_id,),
        seq_ids=(item_id,),
        msa_db_paths=((Path(f"/{source}_msa.lmdb"),),),
        template_db_paths=(Path(f"/{source}_template.lmdb"),),
    )


def test_source_balanced_weights_keep_pdb_internal_distribution() -> None:
    """Source balancing preserves PDB internal type ratios."""
    records = [
        _record("pdb", "pdb-protein"),
        _record("pdb", "pdb-nucleic"),
        _record("dist_short", "short-1"),
        _record("dist_long", "long-1"),
        _record("dist_long", "long-2"),
    ]

    weights = _source_balanced_weights(
        records=records,
        raw_weights=[9.0, 1.0, 1.0, 5.0, 5.0],
        source_weights={"pdb": 0.5, "dist_short": 0.25, "dist_long": 0.25},
        default_source_weight=1.0,
    )

    assert sum(weights[:2]) == pytest.approx(0.5)
    assert weights[0] / weights[1] == pytest.approx(9.0)
    assert weights[2] == pytest.approx(0.25)
    assert sum(weights[3:]) == pytest.approx(0.25)
    assert weights[3] == pytest.approx(weights[4])


def test_configured_source_weights_use_distillation_weight_as_compat_default() -> None:
    """Distillation source weights remain compatibility source defaults."""
    config = BioMolDBV2Config(
        distillation_sources=[
            DistillationSourceConfig(
                name="dist_short",
                cif_db_path=Path("/short_cif.lmdb"),
                weight=0.2,
            ),
        ],
    )

    assert _configured_source_weights(config) == {"dist_short": 0.2}


def test_explicit_source_weights_override_compat_defaults() -> None:
    """Explicit source weights override compatibility defaults."""
    config = BioMolDBV2Config(
        source_weights={"dist_short": 0.7},
        distillation_sources=[
            DistillationSourceConfig(
                name="dist_short",
                cif_db_path=Path("/short_cif.lmdb"),
                weight=0.2,
            ),
        ],
    )

    assert _configured_source_weights(config) == {"dist_short": 0.7}
