from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class CropConfig(BaseModel):
    """Configuration for cropping strategy."""

    contiguous_prob: float = 0.5
    spatial_prob: float = 0.5
    interface_prob: float = 0.0
    interface_simple_prob: float = 0.0
    residue_crop_length: int = 384
    atom_crop_length: int = 4096
    monomer_only: bool = False
    remain_invalid_residues: bool = False


class EdgeWeightConfig(BaseModel):
    """Configuration for edge weighting scheme."""

    PP_edge: float = 1.0  # protein-protein edge
    PN_edge: float = 1.0  # protein-nucleic acid edge
    PL_edge: float = 2 / 3  # protein-ligand edge
    NN_edge: float = 1.0  # nucleic acid-nucleic acid edge
    NL_edge: float = 2 / 3  # nucleic acid-ligand edge
    LL_edge: float = 0.0  # ligand-ligand edge

    # params
    eta: float = 0.9
    decay: float = 0.999
    temperature: float = 1.0
    init_score: float = 1.0
    init_freq: float = 0.0
    device: str = "cpu"
    use_freq: bool = True


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    n_samples: int = 4
    max_msa_depth: int = 512


class MultistateConfig(BaseModel):
    """Configuration for multistate modeling."""

    n_prefilter: int = 128
    n_samples: int = 48
    temperatures: float = 1.0
    consensus_ratio: float = 0.9
    consensus_filter: float = 0.9


class MultiStateDBConfig(BaseModel):
    """Configuration for BioMoldb paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    cluster_ids_path: Path = Path("cluster_ids.txt")
    cluster_id_to_seq_ids_path: Path = Path("cluster_id_to_seq_ids.npz")
    seq_id_to_seq: Path = Path("protein.fasta")
    tmp_dir: Path = Path("./data_tmp/monomer/")
    load_all_msa: bool = False


class BioMolDBConfig(BaseModel):
    """Configuration for BioMolDB paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    edge_id_to_cif_ids_path: Path = Path("edge_id_to_cif_ids.tsv")
    load_all_msa: bool = False
