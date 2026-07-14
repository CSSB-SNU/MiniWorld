from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class SamplerConfig(BaseModel):
    """Configuration for sampling strategy.

    Statistics
        Antibody-Antibody,3096
        Antibody-DNA,27
        Antibody-RNA,30
        Antibody-Protein(L),5091

        DNA-DNA,8045
        DNA-RNA,462
        RNA-RNA,3983
        DNA-DNA/RNA hybrid,40
        DNA/RNA hybrid-DNA/RNA hybrid,43
        DNA/RNA hybrid-RNA,41

        Protein(L)-DNA,9608
        Protein(L)-RNA,27386
        Protein(L)-DNA/RNA hybrid,165
        Protein(L)-Protein(L),43524
        Protein(L)-Ligand,133547

        sole,1462

    # etc_interface:
        Antibody-Ligand,5741
        DNA-Ligand,6201
        RNA-Ligand,4890
        DNA/RNA hybrid-Ligand,197
        Ligand-Ligand,35804
        Ligand-Protein(D),75
        Protein(D)-Protein(D),17
        Protein(D)-Protein(L),57
    """

    # sample number, cluster prob, individual sample score (prob *1e5)
    protein_protein: float = 25.0
    protein_ligand: float = 25
    protein_dna: float = 10
    protein_rna: float = 10
    antibody_protein: float = 15
    dna_dna: float = 5
    rna_rna: float = 5
    dna_rna: float = 0.5
    antibody_antibody: float = 0.5
    antibody_ligand: float = 1.0
    na_ligand: float = 1.0
    etc_interface: float = 1.0
    sole: float = 1.0


class CropConfig(BaseModel):
    """Configuration for cropping strategy."""

    max_tokens: int = 384
    max_atoms: int = 4096
    min_segment_size: int = 1
    max_segment_size: int = 41

    monomer_only: bool = False
    remain_invalid_tokens: bool = False

    bucket_msa_size: int = 128
    bucket_token_size: int = 128
    bucket_atom_size: int = 1024

    chain_crop_prob: float = 0.5


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    max_msa_depth: int = 512
    missing_policy: Literal["gap", "query"] = "gap"
    pairing_mode: Literal["mixed", "paired_only", "no_pairing"] = "mixed"


class TemplateConfig(BaseModel):
    """Configuration for template sampling."""

    n_templates: int = 4
    mask_interchain: bool = True


class BioMolDBConfig(BaseModel):
    """Configuration for BioMolDB paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    template_db_path: Path = Path("template_lmdb")
    edge_id_to_bias_path: Path = Path("edge_id_to_cif_ids.tsv")
    load_all_msa: bool = False
    fingerprint_embedding_path: Path | None = None
    ccd_preprocessed_path: Path | None = None


class DynamicTokenizationConfig(BaseModel):
    """Configuration for dynamic tokenization."""

    minimum_resolution_ratio: list[float] = [
        0.2,
        0.6,
        0.2,
    ]  # [atom, token(0~M), residue]

    sigma_flat_prob: float = 0.3  # prob of sigma=inf (fully uniform tokenization)
    sigma_min: float = 4.0  # Å — lower bound of LogUniform sigma
    sigma_max: float = 8.0  # Å — upper bound; median = sqrt(4*8) ~ 6 Å


class TokenizerConfig(BaseModel):
    """Configuration for the Tokenizer."""

    level: Literal["atom", "dynamic", "lte", "residue"] = "atom"
    dynamic_config: DynamicTokenizationConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("dynamic_config", "dynamic_tokenization"),
    )
    seed: int = 42  # for dynamic tokenization, set seed for reproducibility
