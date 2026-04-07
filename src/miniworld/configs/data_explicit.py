from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


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
    antibody_antibody: float = 0.1  # 3096, 0.3%, 0.1
    antibody_nucleic_acid: float = 20.0  # 27+30=57, 1.12%, 19.7
    antibody_protein: float = 3.0  # 5091, 15%, 2.95

    DNA_DNA: float = 0.25  # 8045, 1.98%, 0.25
    DNA_RNA: float = 5.0  # 462, 2.27%, 4.92
    RNA_RNA: float = 0.5  # 3983, 1.96%, 0.49
    NA_NA: float = 10.0  # 40+43+41=124, 1.22%, 9.84

    protein_nucleic_acid: float = 0.5  # 27386 + 9608 + 165= 36994, 18.2%, 0.49
    protein_protein: float = 1.0  # 43524, 42.8%, 0.98
    protein_ligand: float = 0.1  # 133547, 13.1%, 0.1

    ligand_ligand: float = 0.01  # 35804, 0.35%, 0.01

    etc_interface: float = 0.05  # 5741+6201+4890+197+75+17+57=17178, 0.8%, 0.05
    sole: float = 0.5  # 1462, 0.7%, 0.5


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


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    max_msa_depth: int = 512
    missing_policy: Literal["gap", "query"] = "gap"


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
    fingerprint_vocab_path: Path | None = None


class TokenizerConfig(BaseModel):
    """Configuration for the Tokenizer."""

    level: Literal["atom", "dynamic", "lte", "residue"] = "atom"
    seed: int = 42  # for dynamic tokenization, set seed for reproducibility


class TokenEmbeddingConfig(BaseModel):
    """Configuration for token embedding."""

    embedding_path: Path
    vocab_path: Path | None = None
