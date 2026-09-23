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
    # v1.1 pair-focus policy. Off by default for older experiment configs.
    ab_ag_interface_only: bool = False
    prefer_nonprotein_focus: bool = False


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    max_msa_depth: int = 512
    missing_policy: Literal["gap", "query"] = "gap"
    pairing_mode: Literal["mixed", "paired_only", "no_pairing"] = "mixed"

    # Per-item MSA depth policy applied inside sample_msa():
    #   "uniform" (default): k ~ Uniform[1, min(n_available, max_msa_depth)], then the FIRST
    #                        k rows (best-first prefix). Rows past max_msa_depth are never seen.
    #   "fixed"            : always request max_msa_depth rows (legacy behavior).
    #   "af3"    (v1.2.0)  : k ~ Uniform[1, n_available] over the FULL stored depth (AF3 SI
    #                        2.2), then min(k, max_msa_depth) rows drawn at random per chain
    #                        (AF3 shuffles before it crops). max_msa_depth stays the row budget
    #                        the trunk is fed, so shapes/buckets are unchanged; what changes is
    #                        that deep alignments saturate the budget more often and every
    #                        stored row can be seen.
    sample_depth: Literal["uniform", "fixed", "af3"] = "uniform"
    # v1.2.0: override the policy per DataRecord.source. Unlisted sources use
    # ``sample_depth``. The v1.2 configs set {"pdb": "af3"}: the PDB a3m store holds up to
    # 16k rows (88% of entries exceed 2048), while the distillation stores are capped at
    # 2048 rows anyway, so they keep "uniform".
    sample_depth_by_source: dict[str, Literal["uniform", "fixed", "af3"]] = Field(
        default_factory=dict,
    )

    # v1.2.0: per-source POOL size handed to the trunk (rows the model may draw its
    # per-recycle subset from). Unlisted sources use ``max_msa_depth``. The v1.2 configs
    # set {"pdb": 8192}: the PDB store holds up to 16k rows and our no_pairing stack is
    # the unpaired MSA, whose AF3 budget is 16384 - 8191 paired ~ 8192; the distillation
    # stores hold at most 2048 rows, so a larger pool would only be padding.
    # ``train.bucket_msa_multiple`` must cover the largest pool.
    max_msa_depth_by_source: dict[str, int] = Field(default_factory=dict)

    def depth_for(self, source: str | None) -> int:
        """Pool size for a record of ``source`` (falls back to ``max_msa_depth``)."""
        if source is None:
            return self.max_msa_depth
        return self.max_msa_depth_by_source.get(source, self.max_msa_depth)

    def policy_for(self, source: str | None) -> Literal["uniform", "fixed", "af3"]:
        """Depth policy for a record of ``source`` (falls back to ``sample_depth``)."""
        if source is None:
            return self.sample_depth
        return self.sample_depth_by_source.get(source, self.sample_depth)


class TemplateConfig(BaseModel):
    """Configuration for template sampling."""

    n_templates: int = 4
    mask_interchain: bool = True


class BioMolDBConfig(BaseModel):
    """Configuration for BioMolDB paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    # Optional extra MSA shard tried after a3m_db_path (e.g. PDB RNA-chain MSA,
    # keyed by the same full seq_id). A chain's seq_id is looked up across both,
    # so protein chains hit a3m and RNA chains hit this one.
    a3m_rna_db_path: Path | None = None
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


class TokenEmbeddingConfig(BaseModel):
    """Configuration for token embedding."""

    embedding_path: Path
