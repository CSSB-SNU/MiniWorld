"""Pydantic schema for the inference-time YAML spec."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


_CONTACT_RE = re.compile(
    r"^([A-Za-z0-9]+):(\d+)(?:#(\d+))?-([A-Za-z0-9]+):(\d+)(?:#(\d+))?$",
)


class ContactsSpec(BaseModel):
    """User-provided positive / negative contact pairs.

    Each entry is a string of the form
    ``"<chain_a>:<res_a>[#<tok_a>]-<chain_b>:<res_b>[#<tok_b>]"`` where
    ``chain_*`` are the chain letters declared via ``Chain:<X>`` in the
    fasta headers and ``res_*`` are 1-based residue indices within those
    chains (matching the position in the fasta sequence).

    The optional ``#<tok>`` suffix is the 0-based local token index inside
    that residue, used when the residue tokenizes to more than one token
    (atomized / fragmented). Omitting it (the common case) means "first
    token of the residue" — the default anchor for residue-level tokens.
    """

    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)

    @staticmethod
    def parse_pair(s: str) -> tuple[str, int, int | None, str, int, int | None]:
        """Parse a contact string into ``(chain_a, res_a, tok_a, chain_b, res_b, tok_b)``.

        ``tok_*`` is ``None`` when the ``#<tok>`` suffix is absent.
        """
        m = _CONTACT_RE.match(s.strip())
        if m is None:
            msg = (
                f"Invalid contact pair {s!r}; expected "
                "'<chain>:<res>[#<tok>]-<chain>:<res>[#<tok>]'."
            )
            raise ValueError(msg)
        chain_a, res_a, tok_a, chain_b, res_b, tok_b = m.groups()
        return (
            chain_a,
            int(res_a),
            int(tok_a) if tok_a is not None else None,
            chain_b,
            int(res_b),
            int(tok_b) if tok_b is not None else None,
        )


class ComplexTemplateSpec(BaseModel):
    """One multi-chain (complex) template entry.

    Provide **either** a CIF file path (``cif``) **or** an LMDB cif identifier
    (``cif_id``, e.g. ``"5XYZ_1_1_."``) — the latter is looked up in
    ``InferenceSpec.cif_db``.

    ``chain_map`` keys are **query chain indices** (the same numeric keys
    used in ``InferenceSpec.fasta`` / ``InferenceSpec.a3m``, written as
    strings ``"0"``, ``"1"``, ... — never letters). Values are the
    BioMolDB chain id of the chain in the source structure, with the
    format ``"<label_asym_id>_<operator_id>"`` (e.g. ``"A_1"``, ``"C_2"``).

    * For ``cif`` (Option A) the operator suffix is stripped internally, so
      the value still resolves to ``label_asym_id`` in the raw mmCIF file.
    * For ``cif_id`` (Option B) the value is the full BioMolDB chain id
      stored in ``cifmol.chains.chain_id`` — direct match.

    Both cases resolve to the cif's chain identifier, **not** the
    ``auth_asym_id``. The numeric-key requirement makes this unambiguous
    even when multiple query chains share the same human-readable letter
    via :attr:`InferenceSpec.chain_letters`.
    """

    cif: Path | None = None
    cif_id: str | None = None
    chain_map: dict[str, str]  # query_chain_INDEX (str of int) -> template cif chain id

    @field_validator("chain_map", mode="before")
    @classmethod
    def _coerce_int_keys_chain_map(cls, v: object) -> object:
        return _coerce_int_keys(v)

    @model_validator(mode="after")
    def _check_source(self) -> "ComplexTemplateSpec":
        if (self.cif is None) == (self.cif_id is None):
            msg = (
                "ComplexTemplateSpec must set exactly one of `cif` (file path) "
                "or `cif_id` (LMDB key); got both/neither."
            )
            raise ValueError(msg)
        if not self.chain_map:
            msg = "ComplexTemplateSpec.chain_map must list at least one chain."
            raise ValueError(msg)
        return self


class InferenceSpec(BaseModel):
    """YAML schema for inference inputs.

    Fields:
      name: optional run identifier (used for output filenames). Defaults to the spec stem.
      chain_letters: required ``{chain_index_str: letter}`` mapping. This is
          the **single source of truth** for the chain set: the model's
          chain order is ``sorted(int(k) for k in chain_letters)``. Letters
          are human-readable labels and **may repeat** (e.g.
          ``{"0": "a", "1": "a", "2": "b"}`` for a homo-dimer + monomer);
          chains sharing a letter share the same fasta / a3m entry.
          Used by :attr:`contacts` and :attr:`combine_groups`.
      fasta: ``{chain_letter: fasta_path}``. One chain per file. Keys are
          chain **letters** (matching values of :attr:`chain_letters`), so
          homo-mer copies share a single fasta entry. Every distinct letter
          in :attr:`chain_letters` must be present here. Headers may carry
          ``"> name | <entity_type> | Chain:<X>"`` but the ``Chain:<X>``
          field is ignored — :attr:`chain_letters` is authoritative.
      ccd_db: path to the preprocessed CCD LMDB (``CCD/preprocessed_CCD.lmdb``).
      a3m: ``{chain_letter: a3m_file_path}``. Partial mapping is OK; letters
          without an a3m entry fall back to ``MSA.from_query`` (single-row
          MSA). Keys must be a subset of ``chain_letters`` values.
      template_db: optional path to template LMDB (built by StructCooker;
          keyed by **seq_id**). ``None`` -> no templates.
      template: ``{chain_index_str: seq_id}`` mapping. Each value is the
          LMDB key for that chain's template entry; the loader fetches up
          to ``template_n`` matching ``TemplateMol`` records and stacks
          them into ``ProteinTemplate``. Chains absent from this dict get
          an empty template slot. Keyed by **chain index** (not letter)
          so homo-mer copies can in principle take different templates.
      template_n: number of templates to sample per chain (default 4,
          matches the training dataloader).
      cif_db: optional path to a BioMolDB cif LMDB (the same lookup used by
          the dataloader). Only required when at least one
          ``complex_templates`` entry uses ``cif_id`` (Option B).
      complex_templates: list of multi-chain templates. Each entry shares a
          template slot across the chains listed in ``chain_map``, preserving
          their relative coordinates as a single rigid frame. ``chain_map``
          keys are query **chain indices** (numeric, unique).
      combine_groups: optional list of **numeric chain-index** groups that
          share an SE(3) frame in the diffusion solver. Example:
          ``[[0, 1], [2, 3, 4]]`` makes chains 0+1 move as one rigid body
          and 2+3+4 as another. Chains not listed each get their own
          singleton group. Only the solver's per-chain RT step is
          affected; chain-aware model embeddings and the output CIF still
          use the underlying per-chain ids.
      contacts: optional positive / negative contact pairs. May be given as
          a path to a YAML file (``{positive: [...], negative: [...]}``) or
          inline as the same mapping. Missing / None resolves to empty
          (no constraints).
      template_as_contact: when True, derive inter-chain contacts from
          ``complex_templates`` (CB-CB < 8 Å) and merge them into the
          ``positive`` set used by the model. Useful when you want the
          template's interface geometry to act as a soft constraint
          without writing the contact pairs by hand.
      tokenization: optional path to a per-residue resolution JSON file. See
          ``miniworld.data.inference.tokenization`` for the format. ``None``
          (default) means residue-level for every residue.
      n_trunk_samples: best-of-N axis 1 — number of fresh MSA samples; for
          each, the trunk (msa_module + pairformer + n_recycle) is rerun to
          produce a distinct conditioning. Default 1.
      n_diffusion_samples: best-of-N axis 2 — number of diffusion seeds per
          trunk conditioning. These are processed along the model's
          augmentation axis (``x_t: A B L 3`` with A>=1), not the batch
          axis. Default 1. Total output structures = ``n_trunk_samples *
          n_diffusion_samples``.
      diffusion_batch_size: chunk size along the augmentation axis (how many
          diffusion samples to run in one forward). Used to bound peak GPU
          memory when ``n_diffusion_samples`` is large; the script splits
          the N samples into ``ceil(N / diffusion_batch_size)`` chunks and
          concatenates the results. Clamped to ``n_diffusion_samples`` if
          larger. Default 1.
      save_trajectory: when True (default), write per-step trajectory CIFs
          (``x0hat`` / ``xt`` / ``x_with_noise``) for every produced
          structure. Set to False to skip trajectory I/O when only the
          final predicted structure is needed.
    """

    name: str | None = None
    chain_letters: dict[str, str]
    fasta: dict[str, Path]
    ccd_db: Path
    a3m: dict[str, Path] = Field(default_factory=dict)
    template_db: Path | None = None
    template: dict[str, str] = Field(default_factory=dict)
    template_n: int = 4
    cif_db: Path | None = None
    complex_templates: list[ComplexTemplateSpec] = Field(default_factory=list)
    combine_groups: list[list[int]] = Field(default_factory=list)
    contacts: ContactsSpec = Field(default_factory=ContactsSpec)
    template_as_contact: bool = False

    @field_validator("contacts", mode="before")
    @classmethod
    def _load_contacts(cls, v: object) -> object:
        """Allow ``contacts`` to be given as a path to a YAML file."""
        if v is None:
            return {}
        if isinstance(v, (str, Path)):
            with Path(v).open("r") as f:
                return yaml.safe_load(f) or {}
        return v
    tokenization: Path | None = None  # per-residue resolution JSON; None -> all residue-level
    n_trunk_samples: int = Field(default=1, ge=1)
    n_diffusion_samples: int = Field(default=1, ge=1)
    diffusion_batch_size: int = Field(default=1, ge=1)
    save_trajectory: bool = True

    @field_validator("chain_letters", mode="before")
    @classmethod
    def _coerce_int_keys_chain_letters(cls, v: object) -> object:
        return _coerce_int_keys(v)

    @field_validator("template", mode="before")
    @classmethod
    def _coerce_int_keys_template(cls, v: object) -> object:
        return _coerce_int_keys(v)

    @model_validator(mode="after")
    def _check_letter_keyed_inputs(self) -> "InferenceSpec":
        if not self.chain_letters:
            msg = "InferenceSpec.chain_letters must list at least one chain."
            raise ValueError(msg)
        letters = set(self.chain_letters.values())
        missing_fasta = letters - self.fasta.keys()
        if missing_fasta:
            msg = (
                f"InferenceSpec.fasta is missing entries for chain letters "
                f"{sorted(missing_fasta)} (declared in chain_letters)."
            )
            raise ValueError(msg)
        extra_fasta = self.fasta.keys() - letters
        if extra_fasta:
            msg = (
                f"InferenceSpec.fasta has entries {sorted(extra_fasta)} "
                f"that don't appear as a chain letter in chain_letters."
            )
            raise ValueError(msg)
        extra_a3m = self.a3m.keys() - letters
        if extra_a3m:
            msg = (
                f"InferenceSpec.a3m has entries {sorted(extra_a3m)} "
                f"that don't appear as a chain letter in chain_letters."
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        sampling_path: Path | None = None,
    ) -> "InferenceSpec":
        """Load the spec from a data YAML, optionally overlaying a sampling YAML.

        The data YAML holds target-bound fields (chain_letters, fasta, a3m,
        ccd_db, contacts, combine_groups, tokenization, templates, ...). The
        optional sampling YAML holds per-attempt sampling knobs
        (n_trunk_samples, n_diffusion_samples, diffusion_batch_size,
        save_trajectory). Keys in the sampling YAML overwrite the data YAML.
        """
        with Path(path).open("r") as f:
            data = yaml.safe_load(f) or {}
        if sampling_path is not None:
            with Path(sampling_path).open("r") as f:
                sampling = yaml.safe_load(f) or {}
            data.update(sampling)
        spec = cls.model_validate(data)
        if spec.name is None:
            spec.name = Path(path).stem
        return spec

    def chain_indices(self) -> list[int]:
        """Return chain indices in ascending order (the model's chain order)."""
        return sorted(int(k) for k in self.chain_letters)


def _coerce_int_keys(v: object) -> object:
    """Accept string-int keys ('0', '1') or already-int keys."""
    if not isinstance(v, dict):
        return v
    out: dict[str, object] = {}
    for k, val in v.items():
        key = str(k)
        if not key.lstrip("-").isdigit():
            msg = f"Chain key must be an integer, got {k!r}."
            raise ValueError(msg)
        out[key] = val
    return out
