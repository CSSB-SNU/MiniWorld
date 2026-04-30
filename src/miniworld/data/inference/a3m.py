"""Read a standard a3m text file into a ``miniworld.data.pipeline.MSA``.

The parsing logic mirrors StructCooker's ``a3m_instructions`` module
(`pipelines/instructions/a3m_instructions.py` in
https://github.com/SanggeunParrk/StructCooker), which is the upstream pipeline
that builds the LMDB-stored MSAs MiniWorld consumes. Keeping the two parsers
aligned ensures inference-path inputs use the same conventions as the training
pipeline:

- Sequences: lowercase letters denote insertions and are stripped from the
  aligned columns; the count of insertions at each aligned position is stored
  separately in ``deletions``.
- Headers: three formats are recognised — UniRef-style, UniProt pipe-delimited
  (``OS=...``), and a BFD fallback. Species defaults to ``"N/A"`` when none
  applies.

a3m format reminder:
- The first record is the query sequence (uppercase + ``-`` placeholders).
- Subsequent records align to the query: uppercase / ``-`` characters occupy
  aligned columns; lowercase characters are *insertions* relative to the query
  and do not consume a column.
- Aligned length L = len(uppercase + '-' chars in the query).
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Literal

import numpy as np

from miniworld.data.constants import ResidueMapping
from miniworld.data.pipeline import MSA


PolymerType = Literal["protein", "rna", "dna", "na"]


# ---------------------------------------------------------------------------
# Header parsing — three regex patterns ported from StructCooker
# (pipelines/instructions/a3m_instructions.py: parse_headers).
# ---------------------------------------------------------------------------

# UniRef-style header. Example:
#   >UniRef100_W5NM83 G_PROTEIN_RECEP_F1_2 ... Tax=Lepisosteus oculatus TaxID=7918 RepID=W5NM83_LEPOC
_UNIREF_RE = re.compile(
    r"^(?P<db_name>UniRef\d+)_"
    r"(?P<db_id>\S+).*?Tax=(?P<species>.*?)\s+TaxID=\S+\s+RepID=(?P<rep_id>\S+)",
    re.IGNORECASE,
)

# Pipe-delimited UniProt header. Example:
#   >tr|A0A060WKI3|A0A060WKI3_ONCMY Uncharacterized protein OS=Oncorhynchus mykiss GN=... PE=3 SV=1
_UNIPROT_RE = re.compile(
    r"^(?P<db_name>[^|]+)\|"
    r"(?P<db_id>[^|]+)\|"
    r"(?P<rep_id>[^|]+)\s+.*?OS=(?P<species>.*?)\s+(?=GN=|PE=|SV=)",
    re.IGNORECASE,
)


def parse_header(header: str) -> dict[str, str]:
    """Parse one a3m header line (without the leading '>').

    Returns a dict with keys ``db_name``, ``db_id``, ``species``, ``rep_id``.
    Falls back to a BFD-style record (species="N/A") when neither UniRef nor
    UniProt-pipe patterns match.
    """
    for pattern in (_UNIREF_RE, _UNIPROT_RE):
        match = pattern.search(header)
        if match:
            result = match.groupdict()
            if not result.get("species"):
                result["species"] = "N/A"
            if not result.get("rep_id"):
                result["rep_id"] = "N/A"
            return {
                "db_name": result.get("db_name", "N/A").lower(),
                "db_id": result.get("db_id", "N/A"),
                "species": result.get("species", "N/A"),
                "rep_id": result.get("rep_id", "N/A"),
            }
    # Pattern 3: BFD fallback — no species info recoverable.
    db_id = header.split()[0] if header else "N/A"
    return {
        "db_name": "bfd",
        "db_id": db_id,
        "species": "N/A",
        "rep_id": db_id,
    }


def _parse_headers(headers: list[str]) -> dict[str, np.ndarray]:
    """StructCooker-style header parser; row 0 is treated as the query."""
    database: list[str] = []
    database_id: list[str] = []
    species: list[str] = []
    rep_id: list[str] = []
    for ii, header in enumerate(headers):
        if ii == 0:
            database.append("query")
            database_id.append("query")
            species.append("query")
            rep_id.append("query")
            continue
        parsed = parse_header(header)
        database.append(parsed["db_name"])
        database_id.append(parsed["db_id"])
        species.append(parsed["species"])
        rep_id.append(parsed["rep_id"])
    return {
        "database": np.array(database, dtype=object),
        "database_id": np.array(database_id, dtype=object),
        "species": np.array(species, dtype=object),
        "rep_id": np.array(rep_id, dtype=object),
    }


# ---------------------------------------------------------------------------
# Sequence parsing — also ported from StructCooker (parse_sequence). The
# original strips lowercase via str.translate and computes deletion counts
# from positional offsets.
# ---------------------------------------------------------------------------


_LOWER_TABLE = str.maketrans(dict.fromkeys(string.ascii_lowercase))


def _parse_sequences(
    raw_sequences: list[str],
    polymer: PolymerType,
) -> dict[str, np.ndarray]:
    rm = ResidueMapping()
    max_idx = rm.MAX_INDEX
    if polymer == "protein":
        view = rm.protein
    elif polymer == "rna":
        view = rm.rna
    elif polymer == "dna":
        view = rm.dna
    elif polymer == "na":
        view = rm.na
    else:  # pragma: no cover
        msg = f"Unsupported polymer type: {polymer!r}"
        raise ValueError(msg)

    if not raw_sequences:
        msg = "a3m has no sequence records."
        raise ValueError(msg)
    query_raw = raw_sequences[0]
    # Aligned length = number of uppercase + '-' chars in the query.
    aligned_len = sum(1 for c in query_raw if c.isupper() or c == "-")

    sequences: list[np.ndarray] = []
    deletions: list[np.ndarray] = []
    for raw in raw_sequences:
        # 1 where the char is lowercase (= insertion), 0 elsewhere.
        flags = np.array(
            [0 if c.isupper() or c == "-" else 1 for c in raw],
            dtype=np.int64,
        )
        del_counts = np.zeros(aligned_len, dtype=np.uint8)
        if flags.sum() > 0:
            # Positions of lowercase chars *in the original (insertion-included) string*.
            pos = np.where(flags == 1)[0]
            # Convert to positions in the aligned-only string: shift each by its
            # rank among preceding insertions.
            shifted = pos - np.arange(pos.shape[0])
            uniq, counts = np.unique(shifted, return_counts=True)
            uniq_clamped = np.clip(uniq, 0, aligned_len - 1)
            del_counts[uniq_clamped] = np.clip(counts, 0, 255).astype(np.uint8)
        cleaned = raw.translate(_LOWER_TABLE)
        if len(cleaned) != aligned_len:
            msg = (
                f"a3m sequence has cleaned length {len(cleaned)} != "
                f"query aligned length {aligned_len}."
            )
            raise ValueError(msg)
        encoded = view.map(np.array(list(cleaned)))
        sequences.append(encoded)
        deletions.append(del_counts)

    aligned_sequences = np.stack(sequences).astype(np.int32)
    deletions_arr = np.stack(deletions).astype(np.uint8)
    deletion_mean = (
        2 * np.arctan(deletions_arr.astype(np.float32) / 3) / np.pi
    ).mean(axis=0).astype(np.float32)
    profile = np.eye(max_idx + 1, dtype=np.int32)[aligned_sequences]
    profile = profile.mean(axis=0).astype(np.float32)

    query_chars = np.array(list(query_raw.translate(_LOWER_TABLE)))
    return {
        "query_sequence": query_chars,
        "aligned_sequences": aligned_sequences,
        "deletions": deletions_arr,
        "deletion_mean": deletion_mean,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_a3m_file(path: Path, polymer: PolymerType = "protein") -> MSA:
    """Parse an a3m text file into an ``MSA`` for the given polymer."""
    records = _read_records(Path(path))
    if not records:
        msg = f"a3m file {path} contains no records."
        raise ValueError(msg)
    headers = [h for h, _ in records]
    raw_sequences = [s for _, s in records]
    sequences = _parse_sequences(raw_sequences, polymer)
    parsed_headers = _parse_headers(headers)
    return MSA(
        seq_id=None,
        sequences=sequences,
        headers={"species": parsed_headers["species"]},
    )


def _read_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    cur_header: str | None = None
    cur_lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_header is not None:
                records.append((cur_header, "".join(cur_lines)))
            cur_header = line[1:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_header is not None:
        records.append((cur_header, "".join(cur_lines)))
    return records
