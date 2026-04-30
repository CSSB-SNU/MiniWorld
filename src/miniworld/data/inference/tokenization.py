"""Per-residue tokenization policy for the inference path.

The user-facing JSON file format is::

    {
      "default": 1.0,
      "A:1": 1.0,
      "A:2": 0.5,
      "B:3": 0.0
    }

- Keys are either ``"default"`` (applied to any residue not explicitly listed)
  or ``"<chain_letter>:<residue_1based>"`` matching the contacts notation.
- Values are floats in ``[0, 1]`` with the same semantics as the dataloader's
  ``dynamic_tokenize`` resolution: ``1.0`` = residue-level (one token per
  residue), ``0.0`` = atomize (one token per atom). Intermediate values are
  mapped uniformly onto the discrete ``fragment_ccdmol_all_merges`` levels.

If no file is supplied, the default policy is residue-level for every residue.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


_KEY_RE = re.compile(r"^([A-Za-z0-9]+):(\d+)$")


@dataclass(frozen=True)
class TokenizationPolicy:
    """Per-residue resolution lookup with a fallback default."""

    default: float = 1.0
    per_residue: dict[tuple[str, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_resolution(self.default, "default")
        for (chain_letter, res_idx), v in self.per_residue.items():
            _validate_resolution(v, f"{chain_letter}:{res_idx}")
            if res_idx < 1:
                msg = f"residue index in {chain_letter}:{res_idx} must be >= 1."
                raise ValueError(msg)

    @classmethod
    def from_file(cls, path: Path) -> "TokenizationPolicy":
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            msg = f"tokenization file {path} must be a JSON object."
            raise ValueError(msg)
        default = float(data.get("default", 1.0))
        per_residue: dict[tuple[str, int], float] = {}
        for k, v in data.items():
            if k == "default":
                continue
            m = _KEY_RE.match(str(k))
            if m is None:
                msg = (
                    f"tokenization file {path}: key {k!r} must be 'default' or "
                    f"'<chain_letter>:<residue_1based>'."
                )
                raise ValueError(msg)
            chain_letter, res_str = m.group(1), m.group(2)
            per_residue[(chain_letter, int(res_str))] = float(v)
        return cls(default=default, per_residue=per_residue)

    def resolution(self, chain_letter: str, res_1based: int) -> float:
        return self.per_residue.get((chain_letter, res_1based), self.default)


def _validate_resolution(v: float, where: str) -> None:
    if not (0.0 <= v <= 1.0):
        msg = f"tokenization resolution at {where} must be in [0, 1]; got {v}."
        raise ValueError(msg)
