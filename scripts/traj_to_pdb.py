"""Combine a directory of CIF trajectory frames into a multi-MODEL PDB.

Use this to inspect a diffusion trajectory in PyMOL as an animation:

    python scripts/traj_to_pdb.py \\
        "outputs/.../['2D94']_1_1_._traj/x_with_noise" \\
        /tmp/2d94_x_with_noise.pdb

    pymol /tmp/2d94_x_with_noise.pdb
    # in PyMOL:
    PyMOL> mplay         # play the trajectory
    PyMOL> mset 1 -100   # if you need to reset the movie range

Each ``stepNNN_*.cif`` becomes one ``MODEL`` block, sorted by step number.

Early diffusion steps can have coordinates with |x| > 9999 that overflow the
PDB ``F8.3`` column. We still emit them; PyMOL is lenient about over-wide
ATOM coordinate fields. Use ``--min-step`` to skip the noisiest frames if
another tool chokes.
"""

from __future__ import annotations

import re
from pathlib import Path

import click

_STEP_RE = re.compile(r"step(\d+)")

# Column order of the atom_site loop in the trajectory CIFs produced by this
# project (see e.g. step000_*.cif). 21 fields, whitespace-separated.
_COL_GROUP_PDB = 0
_COL_ID = 1
_COL_TYPE_SYMBOL = 2
_COL_LABEL_ATOM_ID = 3
_COL_LABEL_COMP_ID = 5
_COL_LABEL_ASYM_ID = 6
_COL_CARTN_X = 10
_COL_CARTN_Y = 11
_COL_CARTN_Z = 12
_COL_OCC = 13
_COL_B = 14
_COL_AUTH_SEQ_ID = 16


def _step_num(path: Path) -> int:
    m = _STEP_RE.search(path.stem)
    return int(m.group(1)) if m else 10**9


def _parse_atom_site(cif_path: Path) -> list[tuple]:
    """Return a list of atom rows from the atom_site loop of ``cif_path``.

    Each row is ``(record, serial, elem, name, resname, chain, resseq, x, y,
    z, occ, b)``. The trajectory CIFs only contain a single atom_site loop so
    the parser is intentionally minimal.
    """
    atoms: list[tuple] = []
    in_atom_loop = False
    saw_data_row = False
    with cif_path.open() as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                if saw_data_row:
                    break
                continue
            if s.startswith("loop_"):
                in_atom_loop = False
                saw_data_row = False
                continue
            if s.startswith("_atom_site."):
                in_atom_loop = True
                continue
            if s.startswith("_") or s.startswith("data_"):
                if saw_data_row:
                    break
                continue
            if not in_atom_loop:
                continue
            toks = s.split()
            if len(toks) < 17 or toks[_COL_GROUP_PDB] not in ("ATOM", "HETATM"):
                break
            atoms.append(
                (
                    toks[_COL_GROUP_PDB],
                    int(toks[_COL_ID]),
                    toks[_COL_TYPE_SYMBOL],
                    toks[_COL_LABEL_ATOM_ID],
                    toks[_COL_LABEL_COMP_ID],
                    toks[_COL_LABEL_ASYM_ID],
                    int(toks[_COL_AUTH_SEQ_ID]),
                    float(toks[_COL_CARTN_X]),
                    float(toks[_COL_CARTN_Y]),
                    float(toks[_COL_CARTN_Z]),
                    float(toks[_COL_OCC]),
                    float(toks[_COL_B]),
                )
            )
            saw_data_row = True
    return atoms


def _chain_char(chain: str) -> str:
    # PDB chain ID is a single character. Numeric label_asym_ids ("0","1",...)
    # are mapped A,B,... so PyMOL shows distinct chains.
    if chain.lstrip("-").isdigit():
        return chr(ord("A") + int(chain) % 26)
    return chain[:1] or "A"


def _format_atom(idx: int, atom: tuple) -> str:
    record, _, elem, name, resname, chain, resseq, x, y, z, occ, b = atom
    chain_char = _chain_char(chain)

    # Atom-name placement follows PDB convention: 4-char field where a
    # 1-char element symbol starts at column 14 (leading space).
    if len(name) >= 4:
        name_field = name[:4]
    elif len(elem) == 1:
        name_field = f" {name:<3s}"
    else:
        name_field = f"{name:<4s}"

    record_field = f"{record:<6s}"
    # %8.3f overflows for |coord| > 9999.999; PyMOL still parses these.
    return (
        f"{record_field}{idx:5d} {name_field} {resname:>3s} {chain_char}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}"
        f"          {elem:>2s}\n"
    )


def traj_dir_to_multimodel_pdb(
    traj_dir: Path,
    output_path: Path,
    min_step: int | None = None,
    max_step: int | None = None,
    stride: int = 1,
) -> int:
    """Concatenate ``*.cif`` files in ``traj_dir`` into a multi-MODEL PDB.

    Frames are sorted by the ``stepNNN`` prefix in the filename. Returns the
    number of MODEL blocks written.
    """
    cif_paths = sorted(traj_dir.glob("*.cif"), key=_step_num)
    if not cif_paths:
        msg = f"No .cif files in {traj_dir}"
        raise FileNotFoundError(msg)

    selected: list[Path] = []
    for p in cif_paths:
        n = _step_num(p)
        if min_step is not None and n < min_step:
            continue
        if max_step is not None and n > max_step:
            continue
        selected.append(p)
    selected = selected[::stride]
    if not selected:
        msg = f"All frames filtered out for {traj_dir}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        out.write(f"REMARK   1 trajectory: {traj_dir}\n")
        out.write(f"REMARK   1 frames: {len(selected)} (stride={stride})\n")
        for i, cif_path in enumerate(selected, start=1):
            atoms = _parse_atom_site(cif_path)
            out.write(f"MODEL     {i:>4d}\n")
            for j, atom in enumerate(atoms, start=1):
                out.write(_format_atom(j, atom))
            out.write("ENDMDL\n")
        out.write("END\n")
    return len(selected)


@click.command()
@click.argument(
    "traj_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--min-step", type=int, default=None, help="Skip frames with step < N.")
@click.option("--max-step", type=int, default=None, help="Skip frames with step > N.")
@click.option("--stride", type=int, default=1, help="Take every N-th frame.")
def cli(
    traj_dir: Path,
    output_path: Path,
    min_step: int | None,
    max_step: int | None,
    stride: int,
):
    """Combine CIF frames from TRAJ_DIR into a multi-MODEL PDB at OUTPUT_PATH.

    TRAJ_DIR is a trajectory sub-directory such as
    ``.../['2D94']_1_1_._traj/x_with_noise``.
    """
    n = traj_dir_to_multimodel_pdb(
        traj_dir,
        output_path,
        min_step=min_step,
        max_step=max_step,
        stride=stride,
    )
    click.echo(f"Wrote {n} models -> {output_path}")


if __name__ == "__main__":
    cli()
