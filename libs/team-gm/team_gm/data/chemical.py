import torch

from collections.abc import Mapping
from team_gm.data.atom_names import *

PROT_UNK = "UNK"
DNA_UNK = "DX"
RNA_UNK = "RX"
MASK = "MAS"  # for protein design setup

# fmt: off
PROT_RES_NAMES: tuple[str, ...] = (
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU',
    'LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL', PROT_UNK, MASK)
DNA_RES_NAMES: tuple[str, ...] = ("DA", "DC", "DG", "DT", DNA_UNK)
RNA_RES_NAMES: tuple[str, ...] = ("A", "U", "C", "G", RNA_UNK)
# fmt: on

RES_NAMES = PROT_RES_NAMES + DNA_RES_NAMES + RNA_RES_NAMES
RES_NUM = len(RES_NAMES)

RESCHAR_TO_RESNAME = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",  # TODO: this can't be distinguished Protein, RNA, DNA
    "a": "DA",
    "c": "DC",
    "g": "DG",
    "t": "DT",
    "b": "A",
    "d": "C",
    "h": "G",
    "u": "U",
}

VAN_DER_WAALS_RADIUS: Mapping[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "P": 1.8,
    "S": 1.8,
}


# fmt: off
RES_NAME_TO_ATOM_NAMES: Mapping[str, tuple] = {
    "ALA": (   N,  CA,   C,   O,  CB,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "ARG": (   N,  CA,   C,   O,  CB,  CG,  CD,  NE,  CZ, NH1, NH2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "ASN": (   N,  CA,   C,   O,  CB,  CG, OD1, ND2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "ASP": (   N,  CA,   C,   O,  CB,  CG, OD1, OD2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "CYS": (   N,  CA,   C,   O,  CB,  SG,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "GLN": (   N,  CA,   C,   O,  CB,  CG,  CD, OE1, NE2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "GLU": (   N,  CA,   C,   O,  CB,  CG,  CD, OE1, OE2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "GLY": (   N,  CA,   C,   O,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "HIS": (   N,  CA,   C,   O,  CB,  CG, ND1, CD2, CE1, NE2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "ILE": (   N,  CA,   C,   O,  CB,  CG1,CG2, CD1,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "LEU": (   N,  CA,   C,   O,  CB,  CG, CD1, CD2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "LYS": (   N,  CA,   C,   O,  CB,  CG,  CD,  CE,  NZ,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "MET": (   N,  CA,   C,   O,  CB,  CG,  SD,  CE,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "PHE": (   N,  CA,   C,   O,  CB,  CG, CD1, CD2, CE1, CE2,  CZ,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "PRO": (   N,  CA,   C,   O,  CB,  CG,  CD,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "SER": (   N,  CA,   C,   O,  CB,  OG,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "THR": (   N,  CA,   C,   O,  CB, OG1, CG2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "TRP": (   N,  CA,   C,   O,  CB,  CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "TYR": (   N,  CA,   C,   O,  CB,  CG, CD1, CD2, CE1, CE2, CZ,   OH,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "VAL": (   N,  CA,   C,   O,  CB, CG1, CG2,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "UNK": (   N,  CA,   C,   O,  CB,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "MAS": (   N,  CA,   C,   O,  CB,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "DA" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C2p, C1p,None,  N9,  C4,  N3,  C2,  N1,  C6,  C5,  N7,  C8,None,None,None,None,None,None),
    "DT" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C2p, C1p,None,  N1,  C2,  O2,  N3,  C4,  O4,  C5,  C7,  C6,None,None,None,None,None,None),
    "DC" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C2p, C1p,None,  N1,  C2,  O2,  N3,  C4,  N4,  C5,  C6,None,None,None,None,None,None,None),
    "DG" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C2p, C1p,None,  N9,  C4,  N3,  C2,  N1,  C6,  C5,  N7,  C8,  N2,  O6,None,None,None,None),
    "DX" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C2p, C1p,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
    "A"  : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C1p, C2p, O2p,  N1,  C2,  N3,  C4,  C5,  C6,  N6,  N7,  C8,  N9,None,None,None,None,None),
    "U"  : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C1p, C2p, O2p,  N1,  C2,  O2,  N3,  C4,  O4,  C5,  C6,None,None,None,None,None,None,None),
    "C"  : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C1p, C2p, O2p,  N1,  C2,  O2,  N3,  C4,  N4,  C5,  C6,None,None,None,None,None,None,None),
    "G"  : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C1p, C2p, O2p,  N1,  C2,  N2,  N3,  C4,  C5,  C6,  O6,  N7,  C8,  N9,None,None,None,None),
    "RX" : ( OP1,   P, OP2, O5p, C5p, C4p, O4p, C3p, O3p, C1p, C2p, O2p,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None),
}
# fmt: on

RES_ATOM_NUM = len(RES_NAME_TO_ATOM_NAMES["ALA"])
for atom_names in RES_NAME_TO_ATOM_NAMES.values():
    assert len(atom_names) == RES_ATOM_NUM

PROT_FRAME_ATOM_NAMES: tuple[str, ...] = (N, CA, C)
NA_FRAME_ATOM_NAMES: tuple[str, ...] = (O4p, C4p, C3p)

PROT_ATOM_CENTER_IDX = RES_NAME_TO_ATOM_NAMES["ALA"].index(PROT_FRAME_ATOM_NAMES[1])
NA_ATOM_CENTER_IDX = RES_NAME_TO_ATOM_NAMES["A"].index(NA_FRAME_ATOM_NAMES[1])
for res, atom_names in RES_NAME_TO_ATOM_NAMES.items():
    if res in PROT_RES_NAMES:
        assert PROT_ATOM_CENTER_IDX == atom_names.index(PROT_FRAME_ATOM_NAMES[1])
    elif res in DNA_RES_NAMES or res in RNA_RES_NAMES:
        assert NA_ATOM_CENTER_IDX == atom_names.index(NA_FRAME_ATOM_NAMES[1])
    else:
        raise ValueError(f"Unknown residue name: {res}")

# fmt: off
FRAME_BONDS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "Protein": ((  N, CA), ( CA,  C), (  C,  O), ( CA, CB)),
    "DNA"    : ((  P,OP1), (  P,OP2), (  P,O5p), (O5p,C5p), (C5p,C4p), (C4p,C3p), (C3p,O3p), (C3p,C2p), (C2p,C1p), (C4p,O4p), (O4p,C1p)),
    "RNA"    : ((  P,OP1), (  P,OP2), (  P,O5p), (O5p,C5p), (C5p,C4p), (C4p,C3p), (C3p,O3p), (C3p,C2p), (C2p,O2p), (C2p,C1p), (C4p,O4p), (O4p,C1p)),
}

RES_NAME_TO_NONFRAME_BONDS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "ALA": () , 
    "ARG": (( CB, CG), ( CG, CD), ( CD, NE), ( NE, CZ), ( CZ,NH1), ( CZ,NH2)),
    "ASN": (( CB, CG), ( CG,OD1), ( CG,ND2)),
    "ASP": (( CB, CG), ( CG,OD1), ( CG,OD2)),
    "CYS": (( CB, SG),),  
    "GLN": (( CB, CG), ( CG, CD), ( CD,OE1), ( CD,NE2)),
    "GLU": (( CB, CG), ( CG, CD), ( CD,OE1), ( CD,OE2)),
    "GLY": (),
    "HIS": (( CB, CG), ( CG,ND1), ( CG,CD2), (CD2,CE1), (CE1,NE2)),
    "ILE": (( CB,CG1), ( CB,CG2), (CG1,CD1)),
    "LEU": (( CB, CG), ( CG,CD1), ( CG,CD2)),
    "LYS": (( CB, CG), ( CG, CD), ( CD, CE), ( CE, NZ)),
    "MET": (( CB, CG), ( CG, SD), ( SD, CE)),
    "PHE": (( CB, CG), ( CG,CD1), ( CG,CD2), (CD1,CE1), (CD2,CE2)),
    "PRO": (( CB, CG), ( CG, CD)),
    "SER": (( CB, OG),), 
    "THR": (( CB,OG1), ( CB,CG2)),
    "TRP": (( CB, CG), ( CG,CD1), ( CG,CD2), (CD1,NE1), (CE2,CZ2), (CE3,CZ3)),
    "TYR": (( CB, CG), ( CG,CD1), ( CG,CD2), (CD1,CE1), (CD2,CE2)),
    "VAL": (( CB,CG1), ( CB,CG2)),
    "UNK": (),
    "MAS": (),
    "DA" : ((C1p, N9), ( N9, C4), ( C4, N3), ( N3, C2), ( C2, N1), ( N1, C6), ( C6, C5), ( C5, C4), ( C5, N7), ( N7, C8), ( C8, N9)),
    "DC" : ((C1p, N1), ( N1, C2), ( C2, O2), ( C2, N3), ( N3, C4), ( C4, N4), ( C4, C5), ( C5, C6), ( C6, N1)),
    "DG" : ((C1p, N9), ( N9, C4), ( C4, N3), ( N3, C2), ( C2, N2), ( C2, N1), ( N1, C6), ( C6, O6), ( C6, C5), ( C5, C4), ( C5, N7), ( N7, C8), ( C8, N9)),
    "DT" : ((C1p, N1), ( N1, C2), ( C2, O2), ( C2, N3), ( N3, C4), ( C4, O4), ( C4, C5), ( C5, C6), ( C5, C7)),
    "DX" : (),
    "A"  : ((C1p, N9), ( N9, C4), ( C4, N3), ( N3, C2), ( C2, N1), ( N1, C6), ( C6, N6), ( C6, C5), ( C5, C4), ( C5, N7), ( N7, C8), ( C8, N9)),
    "U"  : ((C1p, N1), ( N1, C2), ( C2, O2), ( C2, N3), ( N3, C4), ( C4, O4), ( C4, C5), ( C5, C6), ( C6, N1)),
    "C"  : ((C1p, N1), ( N1, C2), ( C2, O2), ( C2, N3), ( N3, C4), ( C4, N4), ( C4, C5), ( C5, C6), ( C6, N1)),
    "G"  : ((C1p, N9), ( N9, C4), ( C4, N3), ( N3, C2), ( C2, N2), ( C2, N1), ( N1, C6), ( C6, O6), ( C6, C5), ( C5, C4), ( C5, N7), ( N7, C8), ( C8, N9)),
    "RX"  : (),
}
# fmt: on


def _make_atom_mask() -> torch.BoolTensor:
    """make atom mask from atom names"""
    RES_TYPE_ATOM_MASK = torch.zeros(RES_NUM, RES_ATOM_NUM).bool()
    for res_idx, res_name in enumerate(RES_NAMES):
        atom_names = RES_NAME_TO_ATOM_NAMES[res_name]
        for atom_idx, atom_name in enumerate(atom_names):
            if atom_name is None:
                continue
            RES_TYPE_ATOM_MASK[res_idx, atom_idx] = 1

    return RES_TYPE_ATOM_MASK


def _make_rigidgroup_atom_idx() -> torch.LongTensor:
    """make rigidgroup atom index from frame atoms name"""
    RES_TYPE_RIGIDGROUP_ATOM_IDX = torch.zeros(RES_NUM, 3).long()
    for res_idx, res_name in enumerate(RES_NAMES):
        atom_names = RES_NAME_TO_ATOM_NAMES[res_name]
        if res_name in PROT_RES_NAMES:
            FRAME_ATOM_NAMES = PROT_FRAME_ATOM_NAMES
        elif res_name in DNA_RES_NAMES or res_name in RNA_RES_NAMES:
            FRAME_ATOM_NAMES = NA_FRAME_ATOM_NAMES
        else:
            raise ValueError(f"Unknown residue name: {res_name}")

        # for each frame atom, find the corresponding atom index in the residue atom names
        for frame_idx, frame_atom in enumerate(FRAME_ATOM_NAMES):
            atom_idx = atom_names.index(frame_atom)
            RES_TYPE_RIGIDGROUP_ATOM_IDX[res_idx, frame_idx] = atom_idx

    return RES_TYPE_RIGIDGROUP_ATOM_IDX


def _make_atom_bond_matrix() -> torch.BoolTensor:
    """make atom bond matrix from frame bond list"""
    RES_TYPE_BOND_MATRIX = torch.zeros(RES_NUM, RES_ATOM_NUM, RES_ATOM_NUM).bool()
    for res_idx, res_name in enumerate(RES_NAMES):
        atom_names = RES_NAME_TO_ATOM_NAMES[res_name]
        if res_name in PROT_RES_NAMES:
            chain_type = "Protein"
        elif res_name in DNA_RES_NAMES:
            chain_type = "DNA"
        elif res_name in RNA_RES_NAMES:
            chain_type = "RNA"
        else:
            raise ValueError(f"Unknown residue name: {res_name}")

        for bond in FRAME_BONDS[chain_type] + RES_NAME_TO_NONFRAME_BONDS[res_name]:
            atom1, atom2 = bond
            if res_name == "GLY" and atom1 == "CA" and atom2 == "CB":
                continue
            idx0 = atom_names.index(atom1)
            idx1 = atom_names.index(atom2)
            RES_TYPE_BOND_MATRIX[res_idx, idx0, idx1] = 1
            RES_TYPE_BOND_MATRIX[res_idx, idx1, idx0] = 1

    return RES_TYPE_BOND_MATRIX


def _make_atom_radius() -> torch.FloatTensor:
    """make atom radius from atom names"""
    RES_TYPE_ATOM_RADIUS = torch.zeros(RES_NUM, RES_ATOM_NUM)
    for res_idx, res_name in enumerate(RES_NAMES):
        atom_names = RES_NAME_TO_ATOM_NAMES[res_name]
        for atom_idx, atom_name in enumerate(atom_names):
            if atom_name is None:
                continue
            atom_radius = VAN_DER_WAALS_RADIUS[atom_name.strip()[0].upper()]
            RES_TYPE_ATOM_RADIUS[res_idx, atom_idx] = atom_radius

    return RES_TYPE_ATOM_RADIUS


RES_TYPE_ATOM_MASK = _make_atom_mask()
RES_TYPE_RIGIDGROUP_ATOM_IDX = _make_rigidgroup_atom_idx()
RES_TYPE_BOND_MATRIX = _make_atom_bond_matrix()
RES_TYPE_ATOM_RADIUS = _make_atom_radius()
