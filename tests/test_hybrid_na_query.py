"""DNA/RNA hybrid chains (molecule type N): the restype / MSA query keeps ribo vs deoxy per residue (v2.3.0).

The training data writes a hybrid's one_letter_code as "C(DC)(DG)G"; before v2.3.0 that string went into the NA view, which
is keyed on bare CCD codes, so every deoxy residue became NA_UNKNOWN. The query now comes from the CCD codes."""

from types import SimpleNamespace

import numpy as np

from miniworld.data.constants import ResidueMapping
from miniworld.data.io.load import get_query_sequence


class _Feature:
    def __init__(self, value):
        self.value = np.asarray(value)


class _Chains:
    """The slice of CIFMolAttached.chains that get_query_sequence reads, for one chain."""

    def __init__(self, seq_id, ccd, olc, olc_can):
        self.chain_id = np.array(["A"])
        self._chain = SimpleNamespace(
            seq_id=[SimpleNamespace(value=seq_id)],
            residues=SimpleNamespace(
                chem_comp_id=_Feature(ccd), one_letter_code=_Feature(olc), one_letter_code_can=_Feature(olc_can)
            ),
        )

    def __getitem__(self, mask):
        return self._chain


def test_hybrid_query_keeps_deoxy_and_ribo_residues():
    # PDB 100d chain A as stored in the training CIF DB
    ccd = ["C", "DC", "DG", "DG", "G", "BRU"]
    cif = SimpleNamespace(chains=_Chains("N00000000000000000000", ccd, ["C", "(DC)", "(DG)", "(DG)", "G", "(BRU)"], ["C", "C", "G", "G", "G", "U"]))
    rm = ResidueMapping()
    got = get_query_sequence(cif, "A").tolist()
    assert got == [rm.RNA2NUM["C"], rm.DNA2NUM["DC"], rm.DNA2NUM["DG"], rm.DNA2NUM["DG"], rm.RNA2NUM["G"], rm.NA_UNKNOWN]


def test_dna_and_rna_queries_unchanged():
    rm = ResidueMapping()
    dna = SimpleNamespace(chains=_Chains("D00000000000000000003", ["DC", "DG"], ["(DC)", "(DG)"], ["C", "G"]))
    rna = SimpleNamespace(chains=_Chains("R00000000000000000001", ["A", "U"], ["A", "U"], ["A", "U"]))
    assert get_query_sequence(dna, "A").tolist() == [rm.DNA2NUM["C"], rm.DNA2NUM["G"]]
    assert get_query_sequence(rna, "A").tolist() == [rm.RNA2NUM["A"], rm.RNA2NUM["U"]]
