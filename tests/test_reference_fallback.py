"""Reference geometry, mask and atom-mapping regression tests (CPU only)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from miniworld.data import reference as ref


def molecule(xyz=None, *, stereo=("N", "N"), elements=("C", "O")):
    if xyz is None:
        xyz = [["?", "?", "?"], [30, 20, 10]]
    atoms = NS(
        id=NS(value=np.array(["C1", "O1"])),
        element=NS(value=np.array(elements)),
        charge=NS(value=np.array(["0", "0"])),
        stereo=NS(value=np.array(stereo)),
        model_xyz=NS(value=np.array(xyz, dtype=object)),
        bond_type=NS(value=np.array(["SING"]), src=np.array([0]), dst=np.array([1])),
        bond_stereo=NS(value=np.array(["N"])),
    )
    return NS(atoms=atoms)


@pytest.fixture(autouse=True)
def clear_cache():
    ref._rdkit_conformer.cache_clear()
    yield
    ref._rdkit_conformer.cache_clear()


def test_complete_coordinates_unchanged_and_no_rdkit(monkeypatch):
    mol = molecule([[30, 20, 10], [31.4, 20, 10]])

    def forbidden(*args):
        raise AssertionError("Complete coordinates must not be regenerated")

    monkeypatch.setattr(ref, "_rdkit_conformer", forbidden)
    out = ref.resolve_reference(mol, "COMPLETE")
    assert out.source == "model"
    np.testing.assert_array_equal(
        out.pos, np.asarray(mol.atoms.model_xyz.value, dtype=np.float32)
    )
    assert out.mask.all()


@pytest.mark.parametrize("missing", ["?", ".", np.nan, np.inf, None])
def test_missing_validity_before_zero_fill(missing):
    out = ref.parse_coordinates([[missing, 1, 2], [30, 20, 10]])
    np.testing.assert_array_equal(out.mask, [False, True])
    assert np.isfinite(out.pos).all()
    pos = ref.augment_reference(out.pos, out.mask, np.eye(3), np.array([1, 2, 3]))
    np.testing.assert_array_equal(pos, [[0, 0, 0], [1, 2, 3]])


def test_all_missing_masked_augmentation():
    out = ref.parse_coordinates([["?", "?", "?"]] * 2)
    pos = ref.augment_reference(out.pos, out.mask, np.eye(3), np.ones(3))
    assert not out.mask.any()
    np.testing.assert_array_equal(pos, np.zeros((2, 3)))


def test_rdkit_replaces_whole_frame_caches_and_does_not_mutate_input():
    mol = molecule()
    original = mol.atoms.model_xyz.value.copy()
    out = ref.resolve_reference(mol, "METHANOL")
    assert out.source == "rdkit" and out.mask.all()
    assert 1.0 < np.linalg.norm(out.pos[0] - out.pos[1]) < 1.8
    assert np.linalg.norm(out.pos[1]) < 5  # original known coordinate also replaced
    saved = out.pos.copy()
    out.pos[:] = 999
    again = ref.resolve_reference(mol, "METHANOL")
    np.testing.assert_array_equal(again.pos, saved)
    np.testing.assert_array_equal(mol.atoms.model_xyz.value, original)
    assert ref._rdkit_conformer.cache_info().hits == 1
    ref._rdkit_conformer.cache_clear()
    np.testing.assert_array_equal(ref.resolve_reference(mol, "METHANOL").pos, saved)


def test_failure_keeps_valid_model_atoms_and_masks_missing(monkeypatch):
    from rdkit.Chem import AllChem

    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args: -1)
    out = ref.resolve_reference(molecule(), "FAIL")
    assert out.source == "model_masked"
    np.testing.assert_array_equal(out.mask, [False, True])
    np.testing.assert_array_equal(out.pos, [[0, 0, 0], [30, 20, 10]])


def test_unknown_chemistry_masks_instead_of_inventing_conformer():
    mol = molecule()
    mol.atoms.bond_type.value[:] = "UNKN"
    out = ref.resolve_reference(mol, "UNKNOWN")
    assert out.source == "model_masked"
    np.testing.assert_array_equal(out.mask, [False, True])


def test_mapping_crop_and_unknown_atom():
    mol = molecule()
    out = ref.resolve_reference(mol, "MAP")
    mapped = ref.map_reference(mol, out, ["O1", "C1", "absent"])
    np.testing.assert_array_equal(mapped.pos[:2], out.pos[[1, 0]])
    np.testing.assert_array_equal(mapped.mask, [True, True, False])
    np.testing.assert_array_equal(mapped.pos[2], [0, 0, 0])


def test_training_inference_match_even_when_crop_removed_missing_atom():
    from miniworld.data.features.convert import to_reference_features
    from miniworld.data.inference.ccd import _ccdmol_to_residue
    from miniworld.data.inference.build import (
        _build_reference_arrays,
        _strip_terminal_atoms,
    )

    full = molecule()
    cropped_atoms = NS(
        **{
            name: NS(value=getattr(full.atoms, name).value[[1]])
            for name in ["id", "model_xyz", "element", "charge"]
        }
    )

    class Residues:
        chem_comp_id = NS(value=np.array(["SAMPLE"]))

        def __len__(self):
            return 1

    cropped = NS(
        atoms=cropped_atoms,
        residues=Residues(),
        index_table=NS(atom_to_res=np.zeros(1, dtype=np.int64)),
    )
    train = to_reference_features(
        cropped, np.random.default_rng(31), ccd_mols={"SAMPLE": {0: full}}
    )
    residue = _ccdmol_to_residue("SAMPLE", full)
    stripped, _ = _strip_terminal_atoms([residue], "C1")
    expansion = NS(n_atoms=1, n_residues=1, residues=stripped)
    pos, _, _ = _build_reference_arrays([expansion], np.random.default_rng(31))
    np.testing.assert_allclose(train.pos.numpy().reshape(-1, 3), pos, atol=1e-6)
    assert train.mask.all() and stripped[0].atom_mask.all()


def test_inference_failure_mask_survives_terminal_stripping(monkeypatch):
    from rdkit.Chem import AllChem
    from miniworld.data.inference.ccd import _ccdmol_to_residue
    from miniworld.data.inference.build import (
        _strip_terminal_atoms,
        _build_reference_arrays,
    )

    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args: -1)
    residue = _ccdmol_to_residue("FAIL", molecule())
    stripped, _ = _strip_terminal_atoms([residue], "O1")
    expansion = NS(n_atoms=1, n_residues=1, residues=stripped)
    pos, _, _ = _build_reference_arrays([expansion], np.random.default_rng(31))
    assert not stripped[0].atom_mask.any()
    np.testing.assert_array_equal(pos, [[0, 0, 0]])


def from_smiles(smiles):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    atoms = list(m.GetAtoms())
    bonds = list(m.GetBonds())
    order = {"SINGLE": "SING", "DOUBLE": "DOUB", "TRIPLE": "TRIP", "AROMATIC": "AROM"}
    return NS(
        atoms=NS(
            id=NS(value=np.array([f"A{i}" for i in range(len(atoms))])),
            element=NS(value=np.array([a.GetSymbol() for a in atoms])),
            charge=NS(value=np.array([str(a.GetFormalCharge()) for a in atoms])),
            stereo=NS(
                value=np.array(
                    [
                        a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "N"
                        for a in atoms
                    ]
                )
            ),
            model_xyz=NS(value=np.full((len(atoms), 3), "?", dtype=object)),
            bond_type=NS(
                value=np.array([order[str(b.GetBondType())] for b in bonds]),
                src=np.array([b.GetBeginAtomIdx() for b in bonds]),
                dst=np.array([b.GetEndAtomIdx() for b in bonds]),
            ),
            bond_stereo=NS(
                value=np.array(
                    [
                        {"STEREOE": "E", "STEREOZ": "Z"}.get(str(b.GetStereo()), "N")
                        for b in bonds
                    ]
                )
            ),
        )
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](O)C(=O)O",
        "C[C@@H](O)C(=O)O",
        "F/C=C/F",
        "F/C=C\\F",
        "CC(/Cl)=C(/F)Br",
        "CC(/Cl)=C(\\F)Br",
        "c1ccncc1",
    ],
)
def test_generated_geometry_preserves_stereochemistry(smiles):
    from rdkit import Chem

    mol = from_smiles(smiles)
    out = ref.resolve_reference(mol, smiles)
    assert out.source == "rdkit"
    assert out.mask.all()
    # Infer stereochemistry from the generated geometry independently of the
    # fallback's molecule construction and compare to the original SMILES.
    original = Chem.MolFromSmiles(smiles)
    inferred = Chem.Mol(original)
    Chem.RemoveStereochemistry(inferred)
    conf = Chem.Conformer(len(out.pos))
    for i, xyz in enumerate(out.pos):
        conf.SetAtomPosition(i, xyz.astype(float).tolist())
    inferred.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(inferred)
    assert Chem.MolToSmiles(inferred) == Chem.MolToSmiles(original)


def test_full_component_mapping_after_crop_removed_all_missing_atoms():
    from miniworld.data.features.convert import to_reference_features
    from miniworld.data.inference.ccd import _ccdmol_to_residue
    from miniworld.data.inference.build import (
        _build_reference_arrays,
        _strip_terminal_atoms,
    )

    full = from_smiles("CCO")
    # Only A0 is missing. The other atoms' coordinates deliberately have a
    # different separation to expose a mistaken 'cropped mask all-valid' path.
    full.atoms.model_xyz.value[1:] = [[30, 20, 10], [40, 20, 10]]
    cropped_atoms = NS(
        **{
            name: NS(value=getattr(full.atoms, name).value[[1, 2]])
            for name in ["id", "model_xyz", "element", "charge"]
        }
    )

    class Residues:
        chem_comp_id = NS(value=np.array(["ETHANOL"]))

        def __len__(self):
            return 1

    cropped = NS(
        atoms=cropped_atoms,
        residues=Residues(),
        index_table=NS(atom_to_res=np.zeros(2, dtype=np.int64)),
    )
    train = to_reference_features(
        cropped, np.random.default_rng(31), ccd_mols={"ETHANOL": {0: full}}
    )
    residue = _ccdmol_to_residue("ETHANOL", full)
    stripped, _ = _strip_terminal_atoms([residue], "A0")
    pos, _, _ = _build_reference_arrays(
        [NS(n_atoms=2, n_residues=1, residues=stripped)], np.random.default_rng(31)
    )
    np.testing.assert_allclose(train.pos.numpy().reshape(-1, 3), pos, atol=1e-6)
    assert 1.0 < np.linalg.norm(pos[0] - pos[1]) < 1.8
