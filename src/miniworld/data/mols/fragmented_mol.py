# pyright: reportReturnType=false
from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, View


class FragmentedCCDAtomView(
    View[
        "FragmentedCCDAtomView",
        "FragmentedCCDResidueView",
        "FragmentedCCDChainView",
        "FragmentedCCDMol",
    ],
):
    """View class for FragmentedCCD atoms."""

    @property
    def id(self) -> NodeFeature:
        """Atom IDs. Example: 'N', 'CA', 'C', 'CB', ..."""

    @property
    def element(self) -> NodeFeature:
        """Element symbols of atoms."""

    @property
    def aromatic(self) -> NodeFeature:
        """Aromatic flag."""

    @property
    def stereo(self) -> NodeFeature:
        """Stereochemistry information."""

    @property
    def charge(self) -> NodeFeature:
        """Formal charge of atoms."""

    @property
    def model_xyz(self) -> NodeFeature:
        """Model coordinates of atoms."""

    @property
    def sssr_idx(self) -> NodeFeature:
        """SSSR ring indices."""

    @property
    def hybridization(self) -> NodeFeature:
        """Hybridization states of atoms. Example: 'SP3', 'SP2', etc."""

    @property
    def bond_type(self) -> EdgeFeature:
        """Bond types between atoms. Example: 'SING', 'DOUB', etc."""

    @property
    def bond_aromatic(self) -> EdgeFeature:
        """Aromatic flag for bonds between atoms."""

    @property
    def bond_stereo(self) -> EdgeFeature:
        """Stereochemistry information for bonds between atoms."""

    @property
    def bond_conjugation(self) -> EdgeFeature:
        """Conjugation flag for bonds between atoms."""

    @property
    def bond_aromaticity(self) -> EdgeFeature:
        """Aromaticity flag for bonds between atoms."""


class FragmentedCCDResidueView(
    View[
        "FragmentedCCDAtomView",
        "FragmentedCCDResidueView",
        "FragmentedCCDChainView",
        "FragmentedCCDMol",
    ],
):
    """View class for FragmentedCCD residues."""

    @property
    def residue_id(self) -> NodeFeature:
        """Residue idx."""


class FragmentedCCDChainView(
    View[
        "FragmentedCCDAtomView",
        "FragmentedCCDResidueView",
        "FragmentedCCDChainView",
        "FragmentedCCDMol",
    ],
):
    """View class for FragmentedCCD chains."""

    @property
    def id(self) -> NodeFeature:
        """FragmentedCCD IDs : 0."""


class FragmentedCCDMol(
    BioMol[
        "FragmentedCCDAtomView",
        "FragmentedCCDResidueView",
        "FragmentedCCDChainView",
    ],
):
    """Class for FragmentedCCD molecules."""
