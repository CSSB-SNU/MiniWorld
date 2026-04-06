# pyright: reportReturnType=false
from biomol import BioMol
from biomol.core import EdgeFeature, NodeFeature, View


class CCDAtomView(
    View["CCDAtomView", "CCDResidueView", "CCDChainView", "CCDMol"],
):
    """View class for CCD atoms."""

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


class CCDResidueView(
    View["CCDAtomView", "CCDResidueView", "CCDChainView", "CCDMol"],
):
    """View class for CCD residues."""

    @property
    def id(self) -> NodeFeature:
        """Residue IDs. Example: 'PROTOPORPHYRIN IX CONTAINING FE'."""

    @property
    def formular(self) -> NodeFeature:
        """Formular of the residue. Example: 'C34 H32 Fe N4 O4'."""

    @property
    def rdkit_smiles(self) -> NodeFeature:
        """RDKit SMILES representation of the residue."""


class CCDChainView(
    View["CCDAtomView", "CCDResidueView", "CCDChainView", "CCDMol"],
):
    """View class for CCD chains."""

    @property
    def id(self) -> NodeFeature:
        """CCD IDs. Example: HEM."""


class CCDMol(
    BioMol["CCDAtomView", "CCDResidueView", "CCDChainView"],
):
    """Class for CCD molecules."""

    @property
    def id(self) -> str:
        """CCD ID of the molecule."""
        return self.chains["id"]

    @property
    def name(self) -> str:
        """Name of the molecule."""
        return self.residues["id"]
