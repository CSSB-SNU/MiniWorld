# pyright: reportReturnType=false
from biomol import BioMol
from biomol.core import NodeFeature, View


class TemplateAtomView(
    View["TemplateAtomView", "TemplateResidueView", "TemplateChainView", "TemplateMol"],
):
    """View class for Template atoms."""

    @property
    def id(self) -> NodeFeature:
        """Atom IDs. Example: 'N', 'CA', 'C', 'CB' only."""

    @property
    def xyz(self) -> NodeFeature:
        """XYZ coordinates of atoms."""

    @property
    def b_factor(self) -> NodeFeature:
        """B-factors of atoms."""

    @property
    def occupancy(self) -> NodeFeature:
        """Occupancy of atoms."""


class TemplateResidueView(
    View["TemplateAtomView", "TemplateResidueView", "TemplateChainView", "TemplateMol"],
):
    """View class for Template residues."""

    @property
    def one_letter_code_can(self) -> NodeFeature:
        """One-letter code (canonical)."""

    @property
    def one_letter_code(self) -> NodeFeature:
        """One-letter code (not canonical)."""

    @property
    def cif_idx(self) -> NodeFeature:
        """Template residue indices."""

    @property
    def auth_idx(self) -> NodeFeature:
        """Author residue indices."""

    @property
    def chem_comp_id(self) -> NodeFeature:
        """Chemical component IDs."""

    @property
    def hetero(self) -> NodeFeature:
        """Hetero flag."""


class TemplateChainView(
    View["TemplateAtomView", "TemplateResidueView", "TemplateChainView", "TemplateMol"],
):
    """View class for Template chains."""

    @property
    def entity_id(self) -> NodeFeature:
        """Entity IDs."""

    @property
    def entity_type(self) -> NodeFeature:
        """Entity types. Example: 'polymer', 'non-polymer', etc."""

    @property
    def chain_id(self) -> NodeFeature:
        """Chain IDs. asym_id_oper_id. Example: 'A_1', 'B_1', etc."""

    @property
    def auth_asym_id(self) -> NodeFeature:
        """Author chain IDs. Example: 'A', 'B', etc."""


class TemplateMol(
    BioMol["TemplateAtomView", "TemplateResidueView", "TemplateChainView"],
):
    """Class for Template molecules."""

    @property
    def id(self) -> str:
        """PDB ID of the molecule."""
        return self.metadata["id"]

    @property
    def assembly_id(self) -> str:
        """Assembly ID of the molecule."""
        return self.metadata["assembly_id"]

    @property
    def model_id(self) -> int:
        """Model ID of the molecule."""
        return self.metadata["model_id"]

    @property
    def alt_id(self) -> str:
        """Alternate location ID of the molecule."""
        return self.metadata["alt_id"]
