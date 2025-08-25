from dataclasses import dataclass
import enum
import csv


@enum.unique
class MultiStateType(enum.Enum):
    """
    Enum to represent the type of multi-state protein.
    """

    APO_HOLO = "Apo/Holo"
    ACTIVE_INACTIVE = "Active/Inactive"
    INTRINSIC = "Intrinsically multi-state"
    OLIGOMORPHIC = "Oligomeric multi-state"
    OTHER = "Other"


@dataclass
class MultiStateProteinFormat:
    """
    A class to represent the format of a multi-state protein dataset.
    """

    category: str
    type: MultiStateType
    pdb_ids: list[str]
    ligand_list: list[str]
    description: str | None = None
    source: str | None = None


def to_csv(multi_state_proteins: list[MultiStateProteinFormat], file_path: str):
    """
    Converts a list of MultiStateProteinFormat objects to a CSV file.

    Args:
        multi_state_proteins (list[MultiStateProteinFormat]): List of multi-state protein data.
        file_path (str): Path to save the CSV file.
    """

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(
            ["Category", "Type", "PDB IDs", "Ligand List", "Description", "Source"]
        )

        # Write data
        for protein in multi_state_proteins:
            writer.writerow(
                [
                    protein.category,
                    protein.type,
                    ", ".join(protein.pdb_ids),
                    ", ".join(protein.ligand_list),
                    protein.description or "",
                    protein.source or "",
                ]
            )
