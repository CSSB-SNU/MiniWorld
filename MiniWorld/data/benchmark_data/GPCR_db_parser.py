import csv
import pickle
from collections import defaultdict
from format import MultiStateProteinFormat, MultiStateType


def parse_gpcrdb_csv(file_path, save_path):
    """
    Parses the GPCRdb CSV file and extracts relevant information.

    Returns:
        dict: A dictionary with UniProt IDs as keys and lists of tuples
              containing (state, ligand_id, pdb_id) as values.
    """

    # Dictionary to store results
    result = defaultdict(list)

    # Open the file using the correct encoding
    with open(file_path, encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Use the first row to find column indices
    header = rows[0]
    uniprot_idx = header.index("UniProt")
    chain_ID_idx = header.index("Preferred chain")
    state_idx = header.index("State")
    ligand_idx = header.index("Name")
    pdb_idx = header.index("PDB")

    # Iterate from the second row (skip header)
    for row in rows[1:]:
        try:
            uniprot = row[uniprot_idx].strip()
            state = row[state_idx].strip()
            ligand_id = row[ligand_idx].strip()
            pdb_id = row[pdb_idx].strip()
            chain_id = row[chain_ID_idx].strip()
            pdb_id = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else None
            if uniprot and pdb_id:
                result[uniprot].append((state, ligand_id, pdb_id))
        except IndexError:
            continue  # Skip incomplete or malformed rows

    # multistate_gpcr
    multistate_gpcr = {}
    for k, v in result.items():
        state_list = [state for state, _, _ in v]
        if len(set(state_list)) > 1:
            multistate_gpcr[k] = v

    # Convert to MultiStateProteinFormat
    multi_state_proteins = []
    for uniprot, values in multistate_gpcr.items():
        pdb_ids = [pdb_id for _, _, pdb_id in values]
        ligands = [ligand for _, ligand, _ in values]
        multi_state_proteins.append(
            MultiStateProteinFormat(
                category="GPCR",
                type=MultiStateType.ACTIVE_INACTIVE,
                pdb_ids=pdb_ids,
                ligand_list=ligands,
                description=uniprot,
                source="GPCRdb",
            )
        )

    # Save to pickle file
    with open(save_path, "wb") as f:
        pickle.dump(multi_state_proteins, f)


if __name__ == "__main__":
    file_path = "./benchmark_data/GPCR/GPCRdb_structures.csv"
    save_path = "./benchmark_data/GPCR/multistate_gpcr.pickle"
    parsed_data = parse_gpcrdb_csv(file_path, save_path)
