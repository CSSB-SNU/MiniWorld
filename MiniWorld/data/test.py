
import pickle
from BioMol import DB_PATH


import lmdb
atom_db_env = f"{DB_PATH}/seq_to_str/atom.lmdb"
residue_db_env = f"{DB_PATH}/seq_to_str/residue.lmdb"
def read_seq_lmdb(key: str, level: str = "atom"):
    """
    Read a sequence from the LMDB database.
    """
    if level not in ["atom", "residue"]:
        raise ValueError("level must be either 'atom' or 'residue'.")
    db_env = atom_db_env if level == "atom" else residue_db_env
    env = lmdb.open(db_env, readonly=True)
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            raise ValueError(f"Key {key} not found in the database.")
        data = pickle.loads(data)

    env.close()
    return data


if __name__ == "__main__":
    # Example usage
    try:
        seq_data = read_seq_lmdb("788726", level="residue")
        print(seq_data)
        breakpoint()
    except ValueError as e:
        print(e)