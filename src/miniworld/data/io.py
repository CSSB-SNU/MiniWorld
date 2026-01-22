import re
from pathlib import Path

import lmdb
from biomol.cif import CIFMol
from biomol.core.container import FeatureContainer
from biomol.core.types import BioMolDict
from biomol.core.utils import load_bytes
from datacooker import Cooker, ParsingCache

from miniworld.data.mols.cifmol_attached import CIFMolAttached
from miniworld.data.msa import MSA


def load_seq_cluster(seq_cluster_tsv_path:Path, merge_ab: bool = False) -> dict[str,list[str]]:
    """Load sequence clusters from a TSV file.

    Args:
        seq_cluster_tsv_path (Path): Path to the TSV file containing sequence clusters.
        merge_ab (bool): Whether to merge all 'A' and 'B' clusters into a single cluster. Default is False.

    Returns:
        dict[str,list[str]]: A dictionary mapping cluster IDs to lists of sequence IDs.

    """
    cluster_dict = {}
    if merge_ab:
        cluster_dict = {"Ab_cluster": []}
    with seq_cluster_tsv_path.open("r") as f:
        for line in f:
            cluster_id, seq_id = line.strip().split("\t")
            if cluster_id[0] == "A" and merge_ab:
                cluster_dict["Ab_cluster"].append(seq_id)
                continue
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(seq_id)
    return cluster_dict

def load_seq_id_to_seq(seq_id_to_seq_path:Path) -> dict[str,str]:
    """Load sequence ID to sequence mapping from a TSV file.

    Args:
        seq_id_to_seq_path (Path): Path to the TSV file containing sequence ID to sequence mapping.

    Returns:
        dict[str,str]: A dictionary mapping sequence IDs to sequences.

    """
    seq_dict = {}
    with seq_id_to_seq_path.open("r") as f:
        for line in f:
            seq_id, sequence = line.strip().split("\t")
            seq_dict[seq_id] = sequence
    return seq_dict


def extract_lmdb_keys(env_path: Path) -> list[str]:
    """Extract all keys from the LMDB database."""
    env = lmdb.open(str(env_path), readonly=True, lock=False)
    with env.begin() as txn:
        key_list = [
            key.decode() for key in txn.cursor().iternext(keys=True, values=False)
        ]
    env.close()
    return key_list


def load_raw_data(key: str, env_path: Path) -> bytes | None:
    """Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.

    Returns:
    -------
        bytes
            The data dictionary retrieved from the LMDB database.

    """
    cache = getattr(load_raw_data, "_env_cache", None)
    if cache is None:
        cache = {}
        load_raw_data._env_cache = cache  # pyright: ignore[reportFunctionMemberAccess] # noqa: SLF001

    env_key = str(env_path)
    env = cache.get(env_key)
    if env is None:
        env = lmdb.open(
            env_key,
            readonly=True,
            lock=False,
            max_readers=4096,
            readahead=True,
        )
        cache[env_key] = env

    with env.begin(buffers=True) as txn:
        value = txn.get(key.encode())

    if value is None:
        return None
    return bytes(value)

def load_all_raw_data(env_path: Path) -> dict[str, bytes]:
    """Read all key-value pairs from the LMDB database.

    Args:
        env_path: Path to the LMDB environment.

    Returns:
    -------
        dict
            A dictionary containing all key-value pairs from the LMDB database.

    """
    env = lmdb.open(
        str(env_path),
        readonly=True,
        lock=False,
        max_readers=4096,
        readahead=True,
    )

    data_dict = {}
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = bytes(key).decode()
            value_bytes = bytes(value)
            data_dict[key_str] = value_bytes

    return data_dict



def load_cif(key: str, env_path: Path) -> dict[str, dict[str, CIFMol]]:
    """Read a value from the LMDB database by key.

    Args:
        env_path: Path to the LMDB environment.
        key: Key of the data to retrieve.

    Returns:
    -------
        dict
            The data dictionary retrieved from the LMDB database.

    """
    raw_data = load_raw_data(key, env_path)
    if raw_data is None:
        msg = f"Key '{key}' not found in LMDB database at '{env_path}'."
        raise KeyError(msg)
    value = load_bytes(raw_data)
    value, metadata = value["assembly_dict"], value["metadata_dict"]

    cifmol_dict: dict[str, dict[str, CIFMol]] = {}
    for cif_key, _item in value.items():
        assembly_id, model_id, alt_id = cif_key.split("_")

        md = dict(metadata)
        md["assembly_id"] = assembly_id
        md["model_id"] = model_id
        md["alt_id"] = alt_id

        item = BioMolDict(_item)
        item["metadata"] = md

        cifmol_dict[cif_key] = {"cifmol": CIFMol.from_dict(item)}

    return cifmol_dict



def load_cifmol(db_path:Path, cif_id: str) -> CIFMolAttached:
    """Load CIFMolAttached from LMDB by cif_id."""
    pdb_id, assembly_id, model_id, alt_id, chain_id1, chain_id2 = re.findall(r"\([^)]*\)|[^_]+", cif_id)
    value = load_raw_data(pdb_id, db_path)

    if value is None:
        msg = f"Key '{pdb_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)

    value = load_bytes(value)
    item = value.get(f"{assembly_id}_{model_id}_{alt_id}")
    if item is None:
        msg = f"CIFMolAttached '{cif_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)
    item = item["cifmol_dict"]
    return CIFMolAttached.from_dict(item)



def load_cifmols(db_path:Path, seq_id: str) -> list[CIFMol]:
    """Load CIFMols from the LMDB database for a given seqID."""
    value = load_raw_data(seq_id, db_path)
    if value is None:
        msg = f"Key '{seq_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)
    value = load_bytes(value)

    cifmols = [CIFMol.from_dict(item) for item in value.values()]
    length_list = [len(cifmol.residues) for cifmol in cifmols]
    if len(set(length_list)) != 1:
        msg = f"Different residue lengths found for seqID {seq_id}: {length_list}"
        raise ValueError(msg)
    return cifmols


def load_a3m(key: str, env_path: Path) -> MSA | None:
    """Load MSA from LMDB by key."""
    value = load_raw_data(key, env_path)
    if value is None:
        return None
    msa_container = load_bytes(bytes(value))["msa_container"]
    msa_residue_container = msa_container["residue_container"]
    msa_chain_container = msa_container["chain_container"]
    msa_residue_container = FeatureContainer.from_dict(msa_residue_container)
    msa_chain_container = FeatureContainer.from_dict(msa_chain_container)
    return MSA(
        seq_id=key,
        msa_residue_container=msa_residue_container,
        msa_chain_container=msa_chain_container,
    )


def load_fasta(fasta_path: Path) -> dict[str, str]:
    """Load sequences from a FASTA file."""
    seq_dict = {}
    with fasta_path.open("r") as f:
        seq_id = ""
        sequence_lines = []
        for _line in f:
            line = _line.strip()
            if line.startswith(">"):
                if seq_id:
                    seq_dict[seq_id] = "".join(sequence_lines)
                seq_id = line[1:].split()[0]
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if seq_id:
            seq_dict[seq_id] = "".join(sequence_lines)
    return seq_dict


def parse_signalp(signalp_path: Path) -> tuple[int, int] | None:
    """Parse the signalp output file and extract the sequence IDs."""
    if not signalp_path.exists():
        return None
    with signalp_path.open() as f:
        lines = f.readlines()

    result = lines[1].split("\t")
    return int(result[3]) - 1, int(result[4]) - 1


def base_process(
    data_dict: dict,
    recipe_path: Path,
    targets: list[str] | None = None,
) -> dict:
    """Parse a CIFMol object using a predefined recipe."""
    parse_cache = ParsingCache()
    cooker = Cooker(parse_cache=parse_cache, recipebook=str(recipe_path))
    cooker.prep(data_dict, fields=list(data_dict.keys()))
    cooker.cook()
    return cooker.serve(targets=targets)



def cifmol_process(
    cif_id: str,
    recipe_path: Path,
    cif_db_path: Path,
    targets: list[str] | None = None,
    **external_kwargs: object,
) -> dict:
    """Parse a CIF file using a predefined recipe."""
    cifmol_dict = load_cif(cif_id, env_path=cif_db_path)
    output = {}
    for key in cifmol_dict:
        cifmol = cifmol_dict[key]
        data_dict = {**cifmol, **external_kwargs}
        output[key] = base_process(
            data_dict,
            recipe_path=recipe_path,
            targets=targets,
        )
    return output

def cifmol_chunk_process(
    cif_id_chunk: list[str],
    recipe_path: Path,
    cif_db_path: Path,
    targets: list[str] | None = None,
    **external_kwargs: object,
) -> dict:
    """Parse a chunk of CIF files using a predefined recipe."""
    output = {}
    for cif_id in cif_id_chunk:
        cifmol_dict = load_cif(cif_id, env_path=cif_db_path)
        output[cif_id] = {}
        for key in cifmol_dict:
            cifmol = cifmol_dict[key]
            data_dict = {**cifmol, **external_kwargs}
            output[cif_id][key] = base_process(
                data_dict,
                recipe_path=recipe_path,
                targets=targets,
            )
    return output
