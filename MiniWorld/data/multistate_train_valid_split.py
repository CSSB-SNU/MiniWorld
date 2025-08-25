import os
import pickle
import random
from benchmark_data.format import MultiStateProteinFormat, MultiStateType

def load_GPCR_db(file_path="/home/psk6950/MiniWorld/MiniWorld/data/benchmark_data/multistate_gpcr.pickle"):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_pdb_ID_to_seq_cluster(metadata_path="/home/psk6950/data/BioMolDB_2024Oct21/metadata/metadata_psk_new.csv"):
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    # skip header
    pdb_id_to_seq_cluster = {}
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        pdb_id = parts[0].strip()
        seq_cluster = parts[4].strip()
        pdb_id_to_seq_cluster[pdb_id] = seq_cluster
    return pdb_id_to_seq_cluster

def load_seq_to_cluster(seq_to_cluster_path="/home/psk6950/data/BioMolDB_2024Oct21/cluster/seq_clust/seq_to_cluster.pkl"):
    if not os.path.exists(seq_to_cluster_path):
        raise FileNotFoundError(f"Sequence to cluster mapping file not found: {seq_to_cluster_path}")    
    with open(seq_to_cluster_path, 'rb') as f:
        return pickle.load(f)


def train_valid_split(train_path, valid_path, split_ratio=0.8):
    seq_to_cluster = load_seq_to_cluster()
    pdb_id_to_seq_cluster = load_pdb_ID_to_seq_cluster()

    clusters = set(seq_to_cluster.values())
    cluster_list = list(clusters)

    gpcr_db = load_GPCR_db()
    gpcr_pdb_ids = []
    for protein in gpcr_db:
        gpcr_pdb_ids.extend(protein.pdb_ids)

    # 8XML_R -> 8xml_R
    lower_gpcr_pdb_ids = []
    for pdb_id in gpcr_pdb_ids:
        pdb_id, chain_id = pdb_id.split('_')
        lower_gpcr_pdb_ids.append(f"{pdb_id.lower()}_{chain_id}")

    gpcr_cluster = set()
    for pdb_id in lower_gpcr_pdb_ids:
        if pdb_id in pdb_id_to_seq_cluster:
            seq_cluster = pdb_id_to_seq_cluster[pdb_id]
            gpcr_cluster.add(seq_cluster)

    valid_clusters = gpcr_cluster
    total_clusters = len(cluster_list)
    cluster_list = [c for c in cluster_list if c not in valid_clusters]
    random.shuffle(cluster_list)
    train_clusters_num = (total_clusters - len(valid_clusters)) * 0.8
    train_clusters = cluster_list[:int(train_clusters_num)]
    valid_clusters = list(gpcr_cluster) + cluster_list[int(train_clusters_num):]

    with open(train_path, 'wb') as f:
        pickle.dump(train_clusters, f)
    with open(valid_path, 'wb') as f:
        pickle.dump(valid_clusters, f)


if __name__ == "__main__":
    # Example usage
    train_path = "/home/psk6950/data/BioMolDB_2024Oct21/cluster/seq_clust/train_cluster.pkl"
    valid_path = "/home/psk6950/data/BioMolDB_2024Oct21/cluster/seq_clust/valid_cluster.pkl"
    seq_to_cluster = train_valid_split(train_path, valid_path, split_ratio=0.8)
    breakpoint()
