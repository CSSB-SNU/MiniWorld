from pathlib import Path


def extract_edge_id_to_bias(data_path: Path) -> dict[str, list[str]]:
    # load edge_id to cif_ids mapping
    edge_id_to_bias: dict[str, list[str]] = {}
    with data_path.open("r") as f:
        for _line in f:
            line = _line.strip()
            if line == "":
                continue
            key1, key2, value = line.split("\t")
            edge_id = key1 if key2 == "None" else f"{key1}_{key2}"
            edge_id_to_bias[edge_id] = value.split(",")

    return edge_id_to_bias


def diff(
    edge_id_to_bias1: dict[str, list[str]],
    edge_id_to_bias2: dict[str, list[str]],
) -> dict[str, tuple[list[str], list[str]]]:
    diff_dict: dict[str, tuple[list[str], list[str]]] = {}
    for edge_id in set(edge_id_to_bias1.keys()) | set(edge_id_to_bias2.keys()):
        bias1 = edge_id_to_bias1.get(edge_id, [])
        bias2 = edge_id_to_bias2.get(edge_id, [])
        bias1 = sorted(bias1)
        bias2 = sorted(bias2)
        if bias1 != bias2:
            diff_dict[edge_id] = (bias1, bias2)
    return diff_dict


if __name__ == "__main__":
    data_path1 = Path(
        "/home/psk6950/data/BioMolDB_20260224/metadata/train_edge_node.tsv",
    )
    data_path2 = Path(
        "/home/psk6950/data/BioMolDB_20260224/metadata/train_edge_node_filtered.tsv",
    )

    edge_id_to_bias1 = extract_edge_id_to_bias(data_path1)
    edge_id_to_bias2 = extract_edge_id_to_bias(data_path2)

    differences = diff(edge_id_to_bias1, edge_id_to_bias2)
    test = list(differences.items())[:10]

    breakpoint()

    for edge_id, (bias1, bias2) in differences.items():
        print(f"Edge ID: {edge_id}")
        print(f"  Bias in data1: {bias1}")
        print(f"  Bias in data2: {bias2}")
        print()
