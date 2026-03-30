import json
from pathlib import Path

import numpy as np
import torch
from biomol.core.utils import load_bytes

from miniworld.data.io.load import extract_lmdb_keys, load_raw_data

lmdb_path = Path("/public_data/preprocessed_CCD_fp_emb.lmdb")
out_tensor = Path("/public_data/fp_emb_table.pt")
out_vocab = Path("/public_data/fp_emb_vocab.json")

keys = extract_lmdb_keys(lmdb_path)
keys = sorted(keys)

vocab = {k: i for i, k in enumerate(keys)}
if "UNK" not in vocab:
    raise RuntimeError("UNK key is required in fingerprint LMDB for fallback.")

rows = []
for k in keys:
    raw = load_raw_data(k, lmdb_path)
    if raw is None:
        raise RuntimeError(f"Missing key in LMDB: {k}")
    obj = load_bytes(raw)
    fp = obj["residues"]["nodes"]["fingerprint_embedding"]["value"]  # shape (1, D)
    fp = np.asarray(fp, dtype=np.float32).reshape(-1)
    rows.append(fp)

emb = np.stack(rows, axis=0)  # (V, D), D should be 768 in your DB
torch.save(torch.from_numpy(emb), out_tensor)

with out_vocab.open("w") as f:
    json.dump(vocab, f)

print("saved:", out_tensor, emb.shape)
print("saved:", out_vocab, len(vocab))