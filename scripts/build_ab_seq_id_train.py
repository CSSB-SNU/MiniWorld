"""Build metadata/ab_seq_id_train_20210930.tsv of a BioMolDB release (the antibody seq_id list the data configs name as train_db.antibody_seq_id_path).

Rows of metadata/ab_seq_id.tsv whose seq_id occurs in a train-cif entry that has pdb rows in train_item_20210930.tsv (the release-date <= 2021-09-30 training split:
the train cif lmdb holds only such entries). Reads the lmdb directly; CPU only, ~2 min with 16 workers.

    python scripts/build_ab_seq_id_train.py ROOT NPROC   (ROOT = release root, e.g. /home/psk6950/data/BioMolDB_20261014; point CIF at the node-local copy by editing P)
"""
from zstandard import ZstdDecompressor
P = sys.argv[1].rstrip("/") + "/lmdb/pdb/cif_attached/cif_pdb_attached_train_20210930_res9_chain300.lmdb"
M = sys.argv[1].rstrip("/") + "/metadata/"


def find(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "seq_id" and "chains" in path:
                yield path + (k,), v
            else:
                yield from find(v, path + (k,))


def work(args):
    i, n = args
    env = lmdb.open(P, readonly=True, lock=False, readahead=False)
    out = {}
    dec = ZstdDecompressor()
    with env.begin() as t:
        for j, (k, v) in enumerate(t.cursor()):
            if j % n != i:
                continue
            raw = dec.stream_reader(io.BytesIO(v)).read()
            h = int.from_bytes(raw[:8], "little")
            hd = json.loads(raw[8 : 8 + h])
            spans = {}
            off = 0
            for name, l in hd["arrays"].items():
                spans[name] = (off, l)
                off += l
            payload = memoryview(raw)[8 + h :]
            for path, node in find(hd["template"]):
                a = node["value"] if isinstance(node, dict) and "value" in node else node
                s, l = spans[a]
                arr = np.load(io.BytesIO(payload[s : s + l]), allow_pickle=False)
                out.setdefault(k.decode(), set()).update(map(str, arr.tolist()))
    return out


if __name__ == "__main__":
    n = int(sys.argv[2])
    with mp.Pool(n) as p:
        res = p.map(work, [(i, n) for i in range(n)])
    ab = {l.split("\t")[0] for i, l in enumerate(open(M + "ab_seq_id.tsv")) if i}
    ent = {}
    for o in res:
        ent.update(o)
    used = set()
    for i, l in enumerate(open(M + "train_item_20210930.tsv")):
        if i == 0:
            continue
        p = l.split("\t", 4)
        if p[0] == "pdb":
            used.add(p[3])
    print("entries", len(ent), "entries in train_item", len(used & set(ent)))
    all_ab = set().union(*(v & ab for v in ent.values()))
    used_ab = set().union(*(v & ab for e, v in ent.items() if e in used))
    print("ab all entries", len(all_ab), "ab in train_item entries", len(used_ab))
    with open(M + "ab_seq_id.tsv") as src, open(M + "ab_seq_id_train_20210930.tsv", "w") as dst:
        for i, line in enumerate(src):
            if i == 0 or line.split("\t", 1)[0] in used_ab:
                dst.write(line)
