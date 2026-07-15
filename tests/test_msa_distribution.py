"""MSA DB 분포 분석: msa_depth, msa_depth * seq_len, item size."""

from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np

from miniworld.data.io.load import load_a3m


def main():
    a3m_db_path = Path("/NHNHOME/WORKSPACE/0226010152_A/data/a3m.lmdb")

    env = lmdb.open(str(a3m_db_path), readonly=True, lock=False)

    msa_depths = []
    seq_lens = []
    item_sizes = []  # bytes

    with env.begin() as txn:
        n_entries = txn.stat()["entries"]
        cursor = txn.cursor()
        print(f"Total entries in DB: {n_entries}")
        for i, (key_bytes, value_bytes) in enumerate(cursor):
            if (i + 1) % 1000 == 0:
                print(f"[{i + 1}/{n_entries}]")
            item_sizes.append(len(value_bytes))
            key = key_bytes.decode()
            try:
                msa = load_a3m(key=key, env_path=a3m_db_path)
                if msa is None:
                    continue
                # sequences shape: (L, N_seqs)
                seq_len = msa.sequences.shape[0]
                n_seqs = msa.sequences.shape[1]
                msa_depths.append(n_seqs)
                seq_lens.append(seq_len)
            except Exception as e:
                print(f"Error loading {key}: {e}")
                continue

    env.close()

    msa_depths = np.array(msa_depths)
    seq_lens = np.array(seq_lens)
    item_sizes_kb = np.array(item_sizes) / 1024
    depth_x_len = msa_depths * seq_lens

    print(f"Total entries: {len(item_sizes)}")
    print(f"Successfully loaded: {len(msa_depths)}")
    print(f"msa_depth  — mean: {msa_depths.mean():.1f}, median: {np.median(msa_depths):.1f}, max: {msa_depths.max()}")
    print(f"seq_len    — mean: {seq_lens.mean():.1f}, median: {np.median(seq_lens):.1f}, max: {seq_lens.max()}")
    print(f"depth*len  — mean: {depth_x_len.mean():.1f}, median: {np.median(depth_x_len):.1f}, max: {depth_x_len.max()}")
    print(f"item size  — mean: {item_sizes_kb.mean():.1f}KB, median: {np.median(item_sizes_kb):.1f}KB, max: {item_sizes_kb.max():.1f}KB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. msa_depth distribution
    ax = axes[0, 0]
    ax.hist(msa_depths, bins=100, edgecolor="black", alpha=0.7)
    ax.set_title("MSA Depth Distribution")
    ax.set_xlabel("msa_depth (N_seqs)")
    ax.set_ylabel("Count")
    ax.axvline(np.median(msa_depths), color="red", linestyle="--", label=f"median={np.median(msa_depths):.0f}")
    ax.legend()

    # 2. seq_len distribution
    ax = axes[0, 1]
    ax.hist(seq_lens, bins=100, edgecolor="black", alpha=0.7, color="orange")
    ax.set_title("Sequence Length Distribution")
    ax.set_xlabel("seq_len (L)")
    ax.set_ylabel("Count")
    ax.axvline(np.median(seq_lens), color="red", linestyle="--", label=f"median={np.median(seq_lens):.0f}")
    ax.legend()

    # 3. msa_depth * seq_len distribution
    ax = axes[1, 0]
    ax.hist(depth_x_len, bins=100, edgecolor="black", alpha=0.7, color="green")
    ax.set_title("MSA Depth × Seq Length Distribution")
    ax.set_xlabel("msa_depth × seq_len")
    ax.set_ylabel("Count")
    ax.axvline(np.median(depth_x_len), color="red", linestyle="--", label=f"median={np.median(depth_x_len):.0f}")
    ax.legend()

    # 4. item size distribution
    ax = axes[1, 1]
    ax.hist(item_sizes_kb, bins=100, edgecolor="black", alpha=0.7, color="purple")
    ax.set_title("Item Size Distribution")
    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("Count")
    ax.axvline(np.median(item_sizes_kb), color="red", linestyle="--", label=f"median={np.median(item_sizes_kb):.1f}KB")
    ax.legend()

    plt.tight_layout()
    plt.savefig("tests/msa_distribution.png", dpi=150)
    print("Saved to tests/msa_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
