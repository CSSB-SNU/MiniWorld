from pathlib import Path

import lmdb

from miniworld.data.io.load import load_a3m

if __name__ == "__main__":
    a3m_db_path = Path("/NHNHOME/WORKSPACE/0226010152_A/data/a3m.lmdb")

    # 먼저 LMDB에서 사용 가능한 key 몇 개를 확인
    env = lmdb.open(str(a3m_db_path), readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = []
        for i, (k, _) in enumerate(cursor):
            keys.append(k.decode())
            if i >= 4:
                break
    env.close()
    print(f"Sample keys: {keys}")

    # 첫 번째 key로 load_a3m 테스트
    # seq_id = keys[0]
    seq_id = "P0000740"
    print(f"\nLoading MSA for seq_id: {seq_id}")
    msa = load_a3m(key=seq_id, env_path=a3m_db_path)

    if msa is None:
        print("msa is None")
    else:
        print(f"type: {type(msa)}")
        print(f"seq_id: {msa.seq_id}")
        breakpoint()
