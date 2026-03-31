from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from biomol.core.biomol import BioMol
from biomol.core.container import FeatureContainer
from biomol.core.index import IndexTable
from numpy.typing import NDArray


def fragment_biomol(
    mol: BioMol,
    merge: int = 0,
) -> tuple[BioMol, list[NDArray[np.intp]]]:
    """Fragment a BioMol by cutting rotatable bonds.

    Args:
        mol: Input CCD BioMol with bond_type, bond_conjugation, sssr_idx features.
        merge: Number of consecutive rotatable bonds to keep uncut.
               0 = cut all, 1 = keep pairs, 2 = keep triples, etc.

    Returns:
        fragmented_mol: BioMol where each fragment is a separate residue.
                        Atom features and edges are preserved as-is.
        id_mappings: list of arrays per residue, each containing original atom indices.

    """
    num_atoms = len(mol.atoms)

    # --- Step 1: Identify rotatable bonds ---
    bond_type = mol.atoms.bond_type.value
    bond_conj = mol.atoms.bond_conjugation.value
    src = mol.atoms.bond_type.src
    dst = mol.atoms.bond_type.dst
    sssr = mol.atoms.sssr_idx.value  # (num_atoms, max_rings)

    rotatable = np.zeros(len(bond_type), dtype=bool)
    for i in range(len(bond_type)):
        if bond_type[i] != "SING" or bond_conj[i]:
            continue
        s, d = int(src[i]), int(dst[i])
        src_rings = set(sssr[s][sssr[s] != -1])
        dst_rings = set(sssr[d][sssr[d] != -1])
        if src_rings & dst_rings:
            continue
        rotatable[i] = True

    # --- Step 2: Connected components via non-rotatable bonds ---
    adj: dict[int, list[int]] = defaultdict(list)
    for i in range(len(bond_type)):
        if not rotatable[i]:
            s, d = int(src[i]), int(dst[i])
            adj[s].append(d)
            adj[d].append(s)

    component_of = np.full(num_atoms, -1, dtype=int)
    comp_id = 0
    for atom in range(num_atoms):
        if component_of[atom] != -1:
            continue
        queue = deque([atom])
        component_of[atom] = comp_id
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if component_of[nb] == -1:
                    component_of[nb] = comp_id
                    queue.append(nb)
        comp_id += 1

    num_components = comp_id

    # --- Step 3: Merge components ---
    if merge <= 0 or num_components <= 1:
        # No merging needed
        fragment_of = component_of.copy()
    else:
        # Build component graph
        comp_adj: dict[int, set[int]] = defaultdict(set)
        for i in range(len(bond_type)):
            if rotatable[i]:
                c_s = component_of[int(src[i])]
                c_d = component_of[int(dst[i])]
                if c_s != c_d:
                    comp_adj[c_s].add(c_d)
                    comp_adj[c_d].add(c_s)

        comp_degree = {c: len(comp_adj[c]) for c in range(num_components)}
        comp_size = np.bincount(component_of, minlength=num_components)

        # A component is an anchor if it has more than 1 atom (rings, conjugated groups).
        # Multi-atom components must never be merged together.
        # Single-atom components are never anchors — they can always be merged.
        is_anchor = lambda c: comp_size[c] > 1
        # A component is walkable in a chain only if it's a single atom AND degree <= 2
        is_chain_node = lambda c: comp_size[c] == 1 and comp_degree.get(c, 0) <= 2

        visited_comps = [False] * num_components
        comp_to_fragment: dict[int, int] = {}
        frag_id = 0

        # Process anchors first — each gets its own fragment
        anchors = [c for c in range(num_components) if is_anchor(c)]
        for a in anchors:
            comp_to_fragment[a] = frag_id
            visited_comps[a] = True
            frag_id += 1

        # Walk linear chain segments from anchors and endpoints
        # Find chain starts: anchors with chain neighbors, or degree-1 endpoints
        chain_starts: list[
            tuple[int, int | None]
        ] = []  # (start_comp, from_anchor_or_None)
        for a in anchors:
            for nb in comp_adj.get(a, set()):
                if not visited_comps[nb] and is_chain_node(nb):
                    chain_starts.append((nb, a))

        # Also start from degree-1 endpoints not yet visited
        for c in range(num_components):
            if not visited_comps[c] and is_chain_node(c) and comp_degree.get(c, 0) == 1:
                chain_starts.append((c, None))

        for start, anchor in chain_starts:
            if visited_comps[start]:
                continue
            # Walk the chain
            chain: list[int] = []
            cur = start
            prev = anchor  # where we came from (anchor or None)
            while cur is not None and not visited_comps[cur]:
                chain.append(cur)
                visited_comps[cur] = True
                # Find next in chain
                next_node = None
                for nb in comp_adj.get(cur, set()):
                    if nb != prev and not visited_comps[nb] and is_chain_node(nb):
                        next_node = nb
                        break
                prev = cur
                cur = next_node

            # Detect if chain ends at another anchor
            end_anchor = None
            if chain:
                last_node = chain[-1]
                for nb in comp_adj.get(last_node, set()):
                    if nb in comp_to_fragment and is_anchor(nb) and nb != anchor:
                        end_anchor = nb
                        break

            # Assign chain nodes to fragments
            chunk_size = merge + 1
            if anchor is not None and end_anchor is not None:
                # Chain between two anchors: absorb `merge` nodes from each end
                if len(chain) <= 2 * merge:
                    mid = (len(chain) + 1) // 2
                    front, back = chain[:mid], chain[mid:]
                else:
                    front, back = chain[:merge], chain[-merge:]
                for c in front:
                    comp_to_fragment[c] = comp_to_fragment[anchor]
                for c in back:
                    comp_to_fragment[c] = comp_to_fragment[end_anchor]
                # Middle nodes (if any) get their own chunked fragments
                if len(chain) > 2 * merge:
                    middle = chain[merge:-merge]
                    for i in range(0, len(middle), chunk_size):
                        fid = frag_id
                        frag_id += 1
                        for c in middle[i : i + chunk_size]:
                            comp_to_fragment[c] = fid
            else:
                for i in range(0, len(chain), chunk_size):
                    chunk = chain[i : i + chunk_size]
                    if i == 0 and anchor is not None:
                        fid = comp_to_fragment[anchor]
                    else:
                        fid = frag_id
                        frag_id += 1
                    for c in chunk:
                        comp_to_fragment[c] = fid

        # Handle remaining components (high-degree single atoms, isolated nodes)
        for c in range(num_components):
            if c not in comp_to_fragment:
                # Single-atom components: try to merge with a neighboring anchor
                if comp_size[c] == 1 and merge > 0:
                    for nb in comp_adj.get(c, set()):
                        if nb in comp_to_fragment and is_anchor(nb):
                            comp_to_fragment[c] = comp_to_fragment[nb]
                            break
                if c not in comp_to_fragment:
                    comp_to_fragment[c] = frag_id
                    frag_id += 1

        # Map atoms to fragments
        fragment_of = np.array(
            [comp_to_fragment[component_of[a]] for a in range(num_atoms)],
            dtype=int,
        )

    # --- Step 4: Build output BioMol ---
    # Renumber fragments to be contiguous 0..N-1
    unique_frags, atom_to_res = np.unique(fragment_of, return_inverse=True)
    num_fragments = len(unique_frags)

    atom_to_res = atom_to_res.astype(np.int64)
    res_to_chain = np.zeros(num_fragments, dtype=np.int64)

    index_table = IndexTable.from_parents(atom_to_res, res_to_chain)

    # Keep atom container as-is
    atom_container = mol.atoms.get_container()

    # Build residue container
    residue_container = FeatureContainer.from_dict(
        {
            "nodes": {
                "residue_id": {"value": np.arange(num_fragments, dtype=np.int32)},
            },
            "edges": {},
        },
    )

    # Build chain container (single chain)
    chain_container = FeatureContainer.from_dict(
        {
            "nodes": {
                "chain_id": {"value": np.array([0], dtype=np.int32)},
            },
            "edges": {},
        },
    )

    fragmented_mol = BioMol(
        atom_container=atom_container,
        residue_container=residue_container,
        chain_container=chain_container,
        index_table=index_table,
    )

    # --- Step 5: Build id_mappings ---
    id_mappings: list[NDArray[np.intp]] = []
    for frag in range(num_fragments):
        id_mappings.append(np.where(atom_to_res == frag)[0])

    return fragmented_mol, id_mappings


if __name__ == "__main__":
    import lmdb

    db_path = "/public_data02/mjkang/preprocessed_CCD.lmdb"
    env = lmdb.open(db_path, readonly=True, lock=False)

    with env.begin() as txn:
        mol = BioMol.from_bytes(txn.get(b"HEM"))

    print(f"Original: {len(mol.atoms)} atoms")
    print()

    for merge_val in [0, 1, 2, 5]:
        frag_mol, mappings = fragment_biomol(mol, merge=merge_val)
        print(f"merge={merge_val}: {len(mappings)} fragments")
        for i, m in enumerate(mappings):
            print(f"  fragment {i}: {len(m)} atoms {m.tolist()}")
        total = sum(len(m) for m in mappings)
        assert total == len(mol.atoms), (
            f"Atom count mismatch: {total} != {len(mol.atoms)}"
        )
        # Verify each atom appears exactly once
        all_atoms = np.concatenate(mappings)
        assert len(np.unique(all_atoms)) == len(mol.atoms), "Duplicate or missing atoms"
        print()

    env.close()
    print("All checks passed.")
