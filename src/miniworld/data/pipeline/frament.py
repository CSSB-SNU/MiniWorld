from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
from biomol.core.container import FeatureContainer
from biomol.core.index import IndexTable
from biomol.core.utils import load_bytes
from numpy.typing import NDArray

from miniworld.data.mols import CCDMol, FragmentedCCDMol


def _find_rotatable_bonds(
    bond_type: NDArray,
    bond_conj: NDArray,
    src: NDArray,
    dst: NDArray,
    sssr: NDArray,
) -> NDArray[np.bool_]:
    """Return a boolean mask of rotatable bonds.

    A bond is rotatable if it is a single, non-conjugated bond whose two
    endpoint atoms do not share a ring.
    """
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
    return rotatable


def _find_components(
    num_atoms: int,
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
) -> NDArray[np.intp]:
    """Assign each atom to a connected component via non-rotatable bonds (BFS).

    Returns an array of shape (num_atoms,) with component IDs.
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rotatable)):
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

    return component_of


def _build_comp_adj(
    component_of: NDArray[np.intp],
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
) -> dict[int, list[int]]:
    """Build a component-level adjacency graph using rotatable bonds only."""
    comp_adj_sets: dict[int, set[int]] = defaultdict(set)
    for i in range(len(rotatable)):
        if rotatable[i]:
            c_s = component_of[int(src[i])]
            c_d = component_of[int(dst[i])]
            if c_s != c_d:
                comp_adj_sets[c_s].add(c_d)
                comp_adj_sets[c_d].add(c_s)
    return {k: sorted(v) for k, v in comp_adj_sets.items()}


def _build_component_graph(
    component_of: NDArray[np.intp],
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
) -> tuple[dict[int, list[int]], dict[int, int], NDArray]:
    """Build reusable component-level graph data for all merge levels."""
    num_components = int(component_of.max()) + 1
    comp_adj = _build_comp_adj(component_of, rotatable, src, dst)
    comp_degree = {c: len(comp_adj.get(c, [])) for c in range(num_components)}
    comp_size = np.bincount(component_of, minlength=num_components)
    return comp_adj, comp_degree, comp_size


def _walk_chain(
    start: int,
    anchor: int | None,
    comp_adj: dict[int, list[int]],
    visited_comps: list[bool],
    is_chain_node: Callable[[int], bool],
) -> list[int]:
    """Walk a linear chain of single-atom components starting from *start*.

    Marks visited nodes in visited_comps in-place and returns the ordered chain.
    """
    chain: list[int] = []
    cur: int | None = start
    prev = anchor
    while cur is not None and not visited_comps[cur]:
        chain.append(cur)
        visited_comps[cur] = True
        next_node = None
        for nb in comp_adj.get(cur, set()):
            if nb != prev and not visited_comps[nb] and is_chain_node(nb):
                next_node = nb
                break
        prev = cur
        cur = next_node
    return chain


def _find_end_anchor(
    chain: list[int],
    anchor: int | None,
    comp_to_fragment: dict[int, int],
    comp_adj: dict[int, list[int]],
    is_anchor: Callable[[int], bool],
) -> int | None:
    """Return the anchor at the far end of *chain*, or None if there is none."""
    if not chain:
        return None
    last_node = chain[-1]
    for nb in comp_adj.get(last_node, set()):
        if nb in comp_to_fragment and is_anchor(nb) and nb != anchor:
            return nb
    return None


def _assign_chain_between_anchors(
    chain: list[int],
    anchor: int,
    end_anchor: int,
    comp_to_fragment: dict[int, int],
    merge: int,
    frag_id: int,
) -> int:
    """Assign chain nodes when the chain connects two anchors.

    Absorbs *merge* nodes into each anchor's fragment; middle nodes get new
    chunked fragments.  Returns the next available frag_id.
    """
    chunk_size = merge + 1
    if len(chain) <= 2 * merge:
        mid = (len(chain) + 1) // 2
        front, back = chain[:mid], chain[mid:]
    else:
        front, back = chain[:merge], chain[-merge:]
    for c in front:
        comp_to_fragment[c] = comp_to_fragment[anchor]
    for c in back:
        comp_to_fragment[c] = comp_to_fragment[end_anchor]
    if len(chain) > 2 * merge:
        middle = chain[merge:-merge]
        for i in range(0, len(middle), chunk_size):
            fid = frag_id
            frag_id += 1
            for c in middle[i : i + chunk_size]:
                comp_to_fragment[c] = fid
    return frag_id


def _assign_free_chain(
    chain: list[int],
    anchor: int | None,
    comp_to_fragment: dict[int, int],
    merge: int,
    frag_id: int,
) -> int:
    """Assign chain nodes for a free-ended chain (at most one anchor).

    The first chunk is absorbed into *anchor*'s fragment when anchor is set.
    Returns the next available frag_id.
    """
    chunk_size = merge + 1
    for i in range(0, len(chain), chunk_size):
        chunk = chain[i : i + chunk_size]
        if i == 0 and anchor is not None:
            fid = comp_to_fragment[anchor]
        else:
            fid = frag_id
            frag_id += 1
        for c in chunk:
            comp_to_fragment[c] = fid
    return frag_id


def _handle_remaining_components(
    num_components: int,
    comp_to_fragment: dict[int, int],
    comp_adj: dict[int, list[int]],
    comp_size: NDArray,
    is_anchor: Callable[[int], bool],
    merge: int,
    frag_id: int,
) -> int:
    """Assign fragment IDs to components not yet covered (isolated or high-degree nodes).

    Returns the next available frag_id.
    """
    for c in range(num_components):
        if c in comp_to_fragment:
            continue
        if comp_size[c] == 1 and merge > 0:
            for nb in comp_adj.get(c, set()):
                if nb in comp_to_fragment and is_anchor(nb):
                    comp_to_fragment[c] = comp_to_fragment[nb]
                    break
        if c not in comp_to_fragment:
            comp_to_fragment[c] = frag_id
            frag_id += 1
    return frag_id


def _merge_components(
    component_of: NDArray[np.intp],
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
    merge: int,
) -> NDArray[np.intp]:
    """Merge neighbouring small components up to *merge* bonds deep.

    Returns fragment_of: an array of shape (num_atoms,) mapping each atom to a
    fragment ID.  When merge <= 0, each component stays its own fragment.
    """
    comp_adj, comp_degree, comp_size = _build_component_graph(
        component_of,
        rotatable,
        src,
        dst,
    )
    return _merge_components_from_graph(
        component_of,
        comp_adj,
        comp_degree,
        comp_size,
        merge,
    )


def _merge_components_from_graph(
    component_of: NDArray[np.intp],
    comp_adj: dict[int, list[int]],
    comp_degree: dict[int, int],
    comp_size: NDArray,
    merge: int,
) -> NDArray[np.intp]:
    """Merge components using precomputed component graph data."""
    num_atoms = len(component_of)
    num_components = len(comp_size)

    if merge <= 0 or num_components <= 1:
        return component_of.copy()

    # Anchors: multi-atom components (rings, conjugated groups) — never merged.
    is_anchor = lambda c: bool(comp_size[c] > 1)
    # Chain nodes: single-atom components with at most 2 rotatable-bond neighbours.
    is_chain_node = lambda c: bool(comp_size[c] == 1) and comp_degree.get(c, 0) <= 2

    visited_comps = [False] * num_components
    comp_to_fragment: dict[int, int] = {}
    frag_id = 0

    # Each anchor gets its own fragment.
    anchors = [c for c in range(num_components) if is_anchor(c)]
    for a in anchors:
        comp_to_fragment[a] = frag_id
        visited_comps[a] = True
        frag_id += 1

    # Collect chain-walk starting points.
    chain_starts: list[tuple[int, int | None]] = [
        (nb, a)
        for a in anchors
        for nb in comp_adj.get(a, set())
        if not visited_comps[nb] and is_chain_node(nb)
    ]
    chain_starts.extend(
        (c, None)
        for c in range(num_components)
        if not visited_comps[c] and is_chain_node(c) and comp_degree.get(c, 0) == 1
    )

    for start, anchor in chain_starts:
        if visited_comps[start]:
            continue
        chain = _walk_chain(start, anchor, comp_adj, visited_comps, is_chain_node)
        end_anchor = _find_end_anchor(
            chain,
            anchor,
            comp_to_fragment,
            comp_adj,
            is_anchor,
        )
        if anchor is not None and end_anchor is not None:
            frag_id = _assign_chain_between_anchors(
                chain,
                anchor,
                end_anchor,
                comp_to_fragment,
                merge,
                frag_id,
            )
        else:
            frag_id = _assign_free_chain(
                chain,
                anchor,
                comp_to_fragment,
                merge,
                frag_id,
            )

    frag_id = _handle_remaining_components(
        num_components,
        comp_to_fragment,
        comp_adj,
        comp_size,
        is_anchor,
        merge,
        frag_id,
    )

    return np.array(
        [comp_to_fragment[component_of[a]] for a in range(num_atoms)],
        dtype=int,
    )


def _max_effective_merge_from_components(
    component_of: NDArray[np.intp],
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
) -> int:
    """Return the v1 saturation point from precomputed fragmentation inputs."""
    comp_adj, comp_degree, comp_size = _build_component_graph(
        component_of,
        rotatable,
        src,
        dst,
    )
    return _max_effective_merge_from_graph(comp_adj, comp_degree, comp_size)


def _max_effective_merge_from_graph(
    comp_adj: dict[int, list[int]],
    comp_degree: dict[int, int],
    comp_size: NDArray,
) -> int:
    """Return the v1 saturation point using precomputed component graph data."""
    num_components = len(comp_size)
    is_anchor = lambda c: bool(comp_size[c] > 1)
    is_chain_node = lambda c: bool(comp_size[c] == 1) and comp_degree.get(c, 0) <= 2

    visited = [False] * num_components
    anchors = [c for c in range(num_components) if is_anchor(c)]
    anchor_set: set[int] = set(anchors)
    for a in anchors:
        visited[a] = True

    chain_starts: list[tuple[int, int | None]] = [
        (nb, a)
        for a in anchors
        for nb in comp_adj.get(a, set())
        if not visited[nb] and is_chain_node(nb)
    ]
    chain_starts.extend(
        (c, None)
        for c in range(num_components)
        if not visited[c] and is_chain_node(c) and comp_degree.get(c, 0) == 1
    )

    max_merge = 0
    for start, anchor in chain_starts:
        if visited[start]:
            continue
        chain = _walk_chain(start, anchor, comp_adj, visited, is_chain_node)
        L = len(chain)
        if L == 0:
            continue

        # Check if the far end of the chain terminates at another anchor.
        end_anchor = None
        for nb in comp_adj.get(chain[-1], set()):
            if nb in anchor_set and nb != anchor:
                end_anchor = nb
                break

        sat = (L + 1) // 2 if anchor is not None and end_anchor is not None else L - 1

        max_merge = max(max_merge, sat)

    return max_merge


def _max_effective_merge(mol: CCDMol) -> int:
    """Return the v1 saturation point (internal use only).

    This is the merge value at which the rotatable-bond fragmentation stops
    changing.  The public max_effective_merge adds 1 to account for the v2
    atomize step at merge=0.
    """
    bond_type = mol.atoms.bond_type.value
    bond_conj = mol.atoms.bond_conjugation.value
    src = mol.atoms.bond_type.src
    dst = mol.atoms.bond_type.dst
    sssr = mol.atoms.sssr_idx.value

    rotatable = _find_rotatable_bonds(bond_type, bond_conj, src, dst, sssr)
    component_of = _find_components(len(mol.atoms), rotatable, src, dst)
    return _max_effective_merge_from_components(component_of, rotatable, src, dst)


def max_effective_merge(mol: CCDMol) -> int:
    """Return the maximum merge value that has any effect on fragmentation.

    The v2 scale used by fragment_ccdmol:
      merge=0                  → atomize (each atom is its own fragment)
      merge=1..M               → progressively less fragmented
      merge=M                  → same result as v1 max_effective_merge
      merge=M+1                → one single fragment

    For merge > this value the result is always one fragment.
    """
    return _max_effective_merge(mol) + 1


def _atomize(mol: CCDMol) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Return one fragment per atom, ignoring all bonds."""
    num_atoms = len(mol.atoms)
    atom_to_res = np.arange(num_atoms, dtype=np.int64)
    fragmented_mol = _build_fragmented_mol(mol, atom_to_res, num_atoms)
    id_mappings: list[NDArray[np.intp]] = [
        np.array([i], dtype=np.intp) for i in range(num_atoms)
    ]
    return fragmented_mol, id_mappings


def _merge_all(mol: CCDMol) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Return the entire molecule as a single fragment."""
    num_atoms = len(mol.atoms)
    atom_to_res = np.zeros(num_atoms, dtype=np.int64)
    fragmented_mol = _build_fragmented_mol(mol, atom_to_res, 1)
    id_mappings: list[NDArray[np.intp]] = [np.arange(num_atoms, dtype=np.intp)]
    return fragmented_mol, id_mappings


def _build_fragmented_mol(
    mol: CCDMol,
    atom_to_res: NDArray[np.int64],
    num_fragments: int,
) -> FragmentedCCDMol:
    """Wrap atom/residue/chain data into a FragmentedCCDMol."""
    res_to_chain = np.zeros(num_fragments, dtype=np.int64)
    index_table = IndexTable.from_parents(atom_to_res, res_to_chain)

    atom_container = mol.atoms.get_container()

    residue_container = FeatureContainer.from_dict(
        {
            "nodes": {
                "residue_id": {"value": np.arange(num_fragments, dtype=np.int32)},
            },
            "edges": {},
        },
    )

    chain_container = FeatureContainer.from_dict(
        {
            "nodes": {
                "chain_id": {"value": np.array([0], dtype=np.int32)},
            },
            "edges": {},
        },
    )

    return FragmentedCCDMol(
        atom_container=atom_container,
        residue_container=residue_container,
        chain_container=chain_container,
        index_table=index_table,
    )


def _fragment_ccdmol_v1_from_components(
    mol: CCDMol,
    component_of: NDArray[np.intp],
    rotatable: NDArray[np.bool_],
    src: NDArray,
    dst: NDArray,
    merge: int,
) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Fragment a CCDMol using precomputed v1 fragmentation inputs."""
    comp_adj, comp_degree, comp_size = _build_component_graph(
        component_of,
        rotatable,
        src,
        dst,
    )
    return _fragment_ccdmol_v1_from_graph(
        mol,
        component_of,
        comp_adj,
        comp_degree,
        comp_size,
        merge,
    )


def _fragment_ccdmol_v1_from_graph(
    mol: CCDMol,
    component_of: NDArray[np.intp],
    comp_adj: dict[int, list[int]],
    comp_degree: dict[int, int],
    comp_size: NDArray,
    merge: int,
) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Fragment a CCDMol using precomputed v1 component graph data."""
    fragment_of = _merge_components_from_graph(
        component_of,
        comp_adj,
        comp_degree,
        comp_size,
        merge,
    )

    unique_frags, atom_to_res = np.unique(fragment_of, return_inverse=True)
    num_fragments = len(unique_frags)
    atom_to_res = atom_to_res.astype(np.int64)

    fragmented_mol = _build_fragmented_mol(mol, atom_to_res, num_fragments)
    id_mappings: list[NDArray[np.intp]] = [
        np.where(atom_to_res == frag)[0] for frag in range(num_fragments)
    ]
    return fragmented_mol, id_mappings


def _fragment_ccdmol_v1(
    mol: CCDMol,
    merge: int,
) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Core fragmentation logic (v1 scale: merge=0 cuts all rotatable bonds)."""
    bond_type = mol.atoms.bond_type.value
    bond_conj = mol.atoms.bond_conjugation.value
    src = mol.atoms.bond_type.src
    dst = mol.atoms.bond_type.dst
    sssr = mol.atoms.sssr_idx.value

    rotatable = _find_rotatable_bonds(bond_type, bond_conj, src, dst, sssr)
    component_of = _find_components(len(mol.atoms), rotatable, src, dst)
    return _fragment_ccdmol_v1_from_components(
        mol,
        component_of,
        rotatable,
        src,
        dst,
        merge,
    )


def fragment_ccdmol_all_merges(mol: CCDMol) -> dict[int, FragmentedCCDMol]:
    """Return all v2 merge levels for a molecule, reusing shared work.

    This is intended for dataset startup/precompute paths. Calling
    fragment_ccdmol repeatedly would recompute rotatable bonds, connected
    components, and the effective max merge for every merge level.
    """
    bond_type = mol.atoms.bond_type.value
    bond_conj = mol.atoms.bond_conjugation.value
    src = mol.atoms.bond_type.src
    dst = mol.atoms.bond_type.dst
    sssr = mol.atoms.sssr_idx.value

    rotatable = _find_rotatable_bonds(bond_type, bond_conj, src, dst, sssr)
    component_of = _find_components(len(mol.atoms), rotatable, src, dst)
    comp_adj, comp_degree, comp_size = _build_component_graph(
        component_of,
        rotatable,
        src,
        dst,
    )
    max_merge = _max_effective_merge_from_graph(comp_adj, comp_degree, comp_size) + 1

    atomized, _ = _atomize(mol)
    fragments: dict[int, FragmentedCCDMol] = {0: atomized}

    for merge in range(1, max_merge + 1):
        frag_mol, _ = _fragment_ccdmol_v1_from_graph(
            mol,
            component_of,
            comp_adj,
            comp_degree,
            comp_size,
            merge - 1,
        )
        fragments[merge] = frag_mol

    merged_all, _ = _merge_all(mol)
    fragments[max_merge + 1] = merged_all
    return fragments


def fragment_ccdmol(
    mol: CCDMol,
    merge: int = 0,
) -> tuple[FragmentedCCDMol, list[NDArray[np.intp]]]:
    """Fragment a CCDMol using the v2 merge scale.

    Args:
        mol: Input CCDMol with bond_type, bond_conjugation, sssr_idx features.
        merge: Fragmentation level.
               0          → atomize: each atom is its own fragment
               1          → cut all rotatable bonds
               k (1..M)   → progressively merge chain segments
               M+1        → one single fragment
               (M = max_effective_merge(mol))

    Returns:
        fragmented_mol: CCDMol where each fragment is a separate residue.
                        Atom features and edges are preserved as-is.
        id_mappings: list of arrays per residue, each containing original atom indices.

    """
    if merge <= 0:
        return _atomize(mol)
    if merge > max_effective_merge(mol):
        return _merge_all(mol)
    return _fragment_ccdmol_v1(mol, merge=merge - 1)


if __name__ == "__main__":
    import lmdb

    db_path = "/home/psk6950/data/CCD/preprocessed_CCD.lmdb"
    env = lmdb.open(db_path, readonly=True, lock=False)

    with env.begin() as txn:
        test = txn.get(b"HEM")
        test = load_bytes(test)
        mol = CCDMol.from_bytes(txn.get(b"HEM"))

    num_atoms = len(mol.atoms)
    max_merge = max_effective_merge(mol)
    print(f"Original: {num_atoms} atoms")
    print(f"max_effective_merge = {max_merge}")
    print()

    prev_counts: list[int] | None = None
    for merge_val in range(max_merge + 2):
        _, mappings = fragment_ccdmol(mol, merge=merge_val)
        counts = [len(m) for m in mappings]
        total = sum(counts)
        assert total == num_atoms, f"Atom count mismatch at merge={merge_val}"
        all_atoms = np.concatenate(mappings)
        assert len(np.unique(all_atoms)) == num_atoms, f"Duplicate/missing atoms at merge={merge_val}"
        changed = prev_counts != counts
        tag = "  (same)" if (prev_counts is not None and not changed) else ""
        print(f"merge={merge_val}: {len(mappings)} fragments{tag}")
        prev_counts = counts

    env.close()
    print("\nAll checks passed.")
