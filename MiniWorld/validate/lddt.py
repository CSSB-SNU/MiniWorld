import torch
import numpy as np

from team_gm.data.features_BioMol import Batch
from team_gm.utils import metrics
from jaxtyping import Float, Bool
from collections.abc import Sequence
from BioMol.utils.hierarchy import MoleculeType, PolymerType

Array = np.ndarray | torch.Tensor

def seperate_entity_types(batch: Batch) -> list[int]:
    _, assembly_ID, model_ID, alt_ID = batch.name[0].split('_')

    atom_chain_break = batch.scheme.atom_chain_break[0]
    chain_list = list(atom_chain_break.keys())

    entity_type = []
    entity_list = batch.chain.entity_list[0]
    for entity in entity_list:
        if entity.get_type() == MoleculeType.POLYMER:
            polyer_type = entity.get_polymer_type()
            match polyer_type:
                case PolymerType.PROTEIN:
                    entity_type.append('intra_protein')
                case PolymerType.DNA:
                    entity_type.append('intra_DNA')
                case PolymerType.RNA:
                    entity_type.append('intra_RNA')
                case _:
                    entity_type.append('intra_ligand')
        else:
            entity_type.append('intra_ligand')

    intra_idx_setup = []
    for ii, _entity_type in enumerate(entity_type):
        try:
            chain = chain_list[ii]
        except:
            breakpoint()
        start, end = atom_chain_break[chain]
        atom_idx = torch.arange(start, end+1)
        intra_idx_setup.append( (_entity_type, atom_idx))

    inter_idx_setup = []
    contact_edges = batch.chain.contact_graph[0]['edges']
    same_entity = batch.chain.same_entity[0]
    for edge in contact_edges:
        node1_type = entity_type[edge[0]]
        node2_type = entity_type[edge[1]]
        edge_type = None
        if node1_type == 'intra_protein':
            match node2_type:
                case 'intra_protein':
                    edge_type = 'protein-protein'
                case 'intra_DNA':
                    edge_type = 'protein-DNA'
                case 'intra_RNA':
                    edge_type = 'protein-RNA'
                case 'intra_ligand':
                    edge_type = 'protein-ligand'
        elif node2_type == 'intra_protein':
            match node1_type:
                case 'intra_DNA':
                    edge_type = 'protein-DNA'
                case 'intra_RNA':
                    edge_type = 'protein-RNA'
                case 'intra_ligand':
                    edge_type = 'protein-ligand'
        if edge_type is None:
            # not involved in protein
            continue

        entity_list1 = torch.where(same_entity[edge[0]])[0]
        entity_list2 = torch.where(same_entity[edge[1]])[0]
        atom_idx1_list = []
        atom_idx2_list = []
        for entity1 in entity_list1:
            chain1 = chain_list[entity1]
            start1, end1 = atom_chain_break[chain1]
            atom_idx1 = torch.arange(start1, end1+1)
            atom_idx1_list.append(atom_idx1)
        for entity2 in entity_list2:
            chain2 = chain_list[entity2]
            start2, end2 = atom_chain_break[chain2]
            atom_idx2 = torch.arange(start2, end2+1)
            atom_idx2_list.append(atom_idx2)

        inter_idx_setup.append( (edge_type, atom_idx1_list, atom_idx2_list) )
    return entity_type, intra_idx_setup, inter_idx_setup

def category_lddt(
    batch: Batch,
    pred_atom_pos: Float[Array, "L 3"],
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    ) -> dict[str, list[float]]:

    gt_atom_pos = batch.structure.atom_pos[0]
    atom_mask = batch.structure.atom_mask[0].bool()
    lddt_dict = {
        'intra_protein' : [], # 0
        'intra_DNA' : [], # 1
        'intra_RNA' : [], # 2
        'intra_ligand' : [], # 3
        'protein-protein' : [], # 4
        'protein-DNA' : [], # 5
        'protein-RNA' : [], # 6
        'protein-ligand' : [], # 7
        'total' : [], # 8
    }
    NA_relatted = ['intra_DNA', 'intra_RNA', 'protein-DNA', 'protein-RNA']

    entity_type, intra_idx_setup, inter_idx_setup = seperate_entity_types(batch)
    atom_chain_break = batch.scheme.atom_chain_break[0]

    NA_included = False
    for _type in entity_type:
        if _type in NA_relatted:
            NA_included = True
            break

    for _type, atom_idx in intra_idx_setup:
        if _type is None:
            continue
        pred_pos = pred_atom_pos[atom_idx]
        gt_pos = gt_atom_pos[atom_idx]
        mask = atom_mask[atom_idx]
        if mask.sum() < 10:
            # too small to calculate lddt
            continue
        if _type in NA_relatted:
            max_distance = 30.0
        else:
            max_distance = 15.0
        lddt = metrics.cal_atom_lddt(
            pred_atom_pos=pred_pos,
            gt_atom_pos=gt_pos,
            atom_mask=mask,
            max_distance=max_distance,
            distance_bins=distance_bins,
        )
        lddt_dict[_type].append(lddt)
    for _type, atom_idx1_list, atom_idx2_list in inter_idx_setup:
        if _type is None:
            continue
        if mask.sum() < 10:
            # too small to calculate lddt
            continue
        if _type in NA_relatted:
            max_distance = 30.0
        else:
            max_distance = 15.0
        _lddt_list = []
        for atom_idx1 in atom_idx1_list:
            for atom_idx2 in atom_idx2_list:
                atom_idx = torch.cat( (atom_idx1, atom_idx2), dim=0 )
                pred_pos = pred_atom_pos[atom_idx]
                gt_pos = gt_atom_pos[atom_idx]
                mask = atom_mask[atom_idx]
                lddt = metrics.cal_atom_interface_lddt(
                    pred_atom_pos=pred_pos,
                    gt_atom_pos=gt_pos,
                    atom_mask=mask,
                    chain_break=atom_chain_break,
                    max_distance=max_distance,
                    distance_bins=distance_bins,
                )
        lddt_dict[_type].append(lddt)

    total_lddt = metrics.cal_atom_lddt(
        pred_atom_pos=pred_atom_pos,
        gt_atom_pos=gt_atom_pos,
        atom_mask=atom_mask,
        max_distance=30.0 if NA_included else 15.0,
        distance_bins=distance_bins,
    )
    lddt_dict['total'].append(total_lddt)

    # average for each category
    for key in lddt_dict.keys():
        if len(lddt_dict[key]) > 0:
            lddt_dict[key] = (float(np.mean(lddt_dict[key])), len(lddt_dict[key]))
        else:
            lddt_dict[key] = None
    return lddt_dict
    # 
