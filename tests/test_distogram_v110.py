"""Chemical targets and exact interchain loss/gradient semantics."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniworld.data.features.convert import _build_atom_is_rep
from miniworld.loss.auxiliary import cal_atom_distogram_loss
from miniworld.utils.structure.distance import get_representative_distances


def make_mol(residues):
    names, atom_res, components = [], [], []
    for i, (component, atoms) in enumerate(residues):
        components.append(component)
        names.extend(atoms)
        atom_res.extend([i] * len(atoms))
    return NS(atoms=NS(id=np.array(names)),
              residues=NS(chem_comp_id=NS(value=np.array(components))),
              index_table=NS(atom_to_res=np.array(atom_res)))


@pytest.mark.parametrize('component,expected', [
    ('ALA', 'CB'), ('GLY', 'CA'), ('A', 'C4'), ('G', 'C4'),
    ('DA', 'C4'), ('DG', 'C4'), ('C', 'C2'), ('U', 'C2'), ('DC', 'C2'), ('DT', 'C2'),
])
def test_chemical_representative_and_atom_permutation(component, expected):
    names = ['P', "C4'", 'CA', 'C2', 'CB', 'C4', "C2'"]
    for atoms in [names, names[::-1]]:
        mol = make_mol([(component, atoms)])
        mask = _build_atom_is_rep(mol, np.zeros(len(atoms), dtype=int), np.ones(len(atoms), bool), 1)
        assert mol.atoms.id[mask].tolist() == [expected]
        observed = np.array(atoms) != expected
        assert not _build_atom_is_rep(mol, np.zeros(len(atoms), dtype=int), observed, 1).any()


def test_missing_cb_is_not_replaced_by_ca_or_other_atom():
    mol = make_mol([('ALA', ['N', 'CA', 'C']), ('A', ['P', "C4'"])])
    assert not _build_atom_is_rep(mol, mol.index_table.atom_to_res, np.ones(5, bool), 2).any()


def test_ligand_and_modified_residue_atom_tokens():
    mol = make_mol([('ATP', ['P', 'C4', 'N1']), ('MSE', ['N', 'CA', 'CB', 'SE'])])
    observed = np.array([1, 1, 0, 1, 1, 1, 1], bool)
    np.testing.assert_array_equal(_build_atom_is_rep(mol, np.arange(7), observed, 7), observed)


def inputs(device='cpu'):
    pos = torch.tensor([[[0., 0, 0], [1., 0, 0], [10., 0, 0], [100., 0, 0]]], device=device)
    return (pos, torch.ones(1, 4, dtype=torch.bool, device=device),
            torch.arange(4, device=device)[None], torch.tensor([[0, 0, 1, 1]], device=device))


def loss(logits, values, weight=2.):
    pos, mask, mapping, chains = values
    return cal_atom_distogram_loss(logits, pos, mask, mapping,
        rep_atom_mask=torch.ones_like(mask), token_asym_id=chains, interchain_weight=weight)


def test_interchain_loss_and_gradient_exactly_double_only_cross_chain_pairs():
    torch.manual_seed(5)
    logits = torch.randn(1, 4, 4, 96, requires_grad=True)
    v = inputs()
    plain = loss(logits, v, 1.)
    weighted = loss(logits, v)
    g1, = torch.autograd.grad(plain.sum(), logits, retain_graph=True)
    g2, = torch.autograd.grad(weighted.sum(), logits)
    expected = g1.clone()
    cross = v[3][:, :, None] != v[3][:, None, :]
    expected[cross] *= 2
    torch.testing.assert_close(g2, expected, rtol=0, atol=0)
    # Hand-computed six upper-triangle pairs, including under/overflow bins.
    distances = [1., 10., 100., 9., 99., 90.]
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    target = torch.bucketize(torch.tensor(distances), torch.linspace(2.25,25.75,95))
    ce = F.cross_entropy(torch.stack([logits[0,i,j] for i,j in pairs]), target, reduction='none')
    torch.testing.assert_close(weighted[0], (ce * torch.tensor([1.,2,2,2,2,1])).sum()/6)
    assert target[0] == 0 and target[-1] == 95
    v[3].zero_()
    torch.testing.assert_close(loss(logits, v, 2.), loss(logits, v, 1.))


def test_padding_nan_and_empty_targets():
    pos, mask, mapping, _ = inputs()
    pos[0,1,0] = float('nan')
    mapping[0,2] = -1
    mapping[0,3] = 999
    distances, pairs = get_representative_distances(pos, mask, mapping, 4, mask)
    assert torch.isfinite(distances).all() and pairs.sum() == 1
    logits = torch.randn(1,4,4,96,requires_grad=True)
    value = loss(logits, (pos,mask,mapping,torch.tensor([[0,0,1,1]])))
    value.sum().backward()
    assert value.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_representatives_must_coexist_in_same_conformer_and_padding_is_preserved():
    # Token 0 and 1 occur in disjoint conformers: no valid pair exists.
    pos = torch.zeros(1,2,2,3)
    mask = torch.tensor([[[True,False],[False,True]]])
    logits = torch.randn(1,4,4,96,requires_grad=True)
    value = cal_atom_distogram_loss(logits,pos,mask,torch.tensor([[0,1]]),
        rep_atom_mask=torch.ones(1,2,dtype=torch.bool),
        token_asym_id=torch.tensor([[0,1,0,0]]),interchain_weight=2.)
    value.sum().backward()
    assert value.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_v110_config_composition_and_client_wiring():
    from hydra import compose, initialize_config_dir
    from miniworld.models.distogram_only.client import Client
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(str(root/'configs/miniworld'),version_base=None):
        for name, length in [('phase1a_distogram_v110',384), ('phase1b_distogram_v110',768)]:
            cfg = compose(config_name=name)
            assert cfg.data.crop.max_tokens == length
            assert cfg.model.shared.n_distogram_bins == 96
            lc = Client.LossConfig.model_validate(dict(cfg.loss))
            assert lc.distogram_interchain_weight == 2. and lc.distogram_cb_target
    logits = torch.randn(1,4,4,96,requires_grad=True)
    pos, mask, mapping, chains = inputs()
    batch = NS(msa=None,reference=None,sequence=None,template=None,
               structure=NS(atom_pos=pos,atom_pos_mask=mask,atom_is_rep=mask),
               scheme=NS(atom_to_token_idx_map=mapping,token_asym_id=chains))
    client = NS(model=NS(forward=lambda **kw:logits),config=NS(loss=lc))
    actual,_ = Client.loss_fn(client,batch)
    torch.testing.assert_close(actual,loss(logits,inputs()))
    batch.structure.atom_is_rep = None
    with pytest.raises(ValueError,match='atom_is_rep'):
        Client.loss_fn(client,batch)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_weights_rejected(weight):
    from miniworld.models.distogram_only.client import Client
    with pytest.raises(ValueError):
        Client.LossConfig(distogram_interchain_weight=weight)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_compile_capture_replay_updates_coordinates_masks_and_chains():
    v = inputs('cuda')
    logits = torch.randn(1,4,4,96,device='cuda',requires_grad=True)
    compiled = torch.compile(loss, fullgraph=True, dynamic=False)
    def step():
        result = compiled(logits,v)
        result.sum().backward()
        return result
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            logits.grad = None; step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    logits.grad = None
    with torch.cuda.graph(graph):
        output = step()
    for iteration in range(2):
        if iteration:
            v[0][0,2,0] = 3.
            v[1][0,3] = False
            v[3].copy_(torch.tensor([[0,1,1,0]],device='cuda'))
        logits.grad.zero_(); graph.replay(); torch.cuda.synchronize()
        reference = logits.detach().clone().requires_grad_()
        expected = loss(reference,v); expected.sum().backward()
        torch.testing.assert_close(output,expected)
        torch.testing.assert_close(logits.grad,reference.grad)
