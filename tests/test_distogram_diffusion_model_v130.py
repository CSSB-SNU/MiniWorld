"""Small complete PyTorch trunk check, independent of GPU kernel availability."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel


def test_real_model_gradient_and_target_free_sampling(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_miniworld_distogram_train import _build_precompile_batch

    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        cfg = compose(config_name="distogram_diffusion_medium_v130_bioai", overrides=[
            "model.shared.implementation=pytorch",
            "model.input_feat_embbeder.implementation=pytorch",
            "model.input_feat_embbeder.n_block=1",
            "model.atom_swa.backend=sdpa",
            "model.trunk.msa_module.n_block=1",
            "model.trunk.msa_module.implementation=pytorch",
            "model.trunk.pairformer.n_block=1",
            "model.trunk.pairformer.implementation=pytorch",
            "model.trunk.template_embedder.implementation=PYTORCH",
            "model.trunk.template_embedder.n_block=1",
            "model.trunk.msa_subsample_per_recycle=8",
        ])
    model = MiniSWAModel(MiniSWAModel.Config.model_validate(cfg.model)).train()
    assert model.add_pair_recycle is None and model.distogram_head is None
    batch = _build_precompile_batch(
        device=torch.device("cpu"), msa_depth=16, n_tokens=16,
        n_atoms=64, n_templates=1, num_res_class=32,
    )
    batch.structure.atom_is_rep[:, ::4] = True
    batch.scheme.token_asym_id[:, 8:] = 1
    with torch.no_grad(): model.diffusion.decoder.weight.normal_(std=.01)
    loss, _ = model(
        batch.msa, batch.reference, batch.scheme, batch.sequence,
        batch.structure, batch.template, interchain_weight=2.,
    )
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    for prefix in ("diffusion.encoder", "input_feature_embedder", "pairformer_blocks", "msa_module"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads), prefix
    model.eval()
    torch.manual_seed(130)
    expected = model.sample_distogram(batch.msa, batch.reference, batch.scheme,
        batch.sequence, batch.structure, batch.template, num_steps=2)
    batch.structure.atom_pos.fill_(float("nan"))
    batch.structure.atom_pos_mask.zero_()
    batch.structure.atom_is_rep = None
    torch.manual_seed(130)
    actual = model.sample_distogram(batch.msa, batch.reference, batch.scheme,
        batch.sequence, batch.structure, batch.template, num_steps=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, actual.transpose(-1, -2), rtol=0, atol=0)
