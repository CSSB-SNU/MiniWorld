"""Input feature embedder, forward + backward as one CUDA graph: wall time, kernel-family breakdown, the largest kernels.

The embedder is compiled (``dynamic=False``) and timed on the synthetic phase 1a batch (384 tokens, 4096 atoms, 4 templates). Run it
once per state of the code to build a ladder; each ``--before-*`` flag puts one optimisation back to its previous form:

    python -m benchmarks.embedder.ladder                                   # everything on
    python -m benchmarks.embedder.ladder --before-scatter-add              # the one-hot einsum atom -> token mean
    python -m benchmarks.embedder.ladder --before-fa4-lse                  # FA4 backward recomputes its forward
    MINIWORLD_FUSED_PAIR_INIT=0 python -m benchmarks.embedder.ladder       # the unfused token-pair initialisation

Measured on one B200 (engine ``5d8bb030`` + the embedder branches): 1.84 ms before, 1.41 with the FA4 lse, 1.38 with the scatter-add,
1.08 with the fused token-pair initialisation.
"""

from __future__ import annotations

import argparse

import torch
from hydra import compose, initialize_config_dir

from benchmarks.common import REPO_ROOT, apply_engine_settings, graph_ms, header, kernel_breakdown, use_repo_paths

use_repo_paths()


def one_hot_atom_to_token_mean(self, token_idx, atom_mask, atom_to_token_idx_map, atom_single_rep):
    """The atom -> token mean before the scatter-add: a [B, L_atom, L_token] one-hot mapping and two einsums."""
    atom_single_rep = atom_single_rep * atom_mask[..., None]
    to_add = self.atom_single_rep_to_token_single(atom_single_rep)
    mapping = torch.nn.functional.one_hot(atom_to_token_idx_map, num_classes=int(token_idx.shape[1])).to(to_add.dtype)
    token_sum = torch.einsum("bat,bad->btd", mapping, to_add)
    count = torch.einsum("bat,ba->bt", mapping, atom_mask.to(mapping.dtype))
    return token_sum / count.unsqueeze(-1).clamp(min=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/miniworld/phase1a_distogram_medium_v200.yaml")
    parser.add_argument("--n-block", type=int, default=3, help="SWA blocks of the embedder")
    parser.add_argument("--top", type=int, default=8, help="how many of the largest kernels to list")
    parser.add_argument("--before-scatter-add", action="store_true")
    parser.add_argument("--before-fa4-lse", action="store_true", help="needs an engine with settings.swa_flash_saves_lse")
    parser.add_argument("--engine-setting", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    import miniworld.models.distogram_only.model_mini_swa as model_module
    import miniworld.modules.input_feature_embedder_esmfold2_style as embedder_module
    import run_miniworld_distogram_train as trainer
    from miniworld.models.distogram_only import MiniSWAModel

    apply_engine_settings(args.engine_setting)
    if args.before_fa4_lse:
        apply_engine_settings(["swa_flash_saves_lse=False"])
    if args.before_scatter_add:
        embedder_module.ESMFold2InputAtomAttentionEncoder._scatter_atom_to_token = one_hot_atom_to_token_mean  # noqa: SLF001

    config_path = REPO_ROOT / args.config
    with initialize_config_dir(str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.name, overrides=["train.use_wandb=False", f"model.input_feat_embbeder.n_block={args.n_block}"])
    cfg = trainer.Config.model_validate(cfg)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    model = MiniSWAModel(cfg.model).to(device).train()
    batch = trainer._build_precompile_batch(  # noqa: SLF001
        device=device, msa_depth=2048, n_tokens=384, n_atoms=4096, n_templates=4, num_res_class=cfg.model.shared.num_res_class,
    ).to(device=device)
    embedder = model.input_feature_embedder
    params = [p for p in embedder.parameters() if p.requires_grad]

    def forward():
        token_single_msa = model_module.init_token_single_msa(batch.msa, batch.sequence, num_res_class=cfg.model.shared.num_res_class)
        outputs = embedder(token_single_msa, batch.reference, batch.scheme, batch.structure)
        return [o for o in outputs if isinstance(o, torch.Tensor) and o.requires_grad]

    compiled = torch.compile(forward, dynamic=False)
    upstream: list[torch.Tensor] = []

    def step():
        """Forward through the compiled embedder, backward from fixed upstream gradients (no loss reduction inside the timed graph)."""
        for p in params:
            p.grad = None if p.grad is None else p.grad.zero_()
        outputs = compiled()
        if not upstream:
            upstream.extend(torch.ones_like(o) for o in outputs)
        torch.autograd.backward(outputs, upstream)

    print(header("embedder"), flush=True)
    for p in params:  # the graph accumulates into existing .grad buffers
        p.grad = torch.zeros_like(p)
    wall = graph_ms(step)
    per_kernel, per_family = kernel_breakdown(step)
    total_us = sum(v[1] for v in per_kernel.values())
    launches = sum(v[0] for v in per_kernel.values())
    print(f"EMBEDDER n_block={args.n_block}: graph wall {wall:.2f} ms (forward + backward of the whole embedder), kernel sum {total_us / 1000:.2f} ms, {launches} launches")
    print("EMBEDDER by family:", {k: f"{v[1] / 1000:.2f} ms / {int(v[0])} launches" for k, v in sorted(per_family.items(), key=lambda kv: -kv[1][1])})
    for name, (count, micros) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"EMBEDDER   {micros / 1000:6.3f} ms x{int(count):<3d} {name[:130]}")
    short = sum(int(c) for n, (c, t) in per_kernel.items() if t / c < 10)
    print(f"EMBEDDER launches whose average duration is under 10 us: {short}")


if __name__ == "__main__":
    main()
