from __future__ import annotations

import argparse
import copy
import sys
from typing import TYPE_CHECKING

import torch
from team_gm.modules import (
    DiffusionTransformer,
    ImplementationType,
    MSAModule,
    Pairformer,
)
from torch.testing import assert_close

from miniworld.configs import SharedConfig
from miniworld.data.features.batch import Batch
from miniworld.diffusion import EDMScheduler, EuclideanDiffuser
from miniworld.models.af3_like.model import Model
from miniworld.modules.diffusion_module import DiffusionConditioning, DiffusionModule
from miniworld.modules.input_embedder import InputFeatureEmbedder
from miniworld.modules.msa_util import init_msa, init_token_single_msa

if TYPE_CHECKING:
    from collections.abc import Callable


def _build_shared_config() -> SharedConfig:
    num_res_class = 8
    d_single = 16
    d_profile = 32
    return SharedConfig(
        d_single=d_single,
        d_single_atom=8,
        d_single_token=16,
        d_single_token_input=d_single + num_res_class + d_profile + 1,
        d_pair=8,
        d_pair_atom=4,
        d_time=256,
        r_max=4,
        s_max=2,
        num_res_class=num_res_class,
        n_distogram_bins=7,
        implementation=ImplementationType.PYTORCH,
        use_checkpoint=False,
    )


def _build_model_config() -> Model.Config:
    shared = _build_shared_config()
    atom_dit = DiffusionTransformer.Config(
        d_single=shared.d_single_atom,
        d_cond=shared.d_single_atom,
        d_pair=shared.d_pair_atom,
        n_head=2,
        implementation=ImplementationType.PYTORCH,
        n_block=1,
        n_checkpoint_segments=None,
    )
    token_dit = DiffusionTransformer.Config(
        d_single=shared.d_single_token,
        d_cond=shared.d_single,
        d_pair=shared.d_pair,
        n_head=4,
        implementation=ImplementationType.PYTORCH,
        n_block=1,
        n_checkpoint_segments=None,
    )
    return Model.Config(
        shared=shared,
        input_feat_embbeder=atom_dit,
        trunk=Model.TrunkConfig(
            n_recycle_max=1,
            pairformer=Pairformer.Config(
                d_single=shared.d_single,
                d_pair=shared.d_pair,
                n_head_tri_attention=2,
                n_head_attention=4,
                p_drop=0.0,
                use_self_attention=True,
                implementation=ImplementationType.PYTORCH,
                n_block=1,
                n_checkpoint_segments=None,
            ),
            msa_module=MSAModule.Config(
                d_msa=8,
                d_pair=shared.d_pair,
                d_single_token_input=shared.d_single_token_input,
                d_hidden_msa=4,
                d_hidden_tri_multi=8,
                d_hidden_tri_attention=4,
                n_head_tri_attention=2,
                n_head_attention=4,
                p_drop_msa=0.0,
                p_drop=0.0,
                use_self_attention=False,
                implementation=ImplementationType.PYTORCH,
                num_res_class=shared.num_res_class,
                n_block=1,
                n_checkpoint_segments=None,
            ),
        ),
        diffusion=Model.DiffusionConfig(
            atom_dit=atom_dit,
            token_dit=token_dit,
            dit_cond=DiffusionConditioning.Config(
                n_expand=1,
                n_blocks=1,
            ),
        ),
    )


def _build_batch() -> Batch:
    torch.manual_seed(0)
    shared = _build_shared_config()
    batch = Batch.empty(
        n_temp=1,
        msa_depth=3,
        n_tokens=7,
        n_atoms=11,
    )

    valid_tokens = 5
    valid_atoms = 8
    batch.structure.token_mask[0, :valid_tokens] = True
    batch.structure.atom_mask[0, :valid_atoms] = True
    batch.structure.atom_pos_mask[0, :valid_atoms] = True
    batch.reference.mask[0, :valid_atoms] = 1.0

    batch.sequence.token_type[0, :valid_tokens] = torch.randint(
        0,
        shared.num_res_class,
        (valid_tokens,),
    )

    batch.msa.mask[0, :2] = True
    batch.msa.aligned_sequences[0, :2, :valid_tokens] = torch.randint(
        0,
        shared.num_res_class,
        (2, valid_tokens),
    )
    batch.msa.has_deletion[0, :2, :valid_tokens] = torch.randint(
        0,
        2,
        (2, valid_tokens),
    )
    batch.msa.deletion_value[0, :2, :valid_tokens] = torch.randn(2, valid_tokens)
    batch.msa.profile[0, :valid_tokens] = torch.randn(valid_tokens, 32)
    batch.msa.deletion_mean[0, :valid_tokens] = torch.randn(valid_tokens)

    batch.reference.pos[0, :valid_atoms] = torch.randn(valid_atoms, 3)
    batch.reference.element[0, :valid_atoms] = torch.randint(
        1,
        8,
        (valid_atoms,),
        dtype=torch.float32,
    )
    batch.reference.charge[0, :valid_atoms] = torch.randn(valid_atoms)
    batch.reference.space_uid[0, :valid_atoms] = torch.arange(valid_atoms) % valid_tokens

    batch.structure.atom_pos[0, :valid_atoms] = torch.randn(valid_atoms, 3)

    batch.scheme.token_idx[0] = torch.arange(batch.token_length)
    batch.scheme.token_residue_idx[0] = torch.arange(batch.token_length)
    batch.scheme.token_asym_id[0] = 0
    batch.scheme.token_entity_id[0] = 0
    batch.scheme.token_sym_id[0] = 0
    batch.scheme.atom_to_token_idx_map[0, :valid_atoms] = (
        torch.arange(valid_atoms) % valid_tokens
    )

    return batch


def _build_diffusion_inputs(
    batch: Batch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scheduler = EDMScheduler(EDMScheduler.EDMSchedulerConfig())
    diffuser = EuclideanDiffuser(
        EuclideanDiffuser.EuclideanConfig(seed=1),
        scheduler,
    )
    _, x_t, x_mask, t_emb, _ = diffuser.sample(
        batch.structure.atom_pos,
        mask=batch.structure.atom_mask,
        num_augment=2,
    )
    if x_mask is None:
        msg = "Diffuser.sample returned no mask."
        raise RuntimeError(msg)
    return x_t, x_mask, t_emb


def _randomize_float_parameters(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.no_grad():
        for name, param in module.named_parameters():
            if not param.is_floating_point():
                continue

            if param.ndim == 1 and "weight" in name and ("ln" in name or "norm" in name):
                value = 1.0 + 0.1 * torch.randn(
                    param.shape,
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
            elif param.ndim == 1:
                value = 0.1 * torch.randn(
                    param.shape,
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
            else:
                value = 0.2 * torch.randn(
                    param.shape,
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
            param.copy_(value.to(dtype=param.dtype))

    return module


def _valid_token_pair_mask(token_mask: torch.Tensor) -> torch.Tensor:
    return token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)


def _assert_equal_on_mask(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    assert_close(
        actual[mask],
        expected[mask],
        atol=1e-6,
        rtol=1e-6,
    )


def _perturb_masked_batch(batch: Batch) -> Batch:
    perturbed = copy.deepcopy(batch)
    shared = _build_shared_config()

    token_mask = perturbed.structure.token_mask[0]
    atom_mask = perturbed.structure.atom_mask[0]
    msa_mask = perturbed.msa.mask[0]
    masked_tokens = ~token_mask
    masked_atoms = ~atom_mask
    masked_msa = ~msa_mask

    if masked_tokens.any():
        num_masked_tokens = int(masked_tokens.sum().item())
        idx = torch.arange(num_masked_tokens, dtype=torch.long)
        perturbed.sequence.token_type[0, masked_tokens] = shared.num_res_class - 1
        perturbed.msa.aligned_sequences[0, :, masked_tokens] = (
            shared.num_res_class - 1
        )
        perturbed.msa.has_deletion[0, :, masked_tokens] = 1
        perturbed.msa.deletion_value[0, :, masked_tokens] = 1e3
        perturbed.msa.profile[0, masked_tokens] = -1e3
        perturbed.msa.deletion_mean[0, masked_tokens] = 1e3
        perturbed.scheme.token_idx[0, masked_tokens] = 100 + idx
        perturbed.scheme.token_residue_idx[0, masked_tokens] = 200 + idx
        perturbed.scheme.token_asym_id[0, masked_tokens] = 1
        perturbed.scheme.token_entity_id[0, masked_tokens] = 1
        perturbed.scheme.token_sym_id[0, masked_tokens] = 2

    if masked_msa.any():
        perturbed.msa.aligned_sequences[0, masked_msa, :] = shared.num_res_class - 1
        perturbed.msa.has_deletion[0, masked_msa, :] = 1
        perturbed.msa.deletion_value[0, masked_msa, :] = -1e3

    if masked_atoms.any():
        perturbed.reference.pos[0, masked_atoms] = 1e3
        perturbed.reference.element[0, masked_atoms] = 99.0
        perturbed.reference.charge[0, masked_atoms] = -1e3
        perturbed.reference.space_uid[0, masked_atoms] = 99
        perturbed.structure.atom_pos[0, masked_atoms] = -1e3
        perturbed.scheme.atom_to_token_idx_map[0, masked_atoms] = (
            perturbed.token_length - 1
        )

    return perturbed


def _perturb_masked_single(
    single: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    perturbed = single.clone()
    perturbed[:, ~token_mask[0]] = 1e3
    return perturbed


def _perturb_masked_pair(pair: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    perturbed = pair.clone()
    masked_pair = ~_valid_token_pair_mask(token_mask)
    perturbed[masked_pair] = -1e3
    return perturbed


def _perturb_masked_msa(
    msa: torch.Tensor,
    msa_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    perturbed = msa.clone()
    perturbed[:, ~msa_mask[0]] = 1e3
    perturbed[:, :, ~token_mask[0]] = -1e3
    return perturbed


def _perturb_masked_xt(x_t: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    perturbed = x_t.clone()
    perturbed[~x_mask] = 1e3
    return perturbed


def test_mask_input_feature_embedder() -> None:
    """Masked atom changes must not affect valid input embedder outputs."""
    batch = _build_batch()
    perturbed_batch = _perturb_masked_batch(batch)
    shared = _build_shared_config()
    module = _randomize_float_parameters(
        InputFeatureEmbedder(
            shared_config=shared,
            diffusion_config=DiffusionTransformer.Config(
                d_single=shared.d_single_atom,
                d_cond=shared.d_single_atom,
                d_pair=shared.d_pair_atom,
                n_head=2,
                implementation=ImplementationType.PYTORCH,
                n_block=1,
                n_checkpoint_segments=None,
            ),
        ),
        seed=11,
    ).eval()

    token_single_msa = init_token_single_msa(
        batch.msa,
        batch.sequence,
        num_res_class=shared.num_res_class,
    )
    perturbed_token_single_msa = init_token_single_msa(
        perturbed_batch.msa,
        perturbed_batch.sequence,
        num_res_class=shared.num_res_class,
    )

    with torch.no_grad():
        base_single_input, base_single_init, base_pair = module(
            token_single_msa,
            batch.reference,
            batch.scheme,
            batch.structure,
        )
        pert_single_input, pert_single_init, pert_pair = module(
            perturbed_token_single_msa,
            perturbed_batch.reference,
            perturbed_batch.scheme,
            perturbed_batch.structure,
        )

    token_mask = batch.structure.token_mask
    pair_mask = _valid_token_pair_mask(token_mask)
    _assert_equal_on_mask(pert_single_input, base_single_input, token_mask)
    _assert_equal_on_mask(pert_single_init, base_single_init, token_mask)
    _assert_equal_on_mask(pert_pair, base_pair, pair_mask)


def test_mask_msa_module() -> None:
    """Masked MSA and token changes must not affect valid MSA outputs."""
    batch = _build_batch()
    shared = _build_shared_config()
    module = _randomize_float_parameters(
        MSAModule(
            MSAModule.Config(
                d_msa=8,
                d_pair=shared.d_pair,
                d_single_token_input=shared.d_single_token_input,
                d_hidden_msa=4,
                d_hidden_tri_multi=8,
                d_hidden_tri_attention=4,
                n_head_tri_attention=2,
                n_head_attention=4,
                p_drop_msa=0.0,
                p_drop=0.0,
                use_self_attention=False,
                implementation=ImplementationType.PYTORCH,
                num_res_class=shared.num_res_class,
                n_block=1,
                n_checkpoint_segments=None,
            ),
        ),
        seed=22,
    ).eval()

    msa_feat, msa_mask = init_msa(
        batch.msa,
        num_res_class=shared.num_res_class,
    )
    pair = torch.randn(1, batch.token_length, batch.token_length, shared.d_pair)
    single = torch.randn(1, batch.token_length, shared.d_single_token_input)

    with torch.no_grad():
        base = module(
            msa_feat,
            msa_mask,
            pair,
            single,
            batch.structure.token_mask,
        )
        perturbed = module(
            _perturb_masked_msa(msa_feat, msa_mask, batch.structure.token_mask),
            msa_mask,
            _perturb_masked_pair(pair, batch.structure.token_mask),
            _perturb_masked_single(single, batch.structure.token_mask),
            batch.structure.token_mask,
        )

    _assert_equal_on_mask(
        perturbed,
        base,
        _valid_token_pair_mask(batch.structure.token_mask),
    )


def test_mask_pairformer() -> None:
    """Masked token changes must not affect valid Pairformer outputs."""
    batch = _build_batch()
    shared = _build_shared_config()
    module = _randomize_float_parameters(
        Pairformer(
            Pairformer.Config(
                d_single=shared.d_single,
                d_pair=shared.d_pair,
                n_head_tri_attention=2,
                n_head_attention=4,
                p_drop=0.0,
                use_self_attention=True,
                implementation=ImplementationType.PYTORCH,
                n_block=1,
                n_checkpoint_segments=None,
            ),
        ),
        seed=33,
    ).eval()

    pair = torch.randn(1, batch.token_length, batch.token_length, shared.d_pair)
    single = torch.randn(1, batch.token_length, shared.d_single)

    with torch.no_grad():
        base_pair, base_single = module(pair, single, batch.structure.token_mask)
        pert_pair, pert_single = module(
            _perturb_masked_pair(pair, batch.structure.token_mask),
            _perturb_masked_single(single, batch.structure.token_mask),
            batch.structure.token_mask,
        )

    token_mask = batch.structure.token_mask
    _assert_equal_on_mask(pert_single, base_single, token_mask)
    _assert_equal_on_mask(pert_pair, base_pair, _valid_token_pair_mask(token_mask))


def test_mask_diffusion_module() -> None:
    """Masked token and atom changes must not affect valid diffusion outputs."""
    batch = _build_batch()
    perturbed_batch = _perturb_masked_batch(batch)
    shared = _build_shared_config()
    x_t, x_mask, t_emb = _build_diffusion_inputs(batch)
    module = _randomize_float_parameters(
        DiffusionModule(
            shared_config=shared,
            atom_dit_config=DiffusionTransformer.Config(
                d_single=shared.d_single_atom,
                d_cond=shared.d_single_atom,
                d_pair=shared.d_pair_atom,
                n_head=2,
                implementation=ImplementationType.PYTORCH,
                n_block=1,
                n_checkpoint_segments=None,
            ),
            token_dit_config=DiffusionTransformer.Config(
                d_single=shared.d_single_token,
                d_cond=shared.d_single,
                d_pair=shared.d_pair,
                n_head=4,
                implementation=ImplementationType.PYTORCH,
                n_block=1,
                n_checkpoint_segments=None,
            ),
            dit_cond_config=DiffusionConditioning.Config(
                n_expand=1,
                n_blocks=1,
            ),
        ),
        seed=44,
    ).eval()

    token_single_input = torch.randn(1, batch.token_length, shared.d_single_token_input)
    token_single_trunk = torch.randn(1, batch.token_length, shared.d_single)
    token_pair_trunk = torch.randn(
        1,
        batch.token_length,
        batch.token_length,
        shared.d_pair,
    )

    with torch.no_grad():
        base = module(
            batch.reference,
            batch.scheme,
            batch.structure,
            x_t,
            x_mask,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )
        perturbed = module(
            perturbed_batch.reference,
            perturbed_batch.scheme,
            perturbed_batch.structure,
            _perturb_masked_xt(x_t, x_mask),
            x_mask,
            t_emb,
            _perturb_masked_single(token_single_input, batch.structure.token_mask),
            _perturb_masked_single(token_single_trunk, batch.structure.token_mask),
            _perturb_masked_pair(token_pair_trunk, batch.structure.token_mask),
        )

    _assert_equal_on_mask(perturbed, base, x_mask)


def test_mask_model() -> None:
    """Masked feature changes must not affect valid full-model outputs."""
    batch = _build_batch()
    perturbed_batch = _perturb_masked_batch(batch)
    x_t, x_mask, t_emb = _build_diffusion_inputs(batch)
    model = _randomize_float_parameters(Model(_build_model_config()), seed=55).eval()

    with torch.no_grad():
        base_atom, base_dist = model(
            msa=batch.msa,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
            x_t=x_t,
            x_mask=x_mask,
            t_emb=t_emb,
        )
        pert_atom, pert_dist = model(
            msa=perturbed_batch.msa,
            reference=perturbed_batch.reference,
            scheme=perturbed_batch.scheme,
            sequence=perturbed_batch.sequence,
            structure=perturbed_batch.structure,
            x_t=_perturb_masked_xt(x_t, x_mask),
            x_mask=x_mask,
            t_emb=t_emb,
        )

    _assert_equal_on_mask(pert_atom, base_atom, x_mask)
    _assert_equal_on_mask(
        pert_dist,
        base_dist,
        _valid_token_pair_mask(batch.structure.token_mask),
    )


CASE_ORDER = (
    "input_feature_embedder",
    "msa_module",
    "pairformer",
    "diffusion_module",
    "model",
)

CASES: dict[str, Callable[[], None]] = {
    "input_feature_embedder": test_mask_input_feature_embedder,
    "msa_module": test_mask_msa_module,
    "pairformer": test_mask_pairformer,
    "diffusion_module": test_mask_diffusion_module,
    "model": test_mask_model,
}


def _write_status(tag: str, case: str, detail: str | None = None) -> None:
    message = f"[{tag}] {case}"
    if detail is not None:
        message = f"{message}: {detail}"
    sys.stdout.write(f"{message}\n")


def _run_case(case: str) -> int:
    fn = CASES.get(case)
    if fn is None:
        msg = f"Unknown case: {case}"
        raise ValueError(msg)

    try:
        fn()
    except AssertionError as exc:
        _write_status("FAIL", case, f"{type(exc).__name__}: {exc}")
        return 1

    _write_status("PASS", case)
    return 0


def main() -> int:
    """Run mask invariance checks without invoking pytest directly."""
    parser = argparse.ArgumentParser(
        description="Run mask invariance checks without pytest.",
    )
    parser.add_argument(
        "--case",
        choices=["all", *CASE_ORDER],
        default="input_feature_embedder",
        help="Which check to run.",
    )
    args = parser.parse_args()

    if args.case == "all":
        status = 0
        for case in CASE_ORDER:
            status |= _run_case(case)
        return status

    return _run_case(args.case)


if __name__ == "__main__":
    raise SystemExit(main())
