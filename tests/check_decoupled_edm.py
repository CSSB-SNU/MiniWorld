"""Inspect EDM noising with dataloader2 dynamic tokenization."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    TokenizerConfig,
)
from miniworld.configs.data import DynamicTokenizationConfig
from miniworld.data.dataloader.dataloader2 import BioMolData
from miniworld.diffusion import (
    AF3Solver,
    DecoupledEDMDiffuser,
    DecoupledEDMScheduler,
    DecoupledEDMSolver,
    EDMScheduler,
    EuclideanDiffuser,
)
from miniworld.utils.structure.align import weighted_align
from miniworld.utils.structure.se3 import (
    apply_chain_rt,
    sample_rigid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miniworld.data.features import Batch


LOGGER = logging.getLogger(__name__)
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
DEFAULT_DATA_ROOT = Path("/home/psk6950/data")
DEFAULT_BIOMOL_ROOT = DEFAULT_DATA_ROOT / "BioMolDB_20260224"
DiffuserKind = str
SchedulerT = DecoupledEDMScheduler | EDMScheduler
DiffuserT = DecoupledEDMDiffuser | EuclideanDiffuser


@dataclass(frozen=True)
class MetricRow:
    """A single x0-vs-noisy structure metric row.

    sample_idx is the index inside one num_augment call. repeat_idx counts
    repeated diffuser.sample calls when building distribution plots.
    """

    mode: str
    diffuser_kind: str
    repeat_idx: int
    sample_idx: int
    batch_idx: int
    sigma_y: float
    sigma_rotation: float
    sigma_translation: float
    raw_rmsd: float
    global_rmsd: float
    chainwise_rmsd: float
    chain_rmsds: str


@dataclass(frozen=True)
class PDBFrame:
    """A PDB MODEL frame for sigma-sweep visualization."""

    label: str
    atom_pos: torch.Tensor
    sigma_y: float | None
    sigma_rotation: float | None
    sigma_translation: float | None
    global_rmsd: float | None
    chainwise_rmsd: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Load one dataloader2 batch, run EDM noising, "
            "report RMSDs, and write sigma-sweep multi-model PDB."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/output/decoupled_edm"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--diffuser-kind",
        choices=("decoupled", "edm", "both"),
        default="both",
        help="Which diffusion noising process to inspect.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-atoms", type=int, default=4096)
    parser.add_argument("--max-msa-depth", type=int, default=256)
    parser.add_argument("--sweep-steps", type=int, default=32)
    parser.add_argument(
        "--oracle-steps",
        type=int,
        default=32,
        help="Number of oracle solver reverse steps to write. Set to 0 to skip.",
    )
    parser.add_argument(
        "--pdb-batch-items",
        type=int,
        default=4,
        help=(
            "Number of batch items to write as separate PDB files. "
            "Capped by the loaded batch size."
        ),
    )
    parser.add_argument(
        "--distribution-repeats",
        type=int,
        default=4,
        help=(
            "Number of extra random diffuser.sample calls for RMSD distribution. "
            "Set to 0 to skip."
        ),
    )
    parser.add_argument(
        "--sweep-sigmas",
        default=None,
        help="Comma-separated sigma_y values. Defaults to scheduler sampling schedule.",
    )
    parser.add_argument(
        "--independent-noise",
        action="store_true",
        help="Use a different RNG seed for each sigma sweep point.",
    )
    parser.add_argument("--cif-db-path", type=Path, default=DEFAULT_BIOMOL_ROOT / "cif_attached_train.lmdb")
    parser.add_argument("--a3m-db-path", type=Path, default=DEFAULT_BIOMOL_ROOT / "a3m.lmdb")
    parser.add_argument(
        "--edge-id-to-bias-path",
        type=Path,
        default=DEFAULT_BIOMOL_ROOT / "metadata" / "train_edge_node.tsv",
    )
    parser.add_argument(
        "--template-db-path",
        type=Path,
        default=DEFAULT_BIOMOL_ROOT / "template.lmdb",
    )
    parser.add_argument(
        "--ccd-preprocessed-path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "CCD" / "preprocessed_CCD.lmdb",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_path(path: Path) -> None:
    """Fail early when an input path is missing."""
    if not path.exists():
        msg = f"Required path does not exist: {path}"
        raise FileNotFoundError(msg)


def build_dataset(args: argparse.Namespace) -> BioMolData:
    """Build dataloader2 BioMolData with dynamic tokenization."""
    for path in (
        args.cif_db_path,
        args.a3m_db_path,
        args.edge_id_to_bias_path,
        args.template_db_path,
        args.ccd_preprocessed_path,
    ):
        require_path(path)

    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            max_tokens=args.max_tokens,
            max_atoms=args.max_atoms,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            max_msa_depth=args.max_msa_depth,
            missing_policy="gap",
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=args.cif_db_path,
            a3m_db_path=args.a3m_db_path,
            edge_id_to_bias_path=args.edge_id_to_bias_path,
            template_db_path=args.template_db_path,
            ccd_preprocessed_path=args.ccd_preprocessed_path,
        ),
        tokenizer_config=TokenizerConfig(
            level="dynamic",
            dynamic_config=DynamicTokenizationConfig(
                minimum_resolution_ratio=[0.2, 0.6, 0.2],
                sigma_flat_prob=0.3,
                sigma_min=4.0,
                sigma_max=8.0,
            ),
            seed=args.seed,
        ),
    )
    return BioMolData(config)


def load_batch(args: argparse.Namespace, device: torch.device) -> Batch:
    """Load one batch from dataloader2."""
    dataset = build_dataset(args)
    dataloader = dataset.create_ddp_dataloader(
        rank=0,
        world_size=1,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
        num_workers=args.num_workers,
        bucket_token_multiple=128,
        bucket_atom_multiple=1024,
        batch_size=args.batch_size,
    )
    batch = next(iter(dataloader))
    return batch.to(device=device)


def build_diffuser(
    seed: int,
    diffuser_kind: DiffuserKind,
) -> tuple[SchedulerT, DiffuserT]:
    """Build EDM components."""
    if diffuser_kind == "decoupled":
        decoupled_scheduler = DecoupledEDMScheduler(
            DecoupledEDMScheduler.DecoupledEDMSchedulerConfig(),
        )
        decoupled_diffuser = DecoupledEDMDiffuser(
            config=DecoupledEDMDiffuser.DecoupledEDMConfig(
                seed=seed,
                translation_noise=0.0,
            ),
            scheduler=decoupled_scheduler,
        )
        return decoupled_scheduler, decoupled_diffuser

    if diffuser_kind == "edm":
        edm_scheduler = EDMScheduler(EDMScheduler.EDMSchedulerConfig())
        edm_diffuser = EuclideanDiffuser(
            config=EuclideanDiffuser.EuclideanConfig(seed=seed),
            scheduler=edm_scheduler,
        )
        return edm_scheduler, edm_diffuser

    msg = f"Unsupported diffuser_kind: {diffuser_kind}"
    raise ValueError(msg)


def restore_noisy_positions(
    scheduler: SchedulerT,
    x_input: torch.Tensor,
    sigma_y: torch.Tensor,
) -> torch.Tensor:
    """Invert EDM input scaling to recover physical noisy coordinates."""
    sigma_y = sigma_y.to(device=x_input.device, dtype=x_input.dtype)
    sigma_translation = None
    if isinstance(scheduler, DecoupledEDMScheduler):
        _, sigma_translation = scheduler.convert_to_sigma_rt(sigma_y)
    input_scale = scheduler.input_scale(sigma_y, sigma_translation).to(
        device=x_input.device,
        dtype=x_input.dtype,
    )
    return x_input / input_scale


def rigid_sigmas(
    scheduler: SchedulerT,
    sigma_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma_R/sigma_T or NaN tensors for plain EDM."""
    if isinstance(scheduler, DecoupledEDMScheduler):
        return scheduler.convert_to_sigma_rt(sigma_y)
    return torch.full_like(sigma_y, math.nan), torch.full_like(sigma_y, math.nan)


def raw_rmsd(
    probe_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Calculate unaligned RMSD over masked atoms."""
    valid = mask.bool()
    if int(valid.sum().item()) == 0:
        return math.nan
    diff_sq = (probe_pos - ref_pos).pow(2).sum(dim=-1)
    return float(torch.sqrt(diff_sq[valid].mean()).item())


@torch.no_grad()
def aligned_rmsd(
    probe_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Calculate Kabsch-aligned RMSD over masked atoms."""
    valid = mask.bool()
    finite = torch.isfinite(probe_pos).all(dim=-1) & torch.isfinite(ref_pos).all(dim=-1)
    valid = valid & finite
    if int(valid.sum().item()) < 3:
        return raw_rmsd(probe_pos, ref_pos, valid)

    probe = torch.where(valid[:, None], probe_pos, torch.zeros_like(probe_pos))
    ref = torch.where(valid[:, None], ref_pos, torch.zeros_like(ref_pos))
    aligned = weighted_align(
        probe.unsqueeze(0),
        ref.unsqueeze(0),
        weight=valid.unsqueeze(0).to(dtype=probe.dtype),
    )[0]
    return raw_rmsd(aligned, ref_pos, valid)


def chainwise_aligned_rmsd(
    probe_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    mask: torch.Tensor,
    atom_to_chain_id: torch.Tensor,
) -> tuple[float, str]:
    """Calculate atom-count-weighted chainwise aligned RMSD."""
    valid = mask.bool()
    if int(valid.sum().item()) == 0:
        return math.nan, ""

    chain_ids = torch.unique(atom_to_chain_id[valid]).detach().cpu().tolist()
    chain_ids = sorted(int(chain_id) for chain_id in chain_ids)
    weighted_sum = 0.0
    atom_count = 0
    chain_parts: list[str] = []
    for chain_id in chain_ids:
        chain_mask = valid & (atom_to_chain_id == chain_id)
        n_atoms = int(chain_mask.sum().item())
        rmsd = aligned_rmsd(probe_pos, ref_pos, chain_mask)
        chain_parts.append(f"{chain_id}:{rmsd:.4f}({n_atoms})")
        if math.isfinite(rmsd):
            weighted_sum += rmsd * n_atoms
            atom_count += n_atoms

    chain_mean = weighted_sum / atom_count if atom_count > 0 else math.nan
    return chain_mean, ";".join(chain_parts)


def collect_metrics(
    *,
    mode: str,
    diffuser_kind: DiffuserKind,
    repeat_idx: int,
    scheduler: SchedulerT,
    x0: torch.Tensor,
    x_input: torch.Tensor,
    mask: torch.Tensor,
    sigma_y: torch.Tensor,
    atom_to_chain_id: torch.Tensor,
) -> list[MetricRow]:
    """Collect x0-vs-noisy metrics for every augment and batch item."""
    noisy_pos = restore_noisy_positions(scheduler, x_input, sigma_y)
    sigma_rotation, sigma_translation = rigid_sigmas(scheduler, sigma_y)

    rows: list[MetricRow] = []
    for sample_idx in range(x0.shape[0]):
        for batch_idx in range(x0.shape[1]):
            sample_mask = mask[sample_idx, batch_idx].bool()
            chain_ids = atom_to_chain_id[batch_idx]
            raw = raw_rmsd(
                noisy_pos[sample_idx, batch_idx],
                x0[sample_idx, batch_idx],
                sample_mask,
            )
            global_aligned = aligned_rmsd(
                noisy_pos[sample_idx, batch_idx],
                x0[sample_idx, batch_idx],
                sample_mask,
            )
            chain_mean, chain_text = chainwise_aligned_rmsd(
                noisy_pos[sample_idx, batch_idx],
                x0[sample_idx, batch_idx],
                sample_mask,
                chain_ids,
            )
            rows.append(
                MetricRow(
                    mode=mode,
                    diffuser_kind=diffuser_kind,
                    repeat_idx=repeat_idx,
                    sample_idx=sample_idx,
                    batch_idx=batch_idx,
                    sigma_y=float(sigma_y[sample_idx, batch_idx].reshape(-1)[0].item()),
                    sigma_rotation=float(
                        sigma_rotation[sample_idx, batch_idx].reshape(-1)[0].item(),
                    ),
                    sigma_translation=float(
                        sigma_translation[sample_idx, batch_idx].reshape(-1)[0].item(),
                    ),
                    raw_rmsd=raw,
                    global_rmsd=global_aligned,
                    chainwise_rmsd=chain_mean,
                    chain_rmsds=chain_text,
                ),
            )
    return rows


def _expand_to_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a 1D tensor with one scalar per batch item."""
    value = value.reshape(-1)
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        msg = f"Expected scalar or {batch_size} values, got {value.shape}."
        raise ValueError(msg)
    return value


def add_solver_state(
    *,
    rows: list[MetricRow],
    frames_by_batch: dict[int, list[PDBFrame]],
    mode: str,
    diffuser_kind: DiffuserKind,
    sample_idx: int,
    scheduler: SchedulerT,
    x0: torch.Tensor,
    atom_pos: torch.Tensor,
    mask: torch.Tensor,
    sigma_y: torch.Tensor,
    atom_to_chain_id: torch.Tensor,
    label: str,
) -> None:
    """Record metrics and PDB frames for one solver state."""
    batch_size = atom_pos.shape[0]
    sigma_y = _expand_to_batch(
        sigma_y.to(device=atom_pos.device, dtype=atom_pos.dtype),
        batch_size,
    )
    sigma_rotation, sigma_translation = rigid_sigmas(scheduler, sigma_y)
    sigma_rotation = _expand_to_batch(
        sigma_rotation.to(device=atom_pos.device, dtype=atom_pos.dtype),
        batch_size,
    )
    sigma_translation = _expand_to_batch(
        sigma_translation.to(device=atom_pos.device, dtype=atom_pos.dtype),
        batch_size,
    )

    for batch_idx in range(batch_size):
        sample_mask = mask[batch_idx].bool()
        chain_ids = atom_to_chain_id[batch_idx]
        raw = raw_rmsd(atom_pos[batch_idx], x0[batch_idx], sample_mask)
        global_aligned = aligned_rmsd(atom_pos[batch_idx], x0[batch_idx], sample_mask)
        chain_mean, chain_text = chainwise_aligned_rmsd(
            atom_pos[batch_idx],
            x0[batch_idx],
            sample_mask,
            chain_ids,
        )
        sigma_y_value = float(sigma_y[batch_idx].item())
        sigma_rotation_value = float(sigma_rotation[batch_idx].item())
        sigma_translation_value = float(sigma_translation[batch_idx].item())
        rows.append(
            MetricRow(
                mode=mode,
                diffuser_kind=diffuser_kind,
                repeat_idx=0,
                sample_idx=sample_idx,
                batch_idx=batch_idx,
                sigma_y=sigma_y_value,
                sigma_rotation=sigma_rotation_value,
                sigma_translation=sigma_translation_value,
                raw_rmsd=raw,
                global_rmsd=global_aligned,
                chainwise_rmsd=chain_mean,
                chain_rmsds=chain_text,
            ),
        )
        if batch_idx not in frames_by_batch:
            frames_by_batch[batch_idx] = [
                PDBFrame(
                    label="x0",
                    atom_pos=x0[batch_idx].detach().cpu(),
                    sigma_y=None,
                    sigma_rotation=None,
                    sigma_translation=None,
                    global_rmsd=0.0,
                    chainwise_rmsd=0.0,
                ),
            ]
        frames_by_batch[batch_idx].append(
            PDBFrame(
                label=label,
                atom_pos=atom_pos[batch_idx].detach().cpu(),
                sigma_y=sigma_y_value,
                sigma_rotation=sigma_rotation_value,
                sigma_translation=sigma_translation_value,
                global_rmsd=global_aligned,
                chainwise_rmsd=chain_mean,
            ),
        )


def metric_mask(batch: Batch, sampled_mask: torch.Tensor) -> torch.Tensor:
    """Combine diffuser mask with finite-coordinate atom_pos_mask for metrics."""
    valid_pos_mask = batch.structure.atom_pos_mask.bool().unsqueeze(0)
    return sampled_mask.bool() & valid_pos_mask


def sample_training_like(
    diffuser: DiffuserT,
    batch: Batch,
    num_augment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the corresponding diffuser.sample call."""
    if isinstance(diffuser, DecoupledEDMDiffuser):
        x0, x_input, x_mask, _, sigma_y, _, _, _ = diffuser.sample(
            x0=batch.structure.atom_pos,
            mask=batch.structure.atom_mask,
            atom_to_chain_idx=batch.scheme.atom_to_chain_id,
            num_augment=num_augment,
        )
    else:
        x0_parts: list[torch.Tensor] = []
        x_input_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        sigma_parts: list[torch.Tensor] = []
        for batch_idx in range(batch.shape[0]):
            x0_i, x_input_i, x_mask_i, _, sigma_i = diffuser.sample(
                x0=batch.structure.atom_pos[batch_idx : batch_idx + 1],
                mask=batch.structure.atom_mask[batch_idx : batch_idx + 1],
                num_augment=num_augment,
            )
            if x_mask_i is None:
                msg = "EDM sample unexpectedly returned no mask."
                raise ValueError(msg)
            x0_parts.append(x0_i)
            x_input_parts.append(x_input_i)
            mask_parts.append(x_mask_i)
            sigma_parts.append(sigma_i)
        x0 = torch.cat(x0_parts, dim=1)
        x_input = torch.cat(x_input_parts, dim=1)
        x_mask = torch.cat(mask_parts, dim=1)
        sigma_y = torch.cat(sigma_parts, dim=1)
    if x_mask is None:
        msg = "EDM sample unexpectedly returned no mask."
        raise ValueError(msg)
    return x0, x_input, x_mask, sigma_y


def sample_distribution(
    *,
    diffuser: DiffuserT,
    scheduler: SchedulerT,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    num_augment: int,
    repeats: int,
    seed: int,
) -> list[MetricRow]:
    """Collect RMSD distribution over repeated random diffuser samples."""
    rows: list[MetricRow] = []
    for repeat_idx in range(repeats):
        set_seed(seed + repeat_idx + 1000)
        x0, x_input, mask, sigma_y = sample_training_like(
            diffuser,
            batch,
            num_augment=num_augment,
        )
        rows.extend(
            collect_metrics(
                mode="distribution",
                diffuser_kind=diffuser_kind,
                repeat_idx=repeat_idx,
                scheduler=scheduler,
                x0=x0,
                x_input=x_input,
                mask=metric_mask(batch, mask),
                sigma_y=sigma_y,
                atom_to_chain_id=batch.scheme.atom_to_chain_id,
            ),
        )
    return rows


def run_training_like_check(
    *,
    diffuser: DiffuserT,
    scheduler: SchedulerT,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    num_augment: int,
) -> list[MetricRow]:
    """Run one training-like num_augment sample and collect metrics."""
    train_x0, train_x_input, train_mask, train_sigma_y = sample_training_like(
        diffuser,
        batch,
        num_augment=num_augment,
    )
    return collect_metrics(
        mode="train_like",
        diffuser_kind=diffuser_kind,
        repeat_idx=0,
        scheduler=scheduler,
        x0=train_x0,
        x_input=train_x_input,
        mask=metric_mask(batch, train_mask),
        sigma_y=train_sigma_y,
        atom_to_chain_id=batch.scheme.atom_to_chain_id,
    )


def sample_fixed_sigma(
    diffuser: DiffuserT,
    batch: Batch,
    sigma_y_value: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run diffuser.sample while forcing a scalar sigma_y value."""
    if sigma_y_value <= 0:
        msg = "sigma_y values must be positive; x0 is written separately."
        raise ValueError(msg)

    set_seed(seed)
    scheduler = diffuser.scheduler
    original_sample_noise = scheduler.sample_noise

    def fixed_sample_noise(
        batch_size: int,
        uniform: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del uniform
        sigma_y = torch.full((batch_size,), float(sigma_y_value), dtype=torch.float32)
        if isinstance(scheduler, DecoupledEDMScheduler):
            sigma_rotation, sigma_translation = scheduler.convert_to_sigma_rt(sigma_y)
            return sigma_y, sigma_rotation, sigma_translation
        return sigma_y

    scheduler.sample_noise = fixed_sample_noise  # type: ignore[method-assign]
    try:
        return sample_training_like(diffuser, batch, num_augment=1)
    finally:
        scheduler.sample_noise = original_sample_noise  # type: ignore[method-assign]


def parse_sweep_sigmas(
    raw_sigmas: str | None,
    scheduler: SchedulerT,
    sweep_steps: int,
) -> list[float]:
    """Parse explicit sigma values or default to the scheduler sample schedule."""
    if raw_sigmas:
        sigmas = [float(token) for token in raw_sigmas.split(",") if token.strip()]
    else:
        schedule = scheduler.sampling_time_steps(sweep_steps)[:-1]
        sigmas = [float(value.item()) for value in schedule]
    if not sigmas:
        msg = "At least one sigma value is required."
        raise ValueError(msg)
    if any(sigma <= 0 for sigma in sigmas):
        msg = "All sigma_y values must be positive."
        raise ValueError(msg)
    return sigmas


def run_sigma_sweep(
    *,
    diffuser: DiffuserT,
    scheduler: SchedulerT,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    sigmas: Sequence[float],
    seed: int,
    independent_noise: bool,
) -> tuple[list[MetricRow], dict[int, list[PDBFrame]]]:
    """Run forced-sigma sampling and prepare PDB frames."""
    sweep_rows: list[MetricRow] = []
    pdb_frames_by_batch: dict[int, list[PDBFrame]] = {}

    for sigma_idx, sigma_y_value in enumerate(sigmas):
        sample_seed = seed + sigma_idx + 1 if independent_noise else seed
        x0, x_input, mask, sigma_y = sample_fixed_sigma(
            diffuser,
            batch,
            sigma_y_value=sigma_y_value,
            seed=sample_seed,
        )
        rows = collect_metrics(
            mode="sigma_sweep",
            diffuser_kind=diffuser_kind,
            repeat_idx=0,
            scheduler=scheduler,
            x0=x0,
            x_input=x_input,
            mask=metric_mask(batch, mask),
            sigma_y=sigma_y,
            atom_to_chain_id=batch.scheme.atom_to_chain_id,
        )
        sweep_rows.extend(rows)
        noisy_pos = restore_noisy_positions(scheduler, x_input, sigma_y)
        row_by_batch = {row.batch_idx: row for row in rows}
        for batch_idx, row in row_by_batch.items():
            if batch_idx not in pdb_frames_by_batch:
                pdb_frames_by_batch[batch_idx] = [
                    PDBFrame(
                        label="x0",
                        atom_pos=x0[0, batch_idx].detach().cpu(),
                        sigma_y=None,
                        sigma_rotation=None,
                        sigma_translation=None,
                        global_rmsd=0.0,
                        chainwise_rmsd=0.0,
                    ),
                ]
            pdb_frames_by_batch[batch_idx].append(
                PDBFrame(
                    label=f"sigma_{sigma_idx:03d}",
                    atom_pos=noisy_pos[0, batch_idx].detach().cpu(),
                    sigma_y=row.sigma_y,
                    sigma_rotation=row.sigma_rotation,
                    sigma_translation=row.sigma_translation,
                    global_rmsd=row.global_rmsd,
                    chainwise_rmsd=row.chainwise_rmsd,
                ),
            )

    return sweep_rows, pdb_frames_by_batch


def run_edm_oracle_solver(
    *,
    scheduler: EDMScheduler,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    num_steps: int,
    seed: int,
) -> tuple[list[MetricRow], dict[int, list[PDBFrame]]]:
    """Run an oracle EDM reverse solver trajectory toward the known x0."""
    set_seed(seed)
    solver = AF3Solver(AF3Solver.SolverConfig(seed=seed), scheduler)
    x0 = batch.structure.atom_pos.to(dtype=torch.float32)
    mask = batch.structure.atom_mask.bool() & batch.structure.atom_pos_mask.bool()
    time_steps = scheduler.sampling_time_steps(num_steps).to(device=x0.device)

    sigma_0 = scheduler.sampling_schedule(time_steps[0]).to(
        device=x0.device,
        dtype=x0.dtype,
    )
    x = -torch.randn_like(x0) * sigma_0

    rows: list[MetricRow] = []
    frames_by_batch: dict[int, list[PDBFrame]] = {}
    add_solver_state(
        rows=rows,
        frames_by_batch=frames_by_batch,
        mode="oracle_solver",
        diffuser_kind=diffuser_kind,
        sample_idx=0,
        scheduler=scheduler,
        x0=x0,
        atom_pos=x,
        mask=mask,
        sigma_y=sigma_0.reshape(1),
        atom_to_chain_id=batch.scheme.atom_to_chain_id,
        label="oracle_init",
    )

    for step_idx in range(num_steps):
        sigma_i = scheduler.sampling_schedule(time_steps[step_idx]).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        sigma_next = scheduler.sampling_schedule(time_steps[step_idx + 1]).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        gamma = solver.gamma_0 if sigma_next > solver.gamma_min else 0
        sigma_hat = sigma_i * (1 + gamma)
        added_noise = (
            solver._lambda  # noqa: SLF001
            * torch.sqrt(sigma_hat**2 - sigma_i**2)
            * torch.randn_like(x)
        )
        x = x + added_noise

        c_skip = scheduler.skip_scale(sigma_hat)
        c_out = scheduler.output_scale(sigma_hat)
        x_update = (x0 - c_skip * x) / c_out
        x_denoised = c_skip * x + c_out * x_update
        velocity = (x - x_denoised) / sigma_hat
        x = x + solver.step_scale * (sigma_next - sigma_hat) * velocity

        add_solver_state(
            rows=rows,
            frames_by_batch=frames_by_batch,
            mode="oracle_solver",
            diffuser_kind=diffuser_kind,
            sample_idx=step_idx + 1,
            scheduler=scheduler,
            x0=x0,
            atom_pos=x,
            mask=mask,
            sigma_y=sigma_next.reshape(1),
            atom_to_chain_id=batch.scheme.atom_to_chain_id,
            label=f"oracle_step_{step_idx:03d}",
        )

    return rows, frames_by_batch


def run_decoupled_oracle_solver(
    *,
    scheduler: DecoupledEDMScheduler,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    num_steps: int,
    seed: int,
) -> tuple[list[MetricRow], dict[int, list[PDBFrame]]]:
    """Run an oracle decoupled reverse solver trajectory toward the known x0."""
    set_seed(seed)
    solver = DecoupledEDMSolver(DecoupledEDMSolver.SolverConfig(seed=seed), scheduler)
    x0 = batch.structure.atom_pos.to(dtype=torch.float32)
    mask = batch.structure.atom_mask.bool() & batch.structure.atom_pos_mask.bool()
    atom_to_chain_id = batch.scheme.atom_to_chain_id
    batch_size = x0.shape[0]
    time_steps = scheduler.sampling_time_steps(num_steps).to(device=x0.device)

    sigma_0 = scheduler.sampling_schedule(time_steps[0]).to(
        device=x0.device,
        dtype=x0.dtype,
    )
    sigma_rotation, sigma_translation = scheduler.convert_to_sigma_rt(sigma_0)
    sigma_rotation = _expand_to_batch(
        sigma_rotation.to(device=x0.device, dtype=x0.dtype),
        batch_size,
    )
    sigma_translation = _expand_to_batch(
        sigma_translation.to(device=x0.device, dtype=x0.dtype),
        batch_size,
    )
    y = torch.randn_like(x0) * sigma_0
    group_num = int(atom_to_chain_id.max().item()) + 1
    rotation, translation = sample_rigid(
        sigma_rotation,
        sigma_translation,
        C=group_num,
        device=x0.device,
        dtype=x0.dtype,
    )

    rows: list[MetricRow] = []
    frames_by_batch: dict[int, list[PDBFrame]] = {}
    atom_pos = apply_chain_rt(y, rotation, translation, atom_to_chain_id)
    add_solver_state(
        rows=rows,
        frames_by_batch=frames_by_batch,
        mode="oracle_solver",
        diffuser_kind=diffuser_kind,
        sample_idx=0,
        scheduler=scheduler,
        x0=x0,
        atom_pos=atom_pos,
        mask=mask,
        sigma_y=sigma_0.reshape(1),
        atom_to_chain_id=atom_to_chain_id,
        label="oracle_init",
    )

    for step_idx in range(num_steps):
        sigma_i = scheduler.sampling_schedule(time_steps[step_idx]).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        sigma_next = scheduler.sampling_schedule(time_steps[step_idx + 1]).to(
            device=x0.device,
            dtype=x0.dtype,
        )
        gamma = solver.gamma_0 if sigma_next > solver.gamma_min else 0
        sigma_hat = sigma_i * (1 + gamma)
        sigma_rotation_hat, sigma_translation_hat = scheduler.convert_to_sigma_rt(
            sigma_hat,
        )
        sigma_rotation_hat = _expand_to_batch(
            sigma_rotation_hat.to(device=x0.device, dtype=x0.dtype),
            batch_size,
        )
        sigma_translation_hat = _expand_to_batch(
            sigma_translation_hat.to(device=x0.device, dtype=x0.dtype),
            batch_size,
        )
        rotation_hat, translation_hat = sample_rigid(
            sigma_rotation_hat,
            sigma_translation_hat,
            C=group_num,
            device=x0.device,
            dtype=x0.dtype,
        )

        added_noise = (
            solver._lambda  # noqa: SLF001
            * torch.sqrt(sigma_hat**2 - sigma_i**2)
            * torch.randn_like(y)
        )
        y = y + added_noise

        c_skip = scheduler.skip_scale(sigma_hat)
        c_out = scheduler.output_scale(sigma_hat)
        x_update = (x0 - c_skip * y) / c_out
        x_denoised = c_skip * y + c_out * x_update
        velocity = (y - x_denoised) / sigma_hat
        y = y + solver.step_scale * (sigma_next - sigma_hat) * velocity

        sigma_rotation_next, sigma_translation_next = scheduler.convert_to_sigma_rt(
            sigma_next,
        )
        sigma_rotation_next = _expand_to_batch(
            sigma_rotation_next.to(device=x0.device, dtype=x0.dtype),
            batch_size,
        )
        sigma_translation_next = _expand_to_batch(
            sigma_translation_next.to(device=x0.device, dtype=x0.dtype),
            batch_size,
        )
        rotation, translation = sample_rigid(
            sigma_rotation_next,
            sigma_translation_next,
            C=group_num,
            device=x0.device,
            dtype=x0.dtype,
        )
        atom_pos = apply_chain_rt(y, rotation, translation, atom_to_chain_id)
        add_solver_state(
            rows=rows,
            frames_by_batch=frames_by_batch,
            mode="oracle_solver",
            diffuser_kind=diffuser_kind,
            sample_idx=step_idx + 1,
            scheduler=scheduler,
            x0=x0,
            atom_pos=atom_pos,
            mask=mask,
            sigma_y=sigma_next.reshape(1),
            atom_to_chain_id=atom_to_chain_id,
            label=f"oracle_step_{step_idx:03d}",
        )

    return rows, frames_by_batch


def run_oracle_solver(
    *,
    scheduler: SchedulerT,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    num_steps: int,
    seed: int,
) -> tuple[list[MetricRow], dict[int, list[PDBFrame]]]:
    """Run an oracle reverse solver trajectory for the selected scheduler."""
    if num_steps <= 0:
        return [], {}
    if isinstance(scheduler, DecoupledEDMScheduler):
        return run_decoupled_oracle_solver(
            scheduler=scheduler,
            diffuser_kind=diffuser_kind,
            batch=batch,
            num_steps=num_steps,
            seed=seed,
        )
    return run_edm_oracle_solver(
        scheduler=scheduler,
        diffuser_kind=diffuser_kind,
        batch=batch,
        num_steps=num_steps,
        seed=seed,
    )


def feature_value(obj: Any) -> Any:
    """Unwrap biomol Feature objects."""
    return obj.value if hasattr(obj, "value") else obj


def safe_array_item(values: Any, idx: int, default: str) -> str:
    """Read a Python string from a feature/list/array with bounds fallback."""
    raw_values = feature_value(values)
    try:
        if 0 <= idx < len(raw_values):
            return str(raw_values[idx])
    except TypeError:
        return default
    return default


def safe_bool_item(values: Any, idx: int, default: bool = False) -> bool:
    """Read a Python bool from a feature/list/array with bounds fallback."""
    raw_values = feature_value(values)
    try:
        if 0 <= idx < len(raw_values):
            return bool(raw_values[idx])
    except TypeError:
        return default
    return default


def format_atom_name(atom_name: str) -> str:
    """Format an atom name for fixed-width PDB output."""
    clean_name = atom_name[:4] if atom_name else "X"
    return f"{clean_name:<4s}" if len(clean_name) >= 4 else f" {clean_name:<3s}"


def pdb_line(
    *,
    serial: int,
    record: str,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_seq: int,
    xyz: torch.Tensor,
    b_factor: float,
) -> str:
    """Format one PDB ATOM/HETATM line."""
    x, y, z = xyz.detach().cpu().tolist()
    return (
        f"{record:<6s}{serial:5d} {format_atom_name(atom_name)} "
        f"{residue_name[:3]:>3s} {chain_id:1s}{residue_seq % 10000:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}"
    )


def write_multimodel_pdb(
    batch: Batch,
    frames: Sequence[PDBFrame],
    path: Path,
    diffuser_kind: DiffuserKind,
    batch_idx: int = 0,
    trajectory_name: str = "sigma sweep",
) -> None:
    """Write x0 and sigma-sweep noisy structures as a multi-model PDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (
        batch.structure.atom_mask[batch_idx].bool()
        & batch.structure.atom_pos_mask[batch_idx].bool()
    )
    valid_indices = mask.detach().cpu().nonzero(as_tuple=False).flatten().tolist()
    atom_to_token = batch.scheme.atom_to_token_idx_map[batch_idx]
    token_residue_idx = batch.scheme.token_residue_idx[batch_idx]
    atom_to_chain = batch.scheme.atom_to_chain_id[batch_idx]
    atom_ids = batch.atom_ids[batch_idx]
    chem_comp_ids = batch.chem_comp_ids[batch_idx]
    heteros = batch.heteros[batch_idx]

    with path.open("w") as handle:
        handle.write(
            f"REMARK MiniWorld {diffuser_kind} {trajectory_name}: "
            f"{batch.name[batch_idx]}\n",
        )
        for model_idx, frame in enumerate(frames, start=1):
            handle.write(f"MODEL     {model_idx:4d}\n")
            handle.write(
                "REMARK "
                f"{frame.label} "
                f"sigma_y={frame.sigma_y} "
                f"sigma_rotation={frame.sigma_rotation} "
                f"sigma_translation={frame.sigma_translation} "
                f"global_rmsd={frame.global_rmsd} "
                f"chainwise_rmsd={frame.chainwise_rmsd}\n",
            )
            b_factor = (
                min(math.log10(frame.sigma_y + 1e-8), 99.99)
                if frame.sigma_y is not None
                else 0.0
            )
            for serial, atom_idx in enumerate(valid_indices, start=1):
                token_idx = int(atom_to_token[atom_idx].item())
                residue_idx = int(token_residue_idx[token_idx].item())
                chain_int = int(atom_to_chain[atom_idx].item())
                chain_id = CHAIN_IDS[chain_int % len(CHAIN_IDS)]
                atom_name = safe_array_item(atom_ids, atom_idx, "X")
                residue_name = safe_array_item(chem_comp_ids, residue_idx, "UNK")
                is_hetero = safe_bool_item(heteros, residue_idx)
                record = "HETATM" if is_hetero else "ATOM"
                handle.write(
                    pdb_line(
                        serial=serial,
                        record=record,
                        atom_name=atom_name,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_seq=residue_idx,
                        xyz=frame.atom_pos[atom_idx],
                        b_factor=b_factor,
                    )
                    + "\n",
                )
            handle.write("ENDMDL\n")
        handle.write("END\n")


def write_sigma_sweep_pdbs(
    batch: Batch,
    frames_by_batch: dict[int, list[PDBFrame]],
    output_dir: Path,
    output_prefix: str,
    diffuser_kind: DiffuserKind,
    pdb_batch_items: int,
) -> list[Path]:
    """Write one sigma-sweep multi-model PDB per selected batch item."""
    written_paths: list[Path] = []
    for batch_idx in sorted(frames_by_batch)[:pdb_batch_items]:
        path = (
            output_dir / f"{output_prefix}_sigma_sweep.pdb"
            if batch_idx == 0
            else output_dir / f"{output_prefix}_sigma_sweep_batch{batch_idx:03d}.pdb"
        )
        write_multimodel_pdb(
            batch=batch,
            frames=frames_by_batch[batch_idx],
            path=path,
            diffuser_kind=diffuser_kind,
            batch_idx=batch_idx,
        )
        written_paths.append(path)
    return written_paths


def write_oracle_solver_pdbs(
    batch: Batch,
    frames_by_batch: dict[int, list[PDBFrame]],
    output_dir: Path,
    output_prefix: str,
    diffuser_kind: DiffuserKind,
    pdb_batch_items: int,
) -> list[Path]:
    """Write one oracle-solver trajectory PDB per selected batch item."""
    written_paths: list[Path] = []
    for batch_idx in sorted(frames_by_batch)[:pdb_batch_items]:
        path = (
            output_dir / f"{output_prefix}_oracle_solver.pdb"
            if batch_idx == 0
            else output_dir / f"{output_prefix}_oracle_solver_batch{batch_idx:03d}.pdb"
        )
        write_multimodel_pdb(
            batch=batch,
            frames=frames_by_batch[batch_idx],
            path=path,
            diffuser_kind=diffuser_kind,
            batch_idx=batch_idx,
            trajectory_name="oracle solver trajectory",
        )
        written_paths.append(path)
    return written_paths


def write_metrics_csv(rows: Sequence[MetricRow], path: Path) -> None:
    """Write metric rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(MetricRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def sigma_rt_curve(
    scheduler: DecoupledEDMScheduler,
    num_points: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate sigma_R and sigma_T over the configured sigma_y range."""
    sigma_y_min = scheduler.config.sigma_y_min * scheduler.config.sigma_data
    sigma_y_max = scheduler.config.sigma_y_max * scheduler.config.sigma_data
    sigma_y = torch.logspace(
        math.log10(sigma_y_max),
        math.log10(sigma_y_min),
        steps=num_points,
    )
    sigma_rotation, sigma_translation = scheduler.convert_to_sigma_rt(sigma_y)
    return sigma_y, sigma_rotation, sigma_translation


def write_sigma_rt_curve_csv(
    scheduler: DecoupledEDMScheduler,
    path: Path,
) -> None:
    """Write the sigma_y-to-rigid-noise curve to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sigma_y, sigma_rotation, sigma_translation = sigma_rt_curve(scheduler)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sigma_y", "sigma_rotation", "sigma_translation"],
        )
        writer.writeheader()
        for sig_y, sig_r, sig_t in zip(
            sigma_y.tolist(),
            sigma_rotation.tolist(),
            sigma_translation.tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "sigma_y": sig_y,
                    "sigma_rotation": sig_r,
                    "sigma_translation": sig_t,
                },
            )


def plot_sigma_sweep(rows: Sequence[MetricRow], path: Path) -> None:
    """Plot sigma sweep RMSDs if matplotlib is available."""
    sweep_rows = [row for row in rows if row.mode == "sigma_sweep" and row.batch_idx == 0]
    if not sweep_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping metric plot.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, ax = plt.subplots(figsize=(7, 4))
        sigmas = [row.sigma_y for row in sweep_rows]
        ax.plot(sigmas, [row.raw_rmsd for row in sweep_rows], marker="o", label="raw")
        ax.plot(
            sigmas,
            [row.global_rmsd for row in sweep_rows],
            marker="o",
            label="global aligned",
        )
        ax.plot(
            sigmas,
            [row.chainwise_rmsd for row in sweep_rows],
            marker="o",
            label="chainwise aligned",
        )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("sigma_y")
        ax.set_ylabel("RMSD")
        title_kind = sweep_rows[0].diffuser_kind if sweep_rows else "EDM"
        ax.set_title(f"{title_kind} sigma sweep")
        ax.grid(visible=True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def make_log_bins(values: Sequence[float], max_bins: int = 40) -> np.ndarray | int:
    """Build positive log-spaced histogram bins, with linear fallback."""
    positive_values = [value for value in values if value > 0 and math.isfinite(value)]
    if len(positive_values) < 2:
        return 1
    min_value = min(positive_values)
    max_value = max(positive_values)
    if min_value == max_value:
        return min(5, len(positive_values))
    bin_count = min(max_bins, max(5, int(math.sqrt(len(positive_values)))))
    return np.geomspace(min_value, max_value, num=bin_count + 1)


def plot_rmsd_distribution(rows: Sequence[MetricRow], path: Path) -> None:
    """Plot global and chainwise RMSD distributions."""
    distribution_rows = [row for row in rows if row.mode == "distribution"]
    if not distribution_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping RMSD distribution plot.")
        return

    records = [
        (row.sigma_y, row.global_rmsd, row.chainwise_rmsd)
        for row in distribution_rows
        if math.isfinite(row.global_rmsd) and math.isfinite(row.chainwise_rmsd)
    ]
    global_values = [record[1] for record in records]
    chainwise_values = [record[2] for record in records]
    if not global_values or not chainwise_values:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        global_bins = make_log_bins(global_values)
        chainwise_bins = make_log_bins(chainwise_values)
        axes[0, 0].hist(global_values, bins=global_bins)
        axes[0, 0].set_title("Global aligned RMSD")
        axes[0, 0].set_xlabel("RMSD")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].set_xscale("log")
        axes[0, 1].hist(chainwise_values, bins=chainwise_bins)
        axes[0, 1].set_title("Chainwise aligned RMSD")
        axes[0, 1].set_xlabel("RMSD")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].set_xscale("log")

        sigma_values = [record[0] for record in records]
        axes[1, 0].scatter(sigma_values, global_values, s=12)
        axes[1, 0].set_title("Global RMSD by sigma_y")
        axes[1, 0].set_xlabel("sigma_y")
        axes[1, 0].set_ylabel("RMSD")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].invert_xaxis()
        axes[1, 1].scatter(sigma_values, chainwise_values, s=12)
        axes[1, 1].set_title("Chainwise RMSD by sigma_y")
        axes[1, 1].set_xlabel("sigma_y")
        axes[1, 1].set_ylabel("RMSD")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].invert_xaxis()
        title_kind = distribution_rows[0].diffuser_kind if distribution_rows else "EDM"
        fig.suptitle(f"Repeated random {title_kind} samples")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_sigma_rt_curve(
    scheduler: DecoupledEDMScheduler,
    path: Path,
    sweep_sigmas: Sequence[float] | None = None,
) -> None:
    """Plot sigma_R and sigma_T as functions of sigma_y."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping sigma RT curve plot.")
        return

    sigma_y, sigma_rotation, sigma_translation = sigma_rt_curve(scheduler)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sigma_y, sigma_rotation, label="sigma_R")
        ax.plot(sigma_y, sigma_translation, label="sigma_T")
        if sweep_sigmas:
            sweep_tensor = torch.tensor(sweep_sigmas, dtype=sigma_y.dtype)
            sweep_r, sweep_t = scheduler.convert_to_sigma_rt(sweep_tensor)
            ax.scatter(sweep_tensor, sweep_r, s=12, label="sweep sigma_R")
            ax.scatter(sweep_tensor, sweep_t, s=12, label="sweep sigma_T")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("sigma_y")
        ax.set_ylabel("noise scale")
        ax.set_title("Decoupled EDM rigid noise schedule")
        ax.grid(visible=True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def log_distribution_summary(rows: Sequence[MetricRow]) -> None:
    """Log mean/std/min/max for distribution rows."""
    distribution_rows = [row for row in rows if row.mode == "distribution"]
    if not distribution_rows:
        return

    for metric_name in ("global_rmsd", "chainwise_rmsd"):
        values = torch.tensor(
            [
                getattr(row, metric_name)
                for row in distribution_rows
                if math.isfinite(getattr(row, metric_name))
            ],
        )
        if values.numel() == 0:
            continue
        LOGGER.info(
            (
                "distribution %s: n=%d mean=%.4f std=%.4f "
                "min=%.4f max=%.4f"
            ),
            metric_name,
            values.numel(),
            values.mean().item(),
            values.std(unbiased=False).item(),
            values.min().item(),
            values.max().item(),
        )


def write_outputs(
    *,
    output_dir: Path,
    output_prefix: str,
    diffuser_kind: DiffuserKind,
    batch: Batch,
    all_rows: Sequence[MetricRow],
    distribution_rows: Sequence[MetricRow],
    pdb_frames_by_batch: dict[int, list[PDBFrame]],
    oracle_rows: Sequence[MetricRow],
    oracle_frames_by_batch: dict[int, list[PDBFrame]],
    pdb_batch_items: int,
    scheduler: SchedulerT,
    sigmas: Sequence[float],
) -> None:
    """Write CSV, PDB, and plot outputs."""
    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    distribution_metrics_path = output_dir / f"{output_prefix}_distribution_metrics.csv"
    oracle_metrics_path = output_dir / f"{output_prefix}_oracle_solver_metrics.csv"
    plot_path = output_dir / f"{output_prefix}_sigma_sweep.png"
    distribution_plot_path = output_dir / f"{output_prefix}_rmsd_distribution.png"

    write_metrics_csv(all_rows, metrics_path)
    write_metrics_csv(distribution_rows, distribution_metrics_path)
    pdb_paths = write_sigma_sweep_pdbs(
        batch=batch,
        frames_by_batch=pdb_frames_by_batch,
        output_dir=output_dir,
        output_prefix=output_prefix,
        diffuser_kind=diffuser_kind,
        pdb_batch_items=pdb_batch_items,
    )
    oracle_pdb_paths = []
    if oracle_rows:
        write_metrics_csv(oracle_rows, oracle_metrics_path)
        oracle_pdb_paths = write_oracle_solver_pdbs(
            batch=batch,
            frames_by_batch=oracle_frames_by_batch,
            output_dir=output_dir,
            output_prefix=output_prefix,
            diffuser_kind=diffuser_kind,
            pdb_batch_items=pdb_batch_items,
        )
    plot_sigma_sweep(all_rows, plot_path)
    plot_rmsd_distribution(all_rows, distribution_plot_path)
    if isinstance(scheduler, DecoupledEDMScheduler):
        sigma_rt_curve_path = output_dir / f"{output_prefix}_sigma_rt_curve.png"
        sigma_rt_curve_csv_path = output_dir / f"{output_prefix}_sigma_rt_curve.csv"
        plot_sigma_rt_curve(scheduler, sigma_rt_curve_path, sweep_sigmas=sigmas)
        write_sigma_rt_curve_csv(scheduler, sigma_rt_curve_csv_path)
        LOGGER.info("Wrote sigma RT curve plot: %s", sigma_rt_curve_path)
        LOGGER.info("Wrote sigma RT curve CSV: %s", sigma_rt_curve_csv_path)
    else:
        LOGGER.info("Skipped sigma RT curve for plain EDM.")

    LOGGER.info("Wrote metrics: %s", metrics_path)
    LOGGER.info("Wrote distribution metrics: %s", distribution_metrics_path)
    LOGGER.info("Wrote sigma-sweep PDB models: %s", ", ".join(map(str, pdb_paths)))
    if oracle_rows:
        LOGGER.info("Wrote oracle solver metrics: %s", oracle_metrics_path)
        LOGGER.info(
            "Wrote oracle solver PDB models: %s",
            ", ".join(map(str, oracle_pdb_paths)),
        )
    LOGGER.info("Wrote sigma-sweep plot: %s", plot_path)
    LOGGER.info("Wrote RMSD distribution plot: %s", distribution_plot_path)


def log_rows(label: str, rows: Sequence[MetricRow], limit: int = 12) -> None:
    """Log a compact metric table."""
    LOGGER.info("%s metrics: showing %d/%d rows", label, min(limit, len(rows)), len(rows))
    for row in rows[:limit]:
        LOGGER.info(
            (
                "  kind=%s mode=%s repeat=%d sample=%d batch=%d "
                "sigma_y=%.5g sigma_R=%.5g sigma_T=%.5g raw=%.4f "
                "global=%.4f chainwise=%.4f chains=%s"
            ),
            row.diffuser_kind,
            row.mode,
            row.repeat_idx,
            row.sample_idx,
            row.batch_idx,
            row.sigma_y,
            row.sigma_rotation,
            row.sigma_translation,
            row.raw_rmsd,
            row.global_rmsd,
            row.chainwise_rmsd,
            row.chain_rmsds,
        )


def run_diffuser_check(
    *,
    args: argparse.Namespace,
    batch: Batch,
    diffuser_kind: DiffuserKind,
) -> None:
    """Run one diffuser-kind check."""
    set_seed(args.seed)
    scheduler, diffuser = build_diffuser(args.seed, diffuser_kind)
    output_prefix = "decoupled_edm" if diffuser_kind == "decoupled" else "edm"
    LOGGER.info("Running %s diffusion check.", diffuser_kind)
    train_rows = run_training_like_check(
        diffuser=diffuser,
        scheduler=scheduler,
        diffuser_kind=diffuser_kind,
        batch=batch,
        num_augment=args.num_augment,
    )
    log_rows(f"{diffuser_kind} training-like random sample", train_rows)

    distribution_rows = sample_distribution(
        diffuser=diffuser,
        scheduler=scheduler,
        diffuser_kind=diffuser_kind,
        batch=batch,
        num_augment=args.num_augment,
        repeats=args.distribution_repeats,
        seed=args.seed,
    )
    log_rows(f"{diffuser_kind} repeated random distribution", distribution_rows)
    log_distribution_summary(distribution_rows)

    sigmas = parse_sweep_sigmas(args.sweep_sigmas, scheduler, args.sweep_steps)
    sweep_rows, pdb_frames_by_batch = run_sigma_sweep(
        diffuser=diffuser,
        scheduler=scheduler,
        diffuser_kind=diffuser_kind,
        batch=batch,
        sigmas=sigmas,
        seed=args.seed,
        independent_noise=args.independent_noise,
    )
    log_rows(f"{diffuser_kind} fixed-sigma sweep", sweep_rows, limit=len(sweep_rows))

    oracle_rows, oracle_frames_by_batch = run_oracle_solver(
        scheduler=scheduler,
        diffuser_kind=diffuser_kind,
        batch=batch,
        num_steps=args.oracle_steps,
        seed=args.seed,
    )
    if oracle_rows:
        log_rows(
            f"{diffuser_kind} oracle solver trajectory",
            oracle_rows,
            limit=min(len(oracle_rows), 16),
        )

    write_outputs(
        output_dir=args.output_dir,
        output_prefix=output_prefix,
        diffuser_kind=diffuser_kind,
        batch=batch,
        all_rows=[*train_rows, *distribution_rows, *sweep_rows],
        distribution_rows=distribution_rows,
        pdb_frames_by_batch=pdb_frames_by_batch,
        oracle_rows=oracle_rows,
        oracle_frames_by_batch=oracle_frames_by_batch,
        pdb_batch_items=args.pdb_batch_items,
        scheduler=scheduler,
        sigmas=sigmas,
    )


def selected_diffuser_kinds(args: argparse.Namespace) -> list[DiffuserKind]:
    """Return diffuser kinds requested by CLI."""
    if args.diffuser_kind == "both":
        return ["decoupled", "edm"]
    return [args.diffuser_kind]


def main() -> None:
    """Run EDM checks."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch = load_batch(args, device=device)
    LOGGER.info(
        "Loaded batch name=%s shape=%s token_len=%d atom_len=%d",
        batch.name,
        tuple(batch.shape),
        batch.token_length,
        batch.atom_length,
    )
    for diffuser_kind in selected_diffuser_kinds(args):
        run_diffuser_check(args=args, batch=batch, diffuser_kind=diffuser_kind)


if __name__ == "__main__":
    main()
