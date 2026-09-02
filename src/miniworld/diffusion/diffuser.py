from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from miniworld.diffusion.scheduler import DiffusionScheduler, EDMScheduler
from miniworld.utils.structure.align import weighted_align
from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid

from .scheduler import DecoupledEDMScheduler


def _expand_to_trailing_dims(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-sample scalars over coordinate dimensions."""
    if value.ndim > target.ndim:
        msg = f"Cannot broadcast shape {value.shape} to target shape {target.shape}."
        raise ValueError(msg)
    return value.reshape(*value.shape, *((1,) * (target.ndim - value.ndim)))


class Diffuser(ABC):
    """Base class for defining a diffusion model. (use solver when sampling)."""

    class DiffuserConfig(BaseModel):
        """Configuration for the Diffuser class."""

        method: str = "EDM"
        seed: int = 0
        translation_noise: float = 1.0
        # Add any additional configuration parameters here

    def __init__(
        self,
        config: DiffuserConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @torch.no_grad()
    def random_rotation_and_translation(
        self,
        x: Float[torch.Tensor, "... L 3"],
    ) -> Float[torch.Tensor, "... L 3"]:
        """Apply random rotation and translation to the input tensor."""
        if x.ndim < 2:
            msg = "Input tensor must have at least 2 dimensions."
            raise ValueError(msg)
        if x.shape[-1] != 3:
            msg = "Last dimension of input tensor must be of size 3."
            raise ValueError(msg)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])  # (AB, L, 3) or (B, L, 3)

        n = x.shape[0]
        rot_mats = torch.from_numpy(Rotation.random(n).as_matrix()).to(
            x.device,
            x.dtype,
        )
        translation = (
            torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
            * self.config.translation_noise
        )

        x = torch.bmm(x, rot_mats.transpose(-1, -2))
        x = x + translation
        return x.reshape(*x_shape)

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample noisy input and store preconditioning data."""

    @abstractmethod
    def cal_loss(self, *args: Any, **kwargs: Any) -> Float[torch.Tensor, 1]:
        """Compute loss between model output and ground truth."""


# ruff: noqa: PLR2004
class EuclideanDiffuser(Diffuser, ABC):
    """Diffuser class for Euclidean diffusion process."""

    scheduler: EDMScheduler

    class EuclideanConfig(BaseModel):
        """Configuration for the EuclideanDiffuser class."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0

    def __init__(
        self,
        config: EuclideanConfig,
        scheduler: DiffusionScheduler,
    ) -> None:
        if not isinstance(scheduler, EDMScheduler):
            msg = "EuclideanDiffuser requires an EDMScheduler-compatible scheduler."
            raise TypeError(msg)
        self.config = config
        self.scheduler: EDMScheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32

    def sample(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        mask: Bool[torch.Tensor, "... L"] | None = None,
        num_augment: int = 1,
    ) -> tuple[
        Float[torch.Tensor, "... L 3"],
        Float[torch.Tensor, "... L 3"],
        Bool[torch.Tensor, "... L"] | None,
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
    ]:
        """Add noise to batch.atom_pos and store preconditioning data."""
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)

        batch_size = x0.shape[0]
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)
        if len(x0.shape) == 3:  # x0 : (B, L, 3)
            x0 = x0.expand(num_augment, *x0.shape[1:])
            if mask is not None:
                mask = mask.expand(num_augment, *mask.shape[1:])
        elif len(x0.shape) == 4:  # x0 : (B, N_str, L, 3)
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            x0 = x0.reshape(-1, *x0.shape[2:])
            x0 = x0.repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:])
                mask = mask.repeat(num_expand, 1)

        x0 = self.random_rotation_and_translation(x0)

        # random rotation and translation augmentation
        total_num = x0.shape[0]
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        sigma = self.scheduler.sample_noise(total_num)
        noise = torch.randn_like(x0, device=device, dtype=dtype)
        sigma = sigma.view(sigma_shape).to(device=device, dtype=dtype)
        input_scaling = self.scheduler.input_scale(sigma).to(device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma
        x_input = noisy_x * input_scaling
        t_emb = self.scheduler.noise_condition(sigma).to(device=device, dtype=dtype)

        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma = sigma.view(num_augment, batch_size, *sigma.shape[1:])
        noisy_x = noisy_x.view(num_augment, batch_size, *noisy_x.shape[1:])
        x_input = x_input.view(num_augment, batch_size, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, batch_size, *t_emb.shape[1:])

        return x0, x_input, mask, t_emb, sigma

    def cal_loss(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        x_update: Float[torch.Tensor, "... L 3"],
        sigma: Float[torch.Tensor, ...],
        mask: Bool[torch.Tensor, "... L"] | None = None,
        atom_weight: Float[torch.Tensor, "... L"] | None = None,
    ) -> Float[torch.Tensor, 1]:
        """Compute EDM loss between model prediction and true signal.

        Two independent weights multiply the squared error:

        - `sigma_weight` is one scalar per sample, the EDM weighting of the
          noise level, (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.
        - `atom_weight` is one scalar per atom, w_l of AF3 SI eq. 4 (built by
          `miniworld.loss.auxiliary.cal_atom_loss_weight`), upweighting
          nucleotide and ligand atoms, with invalid atoms zeroed out by `mask`.
          It also weights the rigid alignment of the ground truth onto the
          prediction (AF3 SI eq. 2). Passing None leaves every atom at w_l = 1,
          i.e. the plain masked MSE.
        """
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype,
            )
            raise ValueError(msg)
        input_scaling = self.scheduler.input_scale(sigma)
        input_scaling = input_scaling.to(device=x0.device, dtype=self.dtype)
        noisy_x = x_input / input_scaling

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype)
        noisy_x = noisy_x.to(dtype=dtype)
        c_skip = self.scheduler.skip_scale(sigma).to(dtype=dtype)
        c_out = self.scheduler.output_scale(sigma).to(dtype=dtype)
        sigma_weight = self.scheduler.loss_weight(sigma).to(dtype=dtype)
        if mask is None:
            mask = torch.ones(
                x0.shape[:-1],
                device=x0.device,
                dtype=torch.bool,
            )
        if atom_weight is None:
            atom_weight = torch.ones_like(mask, dtype=dtype)
        atom_weight = atom_weight.to(device=x0.device, dtype=dtype) * mask.to(
            dtype=dtype,
        )

        x_pred = c_skip * noisy_x + c_out * x_update
        # align x0 to x_pred
        x0_aligned = weighted_align(x0, x_pred, weight=atom_weight)
        if torch.isnan((x_pred - x0_aligned).pow(2).mean()):
            torch.save(
                {
                    "sigma": sigma,
                    "c_skip": c_skip,
                    "c_out": c_out,
                    "x_update": x_update,
                    "x_pred": x_pred,
                    "x0_aligned": x0_aligned,
                    "x0": x0,
                    "noisy_x": noisy_x,
                    "mask": mask,
                    "sigma_weight": sigma_weight,
                    "atom_weight": atom_weight,
                },
                "debug_nan_at_loss.pt",
            )
            msg = "NaN detected in the loss calculation."
            raise ValueError(msg)

        x_shape = x0.shape
        loss = (
            (x_pred - x0_aligned).pow(2) * atom_weight.unsqueeze(-1) * sigma_weight
        ).mean()
        return loss, x_pred.reshape(x_shape)


class DecoupledEDMDiffuser(Diffuser):
    """Diffuser class for the decoupled EDM diffusion process."""

    scheduler: DecoupledEDMScheduler

    class DecoupledEDMConfig(BaseModel):
        """Configuration for the DecoupledEDMDiffuser class."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0

    def __init__(
        self,
        config: DecoupledEDMConfig,
        scheduler: DecoupledEDMScheduler,
    ) -> None:
        self.config = config
        self.scheduler: DecoupledEDMScheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32  # diffuser should always use float32

    @staticmethod
    def _build_contact_graph(
        chain_atoms: list[torch.Tensor],
        dist_cutoff: float,
    ) -> torch.Tensor:
        """Return symmetric bool adjacency (n_chains, n_chains) for chains within dist_cutoff.

        Fully vectorized: one cdist over all atoms, then scatter_reduce 'amin' per chain pair.
        Empty (masked-out) chains have no contact with anyone.
        """
        n = len(chain_atoms)
        nonempty = [(i, ca) for i, ca in enumerate(chain_atoms) if len(ca) > 0]
        if len(nonempty) < 2:
            device = chain_atoms[0].device if chain_atoms else torch.device("cpu")
            return torch.zeros(n, n, dtype=torch.bool, device=device)

        _, atoms = zip(*nonempty, strict=True)
        all_atoms = torch.cat(atoms)  # (N_total, 3)
        device = all_atoms.device
        labels = torch.cat(
            [
                torch.full((len(ca),), idx, dtype=torch.long, device=device)
                for idx, ca in nonempty
            ],
        )  # original chain indices 0..n-1

        D = torch.cdist(all_atoms, all_atoms)  # (N_total, N_total)
        pair_idx = labels.unsqueeze(1) * n + labels.unsqueeze(0)  # (N_total, N_total)
        chain_min_dist = torch.full((n * n,), float("inf"), device=device)
        chain_min_dist.scatter_reduce_(
            0,
            pair_idx.flatten(),
            D.flatten(),
            reduce="amin",
            include_self=True,
        )
        contact = chain_min_dist.view(n, n) <= dist_cutoff
        contact.fill_diagonal_(fill_value=False)
        return contact

    @staticmethod
    def _find(x: int, p: list[int]) -> int:
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    @staticmethod
    def _random_kruskal(
        n: int,
        edges: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Random spanning forest via Kruskal with shuffled edge order."""
        parent = list(range(n))
        forest: list[tuple[int, int]] = []
        for idx in torch.randperm(len(edges)).tolist():
            i, j = edges[idx]
            pi = DecoupledEDMDiffuser._find(i, parent)
            pj = DecoupledEDMDiffuser._find(j, parent)
            if pi != pj:
                parent[pi] = pj
                forest.append((i, j))
        return forest

    @staticmethod
    def _compress_components(n: int, kept_edges: list[tuple[int, int]]) -> list[int]:
        """Union-Find on kept edges; return contiguous group index per node."""
        comp = list(range(n))
        for i, j in kept_edges:
            pi = DecoupledEDMDiffuser._find(i, comp)
            pj = DecoupledEDMDiffuser._find(j, comp)
            if pi != pj:
                comp[pi] = pj
        root_to_group: dict[int, int] = {}
        return [
            root_to_group.setdefault(
                DecoupledEDMDiffuser._find(i, comp),
                len(root_to_group),
            )
            for i in range(n)
        ]

    @staticmethod
    def _spanning_forest_cut(contact: torch.Tensor, max_groups: int = 4) -> list[int]:
        """Random spanning forest cut on the contact graph.

        The contact graph may be disconnected — its connected components are
        already separate rigid bodies. This method optionally splits components
        further by randomly cutting spanning-forest edges.

        The target number of groups is sampled first, then forest edges are
        cut until the target is reached.
        """
        n = contact.shape[0]
        edges = contact.triu(diagonal=1).nonzero().tolist()
        forest = DecoupledEDMDiffuser._random_kruskal(n, edges)

        k = n - len(forest)  # number of connected components
        min_target_groups = max(2, k)  # force ≥2 groups when possible
        max_target_groups = min(n, max_groups)
        if max_target_groups < min_target_groups:
            target_groups = k  # cannot merge pre-existing components
        else:
            target_groups = int(
                torch.randint(
                    min_target_groups,
                    max_target_groups + 1,
                    (1,),
                ).item(),
            )

        cut_set: set[int] = set()
        group_count = k
        # In a forest, removing any kept edge increases the component count by 1.
        for idx in torch.randperm(len(forest)).tolist():
            if group_count >= target_groups:
                break
            cut_set.add(idx)
            group_count += 1

        kept = [e for idx, e in enumerate(forest) if idx not in cut_set]
        return DecoupledEDMDiffuser._compress_components(n, kept)

    @torch.no_grad()
    def _randomly_split_chains(
        self,
        x0: Float[torch.Tensor, "B L 3"],
        mask: Bool[torch.Tensor, "B L"] | None,
        atom_to_chain_idx: torch.Tensor,  # (B, L_atom)
        dist_cutoff: float = 6.0,
        max_groups: int = 4,
    ) -> torch.Tensor:  # (B, L_atom)
        """Randomly split chains into target-sampled rigid body groups.

        1. Build contact graph from geometry (chains within dist_cutoff are connected).
        2. Spanning forest of the contact graph defines natural separation boundaries.
        3. Sample the target group count, then cut random forest edges to reach it.

        Returns a new atom_to_chain_idx with contiguous group indices for apply_chain_rt.
        """
        device = atom_to_chain_idx.device
        result = torch.zeros_like(atom_to_chain_idx)

        for b in range(atom_to_chain_idx.shape[0]):
            chain_ids = torch.unique(atom_to_chain_idx[b])
            n_chains = len(chain_ids)
            if n_chains <= 1:
                continue  # result[b] already zeros

            valid = (
                mask[b]
                if mask is not None
                else atom_to_chain_idx[b].new_ones(x0.shape[1], dtype=torch.bool)
            )
            chain_atoms = [
                x0[b][(atom_to_chain_idx[b] == cid) & valid] for cid in chain_ids
            ]

            contact = self._build_contact_graph(chain_atoms, dist_cutoff)
            group_map = self._spanning_forest_cut(contact, max_groups=max_groups)

            lut = torch.zeros(int(chain_ids.max()) + 1, dtype=torch.long, device=device)
            lut[chain_ids] = torch.tensor(group_map, dtype=torch.long, device=device)
            result[b] = lut[atom_to_chain_idx[b]]

        return result

    def sample(  # noqa: PLR0915
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
        atom_to_chain_idx: torch.Tensor,  # [B, L_atom]
        num_augment: int = 1,
    ) -> tuple[
        Float[torch.Tensor, "... L 3"],
        Float[torch.Tensor, "... L 3"],
        Bool[torch.Tensor, "... L"] | None,
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Int[torch.Tensor, ...],
    ]:
        """Add noise to batch.atom_pos and store preconditioning data.

        For now, this assumes a shared atom_chain_break mapping across the batch.
        """
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)
        if atom_to_chain_idx is None:
            msg = "atom_to_chain_idx must be provided for decoupled EDM diffusion."
            raise ValueError(msg)

        batch_size = x0.shape[0]
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)
        if len(x0.shape) == 3:  # x0 : (B, L, 3)
            x0 = x0.unsqueeze(0).expand(num_augment, -1, -1, -1)
            x0 = x0.reshape(num_augment * batch_size, *x0.shape[2:])
            atom_to_chain_idx = atom_to_chain_idx.expand(
                batch_size,
                *atom_to_chain_idx.shape[1:],
            )
            atom_to_chain_idx = atom_to_chain_idx.unsqueeze(0).expand(
                num_augment,
                -1,
                *atom_to_chain_idx.shape[1:],
            )
            atom_to_chain_idx = atom_to_chain_idx.reshape(
                num_augment * batch_size,
                *atom_to_chain_idx.shape[2:],
            )
            if mask is not None:
                mask = mask.unsqueeze(0).expand(num_augment, -1, -1)
                mask = mask.reshape(num_augment * batch_size, *mask.shape[2:])
        elif len(x0.shape) == 4:  # x0 : (B, N_str, L, 3)
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            atom_to_chain_idx = atom_to_chain_idx.unsqueeze(1).expand(
                -1,
                x0.shape[1],
                -1,
            )
            atom_to_chain_idx = atom_to_chain_idx.reshape(
                -1,
                atom_to_chain_idx.shape[-1],
            )
            atom_to_chain_idx = atom_to_chain_idx.repeat(num_expand, 1)
            x0 = x0.reshape(-1, *x0.shape[2:])
            x0 = x0.repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:])
                mask = mask.repeat(num_expand, 1)

        # Apply global augmentation before adding decoupled coordinate/rigid noise.
        x0 = self.random_rotation_and_translation(x0)

        total_num = x0.shape[0]
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        sigma_y, sigma_rotation, sigma_translation = self.scheduler.sample_noise(
            total_num,
        )
        sigma_y = sigma_y.to(device=device, dtype=dtype)
        sigma_rotation = sigma_rotation.to(device=device, dtype=dtype)
        sigma_translation = sigma_translation.to(device=device, dtype=dtype)

        noise = torch.randn_like(x0, device=device, dtype=dtype)
        noisy_x = x0 + noise * sigma_y.view(sigma_shape)

        atom_to_combine = self._randomly_split_chains(x0, mask, atom_to_chain_idx)
        group_num = int(atom_to_combine.max().item()) + 1

        rotation_matrix, translation_vector = sample_rigid(
            sigma_rotation,
            sigma_translation,
            C=group_num,
            device=device,
            dtype=dtype,
        )
        noisy_x = apply_chain_rt(
            noisy_x,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
        )
        sigma_y, sigma_rotation, sigma_translation = [
            x.view(sigma_shape) for x in (sigma_y, sigma_rotation, sigma_translation)
        ]
        input_scaling = self.scheduler.input_scale(sigma_y, sigma_translation).to(
            device=device,
            dtype=dtype,
        )

        t_emb = self.scheduler.noise_condition(sigma_y).to(device=device, dtype=dtype)
        x_input = noisy_x * input_scaling

        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma_y = sigma_y.view(num_augment, batch_size, *sigma_y.shape[1:])
        sigma_rotation = sigma_rotation.view(
            num_augment,
            batch_size,
            *sigma_rotation.shape[1:],
        )
        sigma_translation = sigma_translation.view(
            num_augment,
            batch_size,
            *sigma_translation.shape[1:],
        )
        noisy_x = noisy_x.view(num_augment, batch_size, *noisy_x.shape[1:])
        x_input = x_input.view(num_augment, batch_size, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])

        t_emb = t_emb.view(num_augment, batch_size, *t_emb.shape[1:])

        return (
            x0,
            x_input,
            mask,
            t_emb,
            sigma_y,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
        )

    def get_x0_hat(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        x_update: Float[torch.Tensor, "... L 3"],
        sigma_y: Float[torch.Tensor, ...],
        rotation_matrix: Float[torch.Tensor, ...],
        translation_vector: Float[torch.Tensor, ...],
        atom_to_combine: Int[torch.Tensor, "... L"],
        mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> Float[torch.Tensor, "... L 3"]:
        """Compute the predicted x0 (x0_hat) from the model output and preconditioning data."""
        x_shape = x0.shape
        if x_update.dtype != self.dtype:
            msg = "x_update must be of type float32, but got dtype: " + str(
                x_update.dtype,
            )
            raise ValueError(msg)

        _, sigma_translation = self.scheduler.convert_to_sigma_rt(sigma_y)
        input_scaling = self.scheduler.input_scale(sigma_y, sigma_translation)
        input_scaling = input_scaling.to(device=x0.device, dtype=self.dtype)
        input_scaling = _expand_to_trailing_dims(input_scaling, x_input)
        noisy_x = x_input / input_scaling

        dtype = x_update.dtype

        x0 = x0.to(dtype=dtype).reshape(-1, *x0.shape[-2:])
        noisy_x = noisy_x.to(dtype=dtype).reshape(-1, *noisy_x.shape[-2:])
        x_update = x_update.reshape(-1, *x_update.shape[-2:])
        sigma_y = sigma_y.to(dtype=dtype).reshape(-1)

        c_skip = self.scheduler.skip_scale(sigma_y).to(dtype=dtype).view(-1, 1, 1)
        c_out = self.scheduler.output_scale(sigma_y).to(dtype=dtype).view(-1, 1, 1)
        weight = self.scheduler.loss_weight(sigma_y).to(dtype=dtype).view(-1, 1, 1)

        if mask is None:
            mask = torch.ones(
                x0.shape[:-1],
                device=x0.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.reshape(-1, *mask.shape[-1:])
            weight = weight * mask.unsqueeze(-1)

        noisy_x = apply_chain_rt(
            noisy_x,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
            inverse=True,
        )
        noisy_x = torch.where(mask.unsqueeze(-1), noisy_x, torch.zeros_like(noisy_x))
        x0 = torch.where(mask.unsqueeze(-1), x0, torch.zeros_like(x0))

        x_pred = c_skip * noisy_x + c_out * x_update

        return x_pred.reshape(x_shape)

    def cal_loss(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_pred: Float[torch.Tensor, "... L 3"],
        sigma_y: Float[torch.Tensor, ...],
        mask: Bool[torch.Tensor, "... L"] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, 1]:
        """Compute EDM loss between model prediction and true signal."""
        weight = self.scheduler.loss_weight(sigma_y).to(dtype=dtype).view(-1, 1, 1, 1)

        if mask is None:
            mask = torch.ones(
                x0.shape[:-1],
                device=x0.device,
                dtype=torch.bool,
            )
        else:
            weight = weight * mask.unsqueeze(-1)

        x0 = x0.to(dtype=dtype)
        x0_aligned = weighted_align(x0, x_pred, weight=mask.to(dtype=dtype))
        return ((x_pred - x0_aligned).pow(2) * weight).mean()
