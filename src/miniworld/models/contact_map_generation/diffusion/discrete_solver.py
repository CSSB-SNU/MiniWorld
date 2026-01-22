from collections.abc import Callable

import torch
import torch.nn.functional as F
from team_gm.utils.diffusion import DiffusionSolver

from .discrete_scheduler import D3PMScheduler, SEDDScheduler
from .util import symmetrize_labels, symmetrize_pair


class SEDDSolver(DiffusionSolver):
    """Solver for SEDD discrete diffusion using score-entropy ratios."""

    class SEDDSolverConfig(DiffusionSolver.SolverConfig):
        """Configuration for the SEDDSolver class."""

        method: str = "SEDD"
        enforce_symmetric: bool = True
        min_ratio: float = 1e-5

    def __init__(
        self,
        config: SEDDSolverConfig,
        scheduler: SEDDScheduler,
    ) -> None:
        super().__init__(config, scheduler)
        self.scheduler = scheduler
        self.enforce_symmetric = config.enforce_symmetric
        self.min_ratio = config.min_ratio

    def _base_init_labels(
        self,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        if self.scheduler.config.transition_mode == "absorbing":
            mask_class = self.scheduler.mask_class
            if mask_class is None:
                msg = "mask_class must be set for absorbing transition_mode."
                raise ValueError(msg)
            return torch.full(shape, mask_class, device=device, dtype=torch.long)
        # uniform base
        return torch.randint(
            low=0,
            high=self.scheduler.num_classes,
            size=shape,
            device=device,
        )

    def step(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xt_one_hot: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one ancestral sampling step."""
        scheduler: SEDDScheduler = self.scheduler
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]
        xt_labels = xt_one_hot.argmax(dim=-1)

        sigma_i = scheduler.sigma(t_i)
        t_emb = (
            scheduler.noise_condition(sigma_i).unsqueeze(0).repeat(xt_one_hot.shape[0])
        )
        ratio_pred = model_fn(xt_one_hot, t_emb).clamp_min(self.min_ratio)
        if self.enforce_symmetric:
            ratio_pred = symmetrize_pair(ratio_pred)

        delta_sigma = scheduler.sigma_derivative(t_i, t_next).to(
            device=xt_one_hot.device,
            dtype=xt_one_hot.dtype,
        )
        q = scheduler.base_q(xt_labels)

        off_diag = 1.0 - xt_one_hot
        offdiag_trans = delta_sigma.unsqueeze(-1) * q * ratio_pred * off_diag
        offdiag_mass = offdiag_trans.sum(dim=-1, keepdim=True)
        diag_prob = (1.0 - offdiag_mass).clamp_min(0.0)
        trans = offdiag_trans + xt_one_hot * diag_prob
        trans = torch.clamp(trans, min=0.0)
        trans = trans / trans.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_prob = trans.view(trans.shape[0], -1, trans.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt_next = sampled.view(*xt_labels.shape)
        xt_one_hot_next = torch.nn.functional.one_hot(
            xt_next,
            num_classes=self.scheduler.num_classes,
        ).to(xt_one_hot.dtype)

        if self.enforce_symmetric:
            xt_next = symmetrize_labels(xt_next)
            xt_one_hot_next = torch.nn.functional.one_hot(
                xt_next,
                num_classes=self.scheduler.num_classes,
            ).to(xt_one_hot.dtype)

        return xt_one_hot_next, ratio_pred

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
        """Sample from the diffusion model using ancestral sampling."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)

        init_labels = self._base_init_labels(shape, device)
        xt_one_hot = torch.nn.functional.one_hot(
            init_labels,
            num_classes=self.scheduler.num_classes,
        ).to(torch.float32)
        if self.enforce_symmetric:
            xt_one_hot = symmetrize_pair(xt_one_hot)

        trajectory = []
        ratio_list = []
        for i in range(num_steps):
            xt_one_hot, ratio_pred = self.step(model_fn, xt_one_hot, i, time_steps)
            if return_intermediate:
                trajectory.append(xt_one_hot.clone())
                ratio_list.append(ratio_pred.clone())

        final_labels = xt_one_hot.argmax(dim=-1)
        if return_intermediate:
            return final_labels, trajectory, ratio_list
        return final_labels


class D3PMSolver(DiffusionSolver):
    """Ancestral sampler for D3PM discrete diffusion on pairwise labels."""

    class D3PMSolverConfig(DiffusionSolver.SolverConfig):
        """Configuration for the D3PMSolver class."""

        method: str = "D3PM"
        enforce_symmetric: bool = True

    def __init__(
        self,
        config: D3PMSolverConfig,
        scheduler: D3PMScheduler,
    ) -> None:
        super().__init__(config, scheduler)
        self.scheduler = scheduler
        self.enforce_symmetric = config.enforce_symmetric

    def step(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        xt_one_hot: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one ancestral sampling step."""
        scheduler: D3PMScheduler = self.scheduler
        t = time_steps[t_index].long()
        t_prev = time_steps[t_index + 1].long()
        xt_labels = xt_one_hot.argmax(dim=-1)

        t_emb = scheduler.noise_condition(t).unsqueeze(0).repeat(xt_one_hot.shape[0])
        logits_x0 = model_fn(xt_one_hot, t_emb)

        q_post = scheduler.q_posterior(
            xt_labels,
            logits_x0,
            t,
            t_prev,
        )

        p = q_post.view(q_post.shape[0], -1, q_post.shape[-1])
        B, N, C = p.shape
        p2 = p.reshape(B * N, C)

        if torch.isnan(p2).any():
            raise RuntimeError("q_post has NaN")
        if torch.isinf(p2).any():
            raise RuntimeError("q_post has Inf")
        if (p2 < 0).any():
            raise RuntimeError(f"q_post has negative values, min={p2.min().item()}")

        s = p2.sum(dim=-1)
        bad = (s <= 0).nonzero(as_tuple=False)
        if bad.numel():
            i = bad[0, 0].item()
            b = i // N
            n = i % N
            breakpoint()
            raise RuntimeError(f"q_post row-sum<=0 at (b={b}, n={n}), sum={s[i].item()}")
        q_post = q_post / q_post.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_prob = q_post.view(q_post.shape[0], -1, q_post.shape[-1])
        B, N, C = flat_prob.shape
        sampled = torch.multinomial(
            flat_prob.view(B * N, C),
            num_samples=1,
        ).view(B, N)
        xt_minus_1 = sampled.view(*xt_labels.shape)
        xt_minus_1_one_hot = F.one_hot(
            xt_minus_1,
            num_classes=self.scheduler.num_classes,
        ).to(dtype=xt_one_hot.dtype)

        if self.enforce_symmetric:
            xt_minus_1 = symmetrize_labels(xt_minus_1)
            xt_minus_1_one_hot = F.one_hot(
                xt_minus_1,
                num_classes=self.scheduler.num_classes,
            ).to(dtype=xt_one_hot.dtype)

        return xt_minus_1_one_hot, logits_x0

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: torch.Size,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
        seed: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
        """Sample from the diffusion model using ancestral sampling."""
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        if self.scheduler.config.transition_mode == "absorbing":
            mask_class = self.scheduler.mask_class
            if mask_class is None:
                msg = "mask_class must be set for absorbing transition_mode."
                raise ValueError(msg)
            labels = torch.full(shape, mask_class, device=device, dtype=torch.long)
        else:
            labels = torch.randint(
                low=0,
                high=self.scheduler.num_classes,
                size=shape,
                device=device,
            )
        if self.enforce_symmetric:
            labels = symmetrize_labels(labels)
        self._set_seed(seed)
        xt = F.one_hot(labels, num_classes=self.scheduler.num_classes).to(torch.float32)
        trajectory = []
        logits_list = []
        for i in range(num_steps):
            xt, logits_x0 = self.step(model_fn, xt, i, time_steps)
            if return_intermediate:
                trajectory.append(xt.clone())
                logits_list.append(logits_x0.clone())

        final_labels = xt.argmax(dim=-1)
        if return_intermediate:
            return final_labels, trajectory, logits_list
        return final_labels

