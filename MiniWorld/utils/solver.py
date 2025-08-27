import torch
from team_gm.utils.solver import DiffusionSolver
from MiniWorld.utils.scheduler import DecoupledEDMScheduler
from MiniWorld.utils.se3 import apply_chain_rt, sample_rigid, exp_se3, log_so3, se3_heat_step_delta_sigma, se3_heat_step_sigma


class DecoupledEDMSolver(DiffusionSolver):
    def __init__(self, config: DiffusionSolver.SolverConfig, scheduler: DecoupledEDMScheduler):
        super().__init__(config, scheduler)

        # TODO: move it to config
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self._lambda = 1.003
        self.step_scale = 1.5

    def _set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.random.manual_seed(seed)


    def _add_noise(
        self,
        y: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor, 
        atom_chain_break: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        sigma_Ri, sigma_Ti = self.scheduler.convert_to_sigmaRT(sigma_i)

        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        sigma_Rhat, sigma_That = self.scheduler.convert_to_sigmaRT(t_hat)
        R_hat, T_hat = se3_heat_step_sigma(
            R, T, sigma_Ri, sigma_Ti, sigma_Rhat, sigma_That, eps=1e-12
        )

        print(f"T_hat.norm : {T_hat.norm():.6f}")

        added_noise = self._lambda * (t_hat**2 - sigma_i**2) ** 0.5 * torch.randn_like(y)

        y = y + added_noise
        x_with_noise = apply_chain_rt(y, R_hat, T_hat, atom_chain_break)
        return y, x_with_noise, t_hat

    def y_step(
        self, model_fn: callable, 
        y: torch.Tensor, R : torch.Tensor, T : torch.Tensor,
        t_index: int, time_steps: torch.Tensor, atom_chain_break: dict,
    ) -> torch.Tensor:
        """
        Perform one Euler update in t-space.

        Args:
            model_fn: a callable `f(z, sigma)` → ε̂  (predicted noise at (normalized x, sigma))
            x: current sample, shape, in “noisy” (data) domain
            t_index: integer index in [0 .. len(time_steps)-2]
            time_steps: 1D tensor of time points, length = num_steps + 1

        Returns:
            x_{t_{i+1}} = x_{t_i} + Δt [ α̇(t_i) x_{t_i} - sigmȧ(t_i) · v_data(x_{t_i}, t_i ) ]
        """
        # 1. Get t_i and t_{i+1}, as well as Δt
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        gamma = self.gamma_0 if sigma_next > self.gamma_min else 0
        t_hat = sigma_i * (1 + gamma)
        _, sigma_That = self.scheduler.convert_to_sigmaRT(t_hat)

        # add noise
        y, x_with_noise, t_hat = self._add_noise(y, R, T, t_index, time_steps, atom_chain_break)
        dt = sigma_next - t_hat

        # 4. Query the model for εθ(z_i, sigma_i)
        t_emb = self.scheduler.noise_condition(t_hat)  # noise condition
        c_skip = self.scheduler.skip_scale(t_hat)
        c_out = self.scheduler.output_scale(t_hat)
        c_in = self.scheduler.input_scale(t_hat, sigma_That)
        x_input = x_with_noise * c_in  # normalized input to the model
        x_update = model_fn(x_input, t_emb)

        # x_update = torch.zeros_like(x_update)  # for test

        x_denoised = c_skip * y + c_out * x_update
        # x_denoised = x_update

        # 6. Compute dx/dt at t_i:  dx/dt = α̇(t_i) · x  -  sigmȧ(t_i) · v_data
        v_i = (y - x_denoised) / t_hat

        # 7. One Euler step:  x_{i+1} = x_i + dt * f_i
        y = y + self.step_scale * dt * v_i

        return y, x_update

    def RT_step(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        t_index: int,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one SE3 update step.

        Args:
            R: current rotation matrix
            T: current translation vector
            t_index: integer index in [0 .. len(time_steps)-2]
            time_steps: 1D tensor of time points, length = num_steps + 1

        Returns:
            Updated rotation matrix and translation vector.
        """
        # 1. Get t_i and t_{i+1}
        t_i = time_steps[t_index]
        t_next = time_steps[t_index + 1]

        # 2. Compute the time step
        sigma_i = self.scheduler.sampling_schedule(t_i)  # sigma(t_i)
        sigma_next = self.scheduler.sampling_schedule(t_next)  # sigma(t_{i+1})
        sigma_Ri, sigma_Ti = self.scheduler.convert_to_sigmaRT(sigma_i)
        sigma_Rnext, sigma_Tnext = self.scheduler.convert_to_sigmaRT(sigma_next)
        dt_R, dt_T = sigma_Rnext - sigma_Ri, sigma_Tnext - sigma_Ti

        R_i, T_i = se3_heat_step_delta_sigma(
            R, T, sigma_Ri, sigma_Ti, dt_R, dt_T
        )

        return R_i, T_i

    def step(
        self, model_fn: callable,
        y: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
        t_index: int, time_steps: torch.Tensor, atom_chain_break: dict,
    ) -> torch.Tensor:

        y, x_update = self.y_step(model_fn, y, R, T, t_index, time_steps, atom_chain_break)
        R, T = self.RT_step(R, T, t_index, time_steps)
        diff = (x_update - apply_chain_rt(y, R, T, atom_chain_break)).norm()
        print(f"Step {t_index}: mean abs diff between x_update and y: {diff:.6f}")
        return y, x_update, R, T

    def sample(
        self,
        model_fn: callable,
        shape: torch.Size,
        atom_chain_break: dict,
        num_steps: int,
        device: torch.device,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        A convenience wrapper that:
          1. Builds the time grid: t_0,...,t_N
          2. Initializes x_N as pure Gaussian noise ~ N(0, I)*sigma(t_0)
          3. Loops from i=0..(N-1), calling `step` each time.
          4. Returns x_0 (approximately denoised).

        Args:
            model_fn: f(z, sigma) → ε̂
            shape:   desired output shape (B, C, H, W, …)
            num_steps: how many discretization steps you want (N)
            device:  where to allocate tensors

        Returns:
            A tensor of shape `shape`, representing the decoded sample at t_N (≈0 noise).
        """
        # 1. Build the time grid
        time_steps = self.scheduler.sampling_time_steps(num_steps).to(device)
        # e.g. shape = (num_steps + 1,)

        # 2. The initial noise level is at t_0
        sigma_0 = self.scheduler.sampling_schedule(time_steps[0])
        sigma_R, sigma_T = self.scheduler.convert_to_sigmaRT(time_steps[0])

        B = shape[0]
        sigma_R = sigma_R.expand(B)
        sigma_T = sigma_T.expand(B)

        #    Draw x_N ~ N(0, I) * sigma_0
        y = torch.randn(shape, device=device) * sigma_0
        chain_num = len(atom_chain_break)
        R, T = sample_rigid(sigma_R, sigma_T, C=chain_num, device=device)

        trajectory = []
        hat_list = []

        # 3. Iteratively step from i=0 to N-1
        for i in range(num_steps):
            y, epsilon_hat, R, T = self.step(model_fn, y, R, T, i, time_steps, atom_chain_break)
            if return_intermediate:
                trajectory.append(y.clone())
                hat_list.append(epsilon_hat.clone())

        # 4. Return y at t_N (typically sigma(t_N) ≈ 0, so y is “denoised”)
        if return_intermediate:
            return y, trajectory, hat_list
        else:
            return y
