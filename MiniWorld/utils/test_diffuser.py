import torch
from BioMol.BioMol import BioMol
from MiniWorld.utils.diffuser import DecoupledEDMDiffuser
from MiniWorld.utils.scheduler import DecoupledEDMScheduler
from MiniWorld.utils.solver import DecoupledEDMSolver

def to_mmcif(
    biomol: BioMol,
    xyz: torch.Tensor,
    save_path: str,
):
    biomol.structure.atom_tensor[:, 5:8] = xyz
    biomol.structure.to_mmcif(
        save_path,
    )

def setup_diffusion():
    scheduler_config = DecoupledEDMScheduler.DecoupledEDMSchedulerConfig()
    scheduler = DecoupledEDMScheduler(config=scheduler_config)
    diffuser_config = DecoupledEDMDiffuser.DecoupledConfig(seed=1234)
    diffuser = DecoupledEDMDiffuser(scheduler=scheduler, config=diffuser_config)
    solver = DecoupledEDMSolver(config=diffuser.config, scheduler=scheduler)
    return scheduler, diffuser, solver

def test_forward(pdb_ID):
    scheduler, diffuser, solver = setup_diffusion()
    biomol = BioMol(pdb_ID=pdb_ID)
    biomol.choose('1','1','.')

    xyz = biomol.structure.atom_tensor[:,5:8]
    mask = biomol.structure.atom_tensor[:,4] != 0

    # centering
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    to_mmcif(biomol, xyz, save_path=f"MiniWorld/utils/test_str/{pdb_ID}_clean.cif")

    xyz, mask = xyz.unsqueeze(0), mask.unsqueeze(0)
    atom_chain_break = biomol.structure.atom_chain_break

    noisy_x, sigma_y, sigma_R, sigma_T = diffuser.sample(xyz, mask, atom_chain_break, num_augment = 50)
    noisy_x = noisy_x[:,0]
    sigma_y = sigma_y[:,0]

    save_dir = "MiniWorld/utils/test_str/"
    for bb in range(noisy_x.shape[0]):
        _sigma_y, _sigma_R, _sigma_T = sigma_y[bb], sigma_R[bb], sigma_T[bb]
        _sigma_y = round(_sigma_y.item(), 3)
        _sigma_R = round(_sigma_R.item(), 3)
        _sigma_T = round(_sigma_T.item(), 3)
        print(f"Sample {bb}: sigma_y = {_sigma_y}, sigma_R = {_sigma_R}, sigma_T = {_sigma_T}")
        tt = sigma_y[bb].item()
        tt = round(tt, 3)
        to_mmcif(biomol, noisy_x[bb], save_path=save_dir+f"{pdb_ID}_noisy_{tt}.cif")



def test_solver(pdb_ID):
    scheduler, diffuser, solver = setup_diffusion()
    biomol = BioMol(pdb_ID=pdb_ID)
    biomol.choose('1','1','.')

    xyz = biomol.structure.atom_tensor[:,5:8]
    mask = biomol.structure.atom_tensor[:,4] != 0
    atom_chain_break = biomol.structure.atom_chain_break

    # centering
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    x0 = xyz.unsqueeze(0)

    def orcale_model_fn(x_input, t_emb):
        return x0

    x, trajectory, hat_list = solver.sample(
        orcale_model_fn,
        shape=x0.shape,
        atom_chain_break=atom_chain_break,
        num_steps=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        return_intermediate=True,
    )




if __name__ == "__main__":
    pdb_ID = "2mcg"
    test_forward(pdb_ID)
    test_solver(pdb_ID)
