import torch
from BioMol.BioMol import BioMol
from MiniWorld.utils.diffuser import DecoupledEDMDiffuser
from MiniWorld.utils.scheduler import DecoupledEDMScheduler
from MiniWorld.utils.solver import DecoupledEDMSolver

from Bio.PDB import MMCIFParser, PDBIO
from pathlib import Path


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
    scheduler.draw_sigma(save_path=f"MiniWorld/utils/test_str/{pdb_ID}_sigma.png")
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

    for ii in range(len(trajectory)):
        xyz = trajectory[ii][0]
        to_mmcif(biomol, xyz, save_path=f"MiniWorld/utils/test_str/{pdb_ID}_recon_{ii}.cif")


def visualize_cif():
    cif_dir = Path("/home/psk6950/MiniWorld/MiniWorld/utils/test_str/")
    cif_list = [cif_dir / f"2mcg_recon_{ii}.cif" for ii in range(0, 50)]

    lines = []

    def change_model_id(line, model_id):
        line = line.strip().split()
        line[-1] = str(model_id) + "\n"

        return " ".join(line)

    # reverse
    cif_list = cif_list[::-1]

    for i, cif_file in enumerate(cif_list, start=1):
        with open(cif_file) as file:
            _lines = file.readlines()
        if len(lines) == 0:
            lines.extend(_lines)
            continue

        # find lines starting with "ATOM"
        atom_lines = [change_model_id(line, i) for line in _lines if line.startswith("ATOM")]
        lines.extend(atom_lines)

    with open(cif_dir / "merged_multimodel.cif", "w") as file:
        file.writelines(lines)


def test_scheduler():
    scheduler, diffuser, solver = setup_diffusion()
    scheduler.draw_sigma(save_path=f"MiniWorld/utils/test_str/sigma_train.png", mode="train")
    scheduler.draw_sigma(save_path=f"MiniWorld/utils/test_str/sigma_sampling.png", mode="inference")
    scheduler.draw_sigma(save_path=f"MiniWorld/utils/test_str/sigma_y_to_RT.png", mode="y_to_RT")


if __name__ == "__main__":
    pdb_ID = "2mcg"
    # test_forward(pdb_ID)
    test_scheduler()
    # visualize_cif()
