"""run_miniworld_diffusion_inference.py with exact fp32 matmuls (torch.set_float32_matmul_precision("highest")) instead of the script's "medium" (TF32).
The diffusion module is made fp32 by the hydra override model.diffusion.dtype=fp32 (see dfp32_9jst.sbatch); this wrapper changes nothing else."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402  (sets "medium" at import)

torch.set_float32_matmul_precision("highest")
print("float32 matmul precision:", torch.get_float32_matmul_precision(), flush=True)
inf.cli()
