"""run_miniworld_diffusion_inference.py with engine setting rope_angles_bf16_compat=True (the pre-2026-10-06 RoPE angles) and NOTHING else changed.
Prints the effective setting. Experiment: is this checkpoint adapted to the old bf16 RoPE angles? (docs/known-issues.md)"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402
from miniworld_engine import settings  # noqa: E402

settings.configure(rope_angles_bf16_compat=True)
_Client = inf.Client


class Client(_Client):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        print("rope_angles_bf16_compat =", settings.current().rope_angles_bf16_compat, flush=True)


inf.Client = Client
inf.cli()
