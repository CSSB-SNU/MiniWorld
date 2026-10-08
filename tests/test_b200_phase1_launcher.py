"""scripts/b200_phase1_v200.sh: what it would run on node 1 (gpuq, 4 GPUs) and node 2 (8 GPUs, no gpuq), checked with DRYRUN=1."""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "b200_phase1_v200.sh"


def dry(*args, **env):
    out = subprocess.run(["bash", str(SCRIPT), *args], env={**os.environ, "DRYRUN": "1", **env}, capture_output=True, text=True, check=True).stdout
    header, *lines = out.splitlines()
    command = next(line for line in lines if line.startswith("exec pixi run"))
    return header, command, "\n".join(lines)


def test_node2_runs_all_eight_gpus_directly_with_accumulation_32():
    header, command, body = dry("1a", NODE="2")
    assert header == "NODE=2 USE_GPUQ=0 GPUS=0,1,2,3,4,5,6,7"
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in body and "--nproc_per_node=8" in command and "train.grad_accum_steps=32" in command
    assert "gpuq" not in body


def test_node1_default_is_four_gpus_with_accumulation_64():
    header, command, _ = dry("1a")
    assert header == "NODE=1 USE_GPUQ=1 GPUS=4,5,6,7"
    assert "--nproc_per_node=4" in command and "train.grad_accum_steps=64" in command


def test_the_training_command_is_one_clean_line():
    _, command, _ = dry("1a", NODE="2")
    assert "\\" not in command                                   # a backslash-newline inside the here-document used to leave "\ " (an extra blank argument)
    words = shlex.split(command)
    assert all(w.strip() for w in words)                         # no blank argument
    assert words[words.index("torch.distributed.run") + 4] == "scripts/run_miniworld_distogram_train.py"
    assert "--config" in words and "configs/miniworld/phase1a_distogram_medium_v200_b200.yaml" in words


def test_phase_1b_needs_a_phase_1a_run_to_continue_and_resume_passes_the_checkpoint(tmp_path):
    run = tmp_path / "2026-10-08" / "000000_x"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "last.pt").write_bytes(b"x")
    _, command, _ = dry("1a", NODE="2", RESUME=str(run))
    assert f"--ckpt {run}/checkpoints/last.pt" in command


def test_own_accumulation_and_smoke_overrides():
    _, command, _ = dry("1a", "train.grad_accum_steps=16", NODE="2")
    assert command.count("train.grad_accum_steps=") == 1 and "train.grad_accum_steps=16" in command
    _, smoke, _ = dry("1a", NODE="2", SMOKE="1")
    assert "train.train_item=2048" in smoke and "train.use_wandb=false" in smoke and "smoke-v2.0.0-phase1a-medium-b200" in smoke


@pytest.mark.parametrize("gpus", ["0,1,2", "0,1,2,3,4,5"])
def test_batch_must_divide_by_the_gpu_count(gpus):
    out = subprocess.run(["bash", str(SCRIPT), "1a"], env={**os.environ, "DRYRUN": "1", "NODE": "2", "GPUS": gpus}, capture_output=True, text=True)
    assert out.returncode == 2 and "divisible" in out.stderr
