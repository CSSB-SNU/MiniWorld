"""Qualify expanded masked-front candidates against FP32, with restartable tuning."""
import importlib
import json
from pathlib import Path
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture, native
from miniworld_engine.autotune.cute_config import config_to_kwargs, gated_sm90_candidates

outdir = Path('runs/native_tuning_dev/qualification')
outdir.mkdir(parents=True, exist_ok=True)
settings.configure(run_autotune=True, compile_jobs=12, bench_clear_mb=16, bench_rep_ms=10)
capture.set_round_cache(str(outdir / 'rounds'))
torch.manual_seed(119)
torch.backends.cuda.matmul.allow_tf32 = False
module = importlib.import_module('miniworld_engine.kernels.trimul_inproj.cute.masked_front')
a = torch.randn(264, 128, device='cuda', dtype=torch.bfloat16)
b = torch.randn(128, 512, device='cuda', dtype=a.dtype) * 0.08
mask = torch.ones(1, 264, device='cuda', dtype=torch.float32)
mask[:, ::3] = 0
out, preact = module._masked_front_fake(a, b, mask, True)
ref = a.float() @ b.float()
expected = ref[:, 1::2] * ref[:, ::2].sigmoid() * mask.T
checked = {}

def run(config):
    module._launch(a, b, out, preact, mask, config)
    key = json.dumps(config_to_kwargs(config), sort_keys=True)
    if key not in checked:
        torch.cuda.synchronize()
        error = float((out.float()-expected).norm()/expected.norm())
        pre_error = float((preact.float()-ref).norm()/ref.norm())
        assert error < .025 and pre_error < .025, (config, error, pre_error)
        assert (out[::3] == 0).all()
        checked[key] = {'output_rel_l2': error, 'preact_rel_l2': pre_error}
        (outdir/'correctness.json').write_text(json.dumps(checked, indent=2))

from miniworld_engine.autotune.cute_config import resolve_config
op = 'trimul_inproj_masked_sm90_cute'
bucket = native.tensor_key(a, b, mask, extra=(True,))
configs = gated_sm90_candidates()
winner = resolve_config(op, configs, dtype=str(a.dtype), bucket=bucket,
                        device_index=0, run=run)
run(winner)
capture.dump_shard(str(outdir/'masked_front.json'), unit_complete=True)
# All cached candidates must be usable without compilation/benchmarking after reset.
capture.reset()
def forbidden(_):
    raise AssertionError('warm tuning must not execute a measured candidate')
second = resolve_config(op, configs, dtype=str(a.dtype), bucket=bucket,
                        device_index=0, run=forbidden)
assert second == winner
(outdir/'result.json').write_text(json.dumps({'candidates':len(configs),
    'correct':len(checked), 'winner':config_to_kwargs(winner),
    'warm_reused':capture.skipped_configs(), 'device':torch.cuda.get_device_name()}, indent=2))
print((outdir/'result.json').read_text(), flush=True)
