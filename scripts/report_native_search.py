"""Print the native config inventory without allocating a GPU."""
import argparse
import json
from miniworld_engine.autotune import cute_config as c

parser=argparse.ArgumentParser()
parser.add_argument('--output')
args=parser.parse_args()
report={}
for family,configs in [
    ('gated',c.gated_sm90_candidates()),('plain',c.plain_sm90_candidates()),
    ('M2',c.fused_lnl_candidates()),('LN backward D128',c.lnbwd_candidates(128)),
    ('LN backward D192',c.lnbwd_candidates(192)),('LN backward D256',c.lnbwd_candidates(256)),
    ('TM2',c.tm2_candidates()),
]:
    report[family]={'candidates':len(configs),
        'tiles':sorted({(cfg.tile_m,cfg.tile_n) for cfg in configs}),
        'clusters':sorted({(cfg.cluster_m,cfg.cluster_n) for cfg in configs}),
        'swizzles':sorted({cfg.max_swizzle_size for cfg in configs}),
        'dynamic':sorted({cfg.is_dynamic_persistent for cfg in configs})}
text=json.dumps(report,indent=2)
if args.output:
    from pathlib import Path
    Path(args.output).write_text(text+'\n')
print(text)
