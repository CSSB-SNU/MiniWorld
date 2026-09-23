"""Check that every shipped expanded candidate has representative H100 evidence."""
import json
from pathlib import Path
from miniworld_engine.autotune import cute_config as c

root=Path('runs/native_tuning_dev/qualification')
key=lambda config:json.dumps(config,sort_keys=True)
summary={}
for file,families in [
    ('linear.json',{'layernorm_linear_fwd_foldstats_sm90_cute':c.plain_sm90_candidates(),
                    'layernorm_linear_fwd_sm90_cute':c.fused_lnl_candidates(),
                    'transition_squeeze_residual_sm90_cute':c.plain_sm90_candidates()}),
    ('transition.json',{'transition_swiglu_fwd_sm90_cute':c.gated_sm90_candidates(),
                        'transition_gate_bwd_sm90_cute':c.gated_sm90_candidates()}),
    ('backward.json',{'layernorm_linear_bwd_dx_sm90_cute':c.lnbwd_candidates(128),
                      'transition_bwd_dx_sm90_cute':c.lnbwd_candidates(128)}),
]:
    data=json.loads((root/file).read_text())
    for op,configs in families.items():
        rows={key(r['config']):r for r in data[op]['rows']}
        wanted=[key(c.config_to_kwargs(config)) for config in configs]
        assert all(k in rows and 'relative_l2' in rows[k] for k in wanted),op
        assert all(all(e < .025 for e in rows[k]['relative_l2']) for k in wanted),op
        summary[op]={'qualified_candidates':len(wanted),
                     'max_relative_l2':max(max(rows[k]['relative_l2']) for k in wanted)}
for width in (192,256):
    data=json.loads((root/f'backward_D{width}.json').read_text())
    for op,result in data.items():
        rows={key(r['config']):r for r in result['rows']}
        wanted=[key(c.config_to_kwargs(config)) for config in c.lnbwd_candidates(width)]
        assert all(k in rows and 'relative_l2' in rows[k] for k in wanted),op
        assert all(all(e < .025 for e in rows[k]['relative_l2']) for k in wanted),op
        summary[f'{op}/D{width}']={'qualified_candidates':len(wanted),
            'max_relative_l2':max(max(rows[k]['relative_l2']) for k in wanted)}
masked=json.loads((root/'correctness.json').read_text())
wanted=[key(c.config_to_kwargs(config)) for config in c.gated_sm90_candidates()]
assert all(k in masked and max(masked[k].values()) < .025 for k in wanted)
summary['trimul_inproj_masked_sm90_cute']={'qualified_candidates':len(wanted),
    'max_relative_l2':max(max(masked[k].values()) for k in wanted)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
