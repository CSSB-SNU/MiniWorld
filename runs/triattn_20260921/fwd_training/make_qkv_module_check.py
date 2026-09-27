"""Reuse the full-workload fixtures while changing the experimental boundary."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'module_check.py').read_text()
s=s.replace('from native import extension','from native import extension\nfrom qkv_module import make_forward\nfrom miniworld_engine.kernels.triangle_attention.cuda import ln_backward')
s=s.replace('original = core._tri_attn_fwd','original = ln_backward.forward')
start=s.index('@_compile.opaque')
end=s.index('\n\npkg = ',start)
s=s[:start]+'''candidate=make_forward(a.artifact)
assert not a.installed and a.baseline_artifact is None, 'experimental module boundary only'
''' + s[end:]
s=s.replace('core._tri_attn_fwd = fn','ln_backward.forward = fn')
s=s.replace('core._tri_attn_fwd=fn','ln_backward.forward=fn')
s=s.replace('core._tri_attn_fwd=candidate','ln_backward.forward=candidate')
s=s.replace('core._tri_attn_fwd=forward','ln_backward.forward=forward')
s=s.replace('core._tri_attn_fwd=original','ln_backward.forward=original')
start=s.index("if a.mode == 'qualify':")
end=s.index('for ending in (False,True):',start)
s=s[:start]+'''if a.mode == 'qualify':
    from qkv_gradcheck import run
    run(ext,report,save)

''' + s[end:]
s=s.replace("assert any('training_fwd_stream' in n for n in names),names", "assert any('qkv_attention_resident' in n for n in names),names\n                assert not any('training_fwd_stream' in n for n in names),names")
s=s.replace("assert not any('_attn_fwd' in n for n in names),names", "assert not any('_attn_fwd' in n for n in names),names")
s=s.replace("report.update(rounds=a.rounds, replays=a.replays, baseline_artifact=a.baseline_artifact)",
'''assert manifests['fwd_manifest.json']['artifact']=='cooperative_head2'
report.update(rounds=a.rounds,replays=a.replays,baseline_artifact='installed checkpoint17628',
              fusion_boundary='QKV projection + attention; shared front/attention autograd')''')
(r/'qkv_module_check.py').write_text(s)
