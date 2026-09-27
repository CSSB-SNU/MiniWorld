"""Keep full-module gates but compare incrementally with frozen Q fusion17774."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'qkv_module_check.py').read_text()
s=s.replace('from qkv_candidate import load','from qg_candidate import load,baseline')
s=s.replace("original = getattr(ln_backward,'unfused_forward',ln_backward.forward)", 'original = baseline().forward')
s=s.replace("baseline_artifact='installed checkpoint17628'", "baseline_artifact='frozen checkpoint17774'")
s=s.replace("'QKV projection + attention; shared front/attention autograd'", "'Q and gate projection + attention; shared front/attention autograd'")
s=s.replace("('qkv_module.py','qkv_module_check.py','qkv_gradcheck.py','qkv_candidate.py')", "('qg_module.py','qg_module_check.py','qg_gradcheck.py','qg_candidate.py')")
s=s.replace('from qkv_gradcheck import run','from qg_gradcheck import run')
s=s.replace("'qkv_attention_resident'", "'qg_attention_fused'")
s=s.replace("assert not any('training_fwd_stream' in n for n in names),names", "assert not any('training_fwd_stream' in n or 'qkv_attention_resident' in n for n in names),names")
s=s.replace("'baseline_build'] =", "'baseline_build'] =")
anchor="report.update(staged=a.stage,candidate_python_file=str(candidate_module.__file__),"
s=s.replace(anchor,"report['frozen_baseline_snapshot']=json.loads((Path(__file__).parent/'checkpoint17774/snapshot.json').read_text())\n"+anchor)
(r/'qg_module_check.py').write_text(s)
s=(r/'qkv_module.sbatch').read_text().replace('mw-qkv-attn-module','mw-qg-module').replace('qkv-module','qg-module').replace('qkv_module_check.py','qg_module_check.py')
(r/'qg_module.sbatch').write_text(s)
s=(r/'qkv_sanitize.sbatch').read_text().replace('mw-qkv-sanitize','mw-qg-sanitize').replace('qkv-sanitize','qg-sanitize').replace('qkv_attention_resident','qg_attention_fused').replace('qkv_check.py','qg_check.py').replace('qkv-${TOOL}','qg-${TOOL}')
(r/'qg_sanitize.sbatch').write_text(s)
