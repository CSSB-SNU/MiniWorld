from pathlib import Path
import json,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent
full=json.loads((p/'accuracy-full-comparison.json').read_text());bidir=json.loads((p/'accuracy-bidir-fp32.json').read_text());graph=json.loads((p/'graph-accum-pf16-r4.json').read_text())
suite=ET.parse(p/'accuracy-kernels.xml').getroot();cases=list(suite.iter('testcase'))
counts=dict(tests=len(cases),failures=len(list(suite.iter('failure'))),errors=len(list(suite.iter('error'))),skipped=len(list(suite.iter('skipped'))))
assert counts['failures']==counts['errors']==0
r=dict(kernel_tests=counts,full_model=full,bidir_fp32=bidir,graph=dict(passed=graph['passed'],gradient_tensors=graph['gradient_tensors'],max_gradient_relative_l2=max(c['max_gradient_relative_l2'] for c in graph['checks']),accumulation=[1,2,64],optimizer_stages=['initial','after_optimizer']),jobs=[16677,16679,16680],limitations=['Full-model comparison uses dropout OFF and current forced-Triton reference, not FP32 whole-model oracle.','Dropout ON covered separately by modules and graph versus ordinary execution.','Three optimizer steps on one frozen sample do not establish long-term convergence or validation quality.','FP16/BF16 arithmetic paths need not match bitwise; no claim of <0.01% overall error.','No production training job was changed.'])
(p/'accuracy-summary.json').write_text(json.dumps(r,indent=2))
html='<section id="accuracy-training"><h2>최신 커널 학습 정확도 검증</h2>'
html+=f'<p>GPU tests: {counts}. 양방향 TriMul FP32 비교 + 실제 PF16/recycle4 3-step Adam + dropout ON CUDA graph 누적 검증 완료. 기존 학습 잡 변경 없음.</p>'
html+='<table><tr><th>Step</th><th>Triton loss</th><th>최신 loss</th><th>loss 상대 차이</th><th>전체 gradient 상대 L2</th><th>gradient cosine</th><th>Adam update 상대 L2</th></tr>'
for a in full['rows']:
 html+=f'<tr><td>{a["step"]}</td><td>{a["loss_triton"]:.6f}</td><td>{a["loss_native"]:.6f}</td><td>{100*a["loss_relative_error"]:.4f}%</td><td>{100*a["gradients"]["relative_l2"]:.3f}%</td><td>{a["gradients"]["cosine"]:.6f}</td><td>{100*a["updates"]["relative_l2"]:.3f}%</td></tr>'
html+='</table><p>전체 비교는 dropout OFF, 같은 checkpoint445와 Adam 상태, L384/atom4096/MSA1024입니다. 서로 독립적으로 업데이트하므로 step1~2는 optimizer trajectory 차이도 포함합니다. 436개 gradient 모두 유한. 장기 수렴/validation score 검증은 아닙니다.</p>'
html+=f'<p>Dropout ON graph: 1/2/64회 누적, optimizer 전/후 loss 정확히 동일, 최대 gradient 상대 L2 {100*r["graph"]["max_gradient_relative_l2"]:.5f}%. TriMul FP32 oracle: dropout25%, L384 D64/128 및 L768 D128, gamma=0 추가 케이스; 출력 최대 {100*max(a["errors"]["output"] for a in bidir):.3f}%, 입력 gradient 최대 {100*max(a["errors"]["dx"] for a in bidir):.3f}%, 가중치 gradient 최대 {100*max(v for a in bidir for k,v in a["errors"].items() if k not in ("output","dx")):.3f}%.</p><p><a href="runs/engine2-training-profile-20260923/accuracy-summary.json">전체 검증 결과</a></p></section>'
h=p.parents[1]/'ENGINE_TRAINING_PROFILE.html';s=h.read_text();s=s.replace('<section id="inference-r4-ablation">',html+'<section id="inference-r4-ablation">');h.write_text(s)
print(json.dumps(dict(kernel_tests=counts,graph=r['graph']),indent=2))
