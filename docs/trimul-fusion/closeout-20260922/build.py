from pathlib import Path
import json,hashlib,html,ast,xml.etree.ElementTree as ET
D=Path(__file__).resolve().parent;ROOT=D.parents[2];RUN=ROOT/'runs/trimul_full_latest_20260922'
r=json.loads((RUN/'results.json').read_text());s=json.loads((RUN/'summary.json').read_text());assert r['complete'] and not s['latest_correctness_passed']
for ext in ('md','html'):
 p=ROOT/('TRIMUL_STATUS.'+ext);q=ROOT/('TRIMUL_STATUS_HISTORY_20260922.'+ext)
 if not q.exists():q.write_bytes(p.read_bytes())
# Clarify the pre-existing entry point without changing execution.
p=ROOT/'runs/trimul_training_current.py';src=p.read_text();a=src.index('"""');b=src.index('"""',a+3)+3
src='''"""Retained development adapter; NOT the final measured L384 combination.

Uses cache-policy B1 plus the older joint_lncolumns B7. Execution is preserved.
The final L384 benchmark composition is runs/trimul_full_latest_20260922/policy.py
(Latest). Its input-LN parameter gradients exceed the existing strict limit;
it is a diagnostic candidate, not a promoted default. See TRIMUL_STATUS.md.
"""'''+src[b:];ast.parse(src);p.write_text(src)
text='''# TriMul 개발 마무리 · 2026-09-22

이번 최적화 실험은 종료한다. **성능 실험 완료와 전체 정확도 통과를 구분한다.**
최신 L384 조합은 1.010ms지만 입력 LN gradient 검증이 남아 기본 경로로 승격하지 않았다.
SoL90·B7 252µs 목표 달성을 주장하지 않는다.

## 최종 성능 — 동일 실행 job15526

H100/node01, L384, C128/양방향 H256, BF16, dropout25%, pair mask/residual.

| 구성 | forward | backward | 전체 fwd+bwd |
|---|---:|---:|---:|
| 이전 개선 B7 조합 | 294.400µs | 767.296µs | 1065.824µs |
| 구형 B7을 둔 검사 코드 | 294.592µs | 923.712µs | 1222.288µs |
| 최신 B1 + 최신 단일 B7 후보 | 294.384µs | 712.624µs | **1009.808µs** |

이전 개선 조합 대비 **5.26% 시간 단축 / 1.0555배**. 전체6×300회, 구간5×300회 교차 graph 측정.
full은 직접 측정했으며 분리 구간의 합계가 아니다. optimizer/RNG 생성/CPU dispatch/compile은 제외한다.
과거1048µs·직전1185µs와 직접 나누지 않는다. 이전 개선 구성의 B1/B7 cubin SHA 일치를 확인했다.
L768 최신 B7 조합의 전체 시간은 측정하지 않았다.

## 선택과 적용 상태

| 부분 | 구현 / 위치 | 상태 |
|---|---|---|
| Forward | Anthropic 유래 infer_k1 + save_k3, cuBLAS | 동일한 학습 forward 유지 |
| B1–B4 | runs/trimul_b1_gate_demote_20260922/policy.py | dWproj 저장 지연 + TMA 캐시 정책 선택; L384/768 bit-exact·memcheck·racecheck 통과 |
| B7–B12 | runs/trimul_b7_weight_batch128_20260922/plan.py | K128/ring12/producer32/cluster2 단일 CUDA; L384 개별 검사·sanitizer 통과, 새 전체 검사에서 입력 LN gradient 한도초과 |
| 최종 측정 조합 | runs/trimul_full_latest_20260922/policy.py, Latest | 위 B1/B7을 결합한 진단 후보; 기본 배선 아님 |
| 기존 개발 진입점 | runs/trimul_training_current.py | 기존 joint_lncolumns B7 유지; 최종 측정 조합과 다름을 docstring에 명시 |
| 생산 dispatch | miniworld-engine | 이번 마무리에서 변경 없음 |

B1의 dWproj 누산값은 gate 단계까지 레지스터에 유지한다. dGate TMA 쓰기는 evict_last,
소비할 때는 evict_first를 적용한다. L384는 소비한 x_n도 evict_first다.
B7은 명시적 TMA/WGMMA, 260 CTA/256 threads, shared112KiB, K128 weight stages와 global ring을 쓴다.
단일 호출이어도 global ring/partial scratch 왕복이 존재한다.

## 저장 정책

forward에서 BF16 ab(left/right), 원본 BF16 tri, affine 입력 x_n BF16, 출력 LN mean/rstd FP32를 유지한다.
출력 정규화 activation·projection·gate는 별도 저장하지 않고 backward 안에서 다시 계산한다.
cuBLAS contraction과 기존 수학/반올림 정책을 유지한다.

## 검증과 미해결 항목

최신 조합의3case에서 forward는 정확히 일치하고 모든 경로의 graph/eager 결과는 bit-exact다.
dX와 가중치 gradient는 기존 허용치 내지만 아래 입력 LN 파라미터 gradient는 한도를 넘었다.

| case | gradient | 상대L2 | 기존 한도 |
|---|---|---:|---:|
| 1 | dgamma_in | 8.370424e-6 | 5e-6 |
| 1 | dbeta_in | 6.891240e-6 | 5e-6 |
| 2 | dgamma_in | 9.534856e-6 | 5e-6 |

한도를 완화하지 않았다. 개별 커널 sanitizer 통과가 전체 수치 정확도를 보장하지 않는다.
재개할 때 우선 이 오차를 독립 참조와 비교해 원인을 분리하고, 해결한 뒤 전체3자 벤치와 검증을 다시 수행한다.
L768 최신 단일 B7 적용, 생산 dispatch 승격, SoL90 목표는 완료 항목이 아니다.

## 추가 실험 정리

B1 후속7개 방향·새 후보70개(대조군 포함83개 shape/config)를 기록했다.
TMA 묶음, L2 보존 비율, 타일 분할·부분합 캐시, gate/LN 중첩, 합산 vector load, 동기화 변경을 검사했다.
추가 채택은 없다. 작은 동기화 후보는 B1 0.28% 감소였지만 전체0.07% 차이는 변동과 구분하기 어려웠다.
실패 후보와 cubin/NCU/samples는 재현을 위해 보존한다. 성능을 반복 주장하는 낡은 헤드라인은 첫 화면에서 제외했다.

## 출처와 기조

Anthropic의 biomolecular inference 구현과 TMA/WGMMA primitives를 적극 차용한 학습 확장이다.
자체 추론 구현보다 우수했던 upstream 결과를 계승한다는 기존 기조와 라이선스 표기를 유지한다.
관련 통합 기록: [Anthropic inference](../../../tmp_kernel/ANTHROPIC_INFERENCE.md).

## 재현·증거

- [최종 벤치/실패 수치/telemetry](../../../runs/trimul_full_latest_20260922/README.md)
- [B1 선택과 검증](../../../runs/trimul_b1_gate_demote_20260922/README.md)
- [B7 선택과 전체 검사 제한](../../../runs/trimul_b7_weight_batch128_20260922/README.md)
- [후속 제외 실험과 NCU](../../../runs/trimul_b1_cache_followup_20260922/README.md)
- [고정 증거 목록·SHA-256](manifest.json)
- [누적 상태 보관본](../../../TRIMUL_STATUS_HISTORY_20260922.md)

동일 조건 재현: 저장소 루트에서 `sbatch --job-name=trimul-full-now runs/trimul_full_latest_20260922/bench.sbatch`.
새 실험은 실행하지 않았다. 관련 마지막 실험·검증 잡들은 종료된 상태이며 다른 학습 잡은 건드리지 않았다.
이번 마무리에서 commit/push는 수행하지 않았다.
'''
(D/'README.md').write_text(text)
# Compact native SVG: global-memory boundaries, including global ring traffic.
rows=[
('F1 · infer_k1','x, input LN params, front weights, mask',['xn = LN_in(x)','ab = masked_front_GLU(xn)'],'BF16 ab (left/right; both directions)'),
('F2 · cuBLAS','BF16 ab (left/right)',['tri = bidirectional_contraction(ab)'],'BF16 tri [256,L,L]'),
('F3 · save_k3','x, tri, Wp/Wg, LN params, dropscale',['p = LN_out(tri) @ Wp.T; g = sigmoid(LN_in(x) @ Wg.T)','y = x + dropscale * p * g'],'y; saved affine x_n; output mean/rstd'),
('B1–B4 · b1_fused · one launch','dy, x_n, tri, output stats, Wp/Wg, dropscale',['recompute output LN / p / g; form dP and dG','dWproj + output LN backward; then dWgate'],'dTri, dG, dWp/dWg, output LN grads'),
('B5–B6 · cuBLAS','dTri, saved ab',['dLeft, dRight = contraction_backward(dTri, ab)'],'BF16 dLeft, dRight'),
('B7–B12 · b7_joint · one launch','dLeft/dRight, dG, x_n, x, dy, weights, mask',['recompute front p/g; share derivatives for dX/dW','input LN backward + residual dy + output gate branch'],'dX; four front dW; input LN grads')]
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1350" role="img" aria-labelledby="title desc"><title id="title">TriMul final measured L384 candidate</title><desc id="desc">Inputs and outputs at global memory boundaries. Candidate has an unresolved input LN gradient validation issue.</desc><rect width="1600" height="1350" fill="#f5f8fc"/><g font-family="sans-serif">','<text x="32" y="45" font-size="27" font-weight="bold">TriMul L384 · final measured candidate · 1.010 ms</text>','<text x="32" y="80" font-size="19" fill="#9a4e12">Input LN gradient validation pending; not promoted to default. Formulas are schematic (BF16 rounding omitted).</text>']
for i,(title,inputs,code,outputs) in enumerate(rows):
 y=110+i*180
 svg.append('<text x="32" y="%d" font-size="21" font-weight="bold">%s</text>'%(y+23,html.escape(title)))
 for x,w,color in [(32,355,'#e8f3fc'),(414,752,'#eaf8f0'),(1193,375,'#fff1da')]:svg.append('<rect x="%d" y="%d" width="%d" height="120" rx="9" fill="%s" stroke="#bdd0dd"/>'%(x,y+37,w,color))
 # Text flows into bounded lines, never beyond the box.
 import textwrap
 for x,cap,label,words in [(46,39,'GLOBAL READ',inputs),(1207,40,'GLOBAL WRITE',outputs)]:
  svg.append('<text x="%d" y="%d" font-size="16" font-weight="bold">%s</text>'%(x,y+64,label))
  for j,line in enumerate(textwrap.wrap(words,cap)):svg.append('<text x="%d" y="%d" font-size="16">%s</text>'%(x,y+88+21*j,html.escape(line)))
 svg.append('<text x="430" y="%d" font-size="16" font-weight="bold">ON-CHIP COMPUTATION / SCHEMATIC CODE</text>'%(y+64))
 for j,line in enumerate(code):svg.append('<text x="430" y="%d" font-family="monospace" font-size="15">%s</text>'%(y+92+23*j,html.escape(line)))
 for x in [394,1173]:svg.append('<text x="%d" y="%d" font-size="24">→</text>'%(x,y+110))
svg+=['<rect x="32" y="1200" width="1536" height="110" rx="9" fill="#f5e9f4"/>','<text x="48" y="1228" font-size="18" font-weight="bold">GLOBAL SCRATCH (not eliminated by a single CUDA launch)</text>','<text x="48" y="1256" font-size="17">B1: FP32 dW/LN partial writes and final reads. B7: ring/xring + flags + dW/LN partial writes/reads.</text>','<text x="48" y="1284" font-size="17">Live weight packing and Wp transpose are included in measured time. No output-normalized activation or p/g saves.</text>','</g></svg>']
(D/'selected-L384.svg').write_text('\n'.join(svg));ET.parse(D/'selected-L384.svg')
# Canonical short status; past notes preserved at the same directory level.
(ROOT/'TRIMUL_STATUS.md').write_text('''# TriMul 최적화 마무리 · 2026-09-22

추가 실험 종료. **최신 L384 조합 1.010ms는 정확도 검증이 남은 후보의 진단값**이다.
이전 개선 조합 대비 같은 실행에서5.26% 단축. 입력 LN gradient 상대L2 최대9.535e-6가 기존5e-6 한도를 넘었다.

| 구간 | 최신 후보 |
|---|---:|
| forward | 294.384µs |
| backward | 712.624µs |
| 전체 fwd+bwd 직접 측정 | **1009.808µs** |

H100, 양방향 L384/C128/H256 BF16, dropout25%/mask/residual. optimizer/RNG/CPU dispatch 제외.

- **B1 선택:** dWproj 저장 지연 + TMA 캐시 정책. L384/768 검증 완료.
- **B7 후보:** K128/ring12/producer32/cluster2 단일 CUDA. 전체 입력 LN gradient 검증 미완료.
- **최종 측정 조합:** `runs/trimul_full_latest_20260922/policy.py::Latest`.
- **기존 개발 진입점:** `runs/trimul_training_current.py`는 예전 B7 조합을 유지한다. 최종 측정 조합과 다르다.
- **생산 적용:** 이번 마무리에서 변경 없음. SoL90·252µs 목표 미달.

[최종 정리·구현 선택·재현](docs/trimul-fusion/closeout-20260922/README.md) ·
[최종 배선 SVG](docs/trimul-fusion/closeout-20260922/selected-L384.svg) ·
[전체 재측정 원본](runs/trimul_full_latest_20260922/index.html) ·
[과거 기록](TRIMUL_STATUS_HISTORY_20260922.md)
''')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriMul 개발 마무리</title><style>body{max-width:1200px;margin:24px auto;padding:0 18px;font:16px/1.65 system-ui;background:#f3f6fa;color:#24364c}section{background:white;border-radius:10px;padding:20px;margin:18px 0}h1,h2{margin-top:0}.notice{background:#fff0d4;border-left:5px solid #bc7718;padding:14px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #dce4ee;text-align:left}a{color:#1469a2}.scroll{overflow:auto}img{width:100%;min-width:1000px}summary{cursor:pointer;font-weight:bold}</style><h1>TriMul · 개발 마무리</h1><p>2026-09-22 · 이번 최적화 실험 종료 · L384 양방향 H100</p><p class="notice"><b>최신 후보 1.010ms · 같은 실행의 이전 개선 조합 대비 5.26% 단축.</b><br>입력 LN gradient 검증이 남았습니다: 상대L2 최대9.535×10⁻⁶, 기존 한도5×10⁻⁶. 기본 경로로 승격하지 않았으며 SoL90은 미달입니다.</p><section><h2>최종 동일 실행 비교 · µs</h2><div class="scroll"><table><tr><th>구성</th><th>forward</th><th>backward</th><th>전체 직접 측정</th></tr><tr><td>이전 개선 B7 조합</td><td>294.400</td><td>767.296</td><td>1065.824</td></tr><tr><td>구형 B7 검사 코드</td><td>294.592</td><td>923.712</td><td>1222.288</td></tr><tr><td>최신 B1 + 단일 B7 후보</td><td>294.384</td><td>712.624</td><td><b>1009.808</b></td></tr></table></div><p>BF16 C128/H256 · dropout25% · mask/residual · job15526 · 전체6×300회 교차 CUDA graph.<br>전체는 분리 구간 합계가 아닙니다. optimizer/RNG 생성/CPU dispatch/compile 제외. L768 최신 전체 수치는 없습니다.</p></section><section><h2>선택과 적용 상태</h2><table><tr><th>부분</th><th>정리 결과</th></tr><tr><td>B1–B4</td><td>dWproj 저장 지연 + TMA 캐시 정책 유지. L384/768 bit-exact·memcheck·racecheck 통과.</td></tr><tr><td>B7–B12</td><td>K128/ring12/producer32/cluster2 단일 CUDA 후보. 전체 입력 LN gradient 검증 미완료.</td></tr><tr><td>최종 측정 조합</td><td>runs/trimul_full_latest_20260922/policy.py의 Latest. 진단 후보.</td></tr><tr><td>기존 개발 진입점</td><td>trimul_training_current.py는 예전 joint_lncolumns B7을 유지. 아래 후보와 구분.</td></tr><tr><td>출처</td><td>Anthropic 추론 구현·TMA/WGMMA primitives를 계승한 학습 확장.</td></tr></table></section><section><h2>최종 측정 후보 배선</h2><p>각 행: global 입력 → 내부 계산 → global 출력. 단일 CUDA 호출에도 ring/partial scratch 왕복이 있습니다.</p><div class="scroll"><img src="docs/trimul-fusion/closeout-20260922/selected-L384.svg" alt="최종 측정한 L384 후보의 입력·내부 계산·출력 배선"></div><a href="docs/trimul-fusion/closeout-20260922/selected-L384.svg">SVG 원본</a></section><section><h2>남은 일</h2><p>입력 LN γ/β gradient 차이의 원인을 독립 참조와 대조하고 해결한 뒤 전체 정확도를 재검증합니다. L768 적용·생산 승격·SoL90 목표는 완료되지 않았습니다.</p><p><a href="docs/trimul-fusion/closeout-20260922/README.md">최종 정리·재현</a> · <a href="docs/trimul-fusion/closeout-20260922/manifest.json">증거 목록·SHA-256</a> · <a href="runs/trimul_full_latest_20260922/index.html">전체 벤치·오차 원본</a></p></section><details><summary>과거 실험·상세 그림 보관</summary><p><a href="TRIMUL_STATUS_HISTORY_20260922.html">이전 HTML 현황 전체</a> · <a href="TRIMUL_STATUS_HISTORY_20260922.md">누적 문서</a></p><p><a href="runs/trimul_b1_gate_demote_20260922/index.html">B1 선택 검증</a> · <a href="runs/trimul_b7_weight_batch128_20260922/index.html">B7 선택 검증</a> · <a href="runs/trimul_b1_cache_followup_20260922/index.html">추가70개 후보·NCU</a></p><p>보관본의 “현재/최신” 표기는 당시 시점의 기록입니다. 예전 SVG의 split B7는 최종 단일 B7 후보가 아닙니다.</p></details></html>'''
(ROOT/'TRIMUL_STATUS.html').write_text(page)
files=[RUN/'policy.py',RUN/'bench.py',RUN/'bench.sbatch',RUN/'results.json',RUN/'summary.json',RUN/'historical-cubin-verification.json',ROOT/'runs/trimul_b1_gate_demote_20260922/policy.py',ROOT/'runs/trimul_b7_weight_batch128_20260922/selected.json',ROOT/'runs/trimul_b7_weight_batch128_20260922/plan.py',D/'README.md',D/'selected-L384.svg']
for name in ['b1','b7']:
 for item in r['cubins']['latest_combined'][name]:files.append(Path(item['path']))
for folder in ['trimul_b1_gate_demote_20260922','trimul_b7_weight_batch128_20260922']:
 folder=ROOT/'runs'/folder
 for pattern in ['*.cu','*.inc','*.cuh','*sanitizer*.json']:
  files.extend(folder.glob(pattern))
manifest=dict(date='2026-09-22',phase='experiments_closed_accuracy_issue_open',new_experiments_started=False,default_promoted=False,pushed=False,performance_job=r['job'],latest_full_us=s['times_us']['full']['latest_combined'],latest_correctness_passed=False,pending=['input LN parameter gradient strict tolerance','L768 latest joint B7 validation','production integration','SoL90'],artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))])
(D/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CLOSEOUT',len(manifest['artifacts']),'artifacts')
