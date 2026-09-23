#!/usr/bin/env python3
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'runs/anthropic_ln_ab_20260919';A=ROOT/'runs/anthropic_adoption_20260919'
rows=json.loads((R/'results.json').read_text());refs=json.loads((R/'ln-reference.json').read_text())
labels=[('input_separate_output_separate','입력 분리 / 출력 분리'),('input_separate_output_fused','입력 분리 / 출력 융합'),('input_fused_output_separate','입력 융합 / 출력 분리'),('input_fused_output_fused','입력 융합 / 출력 융합')]
body=''
for key,label in labels:
 body+='<tr><td>'+label+'</td>'+''.join('<td>{:.3f} ms</td>'.format(r['full'][key]['median_us']/1000) for r in rows)+'</tr>'
lnbody=''.join('<tr><td>L{} / {}</td><td>{:.2f}</td><td>{:.2f}</td></tr>'.format(r['N'],'출력 LN' if r['transposed'] else '입력 LN',r['times']['triton']['median_us'],r['times']['anthropic-tma-store1']['median_us']) for r in refs)
section='''<!-- LN_AB_BEGIN --><section id="ln-ab" class="panel"><span class="pill green">새 실험 · 2026-09-19</span>
<h2>Anthropic LayerNorm — 분리 vs 융합</h2>
<p><b>같은 원본 LN 함수와 같은 학습 저장값으로 비교했다. 모두 융합할 때 forward 시간이 L384 6.0%, L768 4.7% 감소했다.</b></p>
<h3>원본에서 확인한 LN 최적화</h3>
<ul><li>MMA fragment 배치 그대로, 4개 lane의 shuffle reduction으로 두 행의 FP32 평균·분산 계산.</li>
<li>BF16 packed 레지스터를 유지하고 각 pass에서 변환. FP32 값을 계속 보관하는 비용을 줄임.</li>
<li>register fence와 2개 affine dependency chain으로 live register 및 spill 제어.</li>
<li>공유 메모리 gamma/beta 벡터 로드, affine 뒤 바로 BF16 packing. 원본 TMA·ldmatrix 입력 배치 유지.</li></ul>
<p><code>ln_fragment</code> 원본을 직접 include했다. SERIAL은 스케줄 차이이며 수식은 같다.
<code>ln_stock</code>은 참조 라이브러리의 합산 순서 재현용으로 별개다.</p>
<h3>학습 forward 전체 — 네 조합</h3>
<div class="table-wrap"><table><thead><tr><th>LN 배치</th><th>L384</th><th>L768</th></tr></thead><tbody>'''+body+'''</tbody></table></div>
<p>B1 · C128/H256 · BF16 · dropout 25% · residual · H100 node02.
8라운드 순서 교대 CUDA Graph 중앙값. <b>모든 학습용 저장 포함. Weight는 미리 packing.</b>
RNG 생성·weight packing·backward·optimizer 시간 제외. 이전 packing 포함 표와 절대 시간 직접 비교 금지.</p>
<div class="flow"><div><b>분리</b>원본 LN → normalized 저장 → K1/F567이 다시 읽음</div>
<div><b>융합</b>K1/K3 안의 원본 LN → 레지스터에서 바로 GEMM 사용<br>backward용 normalized·통계 저장은 그대로</div></div>
<p>입력·출력 normalized 쓰기는 학습 때문에 양쪽 모두 남는다. 융합은 GEMM이 normalized를 다시 읽는 비용을 없앤다.
이론상 제거되는 글로벌 읽기는 L384 108 MiB, L768 432 MiB. L2 적중에 따라 실제 HBM 감소량은 다르다.</p>
<h3>LN 단독 — 원본을 꺼내면 더 빠른가?</h3>
<div class="table-wrap"><table><thead><tr><th>shape</th><th>기존 Triton · µs</th><th>Anthropic LN + TMA load/store · µs</th></tr></thead><tbody>'''+lnbody+'''</tbody></table></div>
<p><b>단독 LN의 우위는 확인되지 않았다.</b> 원본 최적화는 융합 커널 안의 register/MMA 연결에 맞춰져 있다.
분리형은 vector/TMA load, TMA store, thread 수, SERIAL 조합 8개를 비교했다.
GEMM은 기존 schedule 고정으로 전수 재튜닝 결과가 아니다. Triton cache miss는 24개 heuristic 후보.</p>
<h3>검증 및 적용 범위</h3>
<ul><li>L384/768 네 조합의 출력·저장값 12개 모두 비트 일치. L64/72 개별 저장 비교도 통과.</li>
<li>Compute Sanitizer memcheck 0 errors / racecheck 0 hazards.</li>
<li>실험 코드에 두 버전 보존. production dispatch와 기존 학습 잡은 변경하지 않음.</li>
<li>원본 inference와 달리 K3의 입력 LN은 재계산하지 않고 저장한 x_n을 공유. 원본 inference 승리 주장이 아님.</li></ul>
<p><a href="assets/ln-ab-results.json">전체 측정 JSON</a> · <a href="assets/ln-ab-reference.json">LN 단독 JSON</a></p>
<p class="tiny muted">출처: Anthropic uplifting-biomolecular-modeling f4f62fa · Apache-2.0 · native/v5/csrc/tmn_kernels.cuh의 ln_fragment.
로컬 재현: runs/anthropic_ln_ab_20260919/README.md</p></section><!-- LN_AB_END -->'''
for p in (ROOT/'ANTHROPIC_TRIMUL.html',A/'web/trimul.html',A/'site/dist/trimul.html'):
 s=p.read_text();s=re.sub(r'<!-- LN_AB_BEGIN -->.*?<!-- LN_AB_END -->','',s,flags=re.S)
 s=s.replace('<main>','<main>'+section,1)
 if 'href="#ln-ab"' not in s:s=s.replace('<nav>','<nav><a href="#ln-ab">LN 분리·융합 실험</a>',1)
 p.write_text(s)
for d in (A/'web/assets',A/'site/dist/assets'):
 d.mkdir(exist_ok=True,parents=True)
 (d/'ln-ab-results.json').write_text((R/'results.json').read_text())
 (d/'ln-ab-reference.json').write_text((R/'ln-reference.json').read_text())
print('Updated LN experiment in dashboard')
