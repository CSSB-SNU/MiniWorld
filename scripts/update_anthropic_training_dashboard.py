"""Render the current native-forward training baseline in the existing site.

Run after inference dashboard generators, then publish the same private site.
Measurements come from the JSON records, never from claimed target speedups.
"""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/anthropic_adoption_20260919'
TRAIN=ROOT/'runs/anthropic_trimul_training_20260919'
results=json.loads((TRAIN/'native-training-times.json').read_text())
rows=[]
for r in results:
    ts=r['times']
    f=ts['native_training_forward_with_live_pack_dropout_residual']['median_us']/1000
    t=ts['native_training_forward_backward_reference_bwd']['median_us']/1000
    rows.append(f'<tr><td>{"단방향 outgoing" if r["direction"]=="outgoing" else "양방향"}</td><td>{r["N"]}</td><td>{r["H"]}</td><td>{f:.3f}</td><td>{t:.3f}</td></tr>')
section='''<!-- TRAINING_PROGRESS_BEGIN -->
<section id="training-progress" class="panel" style="border:2px solid #087c66">
<div class="eyebrow" style="color:#087c66">CURRENT BASELINE · 2026.09.19 · H100 / NODE02</div>
<h2>Anthropic 원본 forward를 학습에 연결</h2>
<div class="notice"><b>현재: 원본 K1 → cuBLAS → 원본 K3 전체를 사용하는 autograd 학습 기준선.</b><br>
앞서 만든 K3만 차용한 경로와 다르다. 단방향 outgoing/incoming과 양방향에 연결했다.
Backward는 아직 검증용 PyTorch 재계산 + cuBLAS이며, 최적화된 CUDA backward가 아니다.</div>
<h3>현재 학습 forward — 어떤 연산이 어디에 묶였나</h3>
<div class="flow">
<div class="node"><b>Anthropic 원본 K1 · 변경 없음</b>입력 LayerNorm<br>left/right projection + sigmoid gate + mask<br>TMA / WGMMA → a,b 저장</div>
<div class="arrow">→</div><div class="node purple"><b>cuBLAS contraction</b>단방향: 1회<br>양방향: outgoing + incoming 2회<br>하나의 최종 X buffer에 직접 출력</div>
<div class="arrow">→</div><div class="node"><b>Anthropic 원본 K3 · 변경 없음</b>출력 LayerNorm + projection<br>입력 LN 재계산 + output gate<br>TMA / WGMMA → update</div>
<div class="arrow">→</div><div class="node gray"><b>학습용 외부 연산</b>row dropout 25%<br>원래 pair residual 더하기<br>원본 K3에는 새 dropout을 넣지 않음</div>
</div>
<p><b>양방향의 수학:</b> 원본 K1의 H256 출력을 H128씩 나눠 outgoing/incoming contraction을 계산한다.
그 결과 전체 H256에 한 번의 LN_out을 적용하고 원본 K3가 C128로 투영한다.
단방향 원본 모듈 두 개를 더한 것이 아니다. 이 양방향 조립은 engine의 연결 코드다.</p>
<h3>현재 backward — 학습 가능 여부를 검증하는 기준선</h3>
<div class="flow"><div class="node gray"><b>출력 surround 재계산</b>PyTorch FP32 수식<br>LN_out / projection / gate 미분<br>forward의 실제 native X를 소비</div>
<div class="arrow">→</div><div class="node purple"><b>Contraction 미분</b>cuBLAS<br>forward의 실제 native a,b를 소비<br>outgoing/incoming 각각 미분</div>
<div class="arrow">→</div><div class="node gray"><b>입력 surround 재계산</b>PyTorch FP32 수식<br>projection / gate / LN_in 미분<br>입력 및 모든 10개 parameter gradient</div></div>
<p><b>이 backward는 느리다.</b> 원본에는 backward가 없어서 먼저 수식 검증용으로 붙였다.
이를 Anthropic 학습 커널의 성능이라고 부르거나 기존 Triton 대비 승리라고 해석하지 않는다.</p>
<h3>현재 기준선 실측</h3>
<p>BF16 pair · C128 · B1 · dropout 25% · residual 포함. CUDA Graph replay 중앙값, 단위 <b>ms</b>.
Forward에는 매 실행 live weight packing이 포함된다. 고정 dropout scale을 사용하며 RNG 생성과 optimizer는 제외한다.</p>
<div class="table-wrap"><table><thead><tr><th>경로</th><th>L</th><th>전체 H</th><th>학습 forward</th><th>forward + 검증용 backward</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>원본 단방향 update-only · weight pack 재사용 · dropout/residual 제외 시간은 L384 <b>0.155ms</b>, L768 <b>0.603ms</b>였다.
위 학습 forward와 작업량이 다르므로 가속률로 비교하지 않는다. 이번에 원본을 더 빠르게 바꾼 것은 아니다.</p>
<h3>검증 완료</h3>
<ul><li><b>14개 GPU 테스트 통과:</b> original forward 동일성, 모든 gradient, mask/dropout/residual, 단방향·양방향 모듈.</li>
<li>단방향 원본 호출과 update 출력이 비트 단위로 일치: L64·65·384·768 outgoing, L64·65 incoming.</li>
<li>L384·768 단방향/양방향 수식 비교: 출력 상대 L2 최대 0.052%, 모든 gradient 최대 0.290%.</li>
<li>SGD weight 갱신, 여러 forward를 쌓은 뒤 backward, CUDA Graph replay 중 weight/dropout 변경 검증.</li>
<li>실제 GPU trace에서 원본 <code>tmn_k1_*</code> / <code>tmn_k3_*_u</code> 호출 확인.</li></ul>
<h3>사용과 범위</h3>
<pre style="overflow-x:auto"><code>BidirectionalTriangleMultiplication(
    128, implementation="anthropic",
    anthropic_row="native_rebuilt", p_drop=0.25,
)
# 단방향 TriangleMultiplication도 같은 옵션
# TRIMUL_NATIVE_BUILD_DIR 설정, TF32 비활성화 필요</code></pre>
<p>현재 검증 범위: H100 / BF16 pair / B1 / C128 / H128 또는 H256.
Eager와 CUDA Graph는 검증했다. torch.compile fullgraph는 아직 지원하지 않으며 명시적으로 graph break한다.
Backward는 1차 미분만 지원한다. 전체 MiniWorld 모델 학습 실행은 아직 검증하지 않았다.</p>
<h3>다음 작업</h3>
<ol><li>이 원본 forward + 검증용 backward를 정확도 기준으로 고정.</li>
<li>원본 forward의 계산·반올림 규약에 맞춘 빠른 backward를 연결·개발.</li>
<li>live weight packing과 dropout/residual 등 학습 연결 비용을 줄이고 compile/cache 경로 정리.</li>
<li>원본 kernel 자체 개선은 같은 작업량으로 직접 A/B 비교. 세부 NCU stall 분석을 선행.</li></ol>
<details><summary>이전 K3 학습 시제품과 원본 NCU 기록</summary>
<p>이전 <code>implementation="triton", training_output_backend="anthropic_cuda"</code>는 Triton front + 파생 CUDA K3 + 기존 backward다.
그 경로의 Triton 대비 약 1.15배 K3 가속은 Anthropic 원본을 이겼다는 근거가 아니다. 이번 원본 전체 forward 기준선과 분리해서 보관한다.</p>
<p>원본 L768/C128 NCU: K1 209.3µs, K3 211.8µs; HBM 약 2.15/2.14TB/s; DRAM active 약 64%.
세부 stall 자료가 부족하므로 이 값만으로 roofline 도달 여부나 15% 추가 개선 가능성을 확정하지 않는다.
원본은 이미 hand CUDA/PTX·TMA·WGMMA를 사용한다.</p></details>
<p class="tiny muted">원본: <a href="https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e">Anthropic 공개 코드 · f4f62fa</a> (Apache-2.0).
우리의 이전 inference 개발보다 뛰어난 원본의 성과를 인정하고 계승한다. 현재 추가한 것은 engine 연결과 학습 지원 기준선이다.
아래는 기존 추론 분석 스냅샷이다. 이 페이지는 실시간 모니터가 아니다.</p>
</section><!-- TRAINING_PROGRESS_END -->'''
from render_anthropic_saved_section import render
legacy=section.replace('<!-- TRAINING_PROGRESS_BEGIN -->','').replace('<!-- TRAINING_PROGRESS_END -->','').replace('id="training-progress"','id="native-recompute-baseline"')
section=render(ROOT,legacy)

for path in (RUN/'site/dist/trimul.html',RUN/'web/trimul.html',ROOT/'ANTHROPIC_TRIMUL.html'):
    s=path.read_text()
    s=re.sub(r'<!-- TRAINING_PROGRESS_BEGIN -->.*?<!-- TRAINING_PROGRESS_END -->','',s,flags=re.S)
    local_section=section.replace('assets/anthropic-trimul-kernels.svg','ANTHROPIC_TRIMUL_KERNELS.svg') if path.parent==ROOT else section
    s=s.replace('<main>','<main>'+local_section,1)
    s=s.replace('<h1>Anthropic TriMul을 뜯어보면</h1>','<h1>TriMul 커널 개발 현황</h1>')
    if '<a href="#training-progress">' not in s:s=s.replace('<nav>','<nav><a href="#training-progress">현재 학습 개발</a>',1)
    path.write_text(s)
for path in (RUN/'site/dist/index.html',RUN/'web/index.html',ROOT/'ANTHROPIC_STATUS.html'):
    s=path.read_text()
    substitutions={
      '추론 통합·검증 완료 + TriMul K3 학습용 CUDA 확장 구현.':'추론 통합 완료 + TriMul 원본 K1/K3 전체 forward 학습 연결.',
      'K1 학습 이식 및 새 CUDA backward':'빠른 CUDA backward 및 전체 모델 검증',
      '학습 개발 착수: K3 구현 완료':'원본 K1/K3 학습 연결 · backward는 검증용',
      'K3 학습 forward 완료 · K1 / CUDA backward 남음':'원본 전체 forward 학습 연결 · 빠른 backward 남음',
      'K3 학습 확장을 별도 옵션으로 연결했다. Backward는 기존 Triton/cuBLAS를 사용한다. 원본 추론 API는 계속 grad-enabled 호출을 거절한다.':'TriMul 원본 K1/K3를 학습 모듈에 연결했다. 이 기준선의 backward는 PyTorch 재계산/cuBLAS이며 아직 느리다. 원본 low-level 추론 API와 구분한다.',
    }
    for a,b in substitutions.items():s=s.replace(a,b)
    link='ANTHROPIC_TRIMUL.html' if path.parent==ROOT else 'trimul.html'
    notice=f'''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: 기존 융합·저장 정책 유지 + Anthropic 파생 커널</b> · <a href="{link}#training-progress">배선·성능·검증 보기 →</a><br>
    Front/F567은 Anthropic 파생 CUDA, LN과 backward는 기존 경로다. 동일 저장 정책에서 전체 학습 시간 2.8%/3.8% 감소. 원본 내부와 변경 지점의 SVG도 추가했다.</section><!-- TRAINING_LINK_END -->'''
    s=re.sub(r'<!-- TRAINING_LINK_BEGIN -->.*?<!-- TRAINING_LINK_END -->','',s,flags=re.S)
    path.write_text(s.replace('<main>','<main>'+notice,1))
print('Updated saved-policy training dashboard and SVG links')
