"""Current saved-policy training section for the persistent kernel dashboard."""
import json

def render(root, legacy):
    run=root/'runs/anthropic_saved_training_20260919'
    data=json.loads((run/'full-times.json').read_text())
    rows=[]
    for r in data:
        t=r['times']
        for key,label in [('fwd','forward'),('train','forward + backward')]:
            a,b=t['triton_'+key]['median_us'],t['anthropic_saved_'+key]['median_us']
            rows.append(f'<tr><td>{r["N"]}</td><td>{label}</td><td>{a/1000:.3f}</td><td><b>{b/1000:.3f}</b></td><td>{100*(1-b/a):.1f}%</td></tr>')
    return '''<!-- TRAINING_PROGRESS_BEGIN -->
<section id="training-progress" class="panel" style="border:2px solid #087c66">
<div class="eyebrow" style="color:#087c66">CURRENT · 2026.09.19 · H100 / NODE02 · SAME TRAINING SAVE POLICY</div>
<h2>기존 융합·저장 정책 유지 + Anthropic 파생 CUDA</h2>
<div class="notice"><b>학습 경로를 기존 저장 정책으로 연결했다.</b><br>
입력 projection과 F567을 원본 K1/K3에서 파생한 CUDA로 교체했다.
별도 입력/출력 LN과 기존 Triton/cuBLAS backward는 유지한다. 느린 재계산 backward를 쓰지 않는다.</div>
<h3>실제 융합 경계와 저장값</h3>
<div class="flow">
<div class="node gray"><b>F1 · 입력 LN</b>기존 Triton<br>x_n + mean/rstd 저장</div><div class="arrow">→</div>
<div class="node"><b>F2 · K1 파생 CUDA</b>prenormalized x_n 입력<br>projection + gate + mask<br>a/b + preact TMA 저장</div><div class="arrow">→</div>
<div class="node purple"><b>F3 · cuBLAS</b>outgoing + incoming<br>최종 X buffer로 직접 출력</div><div class="arrow">→</div>
<div class="node gray"><b>F4 · 출력 LN</b>기존 Triton<br>normalized X + mean/rstd 저장</div><div class="arrow">→</div>
<div class="node"><b>F567 · K3 파생 CUDA</b>projection + output gate<br>+ dropout + residual<br>projection / gate TMA 저장</div>
</div>
<div class="notice"><b>Backward: 기존 Triton/cuBLAS 그대로.</b> 저장 목록의 shape·dtype·순서가 이전과 일치한다.
입력 projection/gate GEMM 4개와 출력 GEMM 2개를 backward에서 재계산하지 않는다.</div>
<h3 id="kernel-svg">Anthropic 커널은 어떻게 만들어졌나 — SVG</h3>
<p><a href="assets/anthropic-trimul-kernels.svg" target="_blank" rel="noopener">SVG 원본 열기 / 확대 →</a></p>
<img src="assets/anthropic-trimul-kernels.svg" alt="원본 K1과 K3의 HBM, TMA, shared memory, warpgroup, WGMMA 연결 및 학습 파생형 변경 지점" style="display:block;width:100%;height:auto;border:1px solid #dbe4e8;border-radius:12px" loading="lazy">
<p>원본 커널 본체를 직접 변형했다. TMA/WGMMA, producer/consumer, weight ring, MMA/epilogue overlap을 계승한다.
K1/K3 내부의 LN 대신 기존 별도 LN 결과를 읽고, 학습에 필요한 저장 및 dropout/residual을 추가했다.
단순히 아이디어만 참고한 독립 구현이 아니다.</p>
<h3>동일 저장 정책 · 같은 프로세스 A/B</h3>
<p>양방향 C128/H256 · BF16 · B1 · dropout 25% · residual 포함. 단위 <b>ms</b>.
순서를 교대하며 CUDA Graph replay 8라운드 중앙값. Weight packing 포함, RNG 생성/optimizer 제외.</p>
<div class="table-wrap"><table><thead><tr><th>L</th><th>측정 범위</th><th>기존 Triton 학습</th><th>현재 Anthropic 파생형</th><th>시간 감소</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p><b>이 표는 기존 학습 대비 개선이다.</b> Anthropic 원본 inference를 이겼다는 결과는 아니다.
전체 학습 15% 개선은 아직 달성하지 않았다.</p>
<div class="table-wrap"><table><thead><tr><th>초기 커널 호출 비교 · µs</th><th>L384 Triton → 파생형</th><th>L768 Triton → 파생형</th></tr></thead><tbody>
<tr><td>입력 projection · packing 포함</td><td>253.0 → 202.8</td><td>975.7 → 756.0</td></tr>
<tr><td>F567</td><td>106.5 → 108.6 · 약간 느림</td><td>401.1 → 380.4</td></tr></tbody></table></div>
<p class="tiny muted">새 CUDA는 각 커널 초기 6개 후보를 비교했다. Triton cache miss는 24개 heuristic 후보로 튜닝했다.
전수 튜닝 결과가 아니다. 유효 config 공간은 front 58개, F567 H128 30개/H256 12개.
LN을 하지 않는 F567에서 LNSERIAL 축은 제거했다.</p>
<h3>검증 및 NCU</h3>
<ul><li><b>GPU 테스트 14개 통과:</b> 저장 목록 동일성, 단방향/양방향, 전체 gradient, zero mask/dropout/projection, 모듈 SGD/eval, 여러 forward.</li>
<li>fullgraph compile + CUDA Graph에서 weight/dropout 변경 검증. L384/768 출력은 기존 Triton과 이번 입력에서 비트 일치.</li>
<li>Memcheck / synccheck: 0 errors. Racecheck: 0 hazards.</li>
<li>선택한 H128/H256 cubin에 TMA load/store·HGMMA 존재. local memory / LDL / STL 0.</li>
<li>NCU L768/H256: front 754.8µs / 2.59TB/s, F567 396.6µs / 2.61TB/s. DRAM active 77.1% / 78.0%.</li></ul>
<p>NCU 수치만으로 roofline 도달을 단정하지 않는다. F567 L384와 backward 최적화 여지가 남아 있다.</p>
<h3>선택 방법과 범위</h3>
<pre style="overflow-x:auto"><code>BidirectionalTriangleMultiplication(
    128, implementation="anthropic",
    anthropic_row="training_saved", p_drop=0.25,
)
# 단방향 TriangleMultiplication도 같은 row
# global engine_backend="auto" (CUDA이므로 Triton-only 강제와 함께 사용하지 않음)</code></pre>
<p>현재 H100 / C128 / H128 또는 H256 / B1 / BF16 / L이 8의 배수.
전체 모델 수렴, 다른 폭·GPU·배치, engine 전체 autotune/cache 통합은 남아 있다.
원본 전체 forward와 재계산 backward 기준선은 <code>native_rebuilt</code>로 별도 보관한다.</p>
<p class="tiny muted">원본: <a href="https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e">Anthropic · f4f62fa · Apache-2.0</a>.
우리의 이전 inference 개발보다 뛰어난 원본 성과를 인정하고 계승한다. 추가 기여는 engine 통합과 학습 지원이다.</p>
<details><summary>이전 단계: 원본 전체 forward + 검증용 재계산 backward 기록</summary>'''+legacy+'''</details>
</section><!-- TRAINING_PROGRESS_END -->'''
