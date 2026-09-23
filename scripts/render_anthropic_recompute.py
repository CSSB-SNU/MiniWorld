from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/anthropic_ln_inference_20260919';A=ROOT/'runs/anthropic_adoption_20260919'
rs=json.loads((R/'recompute-results.json').read_text());assert len(rs)==2
labels=[('original','원본 융합'),('anthropic-share','분리 · Anthropic LN · x_n 공유'),('anthropic-recompute','분리 · Anthropic LN · K3 재계산'),('triton-share','분리 · Triton LN · x_n 공유'),('triton-recompute','분리 · Triton LN · K3 재계산')]
body=''.join('<tr><td>'+label+'</td>'+''.join('<td>{:.3f} ms</td>'.format(r['times'][k]['median_us']/1000) for r in rs)+'</tr>' for k,label in labels)
sec='''<!-- RECOMPUTE_BEGIN --><section class="panel" id="k3-recompute"><span class="pill green">최신 추가 실험</span><h2>분리형에서 K3만 입력 LN을 재계산하면?</h2>
<p><b>별도 입력 LN과 재튜닝한 K1은 고정.</b> K3만 원본 x를 받아 입력 LN을 다시 계산하는 원본 Anthropic 커널로 교체했다.
원본 융합형·x_n 공유형·K3 재계산형을 같은 프로세스에서 비교했다. 두 종류의 별도 LN 모두 측정했다.</p>
<div class="flow"><div><b>공유형</b>입력 LN → x_n → K1<br>K3: x_n + 원본 residual 별도 TMA</div><div><b>재계산형</b>입력 LN → x_n → K1<br>K3: 원본 x → 입력 LN 재계산 + residual</div></div>
<div class="table-wrap"><table><thead><tr><th>경로</th><th>L384</th><th>L768</th></tr></thead><tbody>'''+body+'''</tbody></table></div>
<p><b>결론:</b> 두 LN 구현 모두 재계산형의 중앙값이 공유형보다 소폭 느렸다. 이번 실험에서 K3 재계산의 이득은 확인되지 않았다. L768의 변동이 있으므로 정확한 작은 비율을 확정적인 성능 차이로 취급하지 않는다.</p><p>H100 node02 · B1 C128/H256 BF16 · 양방향 contraction · mask/residual 포함 · dropout/학습 저장 없음 · weight prepacked.
CUDA Graph 200회 × 20라운드. 실행 순서를 순환·역전. 원본과 K3 재계산형은 원본 설정 고정, 추가 재튜닝 없음.</p>
<p>Anthropic LN 두 경로는 원본과 출력 비트 일치. Triton LN 두 경로도 상대 L2 0.005 미만 검증 통과.
작은 중앙값 차이를 단독으로 속도 우위로 해석하지 말 것. 라운드별 변동 및 두 경로 차이도 JSON에 기록했다.</p>
<p><a href="assets/k3-recompute-results.json">전체 측정·paired 차이 JSON</a>. 로컬 재현: runs/anthropic_ln_inference_20260919/bench_recompute.py</p></section><!-- RECOMPUTE_END -->'''
for p in (ROOT/'ANTHROPIC_TRIMUL.html',A/'web/trimul.html',A/'site/dist/trimul.html'):
 s=p.read_text();s=re.sub(r'<!-- RECOMPUTE_BEGIN -->.*?<!-- RECOMPUTE_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+sec,1)
 if 'href="#k3-recompute"' not in s:s=s.replace('<nav>','<nav><a href="#k3-recompute">K3 재계산 비교</a>',1)
 p.write_text(s)
for d in (A/'web/assets',A/'site/dist/assets'):(d/'k3-recompute-results.json').write_text((R/'recompute-results.json').read_text())
