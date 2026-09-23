#!/usr/bin/env python3
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'runs/anthropic_ln_inference_20260919';A=ROOT/'runs/anthropic_adoption_20260919'
rs=json.loads((R/'results.json').read_text())
body=''
for key,label in [('original','원본 Anthropic 융합 · 설정 고정'),('split-anthropic-ln','입력 분리 · Anthropic LN'),('split-triton-ln','입력 분리 · Triton LN')]:
 body+='<tr><td>'+label+'</td>'+''.join('<td>{:.3f} ms</td>'.format(r['full'][key]['median_us']/1000) for r in rs)+'</tr>'
components=''
for k,label in [('ln-triton','입력 LN · Triton'),('ln-anthropic','입력 LN · 원본에서 추출한 CUDA'),('k1-original','K1 · 원본 융합'),('k1-split','K1 · LN 제거 / 재튜닝'),('k3-original','K3 · 원본 입력 LN 재계산'),('k3-split','K3 · x_n 공유 / 재튜닝')]:
 components+='<tr><td>'+label+'</td>'+''.join('<td>{:.2f}</td>'.format(r['components'][k]['median_us']) for r in rs)+'</tr>'
settings=''.join('<tr><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>'.format(r['N'],r['configs']['k1_split'],r['configs']['k3_split'],r['configs']['ln_triton'],r['configs']['ln_anthropic']) for r in rs)
section='''<!-- INFER_LN_BEGIN --><section class="panel" id="inference-ln"><span class="pill green">최신 · 추론 전용 재측정</span>
<h2>입력 LN 분리 vs 원본 Anthropic 융합</h2>
<p><b>원본 융합형은 재튜닝하지 않았다.</b> 원본 k1_body/k3_body와 C128/H256 배포 설정을 그대로 고정했다.
분리형의 변경된 K1·K3와 별도 LN만 재튜닝했다. 출력 LN은 모든 경로에서 K3 안에 융합한다.</p>
<div class="flow"><div><b>원본 융합</b>K1: 입력 LN + projection/gate → cuBLAS → K3: 입력 LN 재계산 + 출력 LN + projection/gate/residual</div>
<div><b>입력 분리</b>입력 LN 한 번 → x_n 공유 → K1 projection/gate → cuBLAS → K3 출력 LN + projection/gate/residual</div></div>
<p><b>backward용 저장·dropout 없음.</b> 통계, preactivation, projection, gate, normalized contraction을 저장하지 않는다.
분리형 x_n은 forward에 필요한 중간값이다. B1 C128/H256 BF16 · 양방향 packed cuBLAS · H100 node02 · mask/residual 포함.</p>
<p><b>결론:</b> L384는 원본 융합형이 중앙값 기준 약 1.6% 빠르고, L768은 1% 미만 차이로 측정 변동 범위 안이다. 입력 LN 분리의 뚜렷한 추론 이득은 확인되지 않았다.</p><h3>추론 forward 전체</h3><div class="table-wrap"><table><thead><tr><th>경로</th><th>L384</th><th>L768</th></tr></thead><tbody>'''+body+'''</tbody></table></div>
<p>18라운드 순서 순환·역전 × CUDA Graph 200회 중앙값. 각 경로 200회 초기 warmup. Weight는 미리 packing. Python 호출·weight packing 제외.
원본 K1/K3를 같은 양방향 contraction에 연결한 비교이며 upstream 단방향 API 전체 시간과는 다르다.</p>
<h3>커널별 측정 · µs</h3><div class="table-wrap"><table><thead><tr><th>커널</th><th>L384</th><th>L768</th></tr></thead><tbody>'''+components+'''</tbody></table></div>
<p>Anthropic LN은 배포된 독립 LN 제품이 아니라 원본 ln_fragment를 직접 include한 추론용 CUDA 추출판이다.
Triton도 기존 LN에서 mean/rstd 저장만 제거한 추론판이다. 둘 다 normalized 출력만 쓴다.</p>
<h3>탐색 범위와 선택값</h3><ul><li>분리 K1: 유효 타일/ring/chunk 66개, 상위 3개에서 register budget·schedule·warpgroup 시작 간격 추가 탐색.</li>
<li>분리 K3: 타일/ring/accumulator/register 분배/LN schedule 44개씩, residual 글로벌 로드/TMA 로드 두 구현 비교. 최종은 TMA로 shared stage를 재사용.</li>
<li>Triton LN: 등록 설정 800개 명시적 실행. 24개 heuristic 제한 없음.</li><li>Anthropic 파생 LN: vector/TMA load/TMA store/thread/SERIAL 12개.</li></ul>
<div class="table-wrap"><table><thead><tr><th>L</th><th>K1</th><th>K3</th><th>Triton LN</th><th>Anthropic LN</th></tr></thead><tbody>'''+settings+'''</tbody></table></div>
<p class="tiny muted">K1: BI,BJ,slots,Kchunk,schedule,consumer registers,start offset. K3: BI,BJ,slots,accumulators,24/240 split switch,LN serial.
Triton: rows,feature tile,warps,stages. CUDA LN: load type,threads,serial,bulk store.
각 후보 정확도 확인 후 상위 4개를 8라운드 재측정. 유한 탐색 결과이며 전역 최적 보장은 아니다.</p>
<p><a href="assets/inference-ln-results.json">전체 결과·round sample JSON</a> · <a href="assets/inference-ln-validation.json">정확도 기록</a></p>
<p>선택한 경로의 memcheck 0 errors, racecheck 0 hazards. 원본과 Anthropic LN 분리형 출력은 비트 일치하고, Triton LN 경로는 상대 L2 8e-5 수준. 실험 구현은 보존했고 production dispatch는 변경하지 않았다. 이전 저장 포함 학습 forward 실험은 아래 이력으로 남긴다.</p>
</section><!-- INFER_LN_END -->'''
for p in (ROOT/'ANTHROPIC_TRIMUL.html',A/'web/trimul.html',A/'site/dist/trimul.html'):
 s=p.read_text();s=re.sub(r'<!-- INFER_LN_BEGIN -->.*?<!-- INFER_LN_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+section,1)
 if 'href="#inference-ln"' not in s:s=s.replace('<nav>','<nav><a href="#inference-ln">추론 LN 재튜닝</a>',1)
 p.write_text(s)
for d in (A/'web/assets',A/'site/dist/assets'):
 for src,name in [('results.json','inference-ln-results.json'),('validation.json','inference-ln-validation.json')]:
  (d/name).write_text((R/src).read_text())
print('Updated inference LN dashboard')
