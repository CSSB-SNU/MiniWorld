from pathlib import Path
import json,re,shutil,statistics
R=Path('/home/psk6950/MiniWorld');D=R/'runs/anthropic_ln_equal_saves_20260919';A=R/'runs/anthropic_adoption_20260919'
rs=json.loads((D/'results.json').read_text());assert len(rs)==2
rows=''
for r in rs:
 for k,label in [('front','입력 LN + K1 (저장 포함)'),('fwd','전체 forward'),('bwd','backward 단독'),('train','forward + backward 직접 측정')]:
  a=r['times']['fused-'+k]['median_us'];b=r['times']['split-'+k]['median_us']
  rows+=f'<tr><td>{r["N"]}</td><td>{label}</td><td>{a:.2f}</td><td>{b:.2f}</td><td>{(b/a-1)*100:+.2f}%</td></tr>'
body='<div class="table-wrap"><table><thead><tr><th>L</th><th>구간</th><th>융합 (µs)</th><th>분리 (µs)</th><th>분리 시간 증감</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
sec='''<!-- EQUAL_SAVES_BEGIN --><section class="panel" id="equal-saves"><span class="pill green">최신 · 학습 저장 정책 통일</span><h2>입력 LN 융합 vs 분리 — 같은 값을 저장하면?</h2><p>두 경로 모두 <b>x_n, 입력 LN 평균/역표준편차, front preactivation, a/b, 출력 LN 값/통계, projection/gate</b>를 같은 형식으로 저장한다. 출력 LN은 융합하고, K3는 두 경로 모두 저장된 x_n을 읽는다. 동일한 Triton/cuBLAS backward를 사용한다.</p><div class="flow"><div><b>융합형</b>원본 x → K1 [입력 LN + projection/gate]<br>x_n + mean/rstd + preactivation + a/b 저장</div><div><b>분리형</b>원본 x → LN [x_n + mean/rstd 저장]<br>x_n → K1 [projection/gate + preactivation + a/b 저장]</div></div>'''+body+'''<p><b>결론:</b> L384에서는 융합형의 전체 학습 시간이 약 1% 짧고, L768에서는 약 0.1% 차이로 사실상 같다. 동일한 값을 저장할 때 분리형이 더 빠르다는 근거는 없다.</p><p>분리 시간 증감은 (분리/융합 − 1)이며 음수일 때 분리형이 빠르다. 전체 forward+backward는 별도 캡처로 직접 측정한 값이라 단독 구간 시간의 합과 다를 수 있다. 작은 차이는 JSON의 라운드별 변동과 함께 해석해야 한다.</p><p>H100 node02 · B1 C128/H256 BF16 · pair mask · row dropout 25% · residual 포함 · 고정 공통 dropout mask, RNG 제외 · weight prepacked · optimizer 제외. CUDA Graph 80회 × 20 교대 라운드. CPU/autograd dispatch 제외 커널 실행 시간.</p><p><b>튜닝:</b> 저장 코드가 추가된 융합/분리 K1 각각 shape별 58개 설정. 타일·weight ring·K chunk·accumulator schedule 탐색, 상위 4개 재측정. standalone LN은 4개 설정. 기존 register/start-offset 기본값 유지. 변경 없는 공통 backward는 기존 cache와 miss 시 동일한 3개 후보 fallback 사용으로, 전체 backward를 새로 exhaustive tuning한 결과는 아니다.</p><p><b>정확성:</b> 두 경로 forward와 저장값 비트 일치, gradient 비교 및 기존 엔진 reference 통과. 선택 설정에 대해 L64/72 검증, memcheck/racecheck 통과. production 배선은 변경하지 않았다.</p><p><a href="assets/equal-saves-results.json">측정·선택 설정·gradient 원자료</a> · <a href="assets/equal-saves-validation.json">최종 검증</a> · <a href="assets/equal-saves-README.md">방법·재현</a></p></section><!-- EQUAL_SAVES_END -->'''
for p in (R/'ANTHROPIC_TRIMUL.html',A/'web/trimul.html',A/'site/dist/trimul.html'):
 s=p.read_text();s=re.sub(r'<!-- EQUAL_SAVES_BEGIN -->.*?<!-- EQUAL_SAVES_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+sec,1)
 if 'href="#equal-saves"' not in s:s=s.replace('<nav>','<nav><a href="#equal-saves">동일 저장 학습 비교</a>',1)
 p.write_text(s)
for d in (A/'web/assets',A/'site/dist/assets'):
 for f in ('results.json','validation.json','README.md'):shutil.copyfile(D/f,d/('equal-saves-'+f))
print('Updated equal-save comparison')
