from pathlib import Path
import json,re,shutil
R=Path('/home/psk6950/MiniWorld');D=R/'runs/anthropic_b1b4_cuda_20260919';A=R/'runs/anthropic_adoption_20260919'
rows=json.loads((D/'overlap-transform-results.json').read_text())
t=''
for r in rows:
 v=r['times'];t+='<tr><td>'+str(r['L'])+'</td>'+''.join(f'<td>{v[k]["median_us"]:.2f}</td>' for k in ('0','2','3','baseline'))+'</tr>'
sec='''<!-- OVERLAP_BEGIN --><section id="b1b4-overlap"><h2>B1–B4 CUDA overlap 실험</h2><p>node02 H100 · BF16 C128/H256 · dropout 25% · Graph 80회 × 20 교대 라운드 · B1–B4만 측정</p><pre>현재 타일: raw → registers / MMA용 shared → WGMMA
다음 타일:                              ↳ TMA prefetch
변형 3: 다음 타일 gate/dropout 미분도 현재 WGMMA 실행 중 수행</pre><table><thead><tr><th>L</th><th>기존 CUDA µs</th><th>조기 prefetch µs</th><th>미분까지 overlap µs</th><th>Triton/cuBLAS µs</th></tr></thead><tbody>'''+t+'''</tbody></table><p>조기 prefetch는 L768에서 CUDA 대비 약 4.1% 시간 감소. L384는 사실상 동률. 미분까지 겹친 버전은 조기 prefetch보다 느렸다. 추가 shared-memory 없이 raw 버퍼를 재사용하며, 전용 producer warpgroup/multistage 구현은 아직 아니다. 기존 Triton/cuBLAS보다 여전히 느리며 production 경로 변경 없음. 전체 backward는 이번에 재측정하지 않음.</p><p>L64/72/384/768 출력 및 20회 반복/counter reset 통과. <a href="assets/b1b4-overlap-results.json">원자료</a></p></section><!-- OVERLAP_END -->'''
for p in (R/'ANTHROPIC_TRIMUL.html',A/'web/trimul.html',A/'site/dist/trimul.html'):
 s=p.read_text();s=re.sub(r'<!-- OVERLAP_BEGIN -->.*?<!-- OVERLAP_END -->','',s,flags=re.S);p.write_text(s.replace('<main>','<main>'+sec,1))
for p in (A/'web/assets',A/'site/dist/assets'):shutil.copyfile(D/'overlap-transform-results.json',p/'b1b4-overlap-results.json')
