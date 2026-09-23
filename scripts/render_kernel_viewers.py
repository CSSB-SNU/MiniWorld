"""Generate local SVG viewers with phase selection, zoom and direct artifact links."""
from pathlib import Path
import html
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
STYLE='''*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#22384d;font:16px/1.65 system-ui,sans-serif}header{background:white;padding:24px 30px 20px;border-bottom:1px solid #d8e1eb}h1{margin:0 0 8px;font-size:28px}p{margin:6px 0}.sub{color:#51687e}.controls{position:sticky;top:0;display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:12px 26px;background:#edf3faf5;border-bottom:1px solid #ccd8e5;z-index:1}select,button{font:inherit;border:1px solid #aec1d1;padding:6px 10px;border-radius:6px;background:white;color:inherit}label{display:flex;align-items:center;gap:7px}a{color:#1962a0;text-underline-offset:3px}.viewport{overflow:auto;background:white;height:76vh;padding:16px}img{display:block;max-width:none;height:auto}aside{padding:10px 28px;background:#fff7e5;border-bottom:1px solid #ecd9b3}footer{padding:18px 30px}input{width:150px}button,select,input{cursor:pointer}:focus-visible{outline:3px solid #eeaa33;outline-offset:3px}@media(max-width:700px){header{padding:16px}h1{font-size:23px}.controls{padding:10px;gap:10px}}'''

def page(filename,title,intro,entries,links):
    if '--trimul-only' in sys.argv and filename != 'TRIMUL_STATUS.html':
        return
    opts=''.join('<option value="%s">%s</option>'%(i,html.escape(e['label'])) for i,e in enumerate(entries))
    navigation=' · '.join('<a href="%s">%s</a>'%(html.escape(url),html.escape(label)) for label,url in links)
    text='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+html.escape(title)+'''</title><style>'''+STYLE+'''</style></head><body><header><h1>'''+html.escape(title)+'''</h1><p>'''+html.escape(intro)+'''</p><p class="sub">실선 경계 = 명시한 커널의 융합 · 보라 = cuBLAS 호출 · 회색 점선 = compiler/복수 연산 · 노랑 = HBM tensor</p><p>'''+navigation+'''</p></header><nav class="controls" aria-label="그림 선택"><label>그림 <select id="diagram">'''+opts+'''</select></label><label>확대 <input id="zoom" type="range" min="35" max="200" value="100" step="5"><output id="amount">100%</output></label><button id="fit">화면에 맞춤</button><a id="raw" target="_blank" rel="noopener">SVG 원본 열기 ↗</a></nav><aside id="note" aria-live="polite"></aside><main class="viewport" id="viewport"><img id="figure" alt=""></main><footer>소스 함수 본문과 분기 조건을 확인한 2026-09-17 배선도입니다. 실측 범위와 결과는 각 그림의 연결 문서에서 확인할 수 있습니다.</footer><script>
const entries='''+json.dumps(entries,ensure_ascii=False)+''';
const select=document.getElementById('diagram'), fig=document.getElementById('figure'), zoom=document.getElementById('zoom'), area=document.getElementById('viewport');
function resize(){fig.style.width=Math.max(1,fig.naturalWidth)*Number(zoom.value)/100+'px';document.getElementById('amount').textContent=zoom.value+'%';}
function show(){const e=entries[Number(select.value)];document.getElementById('note').textContent=e.note;document.getElementById('raw').href=e.file;fig.alt=e.label;fig.src=e.file;area.scrollTop=0;area.scrollLeft=0;}
fig.addEventListener('load',resize);select.addEventListener('change',show);zoom.addEventListener('input',resize);
document.getElementById('fit').addEventListener('click',()=>{zoom.value=Math.max(35,Math.min(200,Math.floor((area.clientWidth-32)/fig.naturalWidth*20)*5));resize();});show();
</script></body></html>'''
    if filename == 'TRIMUL_STATUS.html':
        text=text.replace('실선 경계 = 명시한 커널의 융합 · 보라 = cuBLAS 호출 · 회색 점선 = compiler/복수 연산 · 노랑 = HBM tensor',
                          '왼쪽 파랑 = HBM 입력 · 가운데 초록/보라 = CUDA/cuBLAS · 오른쪽 주황 = HBM 출력 · 자주 = scratch 왕복')
        text=text.replace('소스 함수 본문과 분기 조건을 확인한 2026-09-17 배선도입니다. 실측 범위와 결과는 각 그림의 연결 문서에서 확인할 수 있습니다.',
                          '2026-09-21 소스·측정 기록 대조. 최근 전체 연결 실험 BF16 tri + 출력 LN 통계 저장 + shared B1 + split_xn_pc1 B7을 표시하며, 엄격 정확도 검증이 남아 엔진 기본값으로 채택한 상태는 아닙니다.')
        text=text.replace('min="35"','min="10"').replace('Math.max(35,','Math.max(10,')
    (ROOT/filename).write_text(text)

page('TRANSITION_STATUS.html','Transition · 연산 → 커널 배선',
     'Triton 정책과 H100 auto의 forward·backward 융합 범위, 중간 버퍼, shape별 분기를 비교합니다.',[
    dict(label='새 Triton residual 융합 · 선택 옵션',file='TRANSITION_RESIDUAL.svg',note='transition_residual_fusion=True: squeeze+residual forward와 LN+residual backward. 기본값 False이며 H100에서 정확도와 속도를 검증했습니다.'),
    dict(label='Forward · 커널 / 융합 경계',file='TRANSITION_FORWARD.svg',note='A: Triton 정책 · B: hand-CUDA b2b · C: CuTe split · D: Triton b2b 대체 경로. 일반 Transition에는 mask/dropout이 없습니다.'),
    dict(label='Backward · 재계산 / gradient / residual',file='TRANSITION_BACKWARD.svg',note='기본 C backward는 separate dA/dB의 GEMM 6회이며, A/B/D stacked 경로는 GEMM 4회입니다. 기본 residual gradient 합산은 LN 커널 밖이며, 새 융합 옵션은 별도 그림을 보세요.'),
    dict(label='Shape별 dispatch 개요',file='TRANSITION.svg',note='H100 BF16 n=4 대표 shape. implementation="triton"만으로 native 경로가 제외되지 않으며 engine_backend="triton" 정책과 구분합니다.')],
    [('Residual 융합 실측','docs/transition-fusion/RESIDUAL.md'),('소스 근거·분기 조건','docs/transition-fusion/README.md'),('TriMul 뷰어','TRIMUL_STATUS.html')])
page('TRIMUL_STATUS.html','TriMul · 현재 Forward / Backward 배선',
     '각 커널을 HBM 읽기 → 내부 계산 → HBM 쓰기로 표시합니다. 내부 계산 박스 왼쪽은 설명·수식, 오른쪽은 PyTorch 대응 코드입니다.',[
    *[dict(label='현재 개발 '+phase+' · L'+str(length),file='TRIMUL_'+phase.upper()+('_L768' if length==768 else '')+'.svg',note='BF16 tri + 출력 LN 통계 저장 + shared B1 + split_xn_pc1 B7 개발 후보. B1의 Phase A/B 두 행은 같은 CUDA 호출입니다. MB/kB는 버퍼 크기이며 실제 HBM 트래픽이 아닙니다. B1/B7의 global partial scratch 왕복과 입력 재읽기를 포함합니다.') for phase in ('Forward','Backward') for length in (384,768)],
    *[dict(label='이전 Triton/CuTe '+phase+' · 보관본',file='docs/trimul-fusion/archive-pre-anthropic-20260921/TRIMUL_'+phase.upper()+'.svg',note='Anthropic 도입 전 8c7d8b39 개발 버전의 역사 기록입니다. 현재 경로를 나타내지 않습니다.') for phase in ('Forward','Backward')]],
    [('현재 배선 설명','TRIMUL_STATUS.md'),('온라인 현황판','https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#current-wiring'),('이전 SVG 보관','docs/trimul-fusion/archive-pre-anthropic-20260921/README.md'),('Transition 뷰어','tmp_kernel/transition/TRANSITION_STATUS.html')])
