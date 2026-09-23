"""Source-backed SVG of native Anthropic TriMul and the saved-policy derivative."""
from pathlib import Path
from html import escape
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
W,H=1600,2070
parts=[f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">Anthropic TriMul: 원본 K1/K3 내부와 기존 저장 정책을 유지한 학습 파생형</title>
<desc id="desc">H100 C128 H256의 128-token CTA 예시. HBM에서 TMA로 shared memory에 입력과 weights를 옮기고, consumer warpgroups가 register LayerNorm과 WGMMA를 수행한다. K1은 gated a/b를, K3는 output projection과 gate를 융합한다. 학습판은 입력과 출력 LN을 별도 Triton 커널로 유지하고 preactivation, projection, gate 저장을 추가하며 기존 backward를 재사용한다.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#405b70"/></marker></defs>
<style>text{{font-family:'Noto Sans CJK KR','Noto Sans KR',Arial,sans-serif;fill:#183344}}.muted{{fill:#526b7c}}.title{{font-size:34px;font-weight:750}}.h2{{font-size:24px;font-weight:700}}.h3{{font-size:20px;font-weight:700}}.body{{font-size:17px}}.small{{font-size:15px}}.mono{{font-family:Consolas,monospace;font-size:15px}}.edge{{fill:none;stroke:#405b70;stroke-width:2.5;marker-end:url(#arrow)}}.dash{{stroke-dasharray:6 5}}</style>
<rect width="1600" height="2070" fill="#f4f7fa"/>''']

def text(x,y,s,cls='body',color=None):
    extra=f' style="fill:{color}"' if color else ''
    parts.append(f'<text x="{x}" y="{y}" class="{cls}"{extra}>{escape(s)}</text>')
def rect(x,y,w,h,fill='#fff',stroke='#cfdae3',rx=12):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
def box(x,y,w,h,title,lines=(),fill='#fff',stroke='#cfdae3',cls='body'):
    rect(x,y,w,h,fill,stroke)
    text(x+16,y+29,title,'h3')
    for i,line in enumerate(lines):text(x+16,y+57+24*i,line,cls)
def arrow(x1,y1,x2,y2,label=None):
    parts.append(f'<path class="edge" d="M{x1},{y1} L{x2},{y2}"/>')
    if label:text((x1+x2)/2+10,(y1+y2)/2-5,label,'small')

text(40,55,'Anthropic TriMul 커널은 어떻게 구성됐나','title')
text(40,88,'원본의 내부 파이프라인과, 기존 융합·저장 정책을 유지한 학습 파생형을 구분해서 보기','body')
text(40,116,'H100 · C128 / H256 · CTA tile 2×64 예시 · 원본 revision f4f62fa · 2026-09-19','small')
for x,fill,label in [(800,'#efe9ff','HBM'),(965,'#e6f1ff','shared memory'),(1190,'#fff2d6','register'),(1390,'#e4f5ed','파생 CUDA')]:
    rect(x,101,17,17,fill,fill,3);text(x+25,115,label,'small')

text(40,163,'1. 원본의 융합 경계','h2')
box(40,185,560,165,'K1 · 입력 LN + gated dual projection',[
    'x = LN_in(z)',
    'a = sigmoid(x W_agᵀ) × (x W_apᵀ) × mask',
    'b = sigmoid(x W_bgᵀ) × (x W_bpᵀ) × mask',
    'a/b를 channel-major BF16 plane으로 출력'],fill='#fff')
box(640,185,290,165,'cuBLAS contraction',[
    'outgoing: A × Bᵀ',
    'incoming: Aᵀ × B',
    '원본 단방향: GEMM 1회',
    '양방향 조립: GEMM 2회'],fill='#f2ecff',cls='small')
box(970,185,590,165,'K3 · 출력 LN + projection + output gate',[
    'p = LN_out(X) W_oᵀ',
    'g = sigmoid(LN_in(z) W_ogᵀ)',
    'update = p × g   /   residual 옵션 지원',
    '원본은 x를 저장하지 않고 여기서 입력 LN을 다시 계산'],fill='#fff')
arrow(602,263,634,263);arrow(932,263,964,263)
text(40,382,'원본의 gate는 FP32 GEMM 누산값에서 계산한 뒤 BF16로 반올림한다. 원본에는 학습 backward가 없다.','small')

text(40,436,'2. CTA 내부: producer가 데이터를 공급하고, consumer가 연산한다','h2')
text(40,465,'이 예시에서 두 커널 모두 384 threads = producer WG0 128개 + consumer WG1/WG2 각 128개','small')
for x in (40,820):rect(x,485,740,875,'#ffffff','#bacbd7',16)
text(62,520,'K1 원본 내부','h2');text(842,520,'K3 원본 내부','h2')
# Original K1
box(60,545,278,100,'HBM · 입력 z', ['BF16 pair + mask','LN gamma/beta'],fill='#efe9ff',cls='small')
box(368,545,392,100,'HBM · projection weights', ['gate 32채널 | projection 32채널','left/right blocks를 순서대로 읽음'],fill='#efe9ff',cls='small')
arrow(198,647,198,677,'TMA');arrow(566,647,566,677,'TMA')
box(60,682,700,79,'WG0 · producer', ['warp 0: 입력 tile 공급    /    warp 1: weight ring 공급'],fill='#edf2f6')
arrow(198,763,198,795);arrow(566,763,566,795)
box(60,799,278,100,'Shared · 입력 staging', ['swizzled [token, C] tile','consumer가 읽으면 다음 tile 적재'],fill='#e6f1ff',cls='small')
box(368,799,392,100,'Shared · weight ring', ['mbarrier full/empty로 slot 소유권 교대','TMA load와 WGMMA 실행을 겹침'],fill='#e6f1ff',cls='small')
arrow(198,901,198,944,'ldmatrix');arrow(566,901,566,944,'WGMMA B operand')
box(60,950,700,155,'WG1 / WG2 · 각 64개 token 처리',[
    '입력 fragment를 register로 읽고 LN_in 수행',
    'WGMMA RS: A = normalized registers / B = shared weights',
    'm64n64: gate 32 + projection 32 → FP32 accumulators',
    'block b+1 MMA 실행 중 block b의 gate·곱·저장 처리'],fill='#fff2d6',cls='small')
arrow(410,1107,410,1143)
box(60,1148,700,83,'Shared · 출력 transpose staging',[
    'sigmoid(gate) × projection × mask → BF16 → stmatrix'],fill='#e6f1ff')
arrow(410,1233,410,1270,'TMA store')
box(60,1275,700,64,'HBM · a/b [2H, N, N]',[],fill='#efe9ff')
# Original K3
box(840,545,310,100,'HBM · X + 원래 z', ['X: contraction 결과','z: output gate의 LN 입력'],fill='#efe9ff',cls='small')
box(1180,545,360,100,'HBM · W_o / W_og', ['output projection / gate weights','LN gamma/beta'],fill='#efe9ff',cls='small')
arrow(997,647,997,677,'TMA');arrow(1357,647,1357,677,'TMA')
box(840,682,700,79,'WG0 · producer', ['warp 0: X/z tile 공급    /    warp 1: projection/gate weight 공급'],fill='#edf2f6',cls='small')
arrow(997,763,997,795);arrow(1357,763,1357,795)
box(840,799,310,100,'Shared · X / z staging', ['별도 barrier로 각 입력 관리','z는 residual에도 재사용 가능'],fill='#e6f1ff',cls='small')
box(1180,799,360,100,'Shared · weight ring', ['projection → gate 순서로 공급','slot 재사용은 소비 완료 후'],fill='#e6f1ff',cls='small')
arrow(997,901,997,944,'ldmatrix');arrow(1357,901,1357,944,'WGMMA B operand')
box(840,950,700,155,'WG1 / WG2 · register LN + 두 GEMM',[
    'X → LN_out register operand / z → LN_in register operand',
    'WGMMA m64n32: projection / output gate 각각 계산',
    'FP32 accumulator → sigmoid(gate) × projection',
    '다음 output block MMA와 현재 block epilogue를 겹침'],fill='#fff2d6',cls='small')
arrow(1190,1107,1190,1143)
box(840,1148,700,83,'Shared · 출력 staging',[
    'update BF16 반올림 → residual 옵션 → stmatrix'],fill='#e6f1ff')
arrow(1190,1233,1190,1270,'TMA store')
box(840,1275,700,64,'HBM · output [N, N, C]',[],fill='#efe9ff')
text(40,1390,'mbarrier = producer/consumer 준비·소비 완료 통지. WGMMA wait = 비동기 MMA 완료 대기. TMA store wait = staging 재사용 보호.','small')
text(40,1415,'TMA store 그림은 H256 / BJ64 설정 기준이다. 일부 다른 원본 설정은 vector global store를 사용한다.','small')

text(40,1463,'3. 현재 학습판: 위 내부 구조를 차용하되, 기존 융합·저장 정책을 유지','h2')
xs=[40,350,660,970,1280]
box(xs[0],1490,280,171,'F1 · 기존 입력 LN', ['Triton 유지','저장: x_n','저장: mean / rstd','K1과 output gate가 공유'],fill='#edf2f6',cls='small')
box(xs[1],1490,280,171,'F2 · K1 파생 CUDA', ['K1 내부 LN 제거','TMA / WGMMA 구조 계승','저장: a/b + preact','preact도 TMA 직접 저장'],fill='#e4f5ed',stroke='#79ad95',cls='small')
box(xs[2],1490,280,171,'F3 · 기존 contraction', ['cuBLAS 유지','outgoing + incoming','저장: X','최종 buffer에 직접 출력'],fill='#f2ecff',cls='small')
box(xs[3],1490,280,171,'F4 · 기존 출력 LN', ['Triton 유지','저장: normalized X','저장: mean / rstd','F567과 별도 커널'],fill='#edf2f6',cls='small')
box(xs[4],1490,280,171,'F567 · K3 파생 CUDA', ['K3 내부 LN들 제거','projection + gate','+ dropout + residual','저장: projection / gate'],fill='#e4f5ed',stroke='#79ad95',cls='small')
for x in xs[:-1]:arrow(x+282,1570,x+304,1570)
for x in xs:arrow(x+140,1663,x+140,1703)
box(40,1708,1520,87,'Backward · 기존 Triton / cuBLAS 그대로',[
    '기존 saved tensor shape·dtype·순서 유지. 위 저장값을 재사용하며, 검증용 경로의 추가 GEMM 6개 재계산은 하지 않는다.'],fill='#edf2f6')
box(40,1824,744,118,'원본에서 계승한 것',[
    '실제 k1_body / k3_body 파생 코드 · TMA · WGMMA · weight ring',
    'producer/consumer 분리 · MMA/epilogue overlap · shared staging'],fill='#fff',cls='small')
box(814,1824,746,118,'학습을 위해 바꾼 것',[
    '별도 LN 결과 사용 · 기존 preact/projection/gate 저장 · dropout/residual',
    '기존 BF16 반올림 위치 유지 · 기존 backward 연결'],fill='#e4f5ed',cls='small')
text(40,1975,'선택한 H128/H256 기본 cubin: TMA load/store + HGMMA 확인, local memory/LDL/STL 0. 모든 후보가 spill-free라는 뜻은 아니다.','small')
text(40,2003,'출처: anthropics/uplifting-biomolecular-modeling · native/pkg/v5/csrc/tmn_kernels.cuh · Apache-2.0','small')
text(40,2029,'파생형: anthropic_saved_front.cu / anthropic_saved_output.cu · 구현과 저장 목록을 대조한 설명도 · 성능 우위의 증명도는 아님','small')
parts.append('</svg>')
out=ROOT/'ANTHROPIC_TRIMUL_KERNELS.svg';out.write_text('\n'.join(parts))
ET.parse(out)
for base in [ROOT/'runs/anthropic_adoption_20260919/site/dist',ROOT/'runs/anthropic_adoption_20260919/web']:
    (base/'assets').mkdir(exist_ok=True)
    shutil.copy2(out,base/'assets/anthropic-trimul-kernels.svg')
print(out)
