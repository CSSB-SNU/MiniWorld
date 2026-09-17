"""Render the installed TriMul inference wiring without importing CUDA packages."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / '.pixi/envs/cu128/lib/python3.10/site-packages/miniworld_engine'
OUT = ROOT / 'docs/trimul-fusion'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
W, H = 2508, 1960
svg = ET.Element('{%s}svg' % NS, width=str(W), height=str(H), viewBox='0 0 %s %s' % (W,H), role='img', **{'aria-labelledby':'title desc'})
defs=ET.SubElement(svg,'{%s}defs'%NS)
marker=ET.SubElement(defs,'{%s}marker'%NS,id='arr',viewBox='0 0 10 10',refX='9',refY='5',markerWidth='7',markerHeight='7',orient='auto-start-reverse')
ET.SubElement(marker,'{%s}path'%NS,d='M 0 0 L 10 5 L 0 10 z',fill='#687a90')

def e(tag, **attrs):
    return ET.SubElement(svg, '{%s}%s' % (NS,tag), {k.replace('_','-'):str(v) for k,v in attrs.items()})
ET.SubElement(svg,'{%s}title'%NS,id='title').text='TriMul 추론: 실제 설치본의 커널 배선과 융합 범위'
ET.SubElement(svg,'{%s}desc'%NS,id='desc').text='Triton과 H100 CuTe의 단방향 및 양방향 추론 비교. Triton 두 경로와 CuTe 단방향은 F4부터 F7까지 Triton 한 커널. CuTe 양방향은 출력 연산 분리. Triton 양방향은 packed_forward로 cat을 제거했으며 H100 CuTe 양방향에는 남아 있다.'
def rect(x,y,w,h,fill='#fff',stroke='#ccd5e1',sw=1,rx=12,dash=None):
    a=dict(x=x,y=y,width=w,height=h,fill=fill,stroke=stroke,stroke_width=sw,rx=rx)
    if dash:a['stroke_dasharray']=dash
    return e('rect',**a)
def text(x,y,s,size=18,color='#24344a',weight='400',anchor='start'):
    n=e('text',x=x,y=y,font_size=size,fill=color,font_weight=weight,text_anchor=anchor,font_family='Noto Sans CJK KR, Noto Sans KR, DejaVu Sans, sans-serif');n.text=s;return n
def lines(x,y,ss,size=17,step=26,color='#24344a'):
    for i,s in enumerate(ss):text(x,y+i*step,s,size,color)
def arrow(x1,y1,x2,y2,dash=False,color='#687a90'):
    return path('M %s %s L %s %s'%(x1,y1,x2,y2),dash,color)
def path(d,dash=False,color='#687a90'):
    a=dict(d=d,fill='none',stroke=color,stroke_width=2,marker_end='url(#arr)')
    if dash:a['stroke_dasharray']='6 5'
    return e('path',**a)
def box(x,y,w,h,title,body,kind='triton',small=16):
    colors={'triton':('#edf5ff','#2775c6'),'cute':('#eaf8f2','#198665'),'blas':('#f6effd','#8658b0'),'copy':('#fff1e9','#c7612b'),'plain':('#fff','#c1cedc')}
    fill,c=colors[kind];rect(x,y,w,h,fill,c,2)
    text(x+16,y+27,title,18,'#334e6b' if kind=='plain' else c,'700');lines(x+16,y+52,body,small,23)
def buf(x,y,w,label):
    rect(x,y,w,43,'#fff7df','#bd9642',1.4,20);text(x+w/2,y+27,label,16,'#785719','500','middle')
def small_box(x,y,w,h,label,kind='cute'):
    c,f={'cute':('#198665','#eaf8f2'),'blas':('#8658b0','#f6effd'),'triton':('#2775c6','#fff')}[kind]
    rect(x,y,w,h,f,c,2,8);lines(x+12,y+22,label,14,21)
rect(0,0,W,H,'#f5f7fb','none',0,0)
text(36,48,'TriMul 추론 · 커널 배선 / 융합 경계',34,'#172a43','700')
text(36,83,'현재 cu128 설치본 · 2026-09-17 · H100 / BF16 / B=1 / d_pair=h=128 / mask가 전달되는 기본 경로',19)
rect(36,106,W-72,76,'#e8eef8','none')
lines(54,134,['진입 조건: model.eval() + torch.no_grad()/inference_mode() → grad OFF, dropscale=None.',
              'eval()만으로는 이 경로가 보장되지 않음. grad ON 또는 dropout scale 존재 시 학습 경로로 분기.'],18,28)
xs=[36,654,1272,1890];cw=582
for i,x in enumerate(xs):
    bi=i%2==1;cute=i>=2;k=256 if bi else 128
    rect(x,202,cw,1390,'#fff','#d8e1ec',1,16)
    text(x+22,237,('H100 CuTe' if cute else 'Triton')+' · '+('양방향' if bi else '단방향'),24,'#172a43','700')
    text(x+22,266,'outgoing + incoming' if bi else 'outgoing 또는 incoming',17,'#60728a')
    buf(x+32,284,cw-64,'pair [M,128]    ·    M=L²')
    arrow(x+cw/2,327,x+cw/2,350)
    box(x+22,350,cw-44,76,'F1 · 입력 LayerNorm · Triton 1 kernel',['layer_norm_fwd_fused  →  x_n = LN(pair)'])
    arrow(x+cw/2,426,x+cw/2,439)
    buf(x+32,439,cw-64,'x_n [M,128] · HBM  /  gate에서도 재사용')
    arrow(x+cw/2,482,x+cw/2,501)
    if not cute:
        box(x+22,501,cw-44,143,'F2 · _bidir_front_kernel · 1 kernel',[
            '4 projections + sigmoid + multiply + mask',
            'left = (x_n @ W_L) × sigmoid(x_n @ W_Lg) × mask',
            'right = (x_n @ W_R) × sigmoid(x_n @ W_Rg) × mask',
            '채널 방향으로 직접 store · save_preact=False'],small=14)
    else:
        rect(x+22,501,cw-44,143,'#f9fcfb','#85aa9f',1.5,10,'5 4')
        text(x+36,524,'F2 · MaskedGatedSm90 · '+('4' if bi else '2')+'개의 별도 kernel',17,'#198665','700')
        labels=['left/out','right/out','left/in','right/in'] if bi else ['left','right']
        bw=122 if bi else 251
        for j,label in enumerate(labels):
            small_box(x+32+j*(bw+10),538,bw,92,[label,'proj + gate','sigmoid × mask','BDLL store'])
    arrow(x+cw/2,644,x+cw/2,658)
    buf(x+32,658,cw-64,('4 tensors: left/right × out/in [128,L,L] · HBM' if i==3 else 'left / right [%s,L,L] · HBM  /  preact 저장 없음'%k))
    arrow(x+cw/2,701,x+cw/2,718)
    if bi:
        small_box(x+22,718,263,76,['F3a · cuBLAS 호출','O_out = L_out @ R_outᵀ',('out=tri[:h] · 최종 slice' if i==1 else 'O_out [128,L,L] · HBM')],'blas')
        small_box(x+297,718,263,76,['F3b · cuBLAS 호출','O_in = L_inᵀ @ R_in',('out=tri[h:] · 최종 slice' if i==1 else 'O_in [128,L,L] · HBM')],'blas')
        arrow(x+153,794,x+153,808);arrow(x+429,794,x+429,808)
        if i==1:
            box(x+22,808,cw-44,69,'packed_forward · 최종 tri에 직접 기록',
                ['2 GEMM 호출 유지 · 중간 출력 / cat 복사 제거'],'plain',15)
        else:
            box(x+22,808,cw-44,69,'torch.cat · 두 출력 → tri',['별도 concat / copy 연산이 소스에 남아 있음'],'copy',15)
        arrow(x+cw/2,877,x+cw/2,890)
    else:
        box(x+22,718,cw-44,102,'F3 · cuBLAS 호출 1개',[
            'outgoing: tri = left @ rightᵀ',
            'incoming: tri = leftᵀ @ right'],'blas')
        arrow(x+cw/2,820,x+cw/2,890)
    buf(x+32,890,cw-64,'tri [%s,L,L] · HBM'%k)
    arrow(x+cw/2,933,x+cw/2,955)
    if i!=3:
        rect(x+22,955,cw-44,368,'#edf5ff','#2775c6',3)
        text(x+38,985,'F4 + F5 + F6 + F7 · Triton 1 kernel',21,'#1a61aa','700')
        text(x+38,1012,'trimul_back_triton → _back_kernel',16,'#1a61aa')
        box(x+38,1029,cw-76,69,'F4 · 출력 LayerNorm',['tri 채널 %s개 정규화 + affine → BF16 norm'%k],'plain',15)
        arrow(x+160,1098,x+160,1116)
        small_box(x+38,1116,244,75,['F5 · projection (tl.dot)','proj = norm @ W_p','FP32 accumulator'],'triton')
        small_box(x+298,1116,244,75,['F6 · gate (tl.dot)','g = sigmoid(x_n @ W_g)','FP32 accumulator'],'triton')
        path('M %s 460 L %s 460 L %s 1150 L %s 1150'%(x+cw-32,x+cw-9,x+cw-9,x+542),True)
        arrow(x+160,1191,x+160,1210);arrow(x+420,1191,x+420,1210)
        box(x+38,1210,cw-76,88,'F7 · multiply + residual → store',[
            'y = pair + proj × g   →   BF16 y',
            'norm / proj / glogit / gate의 별도 HBM buffer 없음'],'plain',14)
        path('M %s 305 L %s 305 L %s 1256 L %s 1256'%(x+32,x+10,x+10,x+38),True)
        text(x+40,1354,'실선 바깥 테두리 전체가 한 GPU 커널',16,'#1a61aa','700')
        text(x+40,1380,'내부 두 GEMM도 이 커널 안에서 실행',16,'#60728a')
        arrow(x+cw/2,1323,x+cw/2,1410)
        buf(x+32,1410,cw-64,'y [M,128] · HBM')
        lines(x+28,1490,(['현재도 F4~F7 융합 사용.','이번 학습 F567 변경과는 별도 커널.'] if not bi else ['F4~F7은 이미 융합되어 있음.','F3a/F3b는 최종 tri slice에 직접 기록.']),17,27)
    else:
        box(x+22,955,cw-44,70,'F4 · 출력 LN + layout store · Triton',['_ln_transpose_dbn_kernel  [256,M] → [M,256]'],small=14)
        buf(x+40,1040,cw-80,'out_normed [M,256] · HBM');arrow(x+cw/2,1025,x+cw/2,1040)
        arrow(x+cw/2,1083,x+cw/2,1096)
        box(x+22,1096,cw-44,65,'F5 · projection · cuBLAS',['proj = out_normed @ W_p'],'blas')
        buf(x+40,1174,cw-80,'proj [M,128] · HBM');arrow(x+cw/2,1161,x+cw/2,1174)
        box(x+22,1236,cw-44,65,'F6 · output gate GEMM · cuBLAS',['glogit = x_n @ W_g'],'blas')
        path('M %s 460 L %s 460 L %s 1268 L %s 1268'%(x+cw-32,x+cw-9,x+cw-9,x+cw-22),True)
        buf(x+40,1314,cw-80,'glogit [M,128] · HBM');arrow(x+cw/2,1301,x+cw/2,1314)
        arrow(x+cw/2,1357,x+cw/2,1374)
        box(x+22,1374,cw-44,90,'F7 · _gate_mul_infer_kernel · Triton',[
            'sigmoid(glogit) × proj + pair → y',
            'gate_elem_infer = F6 GEMM + F7 kernel'],small=15)
        path('M %s 1195 L %s 1195 L %s 1420 L %s 1420'%(x+40,x+16,x+16,x+22),True)
        path('M %s 305 L %s 305 L %s 1440 L %s 1440'%(x+32,x+7,x+7,x+22),True)
        arrow(x+cw/2,1464,x+cw/2,1477);buf(x+32,1477,cw-64,'y [M,128] · HBM')
        text(x+28,1558,'F4/F5/F6/F7 분리 · CuTe F567 미연결',17,'#b14e22','700')
# Scope and legend
rect(36,1615,W-72,112,'#fff','#d8e1ec')
text(54,1646,'mask=None일 때 H100 CuTe의 F2만 달라짐',20,'#172a43','700')
lines(54,1678,['기본 bdll_direct_wide: 방향마다 [gate | projection] wide GEMM × 2 (CuTe/quack) → wide HBM → _glu_wide_kernel × 2 (Triton).',
              '단방향은 2 GEMM + 2 GLU, 양방향은 4 GEMM + 4 GLU. 위 본도는 mask가 전달되어 MaskedGatedSm90가 선택되는 경우.'],18,27)
rect(36,1747,W-72,176,'#e8eef8','none')
text(54,1778,'범례 / 해석 범위',20,'#172a43','700')
lines(54,1810,['파랑 = Triton   ·   초록 = CuTe   ·   보라 = cuBLAS 호출   ·   노랑 = HBM tensor   ·   주황 = concat/copy',
              '실선 외곽 = 한 커널의 융합 범위. 점선 그룹 = 여러 별도 커널. 점선 화살표 = x_n / pair / proj 재사용; 병렬 실행을 뜻하지 않음.',
              '소스 배선 감사이며 런타임 프로파일은 아님. cuBLAS 호출 수 ≠ 내부 GPU launch 수. CuTe의 compile 시 cat 제거 여부는 미실측.',
              'weight packing·cast·mask 준비 등 보조 연산은 생략. 입력 LN은 μ/rstd buffer도 생성. dropout는 추론에서 비활성. B200 경로는 이 그림 범위 밖.'],17,28)
text(36,1946,'Source: installed miniworld_engine / cu128   ·   상세 근거와 SHA-256: docs/trimul-fusion/inference.md + inference-sources.json',15,'#60728a')
OUT.mkdir(parents=True,exist_ok=True)
ET.ElementTree(svg).write(str(ROOT/'TRIMUL_INFERENCE.svg'),encoding='utf-8',xml_declaration=True)
files=['kernels/trimul_inproj/triton/contract.py','modules/triangle_multiplication/module.py','modules/triangle_multiplication/bidirectional.py','modules/dispatch.py','kernels/trimul_inproj/triton/unidirectional.py','kernels/trimul_inproj/triton/bidirectional.py','kernels/trimul_inproj/triton/back.py','kernels/trimul_inproj/triton/gate_elem.py','kernels/tm1/cute/launch.py','kernels/trimul_inproj/cute/masked_front.py','kernels/layernorm/triton/transpose.py','kernels/layernorm/triton/main.py','kernels/trimul_inproj/triton/backward_fused.py']
(OUT/'inference-sources.json').write_text(json.dumps({'date':'2026-09-17','scope':'installed source wiring, not runtime profiling','package':str(PKG),'sha256':{f:hashlib.sha256((PKG/f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
print(ROOT/'TRIMUL_INFERENCE.svg')
