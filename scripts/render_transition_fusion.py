"""Source-audited Transition wiring; stdlib SVG generation, no GPU imports."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'runs/trimul_sm90_parity_20260917/engine'
PKG = ENGINE / 'src/miniworld_engine'
RECORD = ENGINE / 'docs/records/transition-segmented-small-20260918'
DOC = ROOT / 'docs/transition-fusion'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
COLORS = {'triton': ('#2674bd', '#edf5ff'), 'cuda': ('#b45a23', '#fff1e6'),
          'cute': ('#168266', '#eaf8f1'), 'blas': ('#8052ad', '#f5effc'),
          'compiler': ('#697887', '#f1f4f7'), 'plain': ('#b8c7d5', '#ffffff')}

class Diagram:
    def __init__(self, width, height, title, subtitle):
        self.w, self.h = width, height
        self.root = ET.Element('{%s}svg' % NS, width=str(width), height=str(height),
                               viewBox=f'0 0 {width} {height}', role='img', **{'aria-labelledby':'title desc'})
        ET.SubElement(self.root,'{%s}title'%NS,id='title').text=title
        ET.SubElement(self.root,'{%s}desc'%NS,id='desc').text=subtitle
        defs=ET.SubElement(self.root,'{%s}defs'%NS)
        m=ET.SubElement(defs,'{%s}marker'%NS,id='arrow',viewBox='0 0 10 10',refX='9',refY='5',markerWidth='6',markerHeight='6',orient='auto-start-reverse')
        ET.SubElement(m,'{%s}path'%NS,d='M 0 0 L 10 5 L 0 10 z',fill='#6b8192')
        self.rect(0,0,width,height,'#f4f7fb','none')
        self.text(36,49,title,32,weight='700')
        self.text(36,82,subtitle,18,color='#4b6479')
    def el(self,tag,**attrs):
        return ET.SubElement(self.root,'{%s}%s'%(NS,tag),{k.replace('_','-'):str(v) for k,v in attrs.items()})
    def rect(self,x,y,w,h,fill='#fff',stroke='#cbd7e2',dash=False,rx=10,sw=1):
        args=dict(x=x,y=y,width=w,height=h,fill=fill,stroke=stroke,stroke_width=sw,rx=rx)
        if dash:args['stroke_dasharray']='7 5'
        return self.el('rect',**args)
    def text(self,x,y,s,size=18,color='#243a4c',weight='400',anchor='start'):
        e=self.el('text',x=x,y=y,font_size=size,fill=color,font_weight=weight,text_anchor=anchor,
                  font_family='Noto Sans CJK KR, Noto Sans KR, DejaVu Sans, sans-serif')
        e.text=s
        return e
    def lines(self,x,y,items,size=18,step=27,color='#243a4c'):
        for i,line in enumerate(items):self.text(x,y+i*step,line,size,color)
    def arrow(self,x1,y1,x2,y2,dashed=False):
        args=dict(d=f'M {x1} {y1} L {x2} {y2}',fill='none',stroke='#6b8192',stroke_width=2,marker_end='url(#arrow)')
        if dashed:args['stroke_dasharray']='6 5'
        return self.el('path',**args)
    def panel(self,x,y,w,h,title,sub):
        self.rect(x,y,w,h)
        self.text(x+18,y+34,title,23,weight='700')
        self.lines(x+18,y+64,sub,16,24,color='#516a80')
    def kernel(self,x,y,w,title,ops,kind='triton',note=None):
        titles=[title] if isinstance(title,str) else title
        ops=[[op] if isinstance(op,str) else op for op in ops]
        head=20+len(titles)*25
        height=head+sum(18+len(op)*24 for op in ops)+max(0,len(ops)-1)*18+14
        if note:height+=len(note)*23+10
        c,b=COLORS[kind]
        self.rect(x,y,w,height,b,c,dash=kind=='compiler',sw=2)
        for i,t in enumerate(titles):self.text(x+14,y+28+i*25,t,17,c,'700')
        yy=y+head
        for i,op in enumerate(ops):
            hh=18+24*len(op)
            self.rect(x+14,yy,w-28,hh,'#fff',c,rx=5)
            self.lines(x+25,yy+27,op,17,24)
            yy+=hh
            if i<len(ops)-1:self.arrow(x+w/2,yy,x+w/2,yy+18);yy+=18
        if note:self.lines(x+17,yy+28,note,15,23,color=c)
        return y+height
    def buffer(self,x,y,w,lines):
        if isinstance(lines,str):lines=[lines]
        h=20+len(lines)*25
        self.rect(x,y,w,h,'#fff7df','#bf9848',rx=20,sw=1.4)
        self.lines(x+14,y+29,lines,17,25,color='#755b24')
        return y+h
    def next_buffer(self,x,y,w,lines):
        self.arrow(x+w/2,y,x+w/2,y+24)
        return self.buffer(x,y+24,w,lines)
    def next_kernel(self,x,y,w,title,ops,kind='triton',note=None):
        self.arrow(x+w/2,y,x+w/2,y+24)
        return self.kernel(x,y+24,w,title,ops,kind,note)
    def footer(self,y,extra=()):
        self.rect(36,y,self.w-72,self.h-y-26,'#e7eef7','none')
        self.lines(54,y+30,[
            '실선 외곽 = 명시한 커널 내부의 융합. 보라색 = cuBLAS 호출 경계. 회색 점선 = PyTorch / compiler 연산.',
            '노랑 = HBM tensor. 화살표는 의존 관계이며 동시 실행을 뜻하지 않음. cuBLAS 호출 수와 GPU launch 수는 다를 수 있음.',
            'Transition(x)에는 mask 인자·dropout이 없음. 조건부 AdaLN Transition은 별도 모듈이며 이 그림 범위 밖.',
            *extra],17,27)
    def save(self,name):
        ET.ElementTree(self.root).write(str(ROOT/name),encoding='utf-8',xml_declaration=True)


STAMP = '2026-09-18 개발 checkout · H100 / BF16 / n=4 · residual fusion 기본 ON · 설치본/실행 중 학습과 구분'


def forward():
    d=Diagram(2460,1770,'Transition Forward · 현재 residual 융합 배선',STAMP)
    xs=[36,852,1668];w=756
    heads=[('A · Triton b2b / split fallback', ['SM90 BF16 n4 D128/256 M≥16384 → b2b','xn을 저장하고 backward에서 재사용']),
           ('B · H100 hand-CUDA b2b', ['auto D128/256 · M%128=0 · CUDA b2b ON','큰 학습 입력: xn 저장 · 추론: xn 저장 생략']),
           ('C · H100 CuTe expand + squeeze', ['auto D384/512/768 · M≥16384 · M%128=0','explicit cute는 지원 폭 128/256/384/512/768'])]
    for x,(title,sub) in zip(xs,heads):
        d.panel(x,112,w,1380,title,sub)
        y=d.buffer(x+20,229,w-40,'x [M,D] · residual 입력 · N=4D')
        xx=x+20;ww=w-40
        if x==xs[0]:
            y=d.next_kernel(xx,y,ww,'F1 · LayerNorm · Triton', ['xn=LN(x) · γ/β 연산 FP32 유지', 'mean / rstd → backward 재사용'])
            y=d.next_buffer(xx,y,ww,'xn [M,D] · 학습에서는 backward까지 저장')
            y=d.next_kernel(xx,y,ww,['F2+F3+F4+F5 · segmented Triton b2b', 'segmented_residual._kernel → _segmented'], [
                'a/b projection → h=SiLU(a)⊙b · HBM h 없음',
                '같은 CTA 안에서 출력 누산기를 BO 채널씩 분할',
                'squeeze → BF16(acc) + x → y'], note=[
                '추론·학습 동일 forward / 동일 캐시 키',
                'CSV autotune: BM / BN / BK / BO / warps / stages'])
            y=d.next_buffer(xx,y,ww,'학습: xn → 기존 stacked Triton backward')
            y=d.next_kernel(xx,y,ww,'Fallback: 각각 별도 Triton 커널', [
                'D384/512 · 작은 M · 다른 GPU · force_split=True',
                '① LN → xn [M,D] 저장',
                '② expand+SwiGLU → h [M,4D] 저장',
                '③ squeeze+residual → y'], note=[
                '선택된 large-D streamed-K는 split보다 느림',
                '최신 full-K large-D는 재튜닝 필요'])
        else:
            y=d.next_kernel(xx,y,ww,'F1a · _stats_kernel · Triton',['rstd=rsqrt(var+ε), c1=mean×rstd'])
            y=d.next_buffer(xx,y,ww,'x + rstd / c1 [M] · backward까지 저장')
            if x==xs[1]:
                y=d.next_kernel(xx,y,ww,['F1b+F2+F3+F4+F5 · hand-CUDA', 'transition_b2b_rs_wgmma_kernel'],[
                    'xn=(x×rstd−c1)×γ+β · 학습에서는 xn도 HBM store',
                    'a/b projection · TMA / WGMMA','h=SiLU(a)⊙b · tile 내부',
                    'squeeze + residual → y'], 'cuda',note=['b2b CUDA 본체 수정 · residual epilogue 유지',
                    '학습: 같은 커널에서 xn 벡터 store → backward 재사용'])
            else:
                stats_y=y
                y=d.kernel(xx,y+24,ww,'Weight 준비 · _fold_kernel · Triton', ['γ / β와 Wa / Wb → folded B, S, B2'],note=['stats와 독립적인 weight 준비 · 모듈 호출마다 수행'])
                expand_y=y+24
                d.el('path',d=f'M {xx+ww-3} {stats_y} L {xx+ww+8} {stats_y} L {xx+ww+8} {expand_y+25} L {xx+ww} {expand_y+25}',fill='none',stroke='#6b8192',stroke_width=2,marker_end='url(#arrow)')
                y=d.next_kernel(xx,y,ww,'F1b+F2+F3 · GemmLnGatedSm90 · CuTe', ['q=x@Bᵀ · TMA / WGMMA','a/b=q×rstd−c1×S+B2 → h=SiLU(a)⊙b'], 'cute')
                y=d.next_buffer(xx,y,ww,'h [M,N] · forward 임시 HBM')
                y=d.next_kernel(xx,y,ww,'F4+F5 · RoundedResidualSm90 · CuTe', ['acc=h@Wsᵀ · TMA / WGMMA','y=BF16(BF16(acc)+x)'], 'cute',note=['새 epilogue · 별도 z tensor / residual add 없음'])
        y=d.next_buffer(xx,y,ww,['y [M,D]', '큰 학습 입력: xn [M,D]도 backward까지 보관'] if x==xs[1] else 'y [M,D]')
        assert y < 1492, (x,y)
    d.footer(1515,[
        'M=B·L²(pair) 또는 B·L(token). B/C 지원: 정확히 SM90, BF16, n=4, D∈{128,256,384,512,768}, M%128=0.',
        'auto native OFF / shape 미지원 / 작은 wide-D token은 A. explicit cute는 지원 shape에서 C를 선택. legacy fusion=False는 별도.',
        'C의 weight 준비 / stats / expand / squeeze는 각각 별도 경계. 타일링 전체 최적성이나 완전한 native autotune을 주장하지 않음.',
        '소스: kernels/transition/hopper.py · triton/residual.py · cute/squeeze_residual.py · docs/transition-fusion/current-sources.json'])
    d.save('TRANSITION_FORWARD.svg')


def backward():
    d=Diagram(2460,2580,'Transition Backward · 저장값 차이와 residual 융합',STAMP)
    xs=[36,852,1668];w=756;ww=w-40
    heads=[('A · Triton · saved xn / stacked dAB',['forward의 xn 재사용 · 주요 cuBLAS GEMM 4회','D128: B2에서 NORMALIZE=False']),
           ('B · H100 b2b · saved xn / stacked dAB',['큰 학습 입력: x + stats + xn 저장 · cuBLAS GEMM 4회','D128/256: NORMALIZE=False · saved-xn Triton B2']),
           ('C · CuTe forward · 기본 backward',['xn 별도 재계산 + separate dA/dB · GEMM 6회','기본 gate backward는 Triton · CuTe 아님'])]
    for idx,x0 in enumerate(xs):
        d.panel(x0,112,w,2160,*heads[idx])
        x=x0+20;y=d.buffer(x,229,ww,'dy [M,D] · B1 및 최종 LN dx residual 입력')
        dy_y=y
        if idx==2:
            y=d.kernel(x,y+24,ww,'B0 · _xn_recompute_kernel · Triton', ['x / rstd / c1 → xn [M,D] · HBM 저장'])
            xn_y=y;b1_y=y+24
            y=d.kernel(x,b1_y,ww,'B1 · cuBLAS GEMM', ['dh=dy@Ws'], 'blas')
            d.el('path',d=f'M {x+ww-3} {dy_y} L {x+ww+8} {dy_y} L {x+ww+8} {b1_y+25} L {x+ww} {b1_y+25}',fill='none',stroke='#6b8192',stroke_width=2,marker_end='url(#arrow)')
        else:
            y=d.next_kernel(x,y,ww,'B1 · cuBLAS GEMM', ['dh=dy@Ws'], 'blas')
        y=d.next_buffer(x,y,ww,'dh [M,N]')
        if idx==0:
            title=['B2 · _transition_expand_gatebwd_kernel · Triton','_savedxn_stacked launcher · NORMALIZE=False']
            ops=['저장한 xn 로드 → a/b GEMM 재계산','h와 dA / dB 계산','h + packed dAB=[dA | dB] 직접 store']
            note=['L768 profiler: 약 1.244ms · h는 이 경로도 재계산']
        elif idx==1:
            title=['B2 · D128/256: _transition_expand_gatebwd_kernel','Triton · NORMALIZE=False · saved-xn stacked']
            ops=['b2b forward가 저장한 BF16 xn 재사용','a/b GEMM, h, dA / dB 재계산','h + packed dAB store · xn 재계산/재저장 없음']
            note=['L768 profiler: 이전 재계산 1.481 → 저장 재사용 1.229ms',
                  'D256도 저장 모드에서는 Triton saved-xn B2 사용']
        else:
            title=['B2 · _transition_expand_gatebwd_kernel · Triton','_savedxn launcher · NORMALIZE=False']
            ops=['B0의 xn 로드 → a/b/h 재계산','h, dA, dB를 별도 buffer에 저장']
            note=['기본 backward_backend="triton" 기준']
        if idx==2:
            d.el('path',d=f'M {x+3} {xn_y} L {x-8} {xn_y} L {x-8} {y+49} L {x} {y+49}',fill='none',stroke='#6b8192',stroke_width=2,marker_end='url(#arrow)')
        y=d.next_kernel(x,y,ww,title,ops,note=note)
        y=d.next_buffer(x,y,ww,'h [M,N] + '+('dA / dB [M,N]' if idx==2 else 'dAB [M,2N]'))
        branch=y; bw=(ww-18)/2; yy=y+28
        d.arrow(x+ww/4,y,x+ww/4,yy);d.arrow(x+ww*3/4,y,x+ww*3/4,yy)
        e1=d.kernel(x,yy,bw,'B3 · cuBLAS',['dWs=dyᵀ@h'],'blas')
        e2=d.kernel(x+bw+18,yy,bw,'B4 · '+('cuBLAS 2회' if idx==2 else 'cuBLAS 1회'),
                    [['dWa=dAᵀ@xn','dWb=dBᵀ@xn']] if idx==2 else ['dWab=dABᵀ@xn'], 'blas')
        yy=max(e1,e2)+30
        d.el('path',d=f'M {x+ww-3} {branch} L {x+ww+8} {branch} L {x+ww+8} {yy+24} L {x+ww} {yy+24}',
             fill='none',stroke='#6b8192',stroke_width=2,marker_end='url(#arrow)')
        y=d.kernel(x,yy,ww,'B5 · cuBLAS '+('2회 + 별도 add' if idx==2 else '1회'),
                   ['uA=dA@Wa, uB=dB@Wb','dxn=uA+uB'] if idx==2 else ['dxn=dAB@[Wa; Wb]'],
                   'compiler' if idx==2 else 'blas',note=['C의 uA+uB는 residual 합산과 다른 연산'] if idx==2 else ['[Wa; Wb] weight cat / dWa clone 등 보조 연산은 별도'])
        y=d.next_buffer(x,y,ww,'dxn [M,D]')
        if idx==0:
            y=d.next_kernel(x,y,ww,'B6+B7 · _ln_bwd_residual_kernel · Triton',
                            ['LN 미분 → dx_LN, dγ / dβ','dx=BF16(dx_LN)+dy'],note=['identity dy는 dγ / dβ에 더하지 않음'])
        else:
            y=d.next_kernel(x,y,ww,'보조 연산 · mean=c1/rstd 복원',['별도 tensor 연산'], 'compiler')
            y=d.next_kernel(x,y,ww,'B6+B7 · LN dx + residual · 아래 중 한 경로 선택',
                            [['D128 / M≥16384: Triton _ln_bwd_residual_kernel',
                             '그 외 eligible D≤512: CUDA main + affine reduce',
                             'CUDA 미지원/실패: Triton LN residual']], 'compiler',note=[
                             'dx=BF16(dx_LN)+dy · 별도 residual add 없음',
                             'CUDA main의 dx epilogue에 dy; reduce는 dγ / dβ 전용',
                             'Triton이 내는 FP32 affine gradient를 BF16 재변환 없이 반환'])
        y=d.next_buffer(x,y,ww,'dx [M,D] · parameter gradients는 각 branch 결과')
        assert y < 2260,(idx,y)
    d.footer(2295,[
        '현재 A/B는 xn 저장 재사용, C는 xn 재계산. B의 saved-xn 모드는 D128/256 · M≥16384 · gradient가 필요한 학습에서 기본 ON.',
        'B의 B2는 Triton, cuBLAS 4회도 그대로. D128 large pair LN도 Triton. native는 전체 backward 교체를 뜻하지 않음.',
        '프로파일 숫자는 node02 L768/D128의 한 CUDA trace. 전체 학습 시간은 별도 CUDA graph 반복 측정값을 사용.',
        'save_xn=False / 작은 M은 이전 B 재계산 경로. C는 기본 triton backward이며 opt-in cute backward는 별도.',
        '모든 tensor 의존 화살표 / 초기화 / cast를 펼친 전체 launch 목록은 아님. 상세와 제한: docs/transition-fusion/PERFORMANCE.md'])
    d.save('TRANSITION_BACKWARD.svg')


def overview():
    d=Diagram(2220,1670,'Transition · 현재 배선과 학습 이득',STAMP)
    d.panel(36,115,2148,245,'기본값: transition_residual_fusion=True / transition_h100_residual=True',[
        'engine_backend="triton" → A: D128/256 큰 입력 segmented b2b. 나머지 및 force_split=True → split',
        'auto + native 조건 충족 → B: CUDA b2b (D128/256) 또는 C: CuTe (wide D384/512/768, M≥16384)',
        'native 조건: SM90 · BF16 · n=4 · M%128=0. 작은 wide-D / 미지원 shape / native OFF → A.',
        'explicit implementation="cute"는 지원 shape에서 C. D128/256 auto는 transition_cuda_b2b=True도 필요.',
        '이 그림은 로컬 개발본. 설치본 및 실행 중 MiniWorld 학습 경로를 변경했다는 뜻은 아님.'])
    d.text(36,410,'연산 → 커널 연결 · 현재 기본 residual fusion',25,weight='700')
    widths=[220,790,1138];xx=[36,256,1046]
    rows=[['경로','Forward','Backward'],
          ['A · Triton','작은 D: LN → segmented b2b / 나머지: split','saved xn → stacked gatebwd → cuBLAS 4회 → LN+residual'],
          ['B · H100 b2b','stats → [LN+expand+SwiGLU+squeeze+residual]','saved xn → stacked gatebwd → cuBLAS 4회 → LN+residual'],
          ['C · H100 CuTe','stats/fold → [LN+expand+SwiGLU] → [squeeze+residual]','recompute xn → separate gatebwd → cuBLAS 6회 → LN+residual']]
    for j,row in enumerate(rows):
        for i,value in enumerate(row):
            yy=432+j*68;d.rect(xx[i],yy,widths[i],68,'#e7eef7' if j==0 else '#fff',rx=0)
            d.text(xx[i]+13,yy+41,value,17,weight='700' if j==0 else '400')
    d.text(36,756,'최신: split 대비 segmented Triton · 실제 기본 배선 · node02',25,weight='700')
    vals=json.loads((RECORD/'summary.json').read_text())['default_rows']
    for j,l in enumerate([384,768]):
        for dwidth,x,width in [(128,36,1046),(256,1106,1078)]:
            f=next(r for r in vals if r['L']==l and r['D']==dwidth and r['mode']=='inference')['ms']
            t=next(r for r in vals if r['L']==l and r['D']==dwidth and r['mode']=='training')['ms']
            d.panel(x,800+j*210,width,185,f'D{dwidth} / L{l} · split → 새 기본값',[
                f'추론: {f["split"]:.3f} → {f["b2b"]:.3f}ms · {f["split"]/f["b2b"]:.2f}배',
                f'학습: {t["split"]:.3f} → {t["b2b"]:.3f}ms · {t["split"]/t["b2b"]:.3f}배',
                '추론·학습 동일 forward · 기존 Triton backward 재사용'])
    d.panel(36,1240,2148,185,'출력 누산기 분할 · 중간 h는 HBM에 쓰지 않음',[
        'D128: BO=32/64, D256: BO=64/128 탐색. 선택된 큰 입력 타일은 각각 BO64 / BO128.',
        'BM / BN / BK / BO / warps / stages는 CSV·캐시로 관리. runtime 비교도 CUDA Graph 기준.',
        'LN 분리형이 최종 선택: xn은 추론에서 임시 버퍼, 학습에서는 backward까지 보관. FP32 γ/β 유지.'])
    d.footer(1450,[
        'BF16, B1, n4, FP32 affine, nonzero squeeze, static compile (관측 graph 1개), manual CUDA graph; optimizer 제외.',
        'D384/512의 선택된 streamed-K는 split보다 느림. 최신 full-K는 재튜닝 필요. auto CUDA/CuTe는 별도.',
        '소스/측정: docs/transition-fusion/README.md · transition-segmented-small-20260918/timings.md · current-sources.json'])
    d.save('TRANSITION.svg')


def residual():
    d=Diagram(2340,1330,'Transition Residual · 현재 세 경로의 융합 위치',STAMP)
    for idx,x in enumerate([36,812,1588]):
        ww=716;d.panel(x,115,ww,930,['A · Triton','B · H100 hand-CUDA','C · H100 CuTe'][idx],['현재 기본값: residual fusion ON'])
        xx=x+20;w=ww-40
        d.text(xx,235,'FORWARD',22,weight='700')
        y=d.kernel(xx,255,w,['segmented _kernel / fallback: squeeze_residual','transition_b2b_rs_wgmma_kernel','RoundedResidualSm90'][idx].strip(),
                   (['LN tile → expand → SwiGLU','squeeze + residual → y'] if idx==1 else ['squeeze GEMM','BF16(acc) + x → y']),
                   ['triton','cuda','cute'][idx],note=['별도 residual add / 중간 z tensor 없음'])
        d.text(xx,600,'BACKWARD',22,weight='700')
        y=d.kernel(xx,620,w,'LN dx epilogue + identity dy',[
            'LN 미분 → dx_LN','BF16(dx_LN) + dy → dx',
            'dγ / dβ는 LN branch만으로 계산'], 'triton' if idx==0 else 'compiler',note=(
            ['Triton _ln_bwd_residual_kernel'] if idx==0 else
            ['D128 large pair: Triton', '그 외 eligible D≤512: CUDA main + affine reduce', 'CUDA 미지원/실패: Triton residual LN']))
        assert y<1045,(idx,y)
    d.footer(1070,[
        '세 경로 모두 residual을 포함한다. H100의 비교 대상 Triton에도 forward/backward residual 융합이 이미 적용돼 있다.',
        '추가 개발: b2b CUDA 본체에 BF16 xn 벡터 store를 추가하고 saved-xn backward를 연결. residual 융합은 유지.',
        'L384/D128에서 z 또는 dx_LN의 별도 write+read 제거량은 각각 72MiB. 이미 융합된 Triton 대비 추가 절약량이 아니다.',
        'C의 dA@Wa + dB@Wb gradient 합산은 residual과 별개이며 남아 있다. 전체 학습 성능은 TRANSITION.svg 참고.'])
    d.save('TRANSITION_RESIDUAL.svg')


def variants():
    catalog=json.loads((DOC/'variants.json').read_text())
    assert set(catalog['variants']) == {'streamed_k','full_k'}
    d=Diagram(2260,1870,'Transition · 두 버전과 공통 Backward',
              '개발 기준 · D128/256/384/512 · Triton + 새 CUDA fwd/bwd 구현 · 명시적 실험 API · 자동 dispatch와 구분')
    for x,key,title,description in [
        (36,'streamed_k','V1 · streamed-K · BK < D',[
            'hidden 타일마다 K축을 BK씩 나누어 읽고 a/b 누적',
            '입력·가중치 타일을 순차 공급 · 전체 출력 누산기는 유지']),
        (1148,'full_k','V2 · full-K · 전체 K 재사용',[
            '전체 K 입력을 hidden 루프 밖에서 읽고 재사용',
            'D384: CUDA BK384/512 · Triton BK512 · 수식과 융합은 V1과 동일'])]:
        w=1076;d.panel(x,115,w,625,title,description)
        xx=x+18;ww=w-36
        y=d.buffer(xx,245,ww,['공통 LN → xn [M,D] · FP32 affine / BF16 xn · residual x',
                               '학습 저장: xn + x / LN 통계 / γ · 추론은 xn 임시 버퍼'])
        ops=(['hidden 타일마다 K를 BK씩 읽어 expand A/B 누적',
              'SwiGLU → h는 커널 내부에서 squeeze로 전달',
              '분할 출력 누산기 → BF16 squeeze + residual → y'] if key=='streamed_k' else
             ['전체 xn 입력을 먼저 읽고 hidden 타일들에서 재사용',
              'expand A/B → SwiGLU → h는 커널 내부에서 squeeze로 전달',
              '분할 출력 누산기 → BF16 squeeze + residual → y'])
        y=d.next_kernel(xx,y,ww,['CUDA · variant_kernel<false>', 'expand + SwiGLU + squeeze + residual · 하나의 b2b 커널'],ops,kind='cuda',
                        note=['대응 Triton: _segmented / transition_wide_b2b(config=..., BO 포함)',
                              'CUDA: TMA + WGMMA · 출력 WG=1은 RS / 여러 WG는 shared H'])
        y=d.next_buffer(xx,y,ww,'y [M,D] · 두 버전 모두 아래 saved-xn backward를 사용')
        assert y<740,y
    d.panel(36,770,2188,108,'공통 Backward · 수식과 저장 계약 유지',[
        '두 CUDA 버전: saved-xn native gate + 공통 cuBLAS 4회 + native LN/residual backward.',
        '아래 행은 개별 연산/커널 경계. cuBLAS 4회는 각각 별도 호출이며 표의 순서가 새로운 융합을 뜻하지 않음.'])
    widths=[80,710,650,748];xx=[36,116,826,1476]
    rows=[['단계','연산 / 출력','현재 구현','새 CUDA 작업'],
          ['B1','dh = dy @ Ws','cuBLAS','동일 GEMM 유지'],
          ['B2','saved xn → a/b 재계산 + gate 미분 → h, dAB','Triton savedxn_stacked','variant_kernel<true> · dh TMA + vector dAB store'],
          ['B3','dWs = dy.T @ h','cuBLAS','동일 GEMM 유지'],
          ['B4','dWab = dAB.T @ xn → dWa / dWb view','cuBLAS','동일 GEMM 유지'],
          ['B5','dxn = dAB @ [Wa; Wb]','cuBLAS · 앞에 weight torch.cat 있음','동일 GEMM · weight 준비 비용 포함'],
          ['B6','LN 미분 + residual dy → dx, dγ, dβ','Triton LN residual epilogue','native LN+residual dx / 파라미터 reduce · 2커널']]
    for j,row in enumerate(rows):
        for i,value in enumerate(row):
            yy=904+j*60
            d.rect(xx[i],yy,widths[i],60,'#e7eef7' if j==0 else '#fff',rx=0)
            d.text(xx[i]+12,yy+37,value,17,weight='700' if j==0 else '400')
    d.panel(36,1350,2188,248,'완료 기준 · forward만 완료로 표시하지 않음',[
        '① 각 버전: CUDA fwd + saved-xn gate bwd + LN/residual bwd 연결 → 출력 및 6개 gradient 검증',
        '② D128/256/384/512 × L384/768 · inference fwd / training fwd / bwd / fwd+bwd / peak memory 기록',
        '③ variant별 config / cache / source hash 구분 · cuBLAS / weight 준비 비용도 전체 시간에 포함',
        '④ SASS TMA/WGMMA + NCU spill/대기/트래픽 검증 → 측정 결과로만 dispatch 선택',
        '현재: CUDA fwd/bwd·모듈 연결 완료 · GPU 18건 / memcheck 10건 통과 · bounded sweep · 자동 dispatch 미등록',
        '기존 split = 비교 기준/fallback. 기존 CUDA/CuTe 배선은 별도이며 새 두 버전의 완성본으로 취급하지 않음.'])
    d.footer(1625,[
        '새 측정: transition-cuda-variants-20260918/RESULTS.md · 같은 형식 대비 이득과 최선 Triton 대비 성능을 구분.',
        '공통 비교는 LN 분리 / FP32 affine / BF16 activation·weight / residual ON. LN 융합 실험은 별도 표기.',
        '정의·실제 소스·검증 기준: docs/transition-fusion/VARIANTS.md · 기계 판독 목록: variants.json'])
    d.save('TRANSITION_VARIANTS.svg')


def sources():
    paths=['modules/transition/module.py','settings.py','kernels/transition/hopper.py',
           'kernels/transition/triton/residual.py','kernels/transition/triton/fused.py',
           'kernels/transition/triton/b2b_residual.py','kernels/transition/triton/wide_b2b.py',
           'kernels/transition/triton/segmented_b2b.py','kernels/transition/triton/segmented_residual.py',
           'kernels/transition/triton/main.py','kernels/transition/cute/fused.py',
           'kernels/transition/cuda/variants.py','kernels/transition/cuda/transition_variants_kernel.cu',
           'kernels/transition/cuda/transition_variant_norm.cu',
           'kernels/transition/cute/squeeze_residual.py','kernels/transition/cute/gemm_transition_swiglu.py',
           'kernels/layernorm/cuda/layer_norm_cuda_kernel.cu','kernels/trimul_inproj/triton/backward_fused.py']
    (DOC/'current-sources.json').write_text(json.dumps({'date':'2026-09-18','package':str(PKG),
        'scope':'Local development checkout; current residual-fused default. Installed training package unchanged.',
        'development_variants':{'path':'docs/transition-fusion/variants.json',
            'sha256':hashlib.sha256((DOC/'variants.json').read_bytes()).hexdigest(),
            'cuda_status':'implemented experimental; see VARIANTS.md'},
        'sha256':{p:hashlib.sha256((PKG/p).read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')


if __name__=='__main__':
    sources();overview();forward();backward();residual();variants()
