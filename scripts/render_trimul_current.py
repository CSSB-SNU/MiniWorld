"""Render current TriMul kernels as HBM read -> on-chip compute -> HBM write.

The drawing describes shared B1-B4 plus the selected single-launch shared-GP B7 candidate.
Sizes are buffer payloads, not measured DRAM traffic; L2 can service rereads.
"""
from pathlib import Path
import hashlib, html, json, unicodedata
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/trimul_diagrams_20260921'
P=ROOT/'runs/trimul_split_bwd_20260921'
SITE=ROOT/'runs/anthropic_b1b4_pipeline_20260919/site-visuals'
RUN.mkdir(exist_ok=True)
RESULTS={n:json.loads((P/('results-L%d.json'%n)).read_text()) for n in (384,768)}
for record in RESULTS.values():
 for name,digest in record['source_sha256'].items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert json.loads((P/'selection.json').read_text())['candidate']=='split_xn_pc1'
NEW=ROOT/'runs/trimul_ln_policy_v4_20260921'
ACTIVE=ROOT/'runs/trimul_b1_wait_folding_20260921'
assert all((ACTIVE/('results-L%d.json'%n)).exists() for n in (384,768))
for n in (384,768):
 for name,digest in json.loads((ACTIVE/('results-L%d.json'%n)).read_text())['source_sha256'].items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
C={'ink':'#163746','sub':'#52707e','read':'#2879ad','write':'#b36b19','cuda':'#12856e','blas':'#7754ab','scratch':'#a6537b'}

def units(b):
 if b>=1000000:return '%.2f MB'%(b/1000000)
 if b>=1000:return '%.2f kB'%(b/1000)
 return '%d B'%b

def width(s):return sum(1 if unicodedata.east_asian_width(c) in 'WF' else .56 for c in s)
def wrap(s,limit):
 out=[];cur=''
 for c in s:
  if cur and width(cur+c)>limit:out.append(cur.rstrip());cur=c.lstrip()
  else:cur+=c
 if cur:out.append(cur)
 return out


def sketch(phase,label,kernel):
 """Mathematical PyTorch equivalents; not bit-exact BF16 or launch schedules."""
 if kernel.startswith('F.pack'):
  return ['W1 = interleave(WLg, WL, WRg, WR)', 'WT_i = transpose(W_i)'], [
   'WT = [w.T.contiguous() for w in',
   '      (WL, WLg, WR, WRg, Wg)]',
   'g = torch.cat([WLg, WRg]).reshape(-1, 32, 128)',
   'p = torch.cat([WL, WR]).reshape(-1, 32, 128)',
   'W1 = torch.stack([g, p], 1).reshape(1024, 128)',
   'parts = [*WT, W1]',
   'flat = torch.cat([p.flatten() for p in parts])',
   'views = flat.split([p.numel() for p in parts])',
   'packed = [v.view_as(p) for v, p in zip(views, parts)]',
  ]
 if kernel.startswith('K1'):
  return ['L = mask ⊙ (x_n WLᵀ) ⊙ σ(x_n WLgᵀ)', 'R = mask ⊙ (x_n WRᵀ) ⊙ σ(x_n WRgᵀ)'], [
   '# WL/WLg/WR/WRg are logical matrices in W1.',
   'x_n = ln(x, gin, bin)',
   'pL = F.linear(x_n, WL)',
   'pR = F.linear(x_n, WR)',
   'gL = torch.sigmoid(F.linear(x_n, WLg))',
   'gR = torch.sigmoid(F.linear(x_n, WRg))',
   'left  = channels(pL * gL * mask[:, None])',
   'right = channels(pR * gR * mask[:, None])',
   '# HBM outputs: left, right; x_n is not saved.',
  ]
 if kernel.startswith('K3'):
  return ['P = LN_out(tri) Wpᵀ', 'G = σ(LN_in(x) Wgᵀ)', 'y = x + ds ⊙ P ⊙ G'], [
   't = rows(tri)',
   'mu = t.mean(-1, keepdim=True)',
   'rs = torch.rsqrt(((t-mu)**2).mean(-1, True)+1e-5)',
   'xhat = (t-mu)*rs  # on-chip only',
   'xn_out = xhat*gout + bout',
   'proj = F.linear(xn_out, Wp)',
   'x_n = ln(x, gin, bin)  # recompute input LN',
   'gate = torch.sigmoid(F.linear(x_n, Wg))',
   'scale = ds[None].expand(L, L, 128).reshape(M, 128)',
   'y = x + proj * gate * scale',
   '# HBM outputs: y, x_n, FP32 mu and rs.',
   '# xhat/xn_out/proj/gate are not saved.',
  ]
 if kernel.startswith('b7_joint'):
  return ['dW_i = Σ_tiles dQ_iᵀ x_n', 'dN = Σ_i dQ_i W_i + dGate Wg', 'dx = LN_in′(dN; raw x) + dy'], [
   '# ONE launch: each large input loaded once.',
   '# 4 CTAs share derivatives through DSMEM.',
   'for xn, dl, dr in persistent_row_tiles:',
   '    pL, pR, gL, gR = projection_gate(xn)',
   '    dPL = dl * mask * gL',
   '    dGL = dl * mask * pL * gL * (1-gL)',
   '    dPR = dr * mask * gR',
   '    dGR = dr * mask * pR * gR * (1-gR)',
   '    dq = [dGL, dPL, dGR, dPR]  # on-chip',
   '    acc_dW += [q.T @ xn for q in dq]',
   '    dn = sum(q @ w for q, w in zip(dq, W))',
   '    dn += dGate_tile @ Wg',
   '    dx, dg, db = ln_bwd_raw(dn, raw_x, gin)',
   '    store_dx(dx + dy_tile)',
   '    acc_dg += dg; acc_db += db',
   '# Same launch: grid sync, reduce FP32 partials.',
   'store_grads(reduce(acc_dW, acc_dg, acc_db))',
  ]
 if kernel.startswith('transpose'):
  return ['Wp_T = contiguous(Wpᵀ)'], ['Wp_T = Wp.T.contiguous()', '# HBM output: Wp_T (a real copy).']
 if kernel.startswith('b1_fused · Phase A'):
  return ['P = LN_out(tri) Wpᵀ', 'G = σ(x_n Wgᵀ)', 'dP = dy ⊙ ds ⊙ G', 'dGlogit = dy ⊙ ds ⊙ P ⊙ G ⊙ (1−G)', 'partial dWp += dPᵀ xn_out', 'dTri = LN_out′(dP Wp; x̂, rstd)'], [
   '# Per row tile; acc_* are CTA-local accumulators.',
   't = tri_tile.float()  # load original BF16 tri',
   'mu = saved_mu_tile[:, None]',
   'rs = saved_rs_tile[:, None]',
   'xhat = (t - mu) * rs  # no statistics reduction',
   'xn_out = xhat * gout + bout',
   'proj = F.linear(xn_out, Wp)',
   'gate = torch.sigmoid(F.linear(x_n_tile, Wg))',
   'du = dy_tile * dropout_scale_tile',
   'dProj = du * gate',
   'dGate_tile = du * proj * gate * (1 - gate)',
   'acc_dWp += dProj.T @ xn_out',
   'dNorm = dProj @ Wp  # BF16 register fragments',
   'dt, dg, db = ln_bwd(dNorm, xhat, gout, rs)',
   'acc_dg += dg; acc_db += db',
   '# HBM: dGate/dTri tiles; store acc_* at CTA end.',
   '# dNorm GEMM reads dProj in place; no shared copy.',
   '# Prefetch next tri/x_n/stats during LN backward.',
  ]
 if kernel.startswith('b1_fused · Phase B'):
  return ['partial dWg_T += x_nᵀ dGlogit', 'grid barrier → sum of CTA partials', 'dWg / dWp / dγ_out / dβ_out'], [
   '# SAME launch: each CTA reads only its own Phase A rows.',
   '# Read x_n and Phase A output dGate from HBM/L2.',
   '# TMA prefetches the next tile into another buffer.',
   'acc_dWg += x_n_tile.T @ dGate_tile',
   '# CTA end: store acc_dWg; final grid barrier.',
   'dWgate = partial_dWg.sum(0)  # transposed layout',
   'dWproj = partial_dWp.sum(0)',
   'dgout = partial_dg.sum(0)',
   'dbout = partial_db.sum(0)',
   '# No LN, projection or gate recomputation here.',
  ]
 if kernel.endswith('role=1'):
  return ['dP = dA ⊙ mask ⊙ G', 'dGlogit = dA ⊙ mask ⊙ P ⊙ G ⊙ (1−G)', 'dW_P,T = x_nᵀ dP', 'dW_G,T = x_nᵀ dGlogit'], [
   'grads = []',
   'for dA, Wi, Wgi in [',
   '    (rows(dLeft), WL, WLg),',
   '    (rows(dRight), WR, WRg)]:',
   '    p = F.linear(x_n, Wi)   # from packed W1',
   '    g = torch.sigmoid(F.linear(x_n, Wgi))',
   '    da = dA * mask[:, None]',
   '    dp = da * g',
   '    dg = da * p * g * (1 - g)',
   '    grads += [x_n.T @ dp, x_n.T @ dg]',
   'dWL, dWLg, dWR, dWRg = grads',
   '# HBM: four weight gradients, transposed layout.',
   '# Row-tile partials are summed via global scratch.',
  ]
 if kernel.endswith('role=2'):
  return ['d(x_n) = dGlogit,out Wg', '  + Σside (dP Wi + dGlogit Wgi)', 'dx = LN_in′(d(x_n)) + dy', 'dγ_in, dβ_in = LN parameter reductions'], [
   'dxn = dGate @ Wg  # output-gate branch from B1',
   'for dA, Wi, Wgi in [',
   '    (rows(dLeft), WL, WLg),',
   '    (rows(dRight), WR, WRg)]:',
   '    p = F.linear(x_n, Wi)  # recompute again',
   '    g = torch.sigmoid(F.linear(x_n, Wgi))',
   '    da = dA * mask[:, None]',
   '    dp = da * g',
   '    dg = da * p * g * (1 - g)',
   '    dxn = dxn + dp @ Wi + dg @ Wgi',
   'mu = x.mean(-1, keepdim=True)',
   'rs = torch.rsqrt(((x-mu)**2).mean(-1, True)+1e-5)',
   'dx, dgin, dbin = ln_bwd(dxn, (x-mu)*rs, gin, rs)',
   'dx = dx + dy  # residual branch',
   '# HBM: dx, dgin, dbin; no dxn/P/G save.',
  ]
 if kernel.startswith('cuBLAS'):
  if phase=='FORWARD':
   if 'outgoing' in label:return ['T_out = L_out R_outᵀ'], ['torch.bmm(left[:128], right[:128].transpose(1, 2),', '          out=tri[:128])']
   return ['T_in = L_inᵀ R_in'], ['torch.bmm(left[128:].transpose(1, 2), right[128:],','          out=tri[128:])']
  if 'outgoing dLeft' in label:return ['dL_out = dT_out R_out'], ['torch.bmm(dTri[:128], right[:128],','          out=dLeft[:128])']
  if 'outgoing dRight' in label:return ['dR_out = dT_outᵀ L_out'], ['torch.bmm(dTri[:128].transpose(1, 2), left[:128],','          out=dRight[:128])']
  if 'incoming dLeft' in label:return ['dL_in = R_in dT_inᵀ'], ['torch.bmm(right[128:], dTri[128:].transpose(1, 2),','          out=dLeft[128:])']
  return ['dR_in = L_in dT_in'], ['torch.bmm(left[128:], dTri[128:],','          out=dRight[128:])']
 raise ValueError(kernel)

class SVG:
 def __init__(self,phase,n):
  self.phase,self.n=phase,n;self.el=[];self.y=315
  self.rect(0,0,2800,218,'#133343','none',0)
  self.text(40,60,'TriMul '+phase+'  |  HBM 입력 → 커널 → HBM 출력',36,'#ffffff',True)
  self.text(40,103,'L%d · M=L²=%s · C128 / 양방향 H256 · BF16 · dropout 25%% / mask / residual'%(n,format(n*n,',')),23,'#cce0e7')
  self.text(40,145,'표시된 크기 = 버퍼 payload. 실제 HBM 전송량은 L2 hit·CTA 재읽기·spill에 따라 달라진다.',22,'#dfecf0')
  self.text(40,181,'입출력은 global memory 경계 기준  |  커널 내부 shared memory / register 값은 가운데에만 표시',21,'#bad2de')
  self.text(40,259,'HBM READ · 어디서 온 무엇을 읽는가',25,C['read'],True)
  self.text(690,259,'KERNEL · tile 내부 계산',25,C['cuda'],True)
  self.text(2160,259,'HBM WRITE · 무엇을 어디로 넘기는가',25,C['write'],True)
  self.text(40,293,'파랑 화살표: 읽기',18,C['read']);self.text(690,293,'B1의 A/B 두 행 = 같은 호출 · 왼쪽: 설명·수식 / 오른쪽: PyTorch',18,C['sub']);self.text(2160,293,'주황: 쓰기  ·  자주: scratch 재읽기',18,C['write'])
 def rect(self,x,y,w,h,fill,stroke,r=12,dash=False):
  self.el.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" stroke-width="2"%s/>'%(x,y,w,h,r,fill,stroke,' stroke-dasharray="8 5"' if dash else ''))
 def text(self,x,y,s,size=22,color=None,bold=False):
  self.el.append('<text x="%g" y="%g" font-family="Noto Sans CJK KR, sans-serif" font-size="%g" fill="%s"%s>%s</text>'%(x,y,size,color or C['ink'],' font-weight="700"' if bold else '',html.escape(s)))
 def arrow(self,x1,y1,x2,y2,kind):
  self.el.append('<path d="M %g %g L %g %g" fill="none" stroke="%s" stroke-width="3" marker-end="url(#%s)"/>'%(x1,y1,x2,y2,C[kind],kind))
 def block(self,x,y,w,h,title,lines,kind,size=22):
  fill={'read':'#eef6fc','write':'#fff7e7','cuda':'#effaf5','blas':'#f4f0fb','scratch':'#fcf0f7','sub':'#f0f4f6'}[kind]
  self.rect(x,y,w,h,fill,C[kind]);self.text(x+18,y+35,title,23,C[kind],True);yy=y+73
  for line in lines:
   for s in wrap(line,(w-36)/size):
    assert yy<=y+h-14,(title,s,yy,y+h)
    self.text(x+18,yy,s,size);yy+=31
  return yy
 def code(self,x,y,lines,size=20):
  for i,line in enumerate(lines):
   assert len(line)*size*.61<850,(len(line),line)
   self.el.append('<text x="%g" y="%g" xml:space="preserve" font-family="DejaVu Sans Mono, monospace" font-size="%g" fill="%s">%s</text>'%(x,y+i*28,size,'#607a76' if line.lstrip().startswith('#') else '#173d37',html.escape(line)))
 def row(self,label,reads,kernel,ops,writes,height=234,blas=False,scratch=None):
  formula,code=sketch(self.phase,label,kernel)
  ops=[*ops,'── 수식 ──',*formula]
  nlines=sum(len(wrap(line,(490-36)/22)) for line in ops)
  height=max(height,90+nlines*31,88+len(code)*28)
  y=self.y
  self.text(40,y+22,label,21,C['sub'],True);y+=40
  self.block(40,y,570,height,'INPUT · global buffers',reads,'read')
  kind='blas' if blas else 'cuda'
  self.rect(690,y,1390,height,'#f4f0fb' if blas else '#effaf5',C[kind])
  self.el.append('<path d="M1180 %g V%g" stroke="%s" stroke-width="1.5"/>'%(y,y+height,C[kind]))
  self.text(708,y+35,kernel,23,C[kind],True)
  yy=y+73
  for line in ops:
   for line2 in wrap(line,(490-36)/22):
    self.text(708,yy,line2,22);yy+=31
  self.text(1202,y+35,'PyTorch · 수학적 대응 코드',23,C[kind],True)
  self.code(1202,y+73,code)
  self.block(2160,y,600,height,'OUTPUT · global buffers',writes,'write')
  cy=y+height/2;self.arrow(612,cy,678,cy,'read');self.arrow(2082,cy,2148,cy,'write')
  self.text(620,cy-14,'READ',15,C['read'],True);self.text(2085,cy-14,'WRITE',15,C['write'],True)
  self.y=y+height+22
  if scratch:
   self.rect(690,self.y,2070,87,'#fcf0f7',C['scratch'])
   self.text(710,self.y+30,'같은 호출: partial global WRITE → grid 동기화 → READ → 최종 gradient',20,C['scratch'],True)
   self.text(710,self.y+65,scratch,20,C['ink'])
   self.arrow(910,self.y-22,910,self.y,'write')
   self.arrow(1160,self.y,1160,self.y-22,'scratch')
   self.y+=109
 def panel(self,title,lines,height):
  self.block(40,self.y,2720,height,title,lines,'sub',22);self.y+=height+24
 def finish(self,path):
  self.panel('코드 표기: 수식 대응용 PyTorch · 실제 CUDA의 BF16 반올림/타일 순서는 생략',[
   '코드는 모든 연산값과 가중치를 FP32로 해석. 바깥 HBM 박스의 dtype과 구분하며 bit-exact 실행 코드가 아니다.',
   'x, x_n, dy = [M,128]; tri/left/right = [256,L,L]; mask = [M]; ds = [L,128]. M=L², batch=1.',
   'WL/WLg/WR/WRg는 packed W1의 논리 행렬. 코드에 등장해도 원본 가중치를 HBM에서 추가로 읽는다는 뜻은 아니다.',
   'gin/bin = 입력 LN γ/β; gout/bout = 출력 LN γ/β. dGate는 출력 gate logit의 미분이다.',
  ],207)
  y=self.y;self.rect(40,y,2720,520,'#edf5f1',C['cuda'])
  self.text(60,y+36,'공통 PyTorch helper · LN / tensor layout',24,C['cuda'],True)
  self.text(1420,y+36,'LN backward · B1은 tri + 저장 μ/rstd 사용',24,C['cuda'],True)
  self.code(60,y+78,[
   'def rows(t):',
   '    return t.reshape(256, M).T',
   'def channels(t):',
   '    return t.T.reshape(256, L, L)',
   'def ln(z, gamma, beta):',
   '    return F.layer_norm(z, (z.shape[-1],),',
   '                        gamma, beta, eps=1e-5)',
   '# F = torch.nn.functional; torch is imported.',
  ])
  self.code(1420,y+78,[
   'def ln_bwd(dh, h, gamma, rs):',
   '    # h: pre-affine normalized values, FP32',
   '    # No mean/variance or normalization here.',
   '    q = dh * gamma',
   '    dz = rs * (q - q.mean(-1, keepdim=True)',
   '               - h * (q*h).mean(-1, keepdim=True))',
   '    dgamma = (dh*h).sum(0)',
   '    dbeta = dh.sum(0)',
   '    return dz, dgamma, dbeta',
  ])
  self.y+=544
  self.panel('범위 / 확인 근거',[
   '2026-09-22 · B1 v51 + B7 단일 CUDA launch 개발 기본값. 수치·graph·sanitizer 통과. 기존 분리형보다 느림 / production 아님.',
   '출력 LN activation 저장 폐기. BF16 tri 유지 + K3 저장 FP32 μ/rstd로 B1 정규화 값을 재구성.',
   '근거: trimul_b1_wait_folding/{wait_policy.py,b1_fused.cu,lowreg_stats.inc,results-L*.json} / 기존 K3 / trimul_b7_joint_lncolumns_20260922.',
  ],176)
  h=self.y+16
  start='<svg xmlns="http://www.w3.org/2000/svg" width="2800" height="%d" viewBox="0 0 2800 %d" role="img" aria-labelledby="title desc"><title id="title">TriMul %s L%d HBM 입력 출력 지도</title><desc id="desc">각 행은 HBM 읽기, 커널 내부 계산, HBM 쓰기를 왼쪽부터 표시한다. 버퍼 크기는 실제 DRAM 트래픽과 구분한다.</desc><defs>'%(h,h,self.phase,self.n)
  for key in ('read','write','scratch'):start+='<marker id="%s" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="%s"/></marker>'%(key,C[key])
  start+='</defs><rect width="2800" height="%d" fill="#f8fafc"/>'%h
  path.write_text(start+'\n'+'\n'.join(self.el)+'\n</svg>')


def sizes(n):return dict(a=n*n*128*2,t=n*n*256*2,m=n*n,mask=n*n*4,maskb=n*n*2,ds=n*128*2)

def forward(n):
 s=sizes(n);a,t=s['a'],s['t'];g=SVG('FORWARD',n)
 g.row('준비 · 매 호출 weight packing',[
  'WL, WLg, WR, WRg: 각 BF16 [256,128]',
  'Wg: BF16 [128,128]',
  '합계 294.91 kB · 원래 학습 파라미터',
 ],'F.pack · compiled layout',[
  '4개 입력 가중치 + 출력 gate 정렬',
  'transpose / interleave / flat buffer',
  'activation 준비 연산은 아님',
 ],[
  'W1 [1024,128] · 262.14 kB → K1 / B7',
  'WT 5개 중 Wg_T 32.77 kB → B1 / B7 내부 dX',
  '입력 WT 4개 262.14 kB는 이 후보에서 미소비',
  '합계 557.06 kB · 하나의 allocation',
 ],height=207)
 g.row('① F1 + F2 · input LN / input projection + gate',[
  'x: BF16 [M,128] · '+units(a),
  'W1: BF16 [1024,128] · 262.14 kB',
  'mask: FP32 [L,L] · '+units(s['mask']),
  'γ_in / β_in: FP32 [128] ×2 · 1.02 kB',
 ],'K1 · infer_k1 · CUDA',[
  'x → LN → x_n',
  'x_n × W1 → pL,pR,gL,gR',
  'projection × sigmoid(gate) × mask',
  'x_n / p / g: shared·register에만',
 ],[
  'left:  BF16 [256,L,L] · '+units(t),
  'right: BF16 [256,L,L] · '+units(t),
  '둘 다 ab [512,L,L]의 view',
  '→ F3 contraction + B5/B6에서 다시 읽음',
 ],height=234)
 for direction,label,formula in [('out','② F3a · outgoing contraction','left_out × right_outᵀ'),('in','③ F3b · incoming contraction','left_inᵀ × right_in')]:
  g.row(label,[
   'left_'+direction+':  BF16 [128,L,L] · '+units(a),
   'right_'+direction+': BF16 [128,L,L] · '+units(a),
   '← K1이 HBM에 기록한 방향별 slice',
  ],'cuBLAS · bmm 1회',[formula,'GEMM accumulator → BF16','TMA/shared 구현은 cuBLAS 내부'],[
   'tri_'+direction+': BF16 [128,L,L] · '+units(a),
   '→ K3 + B1–B4에서 다시 읽음',
   '최종 tri slice에 직접 store · cat 없음',
  ],height=192,blas=True)
 g.row('④ F4–F7 · output LN / projection + output gate / dropout + residual',[
  'tri: BF16 [256,L,L] · '+units(t)+' ← F3',
  'x: BF16 [M,128] · '+units(a)+' ← 원래 입력',
  'Wp [128,256] + Wg [128,128]: 98.30 kB',
  'γ/β_in,out: FP32 · 합계 3.07 kB',
  'ds: BF16 [L,128] · '+units(s['ds']),
  'ds는 행 방향으로 broadcast하는 dropout scale',
 ],'K3 · save_k3 · CUDA',[
  'tri → LN_out → xn_out → projection',
  'x → LN_in 재계산 → x_n → gate',
  'projection × gate × ds + x',
  'xn_out / projection / gate는 on-chip',
  '출력 LN 평균 / 역표준편차만 저장',
 ],[
  'y: BF16 [M,128] · '+units(a)+' → 다음 모듈',
  'x_n: BF16 [M,128] · '+units(a),
  '→ B1–B4 + 단일 B7가 한 번 읽음',
  'x_n은 K3에서 저장 · K1의 출력을 읽지 않음',
  'μ_out/rstd_out: FP32 [M]×2 · '+units(s['m']*8),
  '→ B1 Phase A에서 TMA로 tri와 함께 읽음',
 ],height=310)
 g.panel('FWD → HBM → BWD: 저장 버퍼의 실제 소비자',[
  'left / right: K1 WRITE → F3 READ → 그대로 유지 → B5/B6 READ. 중간에 별도 복사하지 않음.',
  'tri: F3 WRITE → K3 READ → 그대로 유지 → B1 READ. μ/rstd: K3 WRITE → B1 READ.',
  'x_n: K3 WRITE → B1–B4 READ → 단일 B7 READ. 각 호출의 READ는 독립적.',
  '보존 activation 합계 left + right + tri + x_n + μ/rstd = '+units(3*t+a+s['m']*8)+' (원래 x, 출력 y, mask, 가중치 제외).',
 ],207)
 g.panel('이 그림에서 “저장 안 함”의 의미',[
  '입력 x_n 저장 유지. 출력 x̂ 및 affine xn_out은 미저장. 입력 pL/pR/gL/gR, 출력 projection/gate도 미저장.',
  '이 값들은 뒤의 CUDA kernel 안에서 재계산한다. Gradient 전달 버퍼와 reduction scratch는 BWD 그림에 표시.',
  'B1은 저장 μ/rstd로 (tri−μ)×rstd를 계산한다. 평균/분산 reduction은 다시 하지 않는다.',
  '버퍼 이름이 같아도 읽기 호출이 다르면 load 요청은 발생한다. 실제 DRAM 왕복 횟수는 NCU 실측이 필요하다.',
 ],207)
 return g

def backward(n):
 s=sizes(n);a,t=s['a'],s['t'];g=SVG('BACKWARD',n);ctas=json.loads((ACTIVE/('selected-L%d.json'%n)).read_text())['config']['count']
 g.row('준비 · Wproj 전치 버퍼',[
  'Wp: BF16 [128,256] · 65.54 kB',
  '← 현재 forward의 출력 projection 가중치',
 ],'transpose + contiguous',[
  'Wproj.t().contiguous()',
  '단순 view가 아닌 실제 복사',
 ],['Wp_T: BF16 [256,128] · 65.54 kB','→ B1의 dW/projection 재계산 및 dTri'],height=153)
 g.panel('① B1–B4 · b1_fused · 아래 A/B는 같은 CUDA launch 1회',[
  str(ctas)+' CTA ×256 threads. 각 CTA는 Phase A → local fence → Phase B. 마지막 grid barrier 뒤 전체 partial 합산.',
  '입력 x_n 및 출력 μ/rstd 저장. BF16 tri로 정규화/affine/P/G 재계산; 평균/분산 reduction은 없음.',
  '로드 전용 producer warpgroup은 선택하지 않음. Phase B에서 TMA 이중 버퍼로 다음 타일 로드와 WGMMA를 겹침.',
 ],176)
 group_top=g.y
 g.row('①-A · 공유 재계산 + dWproj / dTri / 출력 LN 미분',[
  'dy: BF16 [M,128] · '+units(a)+' ← 상위 미분',
  'x_n: BF16 [M,128] · '+units(a)+' ← K3 저장',
  'tri: BF16 [256,L,L] · '+units(t)+' ← F3 유지',
  'μ_out/rstd_out: FP32 [M]×2 · '+units(s['m']*8)+' ← K3',
  'Wp_T + Wg_T: BF16 · 합계 98.30 kB',
  'γ_out / β_out: FP32 [256] ×2 · 2.05 kB',
  'ds: BF16 [L,128] · '+units(s['ds']),
 ],'b1_fused · Phase A',[
  '(tri−μ)×rstd → affine · 두 그룹 분담',
  'dGate / dProj를 함께 계산하여 공유',
  '같은 CTA: dWproj + dTri / LN 미분',
  'B4: LN 미분 + γ/β shared 합산',
  'dNorm: 묶음 WGMMA → BF16 레지스터',
  'dTri TMA 쓰기 ↔ γ/β 합산',
  'P/G/dProj: global 저장 없음',
 ],[
  'dTri: BF16 [256,L,L] · '+units(t)+' → B5/B6',
  'dGate: BF16 [M,128] · '+units(a),
  '→ 아래 Phase B에서 READ + 이후 B7 내부 dX',
  'partial dWproj: FP32 · '+units(ctas*32768*4),
  'partial LN: FP32 · '+units(ctas*512*4),
  '부분합 → Phase B 끝의 최종 reduction',
 ],height=285)
 g.panel('같은 호출 내부 · CTA별 Phase 전환 · dGate의 global WRITE → READ',[
  '위 Phase A에서 출력한 dGate를 아래 Phase B가 다시 읽는다. 새 dGate 복사 버퍼를 만들지 않는다.',
  'x_n도 global/L2 재읽기. 각 CTA의 소유 행만 소비하므로 이 경계에 전체 grid barrier는 없음.',
  '마지막에 모든 CTA의 parameter partial을 합치기 전에는 전체 grid barrier를 유지한다.',
 ],176)
 g.row('①-B · dWgate + 모든 parameter gradient 최종 reduction',[
  'x_n: BF16 [M,128] · '+units(a)+' ← K3 저장',
  'dGate: BF16 [M,128] · '+units(a)+' ← Phase A',
  '타일 로드: x_n + dGate 합계 '+units(2*a),
  '최종 reduction: partial dWproj / LN 재읽기',
  '자체 partial dWgate도 WRITE 후 READ',
 ],'b1_fused · Phase B + reduction',[
  'x_nᵀ × dGate → dWgate 부분합',
  'TMA: 다음 타일 LOAD || 현재 WGMMA',
  'partial dWgate: FP32 · 8.65 MB',
  'grid barrier → 모든 partial 합산',
  'LN / projection / gate 재계산 없음',
 ],[
  'dWgate: BF16 [128,128] · 32.77 kB',
  'dWproj: BF16 [128,256] · 65.54 kB',
  'dγ_out / dβ_out: FP32 [256] ×2 · 2.05 kB',
  '→ 최종 parameter gradients',
 ],height=285,scratch='A+B partial: dW 25.95 MB + LN 270.34 kB · 총 WRITE 26.22 MB / READ 26.22 MB')
 g.el.append('<path d="M28 %g H16 V%g H28" fill="none" stroke="%s" stroke-width="6"/>'%(group_top,g.y-10,C['cuda']))
 for direction,side,formula,label in [
  ('out','Left','dTri_out × right_out','② B5 · outgoing dLeft'),
  ('out','Right','dTri_outᵀ × left_out','③ B5 · outgoing dRight'),
  ('in','Left','right_in × dTri_inᵀ','④ B6 · incoming dLeft'),
  ('in','Right','left_in × dTri_in','⑤ B6 · incoming dRight')]:
  operand='right' if side=='Left' else 'left'
  g.row(label,[
   'dTri_'+direction+': BF16 [128,L,L] · '+units(a),
   operand+'_'+direction+': BF16 [128,L,L] · '+units(a),
   '← B1의 dTri + K1에서 보존한 '+operand,
  ],'cuBLAS · bmm 1회',[formula,'contraction의 입력 미분','최종 gradient slice에 직접 store'],[
   'd'+side+'_'+direction+': BF16 [128,L,L] · '+units(a),
   '→ 단일 B7에서 한 번 읽고 dW·dX가 공유',
   '두 방향 합쳐 d'+side+' [256,L,L]',
  ],height=192,blas=True)
 g.row('⑥ B7–B12 · 단일 CUDA launch / dW + dX + input LN + residual',[
  'dLeft + dRight: BF16 [256,L,L] ×2 · '+units(2*t),
  'x_n: BF16 [M,128] · '+units(a)+' ← K3',
  'dGate: BF16 [M,128] · '+units(a)+' ← B1',
  'raw x + dy: BF16 [M,128] ×2 · '+units(2*a),
  'W1 262.14 kB + WT/Wg_T 294.91 kB',
  'mask BF16 '+units(s['maskb'])+' / γ_in FP32 512 B',
  '큰 입력 각 1회 logical load / TMA multicast',
 ],'b7_joint · 1 launch / 4-CTA cluster',[
  'P/G와 dP/dG 한 번 계산 → shared 공유',
  'dW: register에 누적, 중간 dQ는 HBM 미저장',
  'dX: CTA마다 C32 출력 / hidden 전체 합산',
  'input LN도 C32 분할 / 소량 통계만 DSMEM',
  '30 clusters · 512 threads/CTA · shared 224 KiB',
  'register spill 0 · 정확도 통과 / 속도 목표 미달',
 ],[
  'dWL / dWLg / dWR / dWRg: 합계 262.14 kB',
  'dx: BF16 [M,128] · '+units(a)+' → 이전 모듈',
  'dγ_in / dβ_in: FP32 [128] ×2 · 1.02 kB',
  '새 HBM activation 중간 버퍼 없음',
  '호출 안에서 FP32 partial reduction ↓',
 ],height=610,scratch='partial dW: WRITE/READ 각 15.73 MB; partial LN: WRITE/READ 각 30.72 kB; reduction도 같은 launch')
 g.panel('큰 global 버퍼의 재읽기: source에서 확인한 load 범위 (HBM miss 횟수 아님)',[
  'B1: Phase A는 dy / x_n / tri 및 μ/rstd 각 1회. Phase B는 x_n / dGate 각 1회. 정규화는 shared에서 재계산.',
  'B7: x_n은 TMA multicast로 4 CTA가 공유. dLeft/right는 서로 겹치지 않는 H128 slice로 전체 각 1회 읽음.',
  'B7: dP/dG는 한 번 계산 후 DSMEM으로 공유. dW와 dX 사이의 HBM 저장·재읽기나 GP 재계산 없음.',
  'B7: dGate / raw x / residual도 한 번 전송. mask·weight·통계 및 gradient partial의 읽기는 별도.',
  'TMA가 L2에서 만족하면 이 load가 전부 HBM 왕복이 되지는 않는다. 실제 DRAM bytes/SoL 수치는 여기서 추정하지 않음.',
 ],238)
 g.panel('Global reduction scratch도 융합 커널의 입출력이다',[
  'B1 partial: FP32 ['+str(ctas)+',49152] + ['+str(ctas)+',512]. B7 scratch도 위에 WRITE/READ 표시.',
  'B1 및 단일 B7은 각각 호출 내부 grid 동기화 뒤 partial을 다시 읽어 최종 gradient를 쓴다.',
  '소량 atomic counter READ/WRITE: B1 할당 536 B, B7 8 B. Spin-loop 실제 접근량은 별도이며 위 payload에서 제외.',
  'Backward 큰 activation/gradient WRITE: dTri, dGate, dLeft, dRight, dx. 중간 dProj / d(xn_out) / d(x_n)은 미저장.',
 ],207)
 return g

outputs=[]
for n in (384,768):
 for name,draw in [('FORWARD',forward),('BACKWARD',backward)]:
  p=ROOT/('TRIMUL_'+name+('_L768' if n==768 else '')+'.svg');draw(n).finish(p);outputs.append(p)
  (SITE/'dist/assets'/('trimul-current-'+name.lower()+'-L%d.svg'%n)).write_bytes(p.read_bytes())
source_paths=[P/k for k in ('bench.py','b1_fused.cu','b1_pipeline_math.inc','b7_roles.cu','b7_pipe_dw.inc','b7_packed_dx.inc','role_plan.py','saved_plans.py')]
source_paths += [ROOT/'runs/trimul_b7_joint_lncolumns_20260922'/k for k in ('gate_policy.py','plan.py','joint.cu','single_wg.inc','full-check-L384.json','full-check-L768.json','sanitizer-L384.json','sanitizer-L768.json')]
source_paths += [NEW/k for k in ('training.py','replace_plan.py','replace_core.py','final.py','save_k3.cu','b1_fused.cu','lowreg_stats.inc','b1_stream_ln.inc','selected-L384.json','selected-L768.json')]
source_paths += [ACTIVE/k for k in ('wait_policy.py','replace_plan.py','b1_fused.cu','lowreg_stats.inc','bench.py','selected-L384.json','selected-L768.json','results-L384.json','results-L768.json')]
source_paths += [ROOT/'runs/trimul_ln_only_save_20260921/ln_save_core.py',ROOT/'runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/triton/contract.py',ROOT/'runs/anthropic_b7b12_fusion_20260920/training_forward_adapter.py',ROOT/'runs/anthropic_ln_equal_saves_20260919/core_saved.py',ROOT/'runs/trimul_ln_only_save_20260921/build/b98c2847137dbf8211f735fea50436949325bcdf5fe8088a4dc0dfca4dcefd8e.ptxas.log']
manifest=dict(date='2026-09-22',view='HBM read -> kernel [formulas | PyTorch] -> HBM write',candidate='b1_v51_b7_joint_lncolumns',production_ready=False,byte_semantics='buffer payload; not measured DRAM traffic',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},outputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},scratch_bytes={'b1_partial_weights':132*49152*4,'b1_partial_ln':132*512*4,'b7_dw':30*4*2*2*64*128*4,'b7_ln':30*256*4},logical_load_coverage={'b1':{'dy':1,'x_n':2,'tri':1,'xhat_out':0,'mu_out':1,'rstd_out':1,'dGate':1},'b7_joint':{'launches':1,'x_n':1,'dLeft':1,'dRight':1,'dGate':1,'raw_x':1,'residual':1}})
(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(SITE/'dist/assets/trimul-current-wiring.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Rendered current FWD/BWD HBM I/O maps for L384 and L768')
