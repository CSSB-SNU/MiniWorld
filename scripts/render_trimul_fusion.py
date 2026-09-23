from pathlib import Path
import json, subprocess, html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/trimul-fusion'
COLORS={'triton':('#2868ba','#eef5ff'),'cute':('#168269','#eaf8f2'),'blas':('#8655ad','#f7f0ff'),'compiler':('#727e86','#f3f5f6')}
AUDIT=json.loads((OUT/'tiling-audit.json').read_text())
EVIDENCE=ROOT/'runs/trimul_sm90_round2_20260917'
FINAL=json.loads((EVIDENCE/'module/final-evidence.json').read_text())
F567_FINAL=json.loads((EVIDENCE/'f567/final-results.json').read_text())

def measurement(kind,L):
 values={'front':{384:(220.238,183.078),768:(876.068,737.801)},
         'f567':{384:(105.586,95.373),768:(410.061,360.106)},
         'dual':{384:(141.449,132.717),768:(534.529,510.572)}}
 t,c=values[kind][L]
 if kind=='f567':
  row=F567_FINAL[str(L)]['medians_ms'];t=row['triton_bm128']*1000;c=row['cute']*1000
 return f'\nTriton {t:.3f} → CuTe {c:.3f} µs · {t/c:.3f}×'

def scope_title(title,L):
 v=json.loads((ROOT/f'runs/trimul_sm90_dropout_20260917/graph/graph-L{L}.json').read_text())
 assert v['passed'] and v['dropout']==0.25
 t=v['aggregate_medians_ms']['triton'];c=v['aggregate_medians_ms']['sm90']
 return (title+f' · L{L} · BF16 B1 D=h=128\n'
         +f'기준: {FINAL["commit"][:8]} · 동일 융합 / 저장값 · H100 경로는 개발 checkout의 선택 옵션\n'
         +f'전체 학습 FWD+BWD: {t:.3f} → {c:.3f} ms ({t/c:.3f}×) · compile + CUDA graph · dropout=0.25/RNG 포함\n'
         +'청색 Triton / 녹색 CuTe TMA+WGMMA / 보라 cuBLAS / 노랑 HBM · 색상 외곽선 = 커널 경계\n'
         +'아래 config는 실측 후보이며 shape 하드코딩이나 전체 cache 완료를 뜻하지 않음')


def tiling_label(name,L):
 a=AUDIT[name]['axes']
 values=lambda key:'/'.join(map(str,a[key]))
 if name=='B11+B12':
  return '\nCSV: M '+values('BLOCK_M1')+' · K '+values('BLOCK_K')+'\nwarps '+values('num_warps')+' · stages '+values('num_stages')+'\nrow/column reduction · GROUP_M 없음'
 return ('\nCSV: M '+values('BLOCK_M1')+' · N '+values('BLOCK_N')+' · K '+values('BLOCK_K')+
         '\nGROUP_M '+values('GROUP_M')+' · warps '+values('num_warps')+' · stages '+values('num_stages')+
         '\nKP/KG 길이 독립 · 물리 K 타일 공유')

class Graph:
 def __init__(self,name,title):
  self.clusters={}
  self.name=name;self.n=0;self.parts=['digraph G {','graph [rankdir=TB,compound=true,newrank=true,bgcolor="white",pad="0.25",nodesep="0.20",ranksep="0.32",fontname="sans-serif",fontsize=16,label='+json.dumps(title,ensure_ascii=False)+',labelloc=t];','node [shape=box,style="rounded,filled",fillcolor="white",color="#b6c4cf",fontname="sans-serif",fontsize=11,margin="0.10,0.07"];','edge [color="#566d7b",arrowsize=0.65,fontname="sans-serif",fontsize=9];']
 def q(self,s):return json.dumps(s,ensure_ascii=False)
 def node(self,id,label,shape='box',color='#b6c4cf',fill='white'):
  self.parts.append(f'{id} [label={self.q(label)},shape={shape},color="{color}",fillcolor="{fill}"];');return id
 def edge(self,a,b,label='',saved=False):
  target=self.clusters.get(b)
  clip=f',lhead=cluster_{target}' if not saved and target and target!=self.clusters.get(a) else ''
  self.parts.append(f'{a} -> {b} [label={self.q(label)}'+(',style=dashed,color="#84929a",constraint=false' if saved else '')+clip+'];')
 def buffer(self,id,label):return self.node(id,label,'cylinder','#b78b43','#fff7df')
 def input(self,id,label):return self.node(id,label,'oval','#88989d','#f5f7f8')
 def kernel(self,id,title,kind,ops,links=None,multi=False):
  c,b=COLORS[kind];self.parts.append(f'subgraph cluster_{id} {{ label={self.q(title)}; color="{c}"; fillcolor="{b}"; style="'+('rounded,dashed,filled' if multi else 'rounded,filled')+'"; penwidth=2; fontsize=11; fontname="monospace"; margin=12;')
  ids=[]
  for i,op in enumerate(ops):
   ids.append(self.node(id+str(i),op,color=c))
   self.clusters[ids[-1]]=id
  for a,z in (links if links is not None else list(zip(range(len(ops)-1),range(1,len(ops))))):self.edge(ids[a],ids[z])
  self.parts.append('}');return ids
 def write(self):
  dot='\n'.join(self.parts+['}']);(OUT/(self.name+'.dot')).write_text(dot)
  subprocess.run(['dot','-Tsvg',str(OUT/(self.name+'.dot')),'-o',str(OUT/(self.name+'.svg'))],check=True)
  return self.name+'.svg'

def forward(route,L=384):
 parity=route=='parity';new=route in ('new','current');cute=route!='triton';title={'parity':'최신 H100 · Forward · 동일 Triton 융합', 'triton':'Triton 기준 · Forward','h100':'이전 H100 경로 · Forward · 비교용','current':'현재 H100 · Forward · F567 분리','new':'CuTe F567 개발 경로 · Forward · 설치 미적용'}[route]
 g=Graph('forward-'+route+('-L768' if L==768 else ''),scope_title(title,L));g.input('pair','pair');g.input('mask','pair mask');g.input('ds','dropout scale\n난수/scale 생성은 별도')
 ln=g.kernel('ln','F1 · layer_norm_fwd_fused\nTriton · 1 kernel','triton',['μ = mean(pair), rstd = rsqrt(var + ε)','x_n = (pair − μ) × rstd × γin + βin']);g.edge('pair',ln[0]);g.buffer('xn','x_n [M,128] · HBM');g.edge(ln[-1],'xn')
 front=g.kernel('front','F2 · '+('FrontSingleWarpgroupSm90' if parity else 'MaskedGatedSm90' if cute else '_bidir_front_kernel')+'\n'+('CuTe' if cute else 'Triton')+' · 1 kernel'+(measurement('front',L)+'\nM128 / BK64 / BH32 / warps4 / stages2\nA 재사용 · 다음 B TMA ↔ 현재 epilogue\noutput TMA ↔ 다음 GEMM · BF16 packed shuffle' if parity else ''),'cute' if cute else 'triton',['4 projections: pL=x_n·WL, gL=x_n·WLg\npR=x_n·WR, gR=x_n·WRg','σ(gL), σ(gR)','left = BF16(pL × σ(gL)) × mask\nright = BF16(pR × σ(gR)) × mask']);g.edge('xn',front[0]);g.edge('mask',front[-1],saved=True)
 g.buffer('lr','left / right [256,L,L] · HBM');g.buffer('pre','raw pL,gL,pR,gR · HBM\nbackward용 저장');g.edge(front[-1],'lr');g.edge(front[0],'pre','store',True);g.parts.append('{rank=same;lr;pre;}')
 backend='cute' if cute and not parity else 'blas';kn='GemmDefaultSm90' if cute and not parity else 'cuBLAS bmm'
 co=g.kernel('co','F3a · '+kn+'\n별도 GEMM 호출',backend,['tri[:h] = left[:h] @ right[:h]ᵀ'])
 ci=g.kernel('ci','F3b · '+kn+'\n별도 GEMM 호출',backend,['tri[h:] = left[h:]ᵀ @ right[h:]'])
 g.edge('lr',co[0]);g.edge('lr',ci[0]);g.buffer('tri','tri [256,L,L] · HBM\n두 GEMM이 최종 buffer의 다른 slice에 직접 기록\ncat kernel 없음');g.edge(co[-1],'tri');g.edge(ci[-1],'tri')
 lo=g.kernel('lo','F4 · _ln_mat_kernel\nTriton · 1 kernel','triton',['μout, rstdout 계산','xhat = (tri − μout) × rstdout']+([] if new else ['xn_out = xhat × γout + βout']));g.edge('tri',lo[0]);g.buffer('norm',('xhat' if new else 'xn_out')+' [M,256] · HBM\nμout, rstdout도 저장');g.edge(lo[-1],'norm')
 if route in ('triton','new','parity'):
  if new:
   fold=g.kernel('fold','별도 small-weight 준비 · Inductor','compiler',['Wfold = Wp × γout\nbfold = Σk(Wp × βout)'],multi=True)
  fused=g.kernel('f567','F5 + F6 + F7 · '+('ParityF567Sm90' if parity else 'F567Sm90 (신규 검증 경로)' if new else '_output_f567_kernel')+'\n'+('CuTe TMA + WGMMA' if new or parity else 'Triton')+' · 1 kernel'+(measurement('f567',L)+'\nM64 / N64 / K64 / GROUP_M1 / warps4 / stages2\n초기 projection 입력 L2 prefetch ↔ gate GEMM\nprojection TMA ↔ FP32 sigmoid(BF16(logit))\nP/G/Y와 분리된 네 번째 retired tile에 dropout TMA\n공간·정렬 부족 시 기존 읽기 · 추가 shared/HBM 없음' if parity else '' if new else tiling_label('F567',L)), 'cute' if new or parity else 'triton',
   [('proj = xhat @ Wfoldᵀ + bfold' if new else 'proj = xn_out @ Wpᵀ'),'glogit = x_n @ Wg','gate = sigmoid(BF16(glogit))','v = BF16(proj) × gate','v = v × dropout_scale','y = pair + v'],links=[(1,2),(0,3),(2,3),(3,4),(4,5)])
  g.edge('norm',fused[0]);g.edge('xn',fused[1],'x_n 재사용',True)
  if new:g.edge(fold[-1],fused[0])
  g.edge('ds',fused[4],saved=True);g.edge('pair',fused[5],'residual',True)
  g.buffer('saved','proj / gate [M,128] · HBM\nbackward용 store만 · fwd 재읽기 없음')
  g.edge(fused[0],'saved','proj 저장',True);g.edge(fused[2],'saved','BF16 gate 저장',True)
  g.buffer('y','y [M,128] · HBM');g.edge(fused[-1],'y')
  g.node('note',('F2는 목표 달성 · F567 L384 미달 / L768은15% 경계\nF567: dropout LSU global-load sectors→0 (TMA로 이동)\nNCU L768 G4: L2 92.41%, DRAM86.23% · HBM bytes 거의 동일\n현재 N64 선택 포함: registers92 / shared41,984B / spill0\nF2 측정은 이전 검증 유지 · F567은 최신 반복 측정' if parity else 'glogit HBM buffer 제거\n중간 proj의 fwd HBM 재읽기 제거'),'note')
  g.parts.append('{rank=same;saved;y;}')
  g.parts.append('y -> note [style=invis];')
  return g.write()
 if new:
  fold=g.kernel('fold','별도 weight 준비 · Inductor\nGEMM과 융합되지 않음','compiler',['Wfold = Wp × γout\nbfold = Σk(Wp × βout)'],multi=True)
  p=g.kernel('proj','F5 · cuBLAS linear (bias epilogue)\n별도 GEMM 호출','blas',['proj = xhat @ Wfoldᵀ + bfold']);g.edge(fold[-1],p[0])
 else:p=g.kernel('proj','F5 · cuBLAS linear\n별도 GEMM 호출','blas',['proj = xn_out @ Wpᵀ'])
 g.edge('norm',p[0]);g.buffer('projbuf','proj [M,128] · HBM');g.edge(p[-1],'projbuf')
 gg=g.kernel('gateG','F6 · cuBLAS matmul\n별도 GEMM 호출','blas',['glogit = x_n @ Wg']);g.edge('xn',gg[0],'x_n 재사용',True);g.buffer('glogit','glogit [M,128] · HBM');g.edge(gg[-1],'glogit')
 ew=g.kernel('gateEW','F7 · _gate_mul_train_kernel\nTriton · 1 kernel','triton',['gate = sigmoid(glogit)','v = proj × gate','v = v × dropout_scale','y = pair + v']);g.edge('projbuf',ew[1]);g.edge('glogit',ew[0]);g.edge('ds',ew[2],saved=True);g.edge('pair',ew[3],'residual',True)
 g.buffer('y','y · HBM\ngate도 backward용 저장');g.edge(ew[-1],'y');return g.write()

def backward(route,L=384):
 parity=route=='parity';new=route=='new';cute=route!='triton';g=Graph('backward-'+route+('-L768' if L==768 else ''),scope_title({'parity':'최신 H100 · Backward · 동일 Triton 융합','triton':'Triton 기준 · Backward','h100':'이전 H100 경로 · Backward · 비교용','new':'현재 설치된 H100 · Backward'}[route],L));g.input('dy','gy + 저장된 forward buffers')
 ew=g.kernel('gew','B1 · _gate_elem_bwd_ew_kernel\nTriton · 1 kernel','triton',['v = gy × dropout_scale','dproj = v × gate\ndglogit = v × proj × gate × (1 − gate)']);g.edge('dy',ew[0]);g.buffer('dp','dproj [M,128] · HBM');g.buffer('dgl','dglogit [M,128] · HBM');g.edge(ew[-1],'dp');g.edge(ew[-1],'dgl')
 dwg=g.kernel('dwg','B2 · cuBLAS mm\n별도 GEMM 호출','blas',['dWg = x_nᵀ @ dglogit']);g.edge('dgl',dwg[0])
 if new:
  prep=g.kernel('prep','별도 small-weight 연산 · Inductor\n복수 생성 커널 가능','compiler',['s = Σk(Wp × γout)\nb2 = Σk(Wp × βout)'],multi=True)
  cr=g.kernel('cr','B3 · Inductor 생성 Triton reduction\n…div_mul_sub_sum…rows_dgrad…','triton',['c1 = Σn[dproj × (proj − b2)] / 256\nc2 = Σn[dproj × s] / 256']);g.edge('dp',cr[0]);g.edge(prep[-1],cr[0]);g.buffer('corr','c1, c2 [M] · HBM');g.edge(cr[-1],'corr')
  dx=g.kernel('outdx','B4 · _DgradLNRowsSm90\nCuTe · 1 kernel','cute',['acc = dproj @ Wp  [GEMM]','acc = acc × γout','acc = acc − c2 − xhat × c1','dtri = rstdout × acc  [epilogue]']);g.edge('dp',dx[0]);g.edge('corr',dx[2]);g.buffer('dtri','dtri [256,L,L] · HBM\nM-major 직접 store · dx_normed buffer 없음');g.edge(dx[-1],'dtri')
  tw=g.kernel('tw','B5a · cuBLAS mm\n별도 GEMM 호출','blas',['T = dprojᵀ @ xhat']);g.edge('dp',tw[0]);g.buffer('T','T [128,256] · HBM');g.edge(tw[-1],'T')
  tail=g.kernel('tail','B5b · Inductor reductions + GEMV\n여러 커널 / GEMM과 미융합','compiler',['db = Σm dproj','dWp = γout × T + db ⊗ βout\ndγout = Σn(Wp × T)\ndβout = db @ Wp'],multi=True);g.edge('T',tail[1]);g.edge('dp',tail[0])
 else:
  dx=g.kernel('outdx','B3a · cuBLAS matmul\n별도 GEMM 호출','blas',['dx_normed = (Wpᵀ @ dprojᵀ)ᵀ']);g.edge('dp',dx[0]);g.buffer('dxnorm','dx_normed [M,256] · HBM\n큰 중간 gradient buffer');g.edge(dx[-1],'dxnorm')
  dw=g.kernel('dw','B3b · cuBLAS matmul\n별도 GEMM 호출','blas',['dWp = dprojᵀ @ xn_out']);g.edge('dp',dw[0])
  lo=g.kernel('lob',('B4 · _ln_bwd_kernel' if L==384 else 'B4 · _ln_bwd_persistent')+'\nTriton · 1 kernel','triton',['xhat 재계산; u = dx_normed × γout','c1 = mean(u × xhat), c2 = mean(u)','dtri = rstdout × (u − c2 − xhat × c1)',('dγout / dβout atomic reduction' if L==384 else 'dγout / dβout partial sums 기록')]);g.edge('dxnorm',lo[0]);g.buffer('dtri','dtri [256,L,L] · HBM');g.edge(lo[-1],'dtri')
  if L==768:
   g.buffer('partial','pdγ / pdβ [NP,256] · HBM');g.edge(lo[-1],'partial')
   for name,sym in [('rg','γ'),('rb','β')]:
    r=g.kernel(name,'B4-final · ATen reduce_kernel\n각각 별도 sum 호출','compiler',[f'd{sym}out = sum(partial d{sym}, dim=0)']);g.edge('partial',r[0])

 backend='cute' if cute and not parity else 'blas';kn='GemmDefaultSm90' if cute and not parity else 'cuBLAS bmm'
 formulas=['dleft[:h] = dtri[:h] @ right[:h]','dright[:h] = dtri[:h]ᵀ @ left[:h]','dleft[h:] = right[h:] @ dtri[h:]ᵀ','dright[h:] = left[h:] @ dtri[h:]']
 contractions=[]
 for i,formula in enumerate(formulas):
  nodes=g.kernel('cb'+str(i),'B6'+chr(97+i)+' · '+kn+'\n별도 GEMM 호출',backend,[formula]);g.edge('dtri',nodes[0]);contractions+=nodes
 g.buffer('dlr','dleft / dright · HBM\n4 GEMM이 두 최종 buffer의 slice에 직접 기록\nbackward cat 2회 없음')
 for n in contractions:g.edge(n,'dlr')
 dc=g.kernel('dc','B7 · _dconcat_kernel\nTriton · 1 kernel','triton',['dleft/right × pair_mask','gL/R = sigmoid(saved gL/R logits)','dpL/R = dleft/right × gL/R\ndgL/R = dleft/right × pL/R × gL/R × (1−gL/R)','4 gradient를 dconc에 packed store']);g.edge('dlr',dc[0]);g.buffer('dconc','dconc [1024,M] · HBM\n이 buffer는 존재함');g.edge(dc[-1],'dconc')
 wgrad=g.kernel('wgrad','B8 · cuBLAS mm\n4 입력 weight gradient를 묶은 GEMM','blas',['dWs = dconc @ x_n']);g.edge('dconc',wgrad[0]);g.node('dws','dWL,dWLg,dWR,dWRg\nslice / transpose / copy 별도','note');g.edge(wgrad[-1],'dws')
 if route in ('triton','parity'):
  dx=g.kernel('dual','B9 + B10 · '+('DualBackwardSm90\nCuTe TMA + WGMMA · 1 kernel'+measurement('dual',L)+'\nM64 / N128 / K64 / GROUP_M1 / warps4 / stages3\n사용 완료 gate stage에 front tile TMA prefetch\nWGMMA wait + CTA barrier로 reuse 보호\n추가 shared / HBM allocation 없음' if parity else '_input_dual_bwd_kernel\nTriton · 1 kernel'+tiling_label('B9+B10',L)),'cute' if parity else 'triton',['gate_grad = BF16(dglogit @ Wgᵀ)','front_grad = dconcᵀ @ W_stack','dx_n = front_grad + gate_grad'],links=[(0,2),(1,2)])
  g.edge('dgl',dx[0],'gate 분기',True);g.edge('dconc',dx[1]);g.buffer('dxn','dx_n [M,128] · HBM\ndx_gate 중간 buffer 제거');g.edge(dx[-1],'dxn')
  li=g.kernel('lib','B11 + B12 · _ln_bwd_residual_kernel\nTriton · 1 kernel'+tiling_label('B11+B12',L),'triton',['입력 LN xhat 재계산; dx_n × γin','row c1/c2 reduction → dpair_LN\ndγin / dβin atomic reduction','dpair = BF16(dpair_LN) + gy\nresidual은 LN 미분 후에만 더함'])
  g.edge('dxn',li[0]);g.edge('dy',li[2],'residual identity gradient',True);g.input('end','dpair + 모든 parameter gradients');g.edge(li[-1],'end')
  if parity:
   g.node('bnote','B9+B10은 1.15× 목표 미달\nNCU L384: DRAM84.30%, L2 79.92% of peak · TMA 입력 대기가 큼\n137 registers / spill0 · 대역폭의 절대 한계를 입증한 수치는 아님\nB7 dconcat은 GLU 미분 + gradient packing이며 필요한 연산\noutput / 모든 gradient 검증 통과: L384, L768','note');g.parts.append('end -> bnote [style=invis];')
  return g.write()
 gx=g.kernel('gx','B9 · cuBLAS mm\n별도 GEMM 호출','blas',['dx_gate = dglogit @ Wgᵀ']);g.edge('dgl',gx[0],'gate 분기',True);g.buffer('dxgate','dx_gate [M,128] · HBM');g.edge(gx[-1],'dxgate')
 fx=g.kernel('fx','B10 · cuBLAS addmm_\nGEMM + C-add epilogue','blas',['acc = dconcᵀ @ W_stack','dx_n = acc + dx_gate\n같은 dx_gate buffer에 in-place 기록']);g.edge('dconc',fx[0]);g.edge('dxgate',fx[1]);g.buffer('dxn','dx_n [M,128] · HBM');g.edge(fx[-1],'dxn')
 li=g.kernel('lib','B11 · layer_norm_bwd_dx_fused\nTriton · 1 kernel','triton',['입력 LN xhat 재계산; dx_n × γin','row c1/c2 reduction → dpair_LN\ndγin / dβin atomic reduction']);g.edge('dxn',li[0]);g.buffer('dln','dpair_LN · HBM');g.edge(li[-1],'dln')
 add=g.kernel('add','B12 · Inductor Triton add_view\nresidual gradient 누적은 별도 kernel','triton',['dpair = dpair_LN + gy']);g.edge('dln',add[0]);g.edge('dy',add[0],'residual identity gradient',True);g.input('end','dpair + 모든 parameter gradients');g.edge(add[-1],'end');return g.write()


def main():
 import xml.etree.ElementTree as ET
 ns='http://www.w3.org/2000/svg'
 ET.register_namespace('',ns)
 dims={}
 for phase,render in [('forward',forward),('backward',backward)]:
  for L in (384,768):
   graphs=[]
   for route in ('triton','parity'):
    name=render(route,L)
    node=ET.parse(OUT/name).getroot()
    vb=[float(v) for v in node.attrib['viewBox'].split()]
    dims[name[:-4]]=vb[2]
    if route in ('triton','parity'):
     graphs.append((node,vb[2],vb[3]))
   width=sum(g[1] for g in graphs)+40*(len(graphs)-1);height=max(g[2] for g in graphs)
   canvas=ET.Element('{'+ns+'}svg',width=f'{width}pt',height=f'{height}pt',viewBox=f'0 0 {width} {height}')
   ET.SubElement(canvas,'{'+ns+'}rect',x='0',y='0',width=str(width),height=str(height),fill='white')
   for index,(node,_,_) in enumerate(graphs):
    for elem in node.iter():
     if 'id' in elem.attrib:elem.set('id',f'lane{index}_'+elem.attrib['id'])
   x=0
   for node,w,h in graphs:
    node.set('x',str(x));node.set('y','0');node.set('width',str(w));node.set('height',str(h));canvas.append(node);x+=w+40
   ET.ElementTree(canvas).write(ROOT/('TRIMUL_'+phase.upper()+('_L768' if L==768 else '')+'.svg'),encoding='unicode',xml_declaration=True)
 print('Updated Triton / '+FINAL['commit'][:8]+' H100 parity diagrams; legacy artifacts preserved')


if __name__=='__main__':main()
