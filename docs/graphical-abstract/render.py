"""Render editable SVG, PNG and PDF figures from the bundled historical evidence.

Run on a CPU compute node. No CUDA dependency and no benchmark execution.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'evidence.json').read_text())
BG, INK, MUTED, LINE = '#F5F4EF', '#162E3D', '#5C6D75', '#D6DDDA'
TEAL, BLUE, AMBER, VIOLET, RED = '#087F78', '#315CA8', '#C27A20', '#8066AC', '#B85550'
LIGHT = {TEAL: '#E1F0EA', BLUE: '#E7EDF8', AMBER: '#FAEED9', VIOLET: '#EFE9F6', RED: '#F8E8E3'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'savefig.facecolor': BG, 'figure.facecolor': BG})
FIGURES = []


def canvas(h=1080):
    fig = plt.figure(figsize=(18, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1800); ax.set_ylim(h, 0); ax.axis('off')
    return fig, ax


def text(ax, x, y, s, size=13, color=INK, weight='normal', ha='left', va='top', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                   linespacing=1.4, **kw)


def box(ax, x, y, w, h, label='', color=TEAL, fill=None, dashed=False, size=12, lw=1.35):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=10',
                       linewidth=lw, edgecolor=color, facecolor=fill or LIGHT.get(color, 'white'),
                       linestyle=(0, (4, 3)) if dashed else '-')
    ax.add_patch(p)
    if label:
        text(ax, x + w / 2, y + h / 2, label, size, color=INK, ha='center', va='center')
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, dashed=False, lw=1.6, curve=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                linewidth=lw, color=color, linestyle='--' if dashed else '-',
                                connectionstyle=f'arc3,rad={curve}'))


def header(ax, number, title, subtitle):
    text(ax, 55, 30, f'MINIWORLD ENGINE   /   GRAPHICAL ABSTRACT   /   {number}', 10, TEAL, 'bold')
    text(ax, 55, 65, title, 28, INK, 'bold')
    text(ax, 55, 120, subtitle, 12, MUTED)


def footer(ax, h, note):
    ax.plot([55, 1745], [h-62, h-62], color=LINE, lw=1)
    text(ax, 55, h-44, note, 10, MUTED)
    text(ax, 1745, h-44, 'REVIEW DRAFT  /  27 SEP 2026', 9, MUTED, ha='right')


def save(fig, stem):
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(HERE / f'{stem}.{ext}', dpi=160)
    FIGURES.append(fig)


def fusion():
    fig, ax = canvas(1180)
    header(ax, '01', 'Fuse operations around data reuse',
           'Mathematical stages at left; audited execution boundaries at right. Supported paths are shape / dtype / GPU specific.')
    text(ax, 55, 181, 'OPERATOR', 10, MUTED, 'bold')
    text(ax, 245, 181, 'MATHEMATICAL STAGES  ·  NOT A BASELINE LAUNCH COUNT', 10, MUTED, 'bold')
    text(ax, 870, 181, 'IMPLEMENTED GROUPING', 10, TEAL, 'bold')
    rows = [
        ('TriMul', 'Inference  [C1, C3]', 'LN → input projections / sigmoid gates / mask\n→ contraction → LN / projection / gate / residual',
         [('K1\nLN + proj + gate + mask', 265, TEAL, False), ('cuBLAS\ncontraction(s)', 185, BLUE, False), ('K3\nLN + proj + gate + add', 265, TEAL, False)],
         'A/B and contraction output still cross HBM; bidirectional inference uses two BMM calls.'),
        ('Transition', 'BF16 D128; n=4  [C4]', 'LN → expand A/B → SwiGLU\n→ squeeze → residual',
         [('One fused forward kernel\nLN + expand + SwiGLU + squeeze + residual', 775, TEAL, False)],
         'Training saves normalized input / statistics. Backward also has a separate partial reduction.'),
        ('OPM', 'MSA training  [C5]', 'LN → two projections + mask → outer product\n→ permute / mask-count normalization → output projection',
         [('Triton prologue\nLN + projections + mask', 255, TEAL, False), ('cuBLAS\ngrouped product', 185, BLUE, False), ('CUDA epilogue\nlayout + norm + projection', 275, TEAL, False)],
         'Mask-count normalization is computed separately. Save O or recompute it in backward.'),
        ('PWA', 'MSA training  [C6]', 'Pair: LN / projection / mask / softmax\nMSA: LN / value → weighted sum / gate / output / residual',
         [('Pair weights\nLN + proj + softmax', 235, TEAL, False), ('MSA branch\nLN + value', 185, TEAL, False), ('Weighted sum + gate\noutput + residual', 295, TEAL, False)],
         'First two blocks are parallel input branches, not a serial chain. Training also prepares a dropout mask.'),
        ('LNLinear', 'CuTe SM90 path  [C7]', 'LN statistics → normalized activation\n→ linear projection',
         [('One main kernel\nGEMM + row statistics + affine correction epilogue', 775, VIOLET, False)],
         'Avoids materializing the full normalized activation; weight-side preparation is outside this box.'),
        ('TriangleAttention', 'Dispatch-dependent  [C8, C9]', 'LN / QKV / bias → tiled attention\n→ sigmoid gate → output projection',
         [('Projection path', 240, VIOLET, True), ('Tiled attention', 220, VIOLET, True), ('Gated output path', 255, VIOLET, True)],
         'Dashed boxes denote stages, not single launches. Projection and output fusion depend on dispatch.'),
        ('LayerNorm\nRMSNorm', 'Standalone primitives', 'Row statistics / reduction\n→ normalization + optional affine',
         [('Standalone normalization\nrow reduction + normalization / affine', 775, BLUE, True)],
         'A primitive optimization is not evidence of cross-operator fusion. Backward may require partial reductions.'),
    ]
    for i, (name, scope, mathline, blocks, note) in enumerate(rows):
        y = 218 + i*121
        if i % 2 == 0:
            box(ax, 43, y-8, 1710, 113, color=LINE, fill='#FFFFFF', lw=0)
        text(ax, 57, y+4, name.replace('TriangleAttention', 'Triangle\nAttention'), 16, INK, 'bold')
        text(ax, 57, y+58, scope, 9, MUTED)
        text(ax, 245, y+7, mathline, 11)
        x=870
        for j,(label,w,c,dashed) in enumerate([] if name == 'PWA' else blocks):
            box(ax,x,y,w,58,label,c,dashed=dashed,size=11)
            if j and name != 'PWA': arrow(ax,x-24,y+29,x-5,y+29,AMBER)
            x += w+30
        if name == 'PWA':
            box(ax,870,y,335,26,'Pair: LN + projection + softmax',TEAL,size=10)
            box(ax,870,y+32,335,26,'MSA: LN + value projection',TEAL,size=10)
            box(ax,1250,y,395,58,'Weighted sum + gate\noutput + residual',TEAL,size=11)
            arrow(ax,1208,y+13,1246,y+21,AMBER)
            arrow(ax,1208,y+45,1246,y+37,AMBER)
        text(ax, 870, y+70, note, 9.1, MUTED)
    text(ax,55,1082,'SOLID BORDER  audited main kernel / library-call group     DASHED BORDER  execution stage     AMBER ARROW  retained global-memory boundary',10,MUTED)
    footer(ax,1180,'Setup, weight packing, mask preparation and reductions are shown only where called out. No all-operator launch-count claim.')
    save(fig,'01-fusion-map')


def pipeline(ax,x,y,w=1000,compact=False):
    size=10 if compact else 12
    box(ax,x,y,w,260,color=LINE,fill='white')
    text(ax,x+20,y+15,'INSIDE A K1 / K3 CTA  ·  schematic, not a timing trace',size,TEAL,'bold')
    labels=[('HBM\ninput + weights',150,AMBER),('Producer\nTMA loads',155,BLUE),
            ('Shared-memory ring\nready / released slots',210,AMBER),
            ('Consumers\nLN + WGMMA + epilogue',245,TEAL)]
    widths=[a[1] for a in labels]; scale=(w-80)/sum(widths)
    px=x+20
    for label,ww,c in labels:
        ww*=scale
        box(ax,px,y+52,ww,73,label,c,size=size)
        if px>x+20: arrow(ax,px-17,y+87,px-2,y+87)
        px+=ww+14
    text(ax,x+22,y+150,'TRANSFER',9,MUTED,'bold'); text(ax,x+22,y+194,'COMPUTE',9,MUTED,'bold')
    start=x+130; step=(w-160)/4
    for i in range(3):
        box(ax,start+i*step,y+145,step-12,30,f'load {i}',BLUE,size=10,lw=0)
        box(ax,start+(i+0.65)*step,y+188,step-12,30,f'compute {i}',TEAL,size=10,lw=0)
    text(ax,x+w-20,y+237,'Slot depth and consumer count vary by config.',9,MUTED,ha='right')


def algorithm():
    fig,ax=canvas(1220)
    header(ax,'02','TriMul: contract globally, reuse locally',
           'A channel-wise contraction connects two fused endpoints. Training adds explicit saved tensors and a separate backward pipeline.')
    text(ax,55,180,'A  /  THE MATHEMATICS',12,TEAL,'bold')
    box(ax,55,216,650,185,color=LINE,fill='white')
    text(ax,78,232,'Outgoing',13,TEAL,'bold')
    text(ax,78,269,r'$T_{ijc}=\sum_k A_{ikc}\,B_{jkc}$',25)
    text(ax,78,367,'Sum over k; one pair matrix for each channel c.',12,MUTED)
    box(ax,731,216,650,185,color=LINE,fill='white')
    text(ax,754,232,'Incoming',13,BLUE,'bold')
    text(ax,754,269,r'$T_{ijc}=\sum_k A_{kic}\,B_{kjc}$',25)
    text(ax,754,367,'The contracted axis changes; the model semantics stay explicit.',11,MUTED)
    for offset,labels in [(0,('A','Bᵀ','T')),(676,('Aᵀ','B','T'))]:
        for j,label in enumerate(labels):
            gx=420+offset+j*85
            text(ax,gx+27,258,label,11,MUTED,ha='center')
            for row in range(6):
                for col in range(6):
                    active=(row==2 if j==0 else col==3 if j==1 else row==2 and col==3)
                    ax.add_patch(Rectangle((gx+col*9,283+row*9),9,9,
                                 facecolor=[TEAL,BLUE,AMBER][j] if active else '#EDF0ED',
                                 edgecolor='white',linewidth=.6))
            if j<2:text(ax,gx+68,310,'×' if j==0 else '→',12,MUTED,ha='center',va='center')
    box(ax,1407,216,338,185,'Bidirectional path\nseparate channel groups\nfor the two directions',VIOLET,size=14)
    text(ax,55,438,'B  /  THE EXECUTION BOUNDARIES',12,TEAL,'bold')
    blocks=[(55,310,'K1  ·  fused prologue\nLN + projections + gates + mask',TEAL),
            (440,235,'A / B planes\nHBM',AMBER), (750,280,'cuBLAS contraction(s)\nfull reduction over k',BLUE),
            (1105,225,'T planes\nHBM',AMBER),(1405,340,'K3  ·  fused epilogue\nLN + projection + gate + residual',TEAL)]
    for j,(x,w,label,c) in enumerate(blocks):
        box(ax,x,476,w,83,label,c,size=13)
        if j: arrow(ax,blocks[j-1][0]+blocks[j-1][1]+8,518,x-8,518,AMBER,lw=2)
    text(ax,55,578,'These HBM intermediates are retained in the packaged inference path. Kernel fusion does not remove the contraction boundary.',12,MUTED)
    pipeline(ax,55,626,1070)
    box(ax,1155,626,590,260,color=LINE,fill='white')
    text(ax,1178,647,'WHY STOP FUSING HERE?',13,VIOLET,'bold')
    text(ax,1178,693,'• Contraction consumes tiles across the k axis.\n• CTA-local storage is finite.\n• More fusion can raise register / smem pressure.\n• A different boundary requires a different algorithm.',13)
    text(ax,1178,851,'The diagram shows current boundaries, not an impossibility proof.',10,MUTED)
    text(ax,55,925,'C  /  TRAINING TENSOR LIFETIMES  ·  D128 PACKAGED PATH',12,TEAL,'bold')
    box(ax,55,967,395,120,'Forward saves\nA/B · T · normalized x · LN stats\n+ per-forward packed weights',VIOLET,size=13)
    box(ax,505,967,370,120,'Output-side backward (B1)\nreads T / stats / normalized x\nproduces dT + gate gradient',TEAL,size=13)
    box(ax,930,967,330,120,'Contraction backward\nreads A/B and dT\nproduces dA / dB',BLUE,size=13)
    box(ax,1315,967,430,120,'Input-side backward (B7)\nreads dA / dB + gate gradient\nuses normalized x; returns dx / dW',TEAL,size=13)
    for left,right in [(450,505),(875,930),(1260,1315)]: arrow(ax,left+6,1027,right-6,1027,VIOLET,dashed=True)
    text(ax,55,1110,'Saved buffers live across forward and backward; gradients and partial reductions may add launches. D256+ research checkpoints are separate.',11,MUTED)
    footer(ax,1220,'Evidence: h100_inference.py [C1], h100_training.py [C2], tmn_kernels.cuh [C3]. Consumer count depends on tile shape.')
    save(fig,'02-trimul-algorithm')


ROWS=[('TriMul','D128','D128  ·  T'),('Transition','D128; expansion 4','D128, n=4  ·  X'),
      ('OPM','S1024; save O','S1024, save O  ·  O'),('PWA','S1024; D_msa64 / D_pair128','S1024  ·  P'),
      ('RMSNorm','D128','D128  ·  N'),('LayerNorm','D384','D384  ·  N'),
      ('LNLinear','D128 to 16','D128 → 16; portable path  ·  N'),('TriangleAttention',None,'Matched comparison not selected')]


def selected(op,shape,phase):
    return [r for r in DATA['records'] if r['op']==op and r['shape']==shape and r['phase']==phase
            and r['length'] in (384,768)]


def ratio_plot(ax,x,y,w,rows,phase,row_h=72,title=''):
    text(ax,x,y-40,title,16,INK,'bold')
    lo,hi=0.8,12
    pos=lambda v:x+math.log(v/lo)/math.log(hi/lo)*w
    for value in (1,1.5,2,3,5,10):
        ax.plot([pos(value)]*2,[y,y+row_h*len(rows)-8],color=LINE if value!=1 else MUTED,
                lw=1.2 if value==1 else .7,ls='--' if value==1 else '-')
        text(ax,pos(value),y-18,f'{value:g}×',9,MUTED,ha='center')
    for i,(op,shape,_) in enumerate(rows):
        rs=selected(op,shape,phase)
        cy=y+i*row_h+row_h/2
        if not rs:
            text(ax,x+w/2,cy,'—',15,MUTED,ha='center',va='center')
        for r in rs:
            assert lo <= r['speedup'] <= hi, r
            yy=cy+(-10 if r['length']==384 else 12)
            c=TEAL if r['length']==384 else AMBER
            xx=pos(r['speedup'])
            ax.plot([pos(1),xx],[yy,yy],color=c,alpha=.4,lw=2)
            ax.plot(xx,yy,'o' if r['length']==384 else 'D',color=c,ms=6)
            digits=3 if abs(r['speedup']-1)<.02 else 2
            text(ax,xx+10,yy,f'{r["speedup"]:.{digits}f}×',10,c,'bold',va='center')


def performance():
    fig,ax=canvas(1310)
    header(ax,'03','Measure the full operator, not just the fast stage',
           'Historical H100 records • speedup = baseline time / our time • each source keeps its own baseline • no new GPU measurements')
    box(ax,55,170,1690,55,color=LINE,fill='white')
    text(ax,76,188,'●  L384',12,TEAL,'bold'); text(ax,210,188,'◆  L768',12,AMBER,'bold')
    text(ax,360,188,'Shared log axis    /    1× = unchanged    /    >1× = faster    /    — = unavailable',12,MUTED)
    y=293;rh=72
    for i,(op,shape,label) in enumerate(ROWS):
        cy=y+i*rh
        if i%2==0:box(ax,44,cy,1710,rh,color=LINE,fill='white',lw=0)
        text(ax,60,cy+11,op,15,INK,'bold');text(ax,60,cy+40,label,9,MUTED)
    for x,phase,title in [(420,'train_fwd','TRAINING FORWARD'),(870,'bwd','BACKWARD'),(1320,'fb','FORWARD + BACKWARD')]:
        ratio_plot(ax,x,y,370,ROWS,phase,rh,title)
    text(ax,55,891,'T  existing Triton    X  Triton residual    O/P  own engine MSA paths    N  pre-change portable Triton',10,MUTED)
    text(ax,55,915,'Forward means training forward here. F+B is measured directly; backward is never inferred by subtraction. No aggregate speedup is computed.',10,MUTED)
    text(ax,55,960,'TRIMUL WIDTH SCALING',15,INK,'bold')
    text(ax,55,992,'Research checkpoints for D256+\nare not promoted to shared dispatch.\n\nTeal ≥1.5×; blue <1.5×.',11,MUTED)
    for j,(length,phase,label) in enumerate([(384,'bwd','L384  BWD'),(384,'fb','L384  F+B'),(768,'bwd','L768  BWD'),(768,'fb','L768  F+B')]):
        xx=530+j*300;text(ax,xx+120,965,label,12,MUTED,'bold',ha='center')
        for i,d in enumerate((128,256,384,512)):
            r=next(r for r in selected('TriMul',f'D{d}',phase) if r['length']==length)
            yy=999+i*49
            if j==0:text(ax,460,yy+22,f'D{d}',12,INK,'bold',ha='right',va='center')
            color=TEAL if r['speedup']>=1.5 else BLUE
            box(ax,xx,yy,245,40,f'{r["speedup"]:.2f}×',color,size=13,lw=0)
    footer(ax,1310,'Source keys and exact milliseconds: evidence.json / measurements.csv. Missing cells do not mean no optimization exists.')
    save(fig,'03-performance')


def overview():
    fig,ax=canvas(1120)
    header(ax,'00','GPU-aware fusion for MiniWorld',
           'Keep useful intermediates local. Pipeline transfers with compute. Measure forward and backward together.')
    box(ax,55,174,1025,322,color=LINE,fill='white')
    box(ax,1105,174,640,826,color=LINE,fill='white')
    text(ax,79,194,'A  /  FUSION BOUNDARIES',14,TEAL,'bold')
    text(ax,79,234,'TriMul  ·  separate global contraction',13,INK,'bold')
    xs=[85,355,610,840]; ws=[205,190,165,205]
    for x,w,label,c in zip(xs,ws,['K1\nLN / proj / gate','A/B → contraction','T in HBM','K3\nLN / proj / gate'],[TEAL,BLUE,AMBER,TEAL]):
        box(ax,x,270,w,64,label,c,size=12)
    for j in range(3):arrow(ax,xs[j]+ws[j]+5,302,xs[j+1]-5,302,AMBER)
    text(ax,79,367,'Transition  ·  D128, expansion 4, BF16',13,INK,'bold')
    box(ax,85,406,960,60,'One forward kernel: LN → expand → SwiGLU → squeeze → residual',TEAL,size=15)
    text(ax,1127,194,'C  /  MEASURED OPERATOR SPEEDUP',14,TEAL,'bold')
    text(ax,1127,232,'Historical F+B, L384 · family-specific baselines',10,MUTED)
    for i,(op,shape,label) in enumerate(ROWS[:-1]):
        y=290+i*88
        r=next(r for r in selected(op,shape,'fb') if r['length']==384)
        text(ax,1127,y,op,14,INK,'bold')
        text(ax,1722,y,f'{r["speedup"]:.2f}×',20,TEAL,'bold',ha='right')
        text(ax,1127,y+30,f'{r["baseline_ms"]:.3f} → {r["ours_ms"]:.3f} ms  |  {label}',10,MUTED)
        ax.plot([1127,1722],[y+65]*2,color=LINE,lw=.8)
    text(ax,1127,934,'Ratios are not a whole-model speedup.\nPortable Norm results do not measure CUDA fusion.',11,MUTED)
    text(ax,55,530,'B  /  ALGORITHM AND DATA LIFETIME',14,TEAL,'bold')
    pipeline(ax,55,570,1025,compact=True)
    box(ax,55,855,300,145,'SAVE FOR BACKWARD\nA/B, T, normalized x\nand LN statistics',VIOLET,size=13)
    arrow(ax,364,927,413,927,VIOLET,dashed=True)
    box(ax,423,855,300,145,'REUSE IN BACKWARD\noutput-side gradients\n→ contraction gradients',TEAL,size=13)
    arrow(ax,731,927,780,927,VIOLET,dashed=True)
    box(ax,790,855,290,145,'FINISH THE CHAIN\ninput-side gradients\n+ required reductions',BLUE,size=13)
    text(ax,55,1029,'Implementation snapshot: afd54410. Historical checkpoints and baseline identities are documented in the companion figures and evidence bundle.',11,MUTED)
    footer(ax,1120,'Shape-specific SM90 paths. Schematics are not measured launch timelines. v2.1 GPU requalification and missing operator comparisons remain open.')
    save(fig,'00-graphical-abstract')


def exports():
    import csv
    fields=['op','shape','length','phase','baseline_ms','ours_ms','speedup','baseline','status','note','sources']
    with (HERE/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in DATA['records']:writer.writerow({**row,'sources':'|'.join(row['sources'])})
    with PdfPages(HERE/'miniworld-graphical-abstract.pdf') as pdf:
        for fig in [FIGURES[3],*FIGURES[:3]]:pdf.savefig(fig)
        assert pdf.get_pagecount()==4
    template=(HERE/'index.template.html').read_text()
    assert template.count('__EVIDENCE_JSON__')==1
    (HERE/'index.html').write_text(template.replace('__EVIDENCE_JSON__',json.dumps(DATA).replace('</','<\\/')))
    print('Rendered 4 figures as SVG / PNG / PDF; exported a 4-page PDF and source measurement CSV.')


if __name__=='__main__':
    fusion();algorithm();performance();overview();exports()
