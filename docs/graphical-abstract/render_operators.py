"""One standalone figure per operator; render on a CPU compute node."""
import html
import shutil
from render import *

OUT = HERE / 'operators'
OUT.mkdir(exist_ok=True)


def chain(ax, labels, y=255, x=80, width=1640, colors=None, dashed=False):
    gap=38
    w=(width-gap*(len(labels)-1))/len(labels)
    for i,label in enumerate(labels):
        xx=x+i*(w+gap)
        box(ax,xx,y,w,95,label,(colors or [TEAL]*len(labels))[i],size=14,dashed=dashed)
        if i: arrow(ax,xx-gap+6,y+47,xx-6,y+47,AMBER)


def performance_card(ax,op,shape,extra=''):
    box(ax,1120,500,625,380,color=LINE,fill='white')
    text(ax,1145,520,'HISTORICAL MEASUREMENT',13,TEAL,'bold')
    text(ax,1145,558,extra or 'Directly measured full operator F+B',11,MUTED)
    records=selected(op,shape,'fb')
    if not records:
        text(ax,1145,620,'Matched comparison pending',21,INK,'bold')
        text(ax,1145,677,'No speedup assigned to this schematic.',12,MUTED)
    for i,r in enumerate(records):
        yy=615+i*92
        text(ax,1145,yy,f'L{r["length"]}',14,INK,'bold')
        text(ax,1715,yy,f'{r["speedup"]:.2f}×',25,TEAL,'bold',ha='right')
        text(ax,1145,yy+34,f'{r["baseline_ms"]:.3f} → {r["ours_ms"]:.3f} ms',14)
    if records:
        text(ax,1145,820,'Baseline: '+records[0]['baseline'],10,MUTED)
        text(ax,1145,844,'Shape: '+shape,10,MUTED)


def base(index,title,subtitle):
    fig,ax=canvas(1000)
    header(ax,f'OPERATOR {index:02}',title,subtitle)
    text(ax,55,187,'A / EXECUTION AND FUSION',13,TEAL,'bold')
    text(ax,55,459,'B / ALGORITHM AND DATA REUSE',13,TEAL,'bold')
    box(ax,55,500,1035,380,color=LINE,fill='white')
    return fig,ax


def finish(fig,ax,slug,note):
    text(ax,55,910,note,11,MUTED)
    footer(ax,1000,'Audited source snapshot afd54410. Historical measurements; no new GPU benchmark. See ../evidence.json for provenance.')
    for ext in ('svg','png','pdf'): fig.savefig(OUT/f'{slug}.{ext}',dpi=160)
    return fig


def transition():
    fig,ax=base(2,'Transition: keep the expanded activation local',
                'D128 · expansion 4 · BF16 · SM90 specialized path [C4]')
    box(ax,55,228,1690,149,color=TEAL,fill=LIGHT[TEAL])
    text(ax,76,239,'ONE FORWARD KERNEL',10,TEAL,'bold')
    chain(ax,['LayerNorm','Expand A / B','SwiGLU','Squeeze + residual'],y=271,width=1600)
    text(ax,55,399,'The enclosing boundary is one kernel; the inner boxes are mathematical stages.',12,MUTED)
    box(ax,85,549,265,105,'Input tile\nD = 128',BLUE,size=16)
    box(ax,408,532,300,139,'Expanded tile\n4D = 512\nactivation + gate',TEAL,size=16)
    box(ax,765,549,290,105,'Output tile\nD = 128',BLUE,size=16)
    arrow(ax,357,601,402,601);arrow(ax,714,601,759,601)
    text(ax,85,706,'Expanded activations are consumed inside the forward kernel.',15,TEAL,'bold')
    text(ax,85,755,'Training preserves normalized input / statistics.\nBackward uses its main computation plus a separate partial reduction.',14)
    performance_card(ax,'Transition','D128; expansion 4')
    return finish(fig,ax,'transition','The fusion boundary applies to this specialized shape. Other shapes can follow different dispatch paths.')


def opm():
    fig,ax=base(3,'Outer Product Mean: fuse around the sequence reduction',
                'MSA → pair representation · training path [C5]')
    chain(ax,['Triton prologue\nLN + projections + mask','Grouped cuBLAS\nouter-product reduction','CUDA epilogue\nlayout + normalize + project'],colors=[TEAL,BLUE,TEAL])
    text(ax,55,399,'A/B and the product cross global-memory boundaries. Mask-count normalization is computed separately.',12,MUTED)
    text(ax,85,535,r'$O_{ijab}=\sum_s A_{sia}B_{sjb}$',27)
    chain(ax,['MSA rows\ns = 1 … S','Reduce over s\nchannel outer products','Pair features\n(i, j)'],y=634,x=85,width=975,colors=[BLUE,TEAL,BLUE])
    text(ax,85,772,'Backward choice: save O, or recompute O to reduce stored activations.',13,TEAL,'bold')
    text(ax,85,820,'The performance card uses save O; both variants remain in the data bundle.',12,MUTED)
    performance_card(ax,'OPM','S1024; save O')
    return finish(fig,ax,'opm','The sequence reduction is retained as a library call; epilogue fusion combines layout, normalization and projection.')


def pwa():
    fig,ax=base(4,'Pair-Weighted Averaging: join two reusable input branches',
                'Independent pair-weight and MSA-value paths feed a fused output computation [C6]')
    box(ax,80,231,580,72,'Pair → LN + projection + mask / softmax',TEAL,size=15)
    box(ax,80,323,580,72,'MSA → LN + value projection',BLUE,size=15)
    box(ax,850,256,870,113,'Weighted sum + gate + output projection + residual',TEAL,size=16)
    arrow(ax,667,267,843,292,AMBER);arrow(ax,667,359,843,333,AMBER)
    text(ax,85,536,'PAIR WEIGHTS',12,TEAL,'bold');text(ax,560,536,'MSA VALUES',12,BLUE,'bold')
    text(ax,85,580,r'$P_{ijh}=\mathrm{softmax}_{j}(b_{ijh})$',22)
    text(ax,560,580,r'$V_{sjhd}$',25)
    text(ax,85,672,r'$U_{sihd}=\sum_j P_{ijh}V_{sjhd}$',27)
    text(ax,85,769,'Pair weights are reused across MSA rows s.',16,TEAL,'bold')
    text(ax,85,815,'Gate and output projection consume the weighted result in the fused path.',12,MUTED)
    performance_card(ax,'PWA','S1024; D_msa64 / D_pair128')
    return finish(fig,ax,'pwa','Arrows show independent inputs. Training mask preparation and multi-kernel backward are outside the forward boxes.')


def lnlinear():
    fig,ax=base(5,'LayerNormLinear: normalize through the GEMM epilogue',
                'CuTe SM90 main-kernel algorithm [C7] · weight preparation is separate')
    box(ax,55,223,1690,165,color=VIOLET,fill=LIGHT[VIOLET])
    box(ax,80,245,470,55,'Input tile X → GEMM',BLUE,size=15)
    box(ax,80,315,470,55,'Same X → row statistics',TEAL,size=15)
    box(ax,720,254,520,104,'Affine correction\nin the GEMM epilogue',VIOLET,size=16)
    box(ax,1400,254,320,104,'Projected output Y',BLUE,size=16)
    arrow(ax,557,272,713,284);arrow(ax,557,342,713,329);arrow(ax,1247,306,1393,306)
    text(ax,55,409,'One main kernel. No full normalized activation is materialized in HBM.',12,MUTED)
    text(ax,85,531,'ALGEBRAIC REARRANGEMENT',12,VIOLET,'bold')
    text(ax,85,578,r'$Y=r\,[X(\gamma W)-\mu(\mathbf{1}^{T}\gamma W)]+\beta W$',24)
    text(ax,85,664,'GEMM and row statistics use the same input tile.\nMean / reciprocal standard deviation correct the output.\nWeight-side terms are prepared outside the main kernel.',15)
    text(ax,85,803,'The diagram describes CuTe; the portable timing is a different path.',13,VIOLET,'bold')
    performance_card(ax,'LNLinear','D128 to 16','Portable Triton reference ONLY; not the CuTe path')
    return finish(fig,ax,'layernorm-linear','The portable reference does not quantify the CUDA/CuTe fusion shown here. A matched CuTe comparison remains pending.')


def attention():
    fig,ax=base(6,'TriangleAttention: tiled attention between projection stages',
                'Dispatch-dependent module [C8, C9] · dashed boundaries denote stages, not single launches')
    chain(ax,['LN / QKV / bias\nprojection path','Tiled attention\nQKᵀ → softmax → PV','Sigmoid gate\n+ output projection'],colors=[VIOLET,TEAL,VIOLET],dashed=True)
    text(ax,55,399,'Actual projection and gated-output fusion depend on shape, dtype, backend and available native artifacts.',12,MUTED)
    text(ax,85,535,r'$O=\mathrm{softmax}(QK^{T}/\sqrt{d}+B)\,V$',28)
    chain(ax,['Query tile Q','Key / value tiles\nK, V','Output tile O\ntraining: LSE'],y=645,x=85,width=975,colors=[BLUE,TEAL,VIOLET],dashed=True)
    text(ax,85,779,'LSE supports the training path; backward has its own kernel boundaries.',13)
    text(ax,85,824,'Schematic only: tile schedule and exact saved tensors depend on dispatch.',12,MUTED)
    performance_card(ax,'TriangleAttention',None)
    return finish(fig,ax,'triangle-attention','A full-module paired measurement is required before attaching a speedup to this figure.')


def norm(rms=False):
    op='RMSNorm' if rms else 'LayerNorm'
    fig,ax=base(8 if rms else 7,op+': reduce a row, then reuse its statistics',
                'Standalone normalization · portable Triton historical comparison [N0, N1]')
    chain(ax,['Load activation row', 'Mean square' if rms else 'Mean + variance',
              'Reciprocal square root','Normalize + affine'],colors=[BLUE,TEAL,TEAL,BLUE],dashed=True)
    text(ax,55,399,'Boxes show the row algorithm. Backward parameter reductions can require additional kernels.',12,MUTED)
    formula=r'$y_i=x_i\,(\mathrm{mean}(x^2)+\epsilon)^{-1/2}\,\gamma_i$' if rms else r'$y_i=(x_i-\mu)\,(\sigma^2+\epsilon)^{-1/2}\,\gamma_i+\beta_i$'
    text(ax,85,540,formula,26)
    for i in range(16):
        box(ax,90+i*58,644,48,42,'x',BLUE,size=12,lw=0)
    arrow(ax,557,697,557,729,TEAL)
    box(ax,310,736,500,57,'One set of statistics per row',TEAL,size=16)
    text(ax,85,828,'No centering / mean subtraction.' if rms else 'Centering and scaling share row statistics.',13,MUTED)
    performance_card(ax,op,'D128' if rms else 'D384')
    return finish(fig,ax,'rmsnorm' if rms else 'layernorm','Measured gains concern the standalone portable path; they do not establish cross-operator CUDA fusion speedups.')


if __name__=='__main__':
    # TriMul already has a dedicated, detailed algorithm figure.
    algorithm()
    for ext in ('svg','png','pdf'):
        shutil.copyfile(HERE/f'02-trimul-algorithm.{ext}',OUT/f'trimul.{ext}')
    figures=[FIGURES[-1],transition(),opm(),pwa(),lnlinear(),attention(),norm(),norm(True)]
    entries=[('trimul','TriMul'),('transition','Transition'),('opm','OPM'),('pwa','PWA'),
             ('layernorm-linear','LayerNormLinear'),('triangle-attention','TriangleAttention'),
             ('layernorm','LayerNorm'),('rmsnorm','RMSNorm')]
    with PdfPages(OUT/'all-operators.pdf') as pdf:
        for fig in figures: pdf.savefig(fig)
        assert pdf.get_pagecount()==8
    cards=''.join(f'<section id="{slug}"><h2>{html.escape(label)}</h2><p><a href="{slug}.svg">SVG</a> · <a href="{slug}.png">PNG</a> · <a href="{slug}.pdf">PDF</a></p><img loading="lazy" src="{slug}.svg" alt="{html.escape(label)} kernel algorithm and fusion diagram"></section>' for slug,label in entries)
    nav=' · '.join(f'<a href="#{slug}">{label}</a>' for slug,label in entries)
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniWorld — 연산별 그림</title><style>body{font:16px/1.6 system-ui;background:#f5f4ef;color:#162e3d;margin:30px auto;max-width:1600px;padding:0 20px}a{color:#087f78}img{width:100%;height:auto}section{margin:45px 0}nav{position:sticky;top:0;background:#f5f4ef;padding:12px;border-bottom:1px solid #d6ddda}</style><h1>연산별 커널 알고리즘과 융합</h1><p>각 그림은 독립적으로 사용할 수 있는 SVG / PNG / PDF다. 기존 측정값만 사용했으며, 그림의 구현 경로와 성능 측정 경로가 다른 경우 명시했다.</p><p><a href="all-operators.pdf">전체 8페이지 PDF</a> · <a href="../index.html">측정 데이터 보기</a></p><nav>'+nav+'</nav>'+cards+'</html>')
    print('Rendered eight standalone operator figures and eight-page PDF.')
