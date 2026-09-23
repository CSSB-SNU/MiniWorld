# -*- coding: utf-8 -*-
"""Emit the four MSA-module maps in the house style of trimul/TRIMUL_FORWARD.svg.

Run: python tmp_kernel/generate_msa_svg.py   (writes msa_opm/ and msa_pwa/ beside this file)
"""
from xml.sax.saxutils import escape

W = 2800
CR, CK, CW = "#2879ad", "#12856e", "#b36b19"          # read / kernel / write
MUTED, INK, PLAN = "#52707e", "#163746", "#a6537b"
FS = 'font-family="Noto Sans CJK KR, sans-serif"'
FM = 'font-family="DejaVu Sans Mono, monospace"'
RX, RW = 40, 570
KX, KW, KSPLIT = 690, 1390, 1410
WX, WW = 2160, 600


def T(x, y, s, size=22, fill=INK, weight=None, mono=False):
    w = f' font-weight="{weight}"' if weight else ""
    f = FM if mono else FS
    sp = ' xml:space="preserve"' if mono else ""
    return f'<text x="{x}" y="{y}"{sp} {f} font-size="{size}" fill="{fill}"{w}>{escape(s)}</text>\n'


def _w(ch, size):
    """Rendered width of one character: CJK is full-width, Latin about 0.55 em."""
    o = ord(ch)
    full = (0x1100 <= o <= 0x11FF or 0x3000 <= o <= 0x303F or 0x3130 <= o <= 0x318F
            or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7A3 or 0xFF00 <= o <= 0xFF60)
    # the CJK font also draws these at (nearly) full width: dashes, arrows, math signs, middle dot
    wide = (o in (0x00B7, 0x00D7, 0x2013, 0x2014, 0x2015, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026)
            or 0x2190 <= o <= 0x21FF or 0x2200 <= o <= 0x22FF or 0x25A0 <= o <= 0x25FF)
    return size * (1.0 if full else 0.85 if wide else 0.55)


def wrap(lines, px, size=22):
    """Wrap prose to a column width. Korean is full-width, so a line that fits in Latin terms can still
    run past the column divider -- every candidate is measured as it will render, spaces included."""
    def w(t):
        return sum(_w(c, size) for c in t)

    out = []
    for line in lines:
        if not line.strip() or w(line) <= px:
            out.append(line)
            continue
        cur = ""
        for word in line.split(" "):
            cand = (cur + " " + word) if cur else word
            if cur and w(cand) > px:
                out.append(cur)
                cur = "  " + word
            else:
                cur = cand
            while w(cur) > px:                   # one run with nothing left to break on
                keep = ""
                for c in cur:
                    if w(keep + c) > px:
                        break
                    keep += c
                out.append(keep)
                cur = "  " + cur[len(keep):]
        out.append(cur)
    return out


def box(x, y, w, h, fill, stroke):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n'


def row(y, label, read, kern, code, write, planned=False):
    """One HBM read -> kernel -> HBM write band. Returns (svg, height)."""
    read = wrap(read, RW - 36)
    kern = wrap(kern, KSPLIT - KX - 26)
    write = wrap(write, WW - 36)
    n = max(len(read), len(kern), len(write))
    h = max(35 + 38 + 31 * (n - 1), 35 + 38 + 28 * (len(code) - 1)) + 40
    h = max(h, 150)
    s = T(RX, y - 18, label, 21, PLAN if planned else MUTED, 700)
    s += box(RX, y, RW, h, "#eef6fc", CR) + box(KX, y, KW, h, "#effaf5", CK) + box(WX, y, WW, h, "#fff7e7", CW)
    s += f'<path d="M{KSPLIT} {y} V{y + h}" stroke="{CK}" stroke-width="1.5"/>\n'
    s += T(RX + 18, y + 35, "HBM READ", 23, CR, 700)
    s += T(KX + 18, y + 35, "KERNEL · tile 내부", 23, CK, 700)
    s += T(KSPLIT + 22, y + 35, "PyTorch · 수학적 대응", 23, CK, 700)
    s += T(WX + 18, y + 35, "HBM WRITE", 23, CW, 700)
    for i, t in enumerate(read):
        s += T(RX + 18, y + 73 + 31 * i, t)
    for i, t in enumerate(kern):
        s += T(KX + 18, y + 73 + 31 * i, t)
    for i, t in enumerate(code):
        s += T(KSPLIT + 22, y + 73 + 28 * i, t, 20, "#173d37", mono=True)
    for i, t in enumerate(write):
        s += T(WX + 18, y + 73 + 31 * i, t)
    mid = y + h / 2
    s += f'<path d="M {RX + RW + 2} {mid} L {KX - 12} {mid}" fill="none" stroke="{CR}" stroke-width="3" marker-end="url(#read)"/>\n'
    s += f'<path d="M {KX + KW + 2} {mid} L {WX - 12} {mid}" fill="none" stroke="{CW}" stroke-width="3" marker-end="url(#write)"/>\n'
    return s, h + 62


def doc(path, title, sub, notes, rows, planned=False):
    body, y = "", 400
    for r in rows:
        s, dy = row(y, *r, planned=planned)
        body += s
        y += dy
    H = y + 40
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title>'
            f'<desc id="desc">{escape(sub)}</desc><defs>'
            f'<marker id="read" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{CR}"/></marker>'
            f'<marker id="write" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{CW}"/></marker>'
            f'</defs><rect width="{W}" height="{H}" fill="#f8fafc"/>\n')
    band = "#3b2340" if planned else "#133343"
    head += f'<rect x="0" y="0" width="{W}" height="218" fill="{band}"/>\n'
    head += T(40, 60, title, 36, "#ffffff", 700)
    head += T(40, 103, sub, 23, "#cce0e7")
    for i, nline in enumerate(notes):
        head += T(40, 145 + 36 * i, nline, 21, "#dfecf0")
    head += T(40, 340, "HBM READ · 어디서 온 무엇을 읽는가", 25, CR, 700)
    head += T(690, 340, "KERNEL · 계산과 그 PyTorch 대응", 25, CK, 700)
    head += T(2160, 340, "HBM WRITE · 무엇을 어디로 넘기는가", 25, CW, 700)
    head += T(40, 372, "파랑 화살표: 읽기", 18, CR)
    head += T(690, 372, "왼쪽: 설명·수식 / 오른쪽: 같은 일을 하는 PyTorch 문장", 18, MUTED)
    head += T(2160, 372, "주황: 쓰기", 18, CW)
    open(path, "w").write(head + body + "</svg>\n")
    print(path, H)


OUT = __file__.rsplit('/', 1)[0] + '/'

# ------------------------------------------------------------------ OPM FORWARD (measured)
doc(OUT + "msa_opm/MSA_OPM_FORWARD.svg",
    "OuterProductMean FORWARD  |  HBM 입력 → 커널 → HBM 출력",
    "L384 · S1024 · c_msa 64 / c_hidden 32 / c_z 128 · BF16 · H100 실측 0.906 ms",
    ["GEMM은 cuBLAS(76% SoL)에 맡기고 레이아웃 변환만 우리 커널이 한다. 전부 융합하면 같은 GEMM이 26%로 떨어진다.",
     "표시된 크기 = 버퍼 payload. 실제 HBM 전송량은 L2 hit·CTA 재읽기에 따라 달라진다.",
     "기준선(engine 자체 경로) 2.174 ms — permute 0.475 + 나눗셈 0.758 + cast 0.289가 여기서 사라진다."],
    [
     ("준비 · 호출당 한 번, weight packing",
      ["Wa, Wb: BF16 [32,64] 각 4.10 kB", "Wo: BF16 [128,1024] 262.14 kB", "b_o: [128]", "합계 270 kB · 학습 파라미터"],
      ["Wa/Wb는 transpose만 (GEMM 피연산자 정렬)",
       "Wo는 mma.m16n8k16 B-fragment 순서로 pre-swizzle",
       "→ epilogue가 thread당 8 B 로드 하나로 읽는다",
       "bias는 bf16으로 반올림 후 fp32 보관",
       "── 상수이므로 활성값 연산이 아님 ──"],
      ["wa_t = Wa.t().contiguous().to(bf16)",
       "wb_t = Wb.t().contiguous().to(bf16)",
       "wot  = Wo.t().contiguous().to(bf16)   # [1024,128]",
       "bf   = swizzle_b(wot)                 # [64,16,32,4]",
       "bias = b_o.to(bf16).float()"],
      ["wa_t/wb_t 8.19 kB → K_prologue", "bf 262.14 kB → K_epilogue (L2 상주)", "bias 0.51 kB"]),

     ("K_prologue (Triton, 상류 커널) · 0.058 ms",
      ["m: BF16 [1,1024,384,64] 50.33 MB", "mask: [1,1024,384]", "LN γ/β: FP32 [64]"],
      ["m을 한 번 읽어 LN + proj_a + proj_b + mask를 모두 처리",
       "LN 통계는 fp32, 정규화 출력은 bf16 (autocast 지점과 동일)",
       "결과를 GEMM이 원하는 레이아웃으로 바로 쓴다",
       "── 수식 ──",
       "y = LN(m);  a = Wa y ⊙ mask;  b = Wb y ⊙ mask",
       "A2[(i,c), s] = a[s,i,c]   BT[(j,e), s] = b[s,j,e]"],
      ["y = layer_norm(m, (64,), g, b)",
       "a = (y @ Wa.T) * mask[..., None]",
       "b = (y @ Wb.T) * mask[..., None]",
       "A2 = a[0].permute(2,1,0).reshape(384*32, 1024)",
       "BT = b[0].permute(2,1,0).reshape(384*32, 1024)",
       "# 한 커널: 읽기 50.33 MB, 쓰기 50.33 MB"],
      ["A2: BF16 [12288,1024] 25.17 MB", "BT: BF16 [12288,1024] 25.17 MB", "s가 연속 · zero-pad 완료", "→ cuBLAS GEMM"]),

     ("mask count · 0.022 ms",
      ["mask: [1024,384]"],
      ["엔진 모듈과 같은 의미로 fp32 카운트 후 clamp(min=1)",
       "상류 셀은 bf16 카운트라 S>256에서 더 이상 정확하지 않다",
       "── 수식 ──",
       "n[i,j] = max(Σ_s mask[s,i]·mask[s,j], 1)"],
      ["mf = mask[0].float()",
       "norm = (mf.t() @ mf).clamp_(min=1)"],
      ["norm: FP32 [384,384] 0.59 MB", "→ K_epilogue"]),

     ("cuBLAS NT GEMM · 0.412 ms · 309.2 GFLOP → 750 TFLOP/s = BF16 peak의 76%",
      ["A2 25.17 MB", "BT 25.17 MB"],
      ["외적을 GROUPED 레이아웃으로 낸다 — 이 배치만이 matmul이라",
       "물성화할 transpose가 존재하지 않는다",
       "M = N = L·c_hidden = 12,288,  K = S = 1,024",
       "── 수식 ──",
       "O[(i,c),(j,e)] = Σ_s A2[(i,c),s] · BT[(j,e),s]",
       "bf16 피연산자 · fp32 누산 · bf16 출력"],
      ["O = A2 @ BT.t()",
       "# [12288, 12288] bf16",
       "# 이 한 줄이 stock의",
       "#   einsum -> permute -> reshape",
       "# 세 단계를 대신한다"],
      ["O: BF16 [12288,12288] 301.99 MB", "workspace · j축 분할 시 221 MB까지", "→ K_epilogue"]),

     ("K_epilogue (우리 CUDA, csrc/opm_epilogue.cu) · 0.414 ms",
      ["O 301.99 MB (타일 단위)", "norm 0.59 MB", "bf 262.14 kB (L2)", "bias 0.51 kB"],
      ["CTA 하나가 (i 4개 × j 16개) = 64쌍을 맡는다",
       "BJ=16 → O에서 한 번에 읽는 열이 512개 연속(1 kB)",
       "cp.async 이중 버퍼 → ldmatrix.x4 (A) + 8 B 로드 (B)",
       "mma.sync.m16n8k16 · fp32 누산 · 나눗셈은 누산기에서",
       "── 수식 ──",
       "z[i,j,:] = (Σ_{c,e} O[(i,c),(j,e)] · Wo[(c,e),:]) / n[i,j] + b_o",
       "projection이 선형이라 (O/n)·W = (O·W)/n — 반올림 한 번 적다"],
      ["P = O.view(384,32,384,32).permute(0,2,1,3)",
       "P = P.reshape(384,384,1024) / norm[..., None]",
       "z = P @ Wo.T + b_o",
       "# 위 네 줄이 stock에서 네 번의 HBM 패스:",
       "#   permute 0.475 ms + 나눗셈 0.758 ms",
       "#   + cast 0.289 ms + projection 0.117 ms",
       "# 여기서는 한 패스"],
      ["z: BF16 [1,384,384,128] 37.75 MB", "블록이 pair에 residual로 더한다", "→ pair"]),
    ])

# ------------------------------------------------------------------ OPM BACKWARD (planned)
doc(OUT + "msa_opm/MSA_OPM_BACKWARD.svg",
    "OuterProductMean BACKWARD  |  설계 (P2) · 측정값 아님",
    "L384 · S1024 · BF16 · 총 ~1.0 TFLOP = forward의 약 2.9배 · 저장 50 MB / 재계산 302 MB",
    ["forward에서 얻은 규칙을 그대로 뒤집는다: 큰 GEMM은 cuBLAS, 레이아웃을 바꾸는 패스만 우리 커널.",
     "규칙 — 작고 여러 번 쓰는 것은 저장(a, b, norm = 50 MB), 크고 한 번 쓰는 것은 재계산(P = 302 MB).",
     "시간은 아직 측정하지 않았다. FLOP과 트래픽만 설계 근거로 적는다."],
    [
     ("saved tensors · forward가 남기는 것",
      ["forward에서 저장:", "a → A2 25.17 MB", "b → BT 25.17 MB", "norm FP32 0.59 MB", "합계 50.9 MB"],
      ["[N,N,1024] P는 저장하지 않는다 — 302 MB가 블록×recycle 내내 살아있게 된다",
       "대신 cuBLAS GEMM 한 번으로 되살린다 (309.2 GFLOP)",
       "블록에 activation checkpointing이 걸려 있으면 재계산본이 이미 있어 공짜",
       "── 저장 대상 선정 근거 ──",
       "a, b는 dA·dB·dWa·dWb에서 여러 번 읽힌다",
       "P는 dWo 한 곳에서만 쓰인다"],
      ["ctx.save_for_backward(A2, BT, norm)",
       "# P는 저장하지 않는다:",
       "#   302 MB x 4 block x 4 recycle",
       "# = 4.8 GB",
       "O = A2 @ BT.t()   # backward에서 재계산"],
      ["활성값 저장 50.9 MB", "(현재 autograd는 302 MB를 잡는다)"]),

     ("K_dO (forward epilogue를 거꾸로) · 38.7 GFLOP",
      ["dz: BF16 [1,384,384,128] 37.75 MB", "bf (Wo swizzle) 262.14 kB", "norm 0.59 MB"],
      ["forward epilogue와 같은 타일링·같은 pre-swizzle·같은 ldmatrix 경로",
       "읽고 쓰는 방향만 반대다: [N,N,c_z] → GROUPED [(i,c),(j,e)]",
       "── 수식 ──",
       "dO[(i,c),(j,e)] = (Σ_z dz[i,j,z] · Wo[z,(c,e)]) / n[i,j]",
       "fp32 누산 → bf16 저장 (다음 GEMM의 피연산자)"],
      ["dP = (dz / norm[..., None]) @ Wo",
       "dO = dP.view(384,384,32,32)",
       "dO = dO.permute(0,2,1,3).reshape(12288,12288)",
       "# 이 permute를 물성화하지 않는 것이",
       "# 이 커널의 존재 이유"],
      ["dO: BF16 [12288,12288] 301.99 MB", "forward의 O 버퍼를 재사용", "→ cuBLAS dA / dB"]),

     ("cuBLAS dA · dB · 각 309.2 GFLOP · K = 12,288",
      ["dO 301.99 MB", "BT 25.17 MB", "A2 25.17 MB"],
      ["forward와 같은 이유로 cuBLAS에 맡긴다 — 이 모양에서 76% SoL",
       "── 수식 ──",
       "dA[(i,c), s] = Σ_{(j,e)} dO[(i,c),(j,e)] · BT[(j,e), s]",
       "dB[(j,e), s] = Σ_{(i,c)} dO[(i,c),(j,e)] · A2[(i,c), s]",
       "s축 축약이 아니므로 atomics 불필요 · 결정적"],
      ["dA = dO   @ BT",
       "dB = dO.t() @ A2",
       "# 각 [12288, 1024]"],
      ["dA: BF16 [12288,1024] 25.17 MB", "dB: BF16 [12288,1024] 25.17 MB", "→ mask / Linear backward"]),

     ("O 재계산 + K_dWo · 309.2 + 38.7 GFLOP",
      ["A2, BT 50.3 MB", "dz 37.75 MB"],
      ["dWo에는 P가 필요하다 — cuBLAS로 O를 되살린 뒤 스트리밍하며 축약",
       "출력이 [128,1024]로 작아 축약 커널은 메모리 바운드",
       "── 수식 ──",
       "dWo[z,(c,e)] = Σ_{i,j} dz[i,j,z] · O[(i,c),(j,e)] / n[i,j]",
       "db_o[z] = Σ_{i,j} dz[i,j,z]",
       "s축이 아닌 (i,j)축 축약 → split-K 2단 fp32 (atomics 금지)"],
      ["O = A2 @ BT.t()",
       "P = O.view(384,32,384,32).permute(0,2,1,3)",
       "P = P.reshape(-1,1024) / norm.reshape(-1,1)",
       "dWo  = dz.reshape(-1,128).t() @ P",
       "db_o = dz.sum((0,1,2))"],
      ["dWo: [128,1024] 0.26 MB", "db_o: [128]", "→ optimizer"]),

     ("mask · Linear · LayerNorm backward · ~26 GFLOP",
      ["dA, dB 50.3 MB", "y (재계산) 또는 m 50.33 MB", "mask"],
      ["prologue를 거꾸로 — 한 커널로 융합할 후보 (forward prologue의 거울)",
       "── 수식 ──",
       "da = dA ⊙ mask,  db = dB ⊙ mask",
       "dWa = daᵀ y,  dWb = dbᵀ y",
       "dy  = da Wa + db Wb",
       "dm  = LN_bwd(dy, m)   (fp32 통계 재사용)"],
      ["da = dA_t * mask[..., None]",
       "db = dB_t * mask[..., None]",
       "dWa = torch.einsum('smic,smid->cd', da, y)",
       "dy  = da @ Wa + db @ Wb",
       "dm  = layer_norm_backward(dy, m, g, b)"],
      ["dm: BF16 [1,1024,384,64] 50.33 MB", "dWa, dWb: [32,64] 각 4.10 kB", "dγ, dβ: [64]"]),
    ], planned=True)


# ------------------------------------------------------------------ PWA FORWARD (measured)
doc(OUT + "msa_pwa/MSA_PWA_FORWARD.svg",
    "MSAPairWeightedAveraging FORWARD  |  HBM 입력 → 커널 → HBM 출력",
    "L384 · S1024 · c_msa 64 / c_z 128 / head 8 × 32 · BF16 · H100 실측 0.736 ms",
    ["[S,N,N] attention은 어디에도 물성화되지 않는다. w는 [8,384,384] 2.36 MB로 끝난다.",
     "홀쭉한 GEMM을 꽉 찬 GEMM으로 바꾼 것이 핵심 — stock einsum은 출력 N차원이 c=32뿐이라 텐서코어가 10%도 못 돈다.",
     "기준선(engine 자체 경로) 1.513 ms — 그 중 실제 GEMM은 0.172 ms뿐이고 나머지가 레이아웃 복사였다."],
    [
     ("pair 경로 · engine LayerNorm + softmax · 0.050 ms",
      ["z (pair): BF16 [1,384,384,128] 37.75 MB", "Wz: BF16 [8,128] 2.05 kB", "mask: [1,384]"],
      ["상류 셀은 여기서 stock torch LayerNorm을 부른다 — 우리는 engine의 융합 LN으로 바꿨다",
       "NCU 기준 193 µs → 24.5 µs, 그것만으로 전체 1.18배",
       "[N,N,H]까지 줄어든 뒤라 softmax는 7 µs",
       "── 수식 ──",
       "β = Wz · LN_z(z);  β += (1-mask)·(-inf)",
       "w = softmax_j(β)   (fp32 softmax → bf16)"],
      ["zn = ln_pair(z)                 # engine LN",
       "b  = (zn @ Wz.T).permute(0,3,1,2)",
       "b  = b.masked_fill(~mask[:,None,None,:], -inf)",
       "w  = torch.softmax(b, -1).to(bf16)"],
      ["w: BF16 [1,8,384,384] 2.36 MB", "→ K_pwa_fo", "(engine LN 교체분 0.168 ms 절약)"]),

     ("K_ln_vg (Triton, 상류 셀) · 0.258 ms · 452 MB → HBM의 53%",
      ["m (MSA): BF16 [1,1024,384,64] 50.33 MB", "Wv, Wg: BF16 [256,64] 각 32.77 kB", "LN γ/β: FP32 [64]"],
      ["m을 한 번 읽어 LN → value / gate 두 projection → sigmoid까지",
       "stock은 이 자리에서 5.5 + 1.24 + 1.24 + 1.13 ms를 쓴다 (원저자 측정)",
       "대역폭 바운드라 CUDA로 다시 써도 0.283 ms — 가져올 것이 없는 자리",
       "── 수식 ──",
       "mn = LN_m(m)",
       "v = Wv · mn;   g = σ(Wg · mn)"],
      ["mn = layer_norm(m, (64,), g_, b_)",
       "v  = mn @ Wv.T          # [1,1024,384,256]",
       "g  = torch.sigmoid(mn @ Wg.T)",
       "# 읽기 50.33 MB, 쓰기 402.65 MB"],
      ["v: BF16 [1,1024,384,256] 201.33 MB", "g: BF16 [1,1024,384,256] 201.33 MB", "→ K_pwa_fo"]),

     ("K_pwa_fo (Triton, 상류 셀) · 0.381 ms · 90.2 GFLOP · tensor 32% / HBM 36%",
      ["v 201.33 MB (i-tile마다 L2 재사용)", "g 201.33 MB", "w 2.36 MB", "Wo: BF16 [64,256] 32.77 kB"],
      ["CTA = (i 타일 128, s 타일 4), head를 순회하며 j축 K=N contraction",
       "grid 순서를 i-tile 우선으로 두어 v를 L2에서 재사용한다",
       "w를 [H,NPI,NPJ]로 zero-pad → 16 B 정렬, mask 불필요",
       "gate와 출력 projection을 같은 커널의 epilogue에서 처리",
       "head 8개 partial을 fp32로 누산 — stock chunked 경로(bf16)보다 정확하다",
       "── 수식 ──",
       "o[s,i,h,c] = Σ_j w[h,i,j] · v[s,j,h,c]",
       "out = Wo · (g ⊙ o)",
       "launch 338개 → 약 12개"],
      ["o   = torch.einsum('bhij,bsjhd->bsihd', w, v)",
       "og  = g * o.reshape(1,1024,384,256)",
       "out = og @ Wo.T",
       "# [S,N,N] attention도 [S,N,256] o도",
       "# HBM에 나오지 않는다"],
      ["out: BF16 [1,1024,384,64] 50.33 MB", "모듈이 msa에 residual로 더한다", "→ msa"]),
    ])

# ------------------------------------------------------------------ PWA BACKWARD (planned)
doc(OUT + "msa_pwa/MSA_PWA_BACKWARD.svg",
    "MSAPairWeightedAveraging BACKWARD  |  설계 (P3) · 측정값 아님",
    "L384 · S1024 · BF16 · 총 ~296 GFLOP = forward의 약 2배 · 저장 52.7 MB / 재계산 604 MB",
    ["큰 contraction 세 개가 모두 표준 GEMM이라 cuBLAS로 갈 수 있다. 다만 레이아웃 전치 비용이 관건이다.",
     "규칙 — 저장은 m과 w(52.7 MB)뿐, v·g·o(각 201 MB)는 재계산한다.",
     "시간은 아직 측정하지 않았다. FLOP과 트래픽만 설계 근거로 적는다."],
    [
     ("saved tensors + 재계산 · 77.3 GFLOP",
      ["forward에서 저장:", "m BF16 50.33 MB", "w BF16 2.36 MB", "합계 52.69 MB"],
      ["v·g·o를 저장하면 블록당 604 MB — m에서 되살리는 편이 싸다",
       "v, g는 forward의 ln_vg를 그대로 다시 돌린다 (0.258 ms)",
       "o는 forward contraction을 다시 돌린다 (77.3 GFLOP)",
       "── 재계산 ──",
       "mn = LN_m(m);  v = Wv mn;  g = σ(Wg mn)",
       "o[s,i,h,c] = Σ_j w[h,i,j] v[s,j,h,c]"],
      ["ctx.save_for_backward(m, w)",
       "# backward 진입 직후",
       "mn = layer_norm(m, (64,), g_, b_)",
       "v, g = mn @ Wv.T, sigmoid(mn @ Wg.T)",
       "o = einsum('bhij,bsjhd->bsihd', w, v)"],
      ["v, g, o: 각 201.33 MB (재계산)", "활성값 저장은 52.69 MB", "(현재 autograd는 604 MB)"]),

     ("gate · 출력 projection backward · ~26 GFLOP",
      ["dout: BF16 [1,1024,384,64] 50.33 MB", "o, g 402.65 MB", "Wo 32.77 kB"],
      ["forward의 fo epilogue를 거꾸로 — 같은 커널에 융합할 후보",
       "elementwise가 대부분이라 메모리 바운드",
       "── 수식 ──",
       "dWo = (g ⊙ o)ᵀ dout",
       "dog = dout · Wo",
       "dg  = dog ⊙ o,   do = dog ⊙ g",
       "dpre_g = dg ⊙ σ'(pre_g) = dg ⊙ g ⊙ (1-g)"],
      ["og  = g * o",
       "dWo = og.reshape(-1,256).t() @ dout.reshape(-1,64)",
       "dog = dout @ Wo",
       "dg, do = dog * o, dog * g",
       "dpre_g = dg * g * (1 - g)"],
      ["do: BF16 [1,1024,384,256] 201.33 MB", "dpre_g: 201.33 MB", "dWo: [64,256] 32.77 kB"]),

     ("dv · dw · 각 77.3 GFLOP · cuBLAS 후보",
      ["do 201.33 MB", "v 201.33 MB", "w 2.36 MB"],
      ["dv는 forward contraction에서 w만 전치한 같은 모양 — head별 batched GEMM",
       "dw는 K = S·c = 32,768짜리 큰 GEMM이고 출력이 [8,384,384] 2.36 MB로 작다",
       "열쇠: cuBLAS가 요구하는 전치(각 201 MB)가 융합 커널보다 싼지",
       "── 수식 ──",
       "dv[s,j,h,c] = Σ_i w[h,i,j] · do[s,i,h,c]",
       "dw[h,i,j]   = Σ_{s,c} do[s,i,h,c] · v[s,j,h,c]"],
      ["dv = einsum('bhij,bsihd->bsjhd', w, do)",
       "dw = einsum('bsihd,bsjhd->bhij', do, v)",
       "# dw: head별 [384, 32768] x [32768, 384]"],
      ["dv: BF16 201.33 MB", "dw: BF16 [1,8,384,384] 2.36 MB", "→ softmax backward"]),

     ("softmax · pair 경로 backward · ~5 GFLOP",
      ["dw 2.36 MB", "w 2.36 MB", "zn (재계산) 37.75 MB"],
      ["여기부터는 전부 [N,N,H] 크기라 싸다 — torch로 두어도 된다",
       "── 수식 ──",
       "dβ[h,i,j] = w[h,i,j] · (dw[h,i,j] − Σ_j' w[h,i,j'] dw[h,i,j'])",
       "dWz = dβᵀ zn;   dzn = dβ · Wz",
       "dz = LN_z_bwd(dzn, z)"],
      ["dbeta = w * (dw - (w*dw).sum(-1, keepdim=True))",
       "dWz = einsum('bhij,bijc->hc', dbeta, zn)",
       "dzn = einsum('bhij,hc->bijc', dbeta, Wz)",
       "dz  = layer_norm_backward(dzn, z, gz, bz)"],
      ["dz: BF16 [1,384,384,128] 37.75 MB", "dWz: [8,128] 2.05 kB", "→ pair 경로 상류"]),

     ("value / gate projection · LayerNorm backward · ~26 GFLOP",
      ["dv 201.33 MB", "dpre_g 201.33 MB", "mn (재계산) 50.33 MB"],
      ["forward의 ln_vg를 거꾸로 — 한 커널로 융합할 후보",
       "s·i축 축약이므로 split-K 2단 fp32 (atomics 금지, 결정성 요구)",
       "── 수식 ──",
       "dWv = dvᵀ mn,   dWg = dpre_gᵀ mn",
       "dmn = dv · Wv + dpre_g · Wg",
       "dm  = LN_m_bwd(dmn, m)"],
      ["dWv = dv.reshape(-1,256).t() @ mn.reshape(-1,64)",
       "dWg = dpre_g.reshape(-1,256).t() @ mn.reshape(-1,64)",
       "dmn = dv @ Wv + dpre_g @ Wg",
       "dm  = layer_norm_backward(dmn, m, g_, b_)"],
      ["dm: BF16 [1,1024,384,64] 50.33 MB", "dWv, dWg: [256,64] 각 32.77 kB", "residual과 합산되어 상류로"]),
    ], planned=True)
