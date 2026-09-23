# Transition 기존 구현 전체 점검 — 2026-09-18

이 감사 이후 [H100 residual fusion을 구현했다](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-hopper-residual-20260918/README.md). 아래는 구현 전 소스와 이력의 기록이다.

일반 Transition의 현행 소스, H100 b2b v0–v17, split 및 SM100 실험 기록,
관련 Git 이력과 9월의 모듈 벤치를 대조했다. **이번 점검은 정적 감사이며 새 GPU 측정은 아니다.**
소스는 `runs/trimul_sm90_parity_20260917/engine`의 `59bdb335` 기준이다.
ConditionedTransition/AdaLN은 별도 연산이므로 이 목록에 포함하지 않는다.

## 결론

1. **살릴 구현이 있다.** H100 hand-CUDA b2b v14는 당시 Triton b2b보다 약 1.29× 빨랐다.
   CUDA gate backward도 D256/512에서 이겼다. 둘 다 현재 기본 residual-fused 경로에서는 선택되지 않는다.
2. **현재 알고리즘의 직접 대체품인지 따로 봐야 한다.** b2b는 LN부터 squeeze까지 묶으며,
   현재 기본은 LN / expand+SwiGLU / squeeze+residual 세 경계다. b2b의 과거 승리는
   현재 F2 한 커널의 우위를 증명하지 않는다.
3. 현재 B2 개발은 CuTe 코드만 볼 것이 아니라 **packed dAB를 직접 쓰는 CUDA gate backward**를
   먼저 재사용 검토해야 한다. raw x의 LN 재계산을 저장된 xn 입력으로 바꾸는 작업이 필요하다.
4. CuTe backward는 실제 연결·검증·벤치를 한 기록이 있다. D512의 전체 학습에서는 더 느렸다.
5. 현재 기능형 `ops.transition`의 선택 규칙은 모듈의 기본 경로와 다르다. 실험용 dAB+LN
   backward에는 gamma=0을 처리하지 않는 식도 있다. 성능 개발에 앞서 경로/수치 계약 점검 대상이다.

## 기준 수식과 현재 기본 경로

```text
xn = LN(x)
a = xn @ Wa.T; b = xn @ Wb.T
h = SiLU(a) * b
y = x + h @ Ws.T
```

D는 입력 폭, ND=nD는 확장 폭, M은 flatten한 행 수다. Pair는 M=B·L², token은 M=B·L이다.
이 모듈에는 모델 mask/dropout 인자가 없다.

`transition_residual_fusion=True`가 기본이고, 일반 triton/miniworld 모듈의 학습·추론에서
기존 H100 shape 분기보다 먼저 적용된다. 명시적인 `implementation="cute"`는 별도 경로다.

| 단계 | 현재 기본 구현 |
|---|---|
| F1 | 공통 Triton LN → xn 저장 + identity 전달 |
| F2 | Triton dual expand GEMM + SwiGLU → h |
| F3 | Triton squeeze GEMM → BF16 반올림 → residual add |
| B1 | cuBLAS dh = dy @ Ws |
| B2 | Triton 저장 xn에서 a/b/h 재계산 + gate 미분 → h, stacked dAB |
| B3/B4/B5 | cuBLAS dWs / dWab / dxn, 총 3회 |
| B6 | 공통 Triton LN backward → BF16 반올림 → identity gradient add |

B 번호는 이 문서의 Transition 순서이며 TriMul의 번호와 다르다.
Backward의 주요 GEMM은 총 4회다. weight packing과 gradient 초기화 같은 보조 연산은 별도다.

## 구현 목록과 재사용 판정

아래 파일명은 엔진의 `src/miniworld_engine/kernels/transition/` 기준이다.
소스 전체의 상대 경로와 SHA-256은 [파일 목록](implementation-inventory.json)에 있다.

| 구현 / 파일 | 융합 범위·특징 | 현재 연결 / 판정 |
|---|---|---|
| Triton residual split — `triton/residual.py` | F3 squeeze+residual, B6 공통 LN+residual. F2/B2는 기존 커널 재사용 | **현재 기본 비교 기준** |
| Triton normalized split — `triton/main.py` | 정규화된 xn → expand+SwiGLU, cuBLAS squeeze, 별도 residual | residual fusion을 끈 강제 Triton 경로. 유지 |
| Triton raw-input expand — `triton/fused.py` | stats로 LN 적용 + dual expand + SwiGLU → h | fused wrapper의 fallback/API. 별도 squeeze 필요 |
| Triton small-D b2b — 같은 파일 | D≤128에서 LN+expand+SwiGLU+squeeze+residual. stats 별도/내부 옵션 | legacy small-D 경로. covering/K-tiled 분기 이력 주의 |
| Triton K-tiled b2b — 같은 파일 | 큰 D에서 tile로 expand+squeeze, LN 반복, residual 별도 | legacy 학습 일부. wide token에서 크게 느렸음 |
| Triton gate backward — 같은 파일 | raw/saved xn × separate/stacked dAB, h 저장 옵션 | **현재 B2도 이 커널의 saved-xn + stacked 형태**. 서로 다른 4개 커널로 세지 않음 |
| Triton folded-stats LN backward — 같은 파일 | dx + affine gradient, 기본 privatized atomic와 후속 sum | legacy 경로. 현재 B6와 다른 구현 |
| Triton bare SwiGLU FFN — 같은 파일 | LN/residual 없는 별도 연산. public split과 retained b2b A/B | Transition 자체와 구분. 당시 atom 사례에서 split이 빨라 public은 split |
| **H100 CUDA b2b v14** — `cuda/transition_b2b_kernel.cu` | LN+expand+SwiGLU+squeeze(+후속 residual 지원). weight TMA, x cp.async, RS WGMMA squeeze, vector store | legacy D128/256. **재측정 가치 높음**, 현재 융합 경계와 달라 F2 drop-in은 아님 |
| H100 CUDA expand — `cuda/transition_expand_gate_kernel.cu` | LN+dual WGMMA+SwiGLU → h | 직접 API/벤치용. 당시 D128/256/512 모두 CuTe expand보다 느림. 낮은 우선순위 |
| **H100 CUDA gate backward** — `cuda/transition_gatebwd_kernel.cu` | raw x에서 xn/h 재계산 + 미분 → h, stacked dAB, xn | legacy fused BWD의 D256/512 조건부. **현재 B2 재사용 1순위**. saved-xn 입력으로 변경 필요 |
| H100 CuTe expand — `cute/gemm_transition_swiglu.py` | gamma를 weight에 fold, GEMM 뒤 stats 보정 + SwiGLU | legacy wide-D 경로. GEMM scheduling 재사용 후보. 현재 xn BF16 반올림 경계와 다름 |
| H100 CuTe gate backward — `cute/backward_gatebwd.py` | normalized xn dual WGMMA + gate 미분 | opt-in BWD. dh를 M×2ND로 복제하고 interleaved dAB/weight 및 보정 tensor 준비. **전체 wrapper보다 epilogue/pipeline 재사용** |
| **H100 CuTe squeeze+residual** — `cute/squeeze_residual.py` | Quack GEMM의 별도 C operand로 residual, staging copy 없음 | legacy auto는 D512와 row/alignment 조건. **F3 후보**. GEMM BF16 반올림 후 add 계약 검증 필요 |
| CuTe dAB+LN backward — `cute/dab_lnbwd.py` | dxn GEMM+LN dx epilogue로 dxn HBM 제거. affine gradient/residual은 밖 | opt-in, 기본 off, fused dispatcher K≤128. **수치 문제 해결 전 보류** |
| SM100 CuTe forward — `cute/b2b_fwd_sm100.py` | 이름과 달리 LN → tcgen05 expand/SwiGLU → cuBLAS squeeze의 **split** | legacy B200 경로. H100 대상 아님 |
| SM100 CuTe gate backward — `cute/gatebwd_sm100.py` | tcgen05 + dh TMA epilogue, h/dA/dB 분리 출력 | legacy B200 BWD. 분리 후처리 fallback 코드도 있으나 정상 wrapper는 fused 설정 |
| SM100 hand-CUDA b2b — `cuda/transition_b2b_sm100_kernel.cu` | tcgen05 bring-up 및 b2b 실험 | 현 소스에 production Python loader 호출 없음. 과거 매우 느림. 연구 기록 |
| Generic CUDA — `cuda/transition_cuda_kernel.cu`, `.cpp`, `setup.py` | cuBLAS + 분리된 cast/SwiGLU/미분; 호출별 handle 생성/해제 | standalone build/registry driver. H100 TMA/WGMMA 구현 아님. public cuda_transition은 미구현 stub |
| 보조 구현 | `triton/fold.py`, `cute/fused.py`, reference/interface/whole_op, 공통 LN | folding/배선/기준 수식. 별도 완성 Transition 알고리즘으로 세지 않음 |

### 현재 기본에서 우회되는 legacy H100 배선

`transition_residual_fusion=False`, BF16/n=4, 필요한 정렬·빌드 조건 만족 시:

- D128/256: hand-CUDA b2b forward가 학습·추론에 사용될 수 있다.
- D384: 추론 CuTe split, 학습 Triton K-tiled b2b.
- D512/768: CuTe split forward + 기본 Triton separate-dA/dB backward.
  이 backward는 saved-xn stacked 경로의 4회와 달리 주요 GEMM 6회다.
- CUDA gate backward는 `has_xn=False`, D256/512, M%128=0 등 조건이다.
  모듈 D512는 앞서 CuTe로 가므로 기본 legacy 모듈에서 해당 CUDA gate에 도달하지 않는다.
- CuTe squeeze+residual의 자동 선택은 D512, M≥16384, M%128=0, contiguous BF16 등으로 제한된다.
- legacy LN backward에는 native CUDA main+parameter-reduce 두 커널도 있다.
  현재 기본 B6는 공통 Triton residual LN이며 TriMul의 relaxed atomic 개선을 이미 사용한다.

## H100 b2b v0–v17: 실제로 무엇이 성공했나

7월 기록의 대표 microbenchmark는 M=524288, D=128, ND=512, BF16/H100이다.
당시 Triton b2b는 약 538–543 µs였다. 버전별 반복 측정 범위가 다르므로 세부 차이는
원본 notes를 따른다. **현재 Triton residual split과의 재측정 결과가 아니다.**

| 버전 | 시도 | 기록 / 판단 |
|---|---|---|
| v0 | 2 warpgroup, weight TMA, RS squeeze | 658 µs |
| v1 | 전용 producer warpgroup | 752 µs, 회귀 |
| v2 | CUTLASS pipeline 패턴 | 685 µs, 회귀 |
| v3 | BN64→128 | 603.9 µs, 개선 |
| v4 | BN256 | 1823.9 µs, spill·stage 축소로 회귀 |
| v5 | LN parameter load 개선 | 563.3 µs |
| v6 | raw x TMA 및 PTX helper 변경 | 793.3 µs, 회귀. TMA만의 효과를 분리한 실험은 아님 |
| v7 | register uint4 packing | 1294.9 µs, spill |
| v8 | weight buffer 축소 | 786.7 µs, occupancy 이득 없이 overlap 손실 |
| v9 | xn register 보관, global-read RS expand | 571.6 µs |
| v10 | staged x에서 RS expand | 655 µs, scalar LDS 부담 |
| v11 | shared shuffle → STG128 | 448–460 µs, **첫 큰 승리** |
| v12 | XOR swizzle | 441.9–452.9 µs, bank conflict 감소 |
| v13 | warpgroup-local NamedBarrier | 431.8–442.1 µs |
| **v14** | gamma/beta register load | **414.9–426.7 µs, 약 1.26–1.31×** |
| v15 | ND software pipeline, h double buffer | 620.7–635.7 µs, spill, 되돌림 |
| v16 | stats serial 융합 | L1024 모듈 2.05 vs 별도 stats 1.30 ms, 되돌림 |
| v17 | stats warp-parallel 융합 | L384 .243 vs .203 ms; L1024 1.60 vs 1.31 ms, 되돌림 |

v10의 당시 '상한' 추정 직후 v11–v14가 크게 개선됐다. 따라서 옛 기록의
roofline/ceiling 서술을 현재 구현의 절대 한계로 취급하지 않는다.

v14 모듈 통합의 당시 inference는 L384 .2032 vs .2448 ms, L768 .7490 vs .8932 ms였다.
비교 대상은 당시 Triton b2b다. 해당 시기의 PyTorch/초기화 관련 벤치 문제를 고려해
그 기록의 PyTorch 대비 배율은 여기서 사용하지 않았다. 초기 notes의 'inference only' 설명보다
후속 통합 커밋과 현재 코드가 우선하며, 실제로 학습 forward도 지원한다.

## 다른 기존 구현의 성능 근거

### H100 wide-D / gate backward

7월 highdim-and-split의 M=131072 microbench, L2 flush 없음:

| D | CUDA b2b | CuTe expand + squeeze | hand-CUDA expand 단독 | CuTe expand 단독 |
|---:|---:|---:|---:|---:|
| 128 | 112 µs | 211 µs | 152 µs | 149 µs |
| 256 | 409 µs | 461 µs | 430 µs | 341 µs |
| 512 | 빠른 b2b 구성 확보 못함 | 1311 µs | 1698 µs | 929 µs |

D512 b2b 실험은 약 3.1–3.2 ms로 split보다 느렸다. 따라서 small-D 성공을 wide-D로
일괄 확장할 수 없다. D512 CuTe split은 tile/cluster/swizzle/persistence를 실제 탐색했다.
당시 winner는 192×128, ping-pong, cluster(1,2), swizzle8, static persistent였다.
일부 config는 수치 검증에 실패했고 dynamic persistent도 당시 winner가 아니었다.

`2101ee25`의 CUDA gate backward 기록은 D256에서 1.02–1.05×, D512에서 1.07–1.18×,
D128에서는 Triton보다 느렸다. **현재 saved-xn B2와의 비교가 아닌 당시 raw-input 경로 비교다.**

### 9월의 전체 학습 비교: 기존 CuTe backward도 실제 실행했다

[9월 15일 재점검](../transition-performance-recheck.md)은 BF16, static compile/fullgraph,
manual CUDA graph에서 forward+backward를 비교했다. optimizer 제외, 동등 초기값 사용이다.
현재 residual fusion 이전 수치다.

- Pair L768/D512: PyTorch 25.6411 ms, 당시 engine auto 30.3171 ms.
- CuTe backward를 실제 켠 별도 유효 실행: engine 31.8476 ms, PyTorch 25.6737 ms.
  출력/gradient 검사도 수행했다. 이전 환경변수 적용 실패 실행은 이 결과에서 제외됐다.
- D512의 당시 gate recompute는 profile에서 약 7.75 ms였다.
- engine은 activation 재계산으로 forward 후 보관량을 크게 줄였지만,
  CuTe backward의 큰 임시 tensor를 포함한 peak memory까지 낮아진 것은 아니었다.

[Triton 경로별 비교](../transition-triton-comparison.md)에서는 token D384가
legacy auto .21434 ms 대비 Triton split .08240 ms, D768은 .20069 vs .15761 ms였다.
반면 pair D128은 auto 4.10025 vs 당시 split 4.42265 ms였다. **shape별 승패가 다르다.**

[9월 17일 residual fusion](RESIDUAL.md)은 Triton split 학습을
L384 1.165792→1.062112 ms, L768 4.406776→3.996136 ms로 개선했다.
이후 공통 LN 변경도 있으므로 위 옛 native 수치와 직접 나누어 현재 배율을 만들면 안 된다.

### SM100/B200

SM100 notes v1–v3도 확인했다. 초기 LN 제외로 부풀려진 비교는 철회된 기록이다.
raw CUDA b2b는 D128/L384 약 1990 µs, 당시 DSL b2b 254 µs, split 110 µs였다.
후속 진짜 fused DSL 5종도 약 233–246 µs로 split을 넘지 못했다.
이 5종의 일부 실제 코드는 당시 원격 `~/psk/ncu`에만 기록되어 있다.
**현재 checkout에서 없는 원격 실험 소스까지 직접 검증했다고 주장하지 않는다.**

## Config 공간과 불필요한 메모리 이동

- 현재 Triton F2/F3: 각각 선언 조합 1296개, B2: 1080개.
  BM/BN/BK/GROUP_M/warps/stages를 탐색한다. 선언 조합 수는 실행 가능 수나 캐시 완성 수가 아니다.
- hand-CUDA b2b: D128/256, BN64/128, stages1/2/3, min-blocks1/2 등을 탐색하며
  warpgroup=2, KT=D 같은 구조 제약이 있다. expand/gate는 WG1/2, KT64/128 축도 갖는다.
  shared memory와 layout 조건으로 후보를 거른다.
- CuTe SM90: tile/cluster/ping-pong/swizzle/static·dynamic persistent의 native config 관리가 있다.
  예전 고정 winner만 있는 상태는 아니다. 다만 **Triton과 동일한 config 축·범위는 아니다.**
- dAB+LN은 tileM64/128/192 및 clusterM1/2를 실제 튜닝한 이력이 있다. 기본 off를
  곧바로 '튜닝 안 된 커널'로 해석하면 안 된다.
- 현재 dAB는 최종 stacked activation 버퍼로 직접 쓴다. 남은 `cat(Wa,Wb)`는 weight만이며
  D128/n4/BF16에서 256 KiB다. 과거 pair 실측에서는 이를 두 GEMM+add로 나누면 더 느렸다.
- 반면 기존 CuTe gate backward의 dh 복제는 M×2ND여서 입력 길이에 따라 커진다.
  이 wrapper를 현재 B2에 그대로 가져오지 않아야 하는 이유다.

## 발견한 배선·수치 점검 사항

1. **`ops.transition`이 모듈과 다른 경로를 선택한다.**
   `ops/__init__.py`는 `transition/whole_op.py`를 가리킨다. 그 함수는
   `transition_residual_fusion`과 `engine_backend`를 확인하지 않고 H100 D≥256을
   CuTe+별도 residual로 보내는 등 기존 dispatch를 유지한다. 모듈과 같다는 docstring도 오래됐다.
2. **실험용 dAB+LN의 affine gradient 식에 gamma 나눗셈이 있다.**
   `triton/fused.py::_fused_bwd`는 `(dWab - db_ab * beta) / gamma`로 중간량을 복원한다.
   gamma=0 guard가 없어 비유한 값이 생길 수 있고 BF16 dWab 반올림도 증폭될 수 있다.
   소스상 확인한 문제이며 이번에 GPU 재현한 결과는 아니다. 기본 off다.
3. **반올림 경계가 다른 구현이 있다.** CuTe weight-folded expand, fused dAB+LN의
   FP32 accumulator 소비, squeeze epilogue는 현재 BF16 경계와 따로 검증해야 한다.
4. **`implementation="cuda"`와 legacy 경로 내부의 hand-CUDA는 다르다.**
   public cuda_transition은 미구현이며, hand-CUDA b2b/gate는 Triton 계열 wrapper 안에서 선택된다.

## 다음 개발에 적용할 판단

| 순서 | 대상 | 진행 기준 |
|---|---|---|
| 1 | 최신 Triton 기준선 및 경로 통일 | Pair L384/768 D128, token 별도. source/backend/config identity와 커널 trace 기록 |
| 2 | CUDA gate backward → saved-xn B2 | 현재 h + stacked dAB 계약 유지, dh 복제 금지. TMA/WGMMA와 config 축 대조 |
| 3 | CuTe/CUDA expand F2 및 CuTe squeeze F3 | 현재 fusion/반올림 계약을 유지하며 커널별 NCU와 전체 학습 비교 |
| 별도 A/B | CUDA b2b v14 | 기존 성공 구현의 최신 성능 확인. 다른 융합 알고리즘이므로 같은-F2 비교와 분리 |
| 보류 | dAB+LN, wide-D b2b, SM100 실험 | 수치 문제·과거 회귀 이유 해결 또는 대상 GPU 변경 때 재검토 |

## 추적 가능한 근거

- 소스·25개 notes의 경로와 SHA-256: [implementation-inventory.json](implementation-inventory.json).
- H100 b2b v14 `ea5aa3f2`, 통합 `3ae6c43d`, D256 확장 `b2ce8134`, residual `8faf4952`.
- CUDA gate backward `2101ee25`, dAB+LN tile/cluster 실험 `02039d19` / `621ad70c`.
- Triton covering/K-tile 탐색 누락 수정 `79cca4eb` / `f306ab55` — 당시 성능 근거는 A100이므로 H100 수치로 쓰지 않음.
- native config 확장 `c65fe203`, residual fusion `7db83df1`, default-on `fbe2d6ad`, 공통 LN atomic `09196265`.
- [현재 단계별 배선](H100_NEXT.md), [잔차 융합 검증](RESIDUAL.md), [이전 코드 감사](AUDIT.md).

커널 코드는 이번 점검에서 변경하지 않았다. 현재 전체 config의 캐시 완성 여부와
원격에만 있는 SM100 실험 소스는 이번 감사의 검증 범위 밖이다.
