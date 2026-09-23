# Transition 현재 배선 — 2026-09-18

**로컬 개발 checkout**의 residual fusion 기본 경로다. 설치본이나 실행 중 학습에 배포했다는 뜻은 아니다.

개발·측정 대상은 **D128/256/384/512**다. D768은 사용자 요청으로 계획에서 제외했다.
아래 지원 목록은 기존 코드의 실제 분기 조건이며 신규 개발 범위와 구분한다.

## 신규 개발 단위: streamed-K / full-K

**[두 버전 개발 기준](VARIANTS.md)** · [버전 목록](variants.json) · [두 버전 + backward 그림](../../tmp_kernel/transition/TRANSITION_VARIANTS.svg)

신규 구현은 `streamed_k`(BK<D)와 `full_k`(전체 K 입력을 hidden 루프 밖에서 재사용)를 관리한다.
두 CUDA 버전의 **forward + backward + residual 및 명시적 모듈 선택**을 구현했고, GPU 검증 18건과 memcheck 10건(오류 0건)을 통과했다. [신규 구현 및 측정 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918/README.md).
기존 split은 비교 기준/fallback이다. 아래 표는 현재 배선이며 두 새 CUDA 버전의 완성을 뜻하지 않는다.
기존 large-D BK512 sweep은 입력 재사용 개선 전 결과다. 신규 기록은 최신 Triton과 새 CUDA를 다시 측정하며, 아래 기본 배선과 별도인 실험 경로다.

[확대 뷰어](../../tmp_kernel/transition/TRANSITION_STATUS.html) · [선택표·성능](../../tmp_kernel/transition/TRANSITION.svg) · [Forward](../../tmp_kernel/transition/TRANSITION_FORWARD.svg) · [Backward](../../tmp_kernel/transition/TRANSITION_BACKWARD.svg) · [Residual 융합](../../tmp_kernel/transition/TRANSITION_RESIDUAL.svg)

## 현재 연결

일반 Transition: `xn=LN(x); a=xn@Wa.T; b=xn@Wb.T; y=x+(SiLU(a)*b)@Ws.T`.
`D`는 입력 폭, `N=4D`, `M`은 펼친 행 수(pair이면 `B*L²`). 이 모듈에는 mask·dropout이 없다.

기본값은 `transition_residual_fusion=True`, `transition_h100_residual=True`, `transition_h100_save_xn=True`, `transition_triton_b2b=True`다.

| 경로 | Forward | 저장 / backward |
|---|---|---|
| A1: Triton SM90 D128/256, n4, BF16, M≥16384 | Triton LN → segmented b2b(expand+SwiGLU+squeeze+residual) | 추론도 xn 임시 작성 · 학습은 xn 보관 → 기존 stacked gate backward → cuBLAS 4회 → Triton LN+residual |
| A2: 나머지 Triton shape / force_split | Triton LN → expand+SwiGLU → squeeze+residual | xn 저장 → 같은 backward |
| B: H100 D128/256 | stats → CUDA b2b(LN+expand+SwiGLU+squeeze+residual) | 큰 학습 입력에서 x/stats/xn 저장 → saved-xn stacked gate backward → cuBLAS 4회 → LN+residual |
| C: H100 wide-D | stats/fold → CuTe LN-folded expand+SwiGLU → CuTe squeeze+residual | xn 별도 재계산 → separate dA/dB gate backward → cuBLAS 6회 → LN+residual |

- `engine_backend="triton"`: 지원 shape는 A1, 나머지는 A2. `transition_force_split=True` 또는 `transition_triton_b2b=False`는 A2.
- A1은 추론·학습에서 같은 forward를 사용한다. 기본 분리형은 같은 캐시 키를 쓰며 BM/BN/BK/BO/warps/stages는 CSV로 관리한다. 실험용 LN 융합형은 정규화·저장 모드를 별도 키로 구분한다. 1차원 row grid라 `GROUP_M`은 적용할 output-column grid가 없다.
- Auto native 지원: 정확히 SM90, BF16, n=4, D∈{128,256,384,512,768}, M%128=0.
- Auto D128/256: `transition_cuda_b2b=True`이면 B.
- Auto D384/512/768: M≥16384이면 C. 작은 wide token / 미지원 shape / native OFF는 A.
- Explicit `implementation="cute"`: 지원 shape에서 C. 위 auto 정책과 별도 선택이다.
- B의 D128/256 큰 학습 입력(M≥16384)은 xn을 같은 CUDA 커널에서 저장하고 Triton saved-xn backward를 쓴다. 추론에서는 저장하지 않는다. `transition_h100_save_xn=False` 또는 작은 M은 이전 재계산 경로다.
- B/C의 D128·M≥16384 LN backward는 측정상 빠른 Triton residual 커널이다. 그 외 eligible D≤512는 CUDA main+parameter reduce, 미지원/실패 시 Triton residual LN이다.
- C의 기본 backward는 Triton separate dA/dB다. Opt-in CuTe backward는 별도 경로다.

**세 경로 모두 residual 융합을 사용한다.** C의 `dA@Wa + dB@Wb`는 residual이 아닌 두 gradient의 합산이며 남아 있다.

## 성능과 검증

[최신 D128/256 segmented Triton](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-segmented-small-20260918/README.md):
두 폭 모두 LN 분리 + segmented b2b를 추론·학습 공통 기본 경로로 연결했다.
D128은 split 대비 추론 1.67–1.74배 / 학습 1.12–1.13배,
D256은 추론 1.28–1.35배 / 학습 1.07배다. FP32 affine을 유지한다.

[이전 full-output 기본 연결 및 큰 D 실험](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-triton-unified-forward-20260918/README.md):
D256 full-output b2b의 퇴행은 후속 segmented 구현에서 개선했다.
D384/512의 당시 추가 변형 3종은 split보다 느려 실험 호출로 유지한다.
이 기록은 후속 full-K 입력 재사용 개선 이후의 large-D 성능을 판정하지 않는다.

[D128/256 동일 b2b CUDA·Triton 비교](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-small-b2b-comparison-20260918/README.md):
LN·xn 저장·backward를 동일하게 맞춰도 CUDA 추론은 D128 약 1.14배, D256 약 1.36–1.38배 빠르다.
두 구현 모두 WGMMA를 사용하며 TMA·타일 크기·spill 차이는 실제 SASS와 NCU로 확인했다.

[Triton D384/512 b2b 구현·튜닝 결과](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-triton-wide-b2b-20260918/README.md):
LN 분리형과 LN 융합형을 구현했지만 기존 split보다 느려 실험용 명시 호출로 유지한다.
선택 커널의 WGMMA/HGMMA와 NCU 지표까지 확인했다. D384/512의 자동 배선은 split을 유지한다. D128/256은 후속 segmented 구현 A1으로 연결했다.

[D384/512 신규 실측과 b2b 제한 근거](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-wide-bench-20260918/README.md):
현재 H100 CuTe 경로는 D384 추론 1.06배, 학습은 거의 동률이다.
D512 추론은 1.24–1.30배, 학습은 1.04–1.05배다. L384/L768 pair 입력 기준이며 전체 config 탐색 결과는 아니다.

[학습 이득이 작은 이유와 비용 분해](PERFORMANCE.md) · [원본 구현·검증 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-b2b-savedxn-20260918/README.md)

추가 개발 후 D128의 최종 auto/Triton 속도비: L384 추론 1.71배 / 학습 1.11배, L768 추론 1.82배 / 학습 1.13배.
CUDA b2b가 xn을 저장하도록 바뀌어 이전 H100 backward의 추가 재계산을 제거했다. D128에서 L384 36MiB, L768 144MiB를 backward까지 보관한다.

## 근거 / 재생성

- [현재 개발본 소스 SHA-256](current-sources.json)
- [이전 설치본 배선 기록](LEGACY_20260917.md): fusion=False의 옛 shape 표와 구현 설명. 현재 기본값 표가 아니다.
- [Triton residual 구현 당시 기록](RESIDUAL.md), [기존 구현 감사](IMPLEMENTATIONS.md)

```sh
python3 scripts/render_transition_fusion.py
```

스크립트는 GPU 패키지 import 없이 개발본 소스와 node02 측정 JSON을 읽는다.
실선은 단일 커널의 융합 경계, 보라색은 cuBLAS 호출, 회색 점선은 복수 연산/선택 설명, 노랑은 HBM tensor다.
모든 입력·초기화·cast의 전체 launch 목록이나 config 최적성 검증을 뜻하지 않는다.
