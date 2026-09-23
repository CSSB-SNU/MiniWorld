# Transition 두 버전 개발 기준

2026-09-18. 개발 단위는 **forward + backward + residual**이다.
두 버전은 같은 수식과 융합 경계를 유지하고, K축 입력을 읽고 재사용하는 방식으로 구분한다.

| 고정 이름 | `streamed_k` | `full_k` |
|---|---|---|
| 입력 처리 | hidden 타일마다 K를 BK 단위로 읽어 누적 | 전체 K 입력을 hidden 루프 밖에서 읽고 재사용 |
| BK 조건 | `BK < D` | Triton: `BK = next_power_of_2(D)` / CUDA: `BK = D` 또는 padding |
| D384 예 | BK64/128/256 | Triton BK512 / CUDA BK384 또는 BK512 |
| forward 융합 | expand A/B + SwiGLU + squeeze + residual | 동일 |
| 학습 저장 | xn 및 입력 LN backward에 필요한 x/통계/γ | 동일 |
| backward 수식/융합 | 아래 공통 saved-xn stacked backward | 동일; 중복 코드를 강제로 만들지 않음 |
| Triton 구현 | `_segmented`의 `BK < D` 분기 | `_segmented`의 `BK >= D` 분기 |
| 새 CUDA 구현 상태 | **fwd+bwd·모듈 연결 및 검증 완료**, bounded sweep | **fwd+bwd·모듈 연결 및 검증 완료**, bounded sweep |

기존 split은 비교 기준과 fallback으로 유지한다. 세 번째 신규 개발 버전으로 확대하지 않는다.
LN 분리/융합, 추론/학습, CUDA/Triton은 위 두 버전 이름과 별도 축이다.
기존 full-output/packed/concat 실험도 새 버전 이름으로 섞지 않는다.

## 공통 연산 계약

```
xn = BF16(LN(x; gamma, beta))
a = xn @ Wa.T
b = xn @ Wb.T
h = BF16(SiLU(a) * b)
y = BF16(BF16(h @ Ws.T) + x)
```

- 대상: H100/node02, D128/256/384/512, expansion=4. D768은 계획에 포함하지 않는다.
- 기준 dtype: 입력·projection BF16, LN affine FP32. gamma/beta를 BF16으로 바꿔 성능을 비교하지 않는다.
- 1차 CUDA 비교는 **Triton과 동일하게 LN 분리**로 맞춘다. LN까지 합치는 별도 실험은 `ln_mode`를 기록해 구분한다.
- 두 버전 모두 일반 Transition이다. mask/dropout이 없는 현재 수식을 따른다. AdaLN/conditioned Transition은 별도다.
- 추론과 학습의 forward 수식·residual rounding 경계를 맞춘다. 저장 정책은 명시한다.
- 학습에서 xn을 backward까지 저장하며, 큰 h/a/b를 forward에서 추가로 저장하는 구현은 같은 조건의 비교로 취급하지 않는다.
- full-K는 논리적 입력 재사용을 의미한다. 실제 load, shared-memory 배치, spill 및 WGMMA 사용은 cubin/NCU로 검증한다.

## Backward를 포함한 공통 개발 단위

현재 LN 분리 Triton 두 버전은 `transition_wide_b2b` → `_NormalizedB2B` →
`swiglu_squeeze_backward`와 `input_ln_residual_bwd`를 공유한다.

| 단계 | 연산 / 출력 | 현재 구현 | 새 CUDA 버전의 작업 범위 |
|---|---|---|---|
| B1 | `dh = dy @ Ws` | cuBLAS | 동일 GEMM 유지 |
| B2 | saved xn으로 a/b 재계산 + SwiGLU 미분 → `h`, `dAB=[dA\|dB]` | Triton `_transition_expand_gatebwd_savedxn_stacked` | CUDA TMA/WGMMA로 같은 융합·출력 layout 구현 |
| B3 | `dWs = dy.T @ h` | cuBLAS | 동일 GEMM 유지 |
| B4 | `dWab = dAB.T @ xn` → dWa/dWb view | cuBLAS | 동일 GEMM 유지 |
| B5 | `dxn = dAB @ [Wa;Wb]` | cuBLAS | 동일 GEMM 유지; weight pack/cat 비용도 측정 |
| B6 | LN 미분 + residual identity gradient → dx, dγ, dβ | Triton residual LN backward | 기존 CUDA residual LN 구현 검증·재사용/개선 |

현재 B5 앞에는 `[Wa;Wb]`를 만드는 `torch.cat`이 있다. 제거되었다고 표기하지 않는다.
네 cuBLAS GEMM의 비용과 weight 준비 비용도 전체 학습 시간에 포함한다.
B3/B4/B5는 각각 별도 GEMM이며 위 표가 새로운 융합을 의미하지 않는다.

Residual은 forward squeeze epilogue와 backward LN dx epilogue 양쪽에 포함한다.
identity `dy`는 dγ/dβ에 더하지 않는다. gamma=0도 정상적으로 미분해야 한다.

새 CUDA backward는 두 버전의 같은 수학/저장 계약을 공유한다.
gate 재계산의 streamed/full-K 스케줄이 달라지는 경우에만 해당 구현과 튜닝 결과를 분리한다.
cuBLAS와 동일한 LN/reduction 코드를 버전별로 복사할 필요는 없다.

## CUDA 개발 순서와 완료 기준

**각 버전의 fwd만 완료하고 다음 버전으로 넘어가지 않는다.**

1. 공통 saved-xn 입력/출력, FP32 affine, residual rounding, stacked dAB 계약을 고정한다.
2. `streamed_k`: CUDA forward + gate backward + LN/residual backward를 연결하고, 출력과 여섯 gradient를 함께 검증한다.
3. `full_k`: 같은 계약으로 forward/backward를 구현·검증한다. 공통 backward는 재사용한다.
4. 두 버전을 같은 D/L/입력/저장 정책에서 튜닝하고, fwd/bwd/전체 학습을 함께 비교한다.
5. 실제 SASS의 TMA/WGMMA, NCU의 spill/대기/트래픽으로 병목을 확인하고 다시 최적화한다.
6. 검증·측정된 결과로만 shape별 dispatch를 선택한다. 기존 CUDA/CuTe 호출을 새 CUDA 버전으로 표시하지 않는다.

기존 CUDA `transition_b2b_kernel.cu`는 D128/256의 fused-LN/full-K forward 재사용 후보다.
기존 `transition_gatebwd_kernel.cu`는 LN 재계산형 backward 재사용 후보다.
둘 다 새로운 LN 분리/saved-xn/FP32 affine 계약에 대한 완성본이 아니다.
지원 폭·반올림·저장 정책·dtype부터 대조한 후 필요한 부분을 이식한다.

## 튜닝과 결과 관리

- BM, BN, BK, BO, warps/warpgroups, stages를 각 버전에서 관리한다. 타일 방문 순서가 있는 2-D GEMM은 GROUP_M도 관리한다.
- full-K의 BK는 D로 결정되고, streamed-K는 `BK<D`인 후보를 탐색한다. 공통으로 의미 있는 축의 범위를 맞춘다.
- CUDA 명령/layout 제약으로 제외한 후보는 이유를 기록한다. shared-memory 초과, 컴파일 실패, 수치 실패를 빠른 후보 목록에서 조용히 숨기지 않는다.
- backend/variant/direction/source hash/shape/dtypes/LN 모드/저장 정책/설정을 기록한다. 미래 캐시를 등록할 때도 두 variant의 결과가 덮어써지지 않게 한다.
- L384/768을 우선 측정하며 D128/256/384/512 각각을 평가한다. 과거 다른 구현의 승자를 최신 버전의 튜닝 완료 근거로 쓰지 않는다.
- 성능 결과에는 순수 inference fwd, training fwd, bwd, fwd+bwd, peak memory를 함께 싣는다. cuBLAS와 커널별 시간도 분리한다.
- 정적 shape compile + CUDA Graph 조건을 맞춘 module benchmark를 최종 비교로 사용한다. NCU replay 시간을 module 성능으로 사용하지 않는다.
- 출력 + dx/dγ/dβ/dWa/dWb/dWs, 비영 squeeze, gamma=0, residual identity, tail row, compile/graph, memcheck를 검증한다. 허용 수치오차와 실제 오차를 함께 기록한다.

## 현재 증거의 범위

- `streamed_k` large-D 선택 설정(BM64/BN128/BK64/BO128/W8/S3)은 [2026-09-18 NCU](../../runs/transition_large_ncu_20260918/README.md)에서 확인했다. 이 프로파일은 forward만 측정했다.
- `full_k` small-D는 [입력 재사용 개선 이후](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-segmented-small-20260918/README.md) 튜닝·학습 검증했다.
- 과거 large-D BK512 sweep은 입력 재사용 개선 전이다. 새 CUDA 기록에서 최신 Triton full-K도 다시 측정했다. 두 backend 모두 제한된 후보 비교이며 전역 최적성을 주장하지 않는다.
- 두 새 CUDA 버전은 `Transition(implementation="cuda", cuda_variant=..., cuda_forward_config=..., cuda_backward_config=...)` 또는 `cuda/variants.py:transition`으로 명시적으로 호출한다. `engine_backend="auto"`에서 선택하며, Triton 강제 모드는 명시 CUDA와 충돌을 보고한다. 별도 native LN → TMA/WGMMA b2b → saved-xn native gate backward → cuBLAS 4회 → native LN/residual backward로 연결했다.
- 출력 warpgroup 1개는 기존 CUDA의 WGMMA RS register 전달을 재사용한다. 여러 warpgroup은 shared H를 공유하며, 마지막 A 가중치 stage를 재사용할 수 있으면 별도 H 버퍼를 할당하지 않는다.
- CUDA backward는 dh도 TMA로 읽고 h/dAB를 128-bit vector store로 내보낸다. LN은 FP32 affine을 유지하며 dγ/dβ에 residual dy를 섞지 않는다.
- [새 CUDA 구현·검증 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918/README.md). GPU 18건과 unfiltered memcheck 10건(오류 0건), import/dispatch 47건을 통과했다. native bounded sweep과 large-D 후속 후보 비교를 완료했다. 자동 production dispatch와 build-all 캐시에 새 버전을 등록한 상태는 아니다.

현재 기계 판독 목록: [variants.json](variants.json).
두 버전과 공통 backward 그림: [TRANSITION_VARIANTS.svg](../../tmp_kernel/transition/TRANSITION_VARIANTS.svg).
