# Transition H100 작업 시작점 — 2026-09-18

## 현재 개발 범위 (2026-09-18 갱신)

- 대상 채널 폭은 **D128/256/384/512**다. **D768 신규 구현·최적화·벤치 계획은 제외한다.**
- L768은 행 수를 정하는 길이이며 D768과 다르다. L384/L768 비교는 유지한다.
- D384/512는 현재 Triton과 H100 CuTe 경로를 같은 공식 모듈 fixture로 비교한다.
  현재 CUDA b2b는 D128/256만 구현되어 있어 wide-D b2b 수치를 대신 기입하지 않는다.
- **측정 완료:** [node02 D384/512 결과](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-wide-bench-20260918/README.md).
  D384 학습은 거의 동률이고 D512 학습은 H100이 약 4–5% 빠르다.
- **Triton b2b 시도 완료:** [두 구현 및 튜닝·NCU 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-triton-wide-b2b-20260918/README.md).
  현재 구조는 split보다 느리다. Wide-D H100 b2b 후속 개발은 producer/consumer 또는
  출력 누적값 분산 방식을 바꾸는 설계가 필요하며, 단순 이식은 성능 개선 근거가 없다.
- 아래는 초기 감사 기록이다. 현재 배선은 [README](README.md)를 따른다.

이 문서는 구현 전 점검이다. 후속 [H100 residual fusion 구현 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-hopper-residual-20260918/README.md)에 현재 배선과 실측을 정리했다.

TriMul 개발 checkout `runs/trimul_sm90_parity_20260917/engine`의 현재 소스를 확인했다.
정적 배선 감사이며 이번 단계에서 GPU 재측정이나 커널 변경은 하지 않았다.
일반 Transition, BF16, expansion=4가 대상이다. ConditionedTransition과 구분한다.

## 현재 기본 배선

`transition_residual_fusion=True`가 기본값이며 학습/추론 모두 이전 H100 자동 분기보다
먼저 검사한다. 일반 triton/miniworld 모듈은 아래 경로를 사용한다.
명시적인 `implementation="cute"`는 별도 경로다.

| 단계 | 연산 | 현재 구현 |
|---|---|---|
| F1 | LN → xn 저장, identity 전달 | 공통 Triton input_ln_residual |
| F2 | expand GEMM 두 개 + SwiGLU → h | transition_fwd_kernel |
| F3 | squeeze GEMM + residual → y | _squeeze_residual_kernel |
| B1 | dh = dy @ Ws | cuBLAS |
| B2 | xn에서 a/b/h 재계산 + SwiGLU 미분 → h, packed dAB | _transition_expand_gatebwd_kernel |
| B3 | dWs = dy.T @ h | cuBLAS |
| B4 | dWab = dAB.T @ xn | cuBLAS |
| B5 | dxn = dAB @ packed weights | cuBLAS |
| B6 | LN backward + identity gradient | 공통 Triton _ln_bwd_residual_kernel |

위 B 번호는 Transition의 순서이며 TriMul의 같은 번호와 다른 연산이다.
B2는 NORMALIZE=False/STORE_H=True/STACK_DAB=True로 실행한다.
F3와 B6 모두 기존 BF16 반올림 뒤 residual을 더하는 순서를 유지한다.
Backward의 중심은 GEMM 호출 4회, gate 재계산 1개, LN+residual 1개다.
weight packing/gradient 초기화 등의 보조 launch는 별도다.
이 모듈 자체에는 dropout이나 모델 mask 인자가 없다.

## H100 구현 후보

후속 [기존 구현 전체 감사](IMPLEMENTATIONS.md)에서 hand-CUDA b2b v14의 과거 승리와
CUDA gate backward의 D256/512 개선 기록을 확인했다. 아래 B2 후보를 이에 맞춰 보강했다.

| 우선 후보 | 기존 구현에서 활용할 부분 | 그대로 연결하기 어려운 점 |
|---|---|---|
| B2 | 우선 hand-CUDA stacked gate backward, CuTe WGMMA epilogue도 비교 | CUDA는 raw x의 LN 재계산을 saved-xn 입력으로 바꿔야 한다. CuTe wrapper는 dh 복제/보정 tensor/interleaved 출력 준비를 제거하거나 변경해야 한다 |
| F2 | CuTe dual GEMM + SwiGLU | 기존 구현은 LN을 weight에 접어 GEMM 뒤 보정한다. 현재 LN → BF16 xn → GEMM 경계와 다름 |
| F3 | CuTe squeeze GEMM, 별도 residual C operand | GEMM 결과의 BF16 반올림 뒤 residual을 더하는 계약을 확인해야 하며 config 공간도 현재 Triton과 다름 |

이는 구현 후보의 우선순위이며 측정된 병목 순위가 아니다. F1/B6는 유지하고 세 GEMM
후보의 NCU 비중부터 확인하는 것이 합리적이다. B6에는 TriMul의 relaxed atomic 개선이
이미 연결되어 있다. 예전 hand-CUDA b2b는 LN부터 squeeze까지 융합 경계가 더 넓고,
hand-CUDA gate backward는 raw x에서 LN을 재계산하므로 직접 대체품으로 보지 않는다.
WGMMA와 일부 TMA 로딩 구현은 재사용 검토 대상이다.

## Config와 복사

F2/F3 각각 1,296개, B2 1,080개의 선언 조합이다.
BM/BN/BK/GROUP_M/warps/stages 축을 모두 가진다. 이는 튜닝 완료 수가 아니다.
같은 config 축과 의미를 유지하며 H100 구현의 불가능한 설정은 명시적으로 제외해야 한다.

현재 `torch.cat((Wa,Wb),0)`는 activation이 아닌 weight packing이다.
D128/n4/BF16에서 256KiB이고 L에 따라 커지지 않는다. dAB는 최종 버퍼에 직접 쓴다.
과거 pair D128 실측에서는 이를 GEMM 두 개+add로 나누는 것이 오히려 느렸다.

## 다음 측정 기준

Pair D128의 L384/L768부터 확인하고 D384/D512를 추가 비교한다.
D768은 개발 및 측정 계획에서 제외한다.
동일한 저장값·반올림·잔차 계약을 유지하고 현재 Triton과 대조한다.
공식 fixture의 squeeze는 기본 zero init이므로 비영 projection으로 전체 gradient를
별도 검증한다. 공통 LN 개선 이후의 최신 커널별/전체 시간은 아직 재측정하지 않았다.

근거: [소스 해시와 config 축](h100-next-sources.json),
[잔차 융합 검증](RESIDUAL.md), [weight cat 실측을 포함한 이전 감사](AUDIT.md).
