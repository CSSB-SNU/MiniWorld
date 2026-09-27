# CUDA config 공간 확장 방향

2026-09-27 사용자 합의. v2.1 후속 개발 계획이며, 아래 확장이 구현되거나 GPU에서
검증됐다는 의미는 아니다. 현재 구현 범위는 engine `docs/releases/2.1.0.md`를 따른다.

## 탐색 단계

1. **Default**: 기존 측정 설정을 포함하는 작은 후보 공간. 일반 빌드의 출발점이다.
2. **Expanded**: 실제 구현이 지원하는 설정을 더 넓게 탐색한다. shared memory,
   register, layout 및 명령어 제약으로 불가능한 후보를 사전 제거한다.
3. **구조 variant**: 기존 설정 탐색으로 해결되지 않는 병목에는 새로운 CUDA 구현을
   추가한다. fusion 범위, producer/consumer 배치, 저장·재계산, reduction 방식을 다룬다.

CUDA/CuTe의 현재 공간은 위 단계로 통일돼 있지 않다. 작은 후보 집합, 넓은 전역 공간,
수동 측정표와 고정 schedule이 공존한다. 앞으로 계열별로 이 구분을 명시한다.
사용하지 않는 knob를 추가하거나 후보 개수만 늘리는 것은 목표가 아니다.

## 계열별 확장 후보

| 계열 | 기존 구현 안에서 먼저 검토할 축 | 구조 변경이 필요한 확장 |
|---|---|---|
| Fused Transition | CTA 수, DW replica 범위를 더 촘촘하게 탐색 | tile·pipeline 및 역할 배치 변경 |
| TriMul 추론 | 구현된 K1/K3 variant 조합 | 새 tile·buffer slot·누적 방식 variant |
| CUDA LayerNorm | block, warps, waves, reduction launch 설정 | reduction 분할 및 주변 연산 fusion |
| CuTe GEMM | 기존 공간이 놓치는 shape와 설정 조합 | 기존 template이 표현하지 못하는 구조 |

각 축이 현재 코드에서 실제로 지원되는지는 확장 전에 확인한다. 구조에 고정된 값은
숫자만 바꿔도 되는 독립 config 축으로 취급하지 않는다.

## 작업 기준

- 학습 token 길이는 **384·768**에 집중한다. atom·MPNN edge 길이는 별개다.
- 추론은 **기존 shape 공간을 유지**하며 GPU·shape별 설정과 variant를 선택한다.
- 해당 GPU/shape/mode에서 CUDA가 선택되면 쓰이지 않는 Triton 대안을 별도로 빌드하지
  않는다. 실제 실행되는 Triton 연산과 fallback 경로는 필요한 캐시를 유지한다.
- Triton의 작은 default와 명시적 global 탐색, smem 예측기는 유지한다.
- 대규모 반복 학습·추론에는 GPU별 fused CUDA 구현을 적극 검토한다. 특히 backward의
  중간 텐서 메모리 왕복을 줄이되, 모든 연산이 memory-bound이거나 모든 fusion이
  유리하다고 가정하지 않는다.
- 기존 측정 설정을 보존하고 실제 dispatch, 독립적인 F+B 정확성, changed-input
  CUDA Graph replay, sanitizer, 같은 GPU에서의 baseline/candidate 시간을 확인한다.
  부분 커널 시간만으로 승격하지 않고 전체 연산 및 F+B 효과를 판단한다.

GPU 자원이 없는 동안에는 후보 정의와 코드 정비까지만 진행할 수 있다.
후보 수가 적다는 사실은 최적화가 끝났다는 증거가 아니다.
