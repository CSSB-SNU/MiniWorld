# 왜 Transition 학습 이득은 약 5%인가

## 결론

**H100 forward는 빨라졌지만, backward는 그대로인 것이 아니라 xn 재계산 때문에 일부 더 비싸다.**
또 큰 비용인 gate-backward와 cuBLAS GEMM 대부분은 이번 residual 융합에서 바뀌지 않았다.
기준 Triton에도 이미 squeeze/residual과 LN/residual이 융합되어 있다.

## 동일 모듈 벤치마크

node02 H100, B1, D128, BF16, n4, nonzero squeeze, static compile(`dynamic=False`, partial allowed, 관측 graph 1개), manual CUDA graph.
학습은 forward+backward이며 optimizer 제외. 두 capture의 median을 평균했다.

| L | 추론 FWD ms: Triton → H100 | 학습−추론 추정 ms: Triton → H100 | 전체 학습 ms: Triton → H100 | 전체 절약 ms |
|---|---:|---:|---:|---:|
| 384 | 0.275 → 0.160 | 0.774 → 0.836 | 1.049 → 0.996 | 0.053 |
| 768 | 1.036 → 0.567 | 2.939 → 3.198 | 3.975 → 3.765 | 0.210 |

**차감 열은 BWD 직접 타이밍이 아니다.** 독립적인 추론/학습 측정의 차이를 활용한 비용 추정이다.
L768에서 추론 FWD 절약 약 0.469ms 중 약 0.259ms가 차감 추정의 BWD 증가로 상쇄되어 전체 절약은 약 0.210ms다.
단순히 “backward는 똑같으니 forward 비중 때문에 희석된다”로만 설명하면 이 차이를 놓친다.

## 실제 소스의 차이

- Triton: `input_ln_residual`이 만든 **xn을 forward에서 저장**한다. Backward는 `_transition_expand_gatebwd_savedxn_stacked` → `NORMALIZE=False`로 이를 읽는다.
- H100 b2b: forward에서는 전체 xn/h tensor를 저장하지 않는다. `_fused_bwd(... has_xn=False ...)` → `_transition_expand_gatebwd_stacked` → **`NORMALIZE=True`**로 xn을 tile마다 재계산하고, weight gradient에 사용할 xn도 HBM에 쓴다.
- **a/b/h 재계산은 두 경로 모두 한다.** H100에만 있는 추가 차이는 xn 정규화 재계산·출력이며, 두 specialization의 config 선택도 다를 수 있다.
- D128의 gate-backward는 H100 경로에서도 Triton이다. 큰 pair의 LN residual도 빠른 Triton 구현을 선택한다. 주요 cuBLAS GEMM 4회도 유지한다.

## L768 CUDA trace

최종 측정 JSON의 `traces["1-training-triton"]` / `traces["1-training-h100"]`.

| 항목 | Triton 정책 | H100 정책 |
|---|---:|---:|
| `_transition_expand_gatebwd_kernel` | 1.244ms | 1.485ms |
| 주요 cuBLAS GEMM + split-K reduce 합 | 1.476ms | 1.473ms |
| `_ln_bwd_residual_kernel` | 0.231ms | 0.228ms |

Gate-backward specialization은 약 **19% / 0.241ms** 더 느리다. 저장 정책 차이와 일치하지만,
정규화 자체·추가 store·타일 설정 각각의 기여를 이 trace만으로 분리하지는 않았다.
H100 trace에서 gate-backward 약 39.4%, cuBLAS 약 39.1%, LN residual 약 6.1%다.
즉 약 79%는 gate 재계산과 GEMM에 쓰이며, residual epilogue 추가만으로 크게 줄어들 영역이 아니다.

프로파일은 torch.profiler의 한 번 실행이며 CUDA graph 반복 측정의 phase 분해가 아니다.
Triton trace에는 첫 input LN이 누락되어 전체 합을 완전한 학습 시간으로 쓰지 않는다.
L384/D512 등 일부 trace는 CUDA 이벤트가 비어 있어 비율 계산에 사용하지 않았다.
NCU의 stall/roofline 분석을 새로 수행한 결과가 아니다.

## 다음 최적화 지점

1. H100 b2b에서 xn을 함께 저장하고 saved-xn backward를 쓰는 후보와 현재 재계산 방식을 학습 전체 시간으로 비교한다. 추가 forward store 때문에 무조건 이긴다는 보장은 없다.
2. 현재 약 39%인 D128 gate-backward를 별도로 최적화한다. 이것이 아직 H100 native로 대체되지 않은 큰 부분이다.
3. cuBLAS와 LN residual은 이번 trace에서 두 경로가 거의 같다. 이들에 이번 약 0.26ms 추가 비용의 원인을 돌릴 근거는 없다.

[원본 summary](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-hopper-residual-20260918/summary.json) · [L768 trace 포함 JSON](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-hopper-residual-20260918/bench-L768-D128.json) · [Backward SVG](../../tmp_kernel/transition/TRANSITION_BACKWARD.svg)
