# Transition residual 융합 — 2026-09-17

**[융합 전후 SVG](../../tmp_kernel/transition/TRANSITION_RESIDUAL.svg)** · [확대 뷰어](../../tmp_kernel/transition/TRANSITION_STATUS.html)

일반 BF16 Transition의 Triton 경로에 forward와 backward residual 융합을 구현했다.
설치본과 재설치 patch에 반영했으며, **`transition_residual_fusion=True`가 기본값**이다.
새 프로세스의 일반 Transition은 기본적으로 이 Triton 융합 경로를 사용한다.
기존 CuTe/CUDA 자동 분기로 돌아가려면 `transition_residual_fusion=False`를 명시한다.
실행 중인 프로세스가 이미 읽은 설정은 소급 변경되지 않는다.

```python
from miniworld_engine import settings
settings.configure(engine_backend="triton", transition_residual_fusion=True)
# 모델 생성/compile 전에 설정
```

## 무엇을 합쳤는가

| 구간 | 기존 | 새 경로 |
|---|---|---|
| Forward | cuBLAS squeeze → z HBM → residual add | `_squeeze_residual_kernel`: GEMM + residual epilogue |
| Backward | LN backward → dx_LN HBM → residual gradient add | `_ln_bwd_residual_kernel`: LN dx + residual epilogue |

Backward는 TriMul에서 사용하던 **일반 LayerNorm+residual 커널을 재사용**했다.
`input_ln_residual`이 normalized activation과 identity를 두 출력으로 내보내고,
그 두 gradient가 LN custom Function 안에서 만난다. Identity gradient는 **dx에만** 더하며
LN의 dgamma/dbeta에는 더하지 않는다. SwiGLU 재계산, packed dAB, weight cat,
dh/dWs/dWab/dxn GEMM 구성은 유지한다.

Forward는 `BF16(GEMM)`을 먼저 반올림한 뒤 residual을 더한다. Backward도 LN dx를
BF16으로 먼저 반올림한 뒤 identity gradient를 더한다. 기존 반올림 경계를 없애서
수식이 달라지는 최적화가 아니다. 서로 다른 GEMM 구현의 FP32 누적 순서에 따른 차이는
수치 검증으로 확인했다.

Pair L384 / D128 / B1 / BF16에서 한 `[M,D]` 버퍼는 36MiB다.
각 방향에서 중간 버퍼의 write+read **72MiB**, 합계 **144MiB의 논리적 메모리 트래픽**을 제거한다.
이는 HBM 계측값이 아닌 tensor 크기 기반 계산이며, 실제 DRAM 트래픽은 cache hit에 따라 다르다.

## 모듈 벤치 결과

H100 80GB, BF16, B1, D128, n4, **Transition 1개 전체**, optimizer 제외.
엔진의 `bench_module_transition` 입력/모델/초기화/정확도 검사를 사용했다.
두 경로 모두 `torch.compile(dynamic=False)`이며 Inductor 내부 CUDA graph는 껐다.
그래프 측정은 하네스의 manual CUDA graph capture/replay다.
컴파일·autotune·warmup은 시간에서 제외하고, 12회 교대 측정의 median을 사용했다.

| L | 측정 구간 | CUDA graph | 기존 ms | 융합 ms | 속도비 |
|---:|---|---|---:|---:|---:|
| 128 | 추론 FWD | manual | 0.038688 | 0.036032 | 1.074× |
| 128 | 학습 FWD+BWD | manual | 0.157200 | 0.145888 | 1.078× |
| 128 | 학습 FWD+BWD | off | 0.794272 | 0.754272 | 1.053× |
| 384 | 추론 FWD | manual | 0.302056 | 0.277504 | 1.088× |
| 384 | 학습 FWD+BWD | manual | 1.165792 | 1.062112 | 1.098× |
| 384 | 학습 FWD+BWD | off | 1.181168 | 1.077256 | 1.096× |
| 768 | 추론 FWD | manual | 1.131656 | 1.032384 | 1.096× |
| 768 | 학습 FWD+BWD | manual | 4.406776 | 3.996136 | 1.103× |
| 768 | 학습 FWD+BWD | off | 4.430032 | 4.012112 | 1.104× |

수치는 PyTorch 대비가 아니라 **기존 Triton split 경로 대비**다. 공식 하네스는 squeeze를
zero initialization하므로, 성능 검사와 별도로 **모든 projection을 비영 가중치로 채운** 검증을 했다.
Token D384/D768은 정확도를 확인했으며 위 표의 Pair D128 성능을 그쪽에 일반화하지 않는다.

교대 측정 wrapper가 `graph.replay`만 보관하면 하네스 반환 후 입력/model tensor가 해제될 수 있다.
L768에서 이 수명 문제를 발견해 원래 callable·입력·parameters를 유지하도록 수정하고,
**위 9개 결과를 모두 다시 측정**했다. 이전 결과는 `invalid-operand-lifetime/`에 분리했고 표에 사용하지 않았다.
공식 하네스 자체는 원래 측정하는 동안 소유자를 유지한다.

## Profiler 확인

L384에서 eager와 static compile 모두 FWD는 LN → expand/SwiGLU → 새 squeeze/residual의
**세 커널**이었다. BWD 마지막은 `_ln_bwd_residual_kernel`이었다.
기존 `triton_poi_fused_add_view`와 `triton_poi_fused_add` residual 커널은 사라졌다.

## Config / cache 연결

- BM 32/64/128, BN 32/64/128/256, BK 16/32/64, GROUP_M 1/4,
  warps 1/2/4, stages 1..6: **1,296개 grid 후보**.
- 각 기존 config set에도 새 커널 항목을 추가했다. launch에서 타일을 하드코딩하지 않는다.
- M/N/K tail mask와 입력/weight/residual 각각의 stride를 처리한다.
- registry, driver, checker, builder switch, **실제 Transition module shape/options 목록**에 연결했다.
- 새 forward 커널의 완전한 cache sweep을 수행한 결과는 아니다. 이번 실행의 cache miss는
  엔진의 기존 **heuristic 24개 후보 탐색** 정책을 사용했다. 전체 config 공간을 다 튜닝했다는 뜻이 아니다.
- 기존 cache build 잡은 이 소스를 포함하지 않는다. 새 옵션 경로는 후속 빌드에서 별도로 튜닝해야 한다.

## 검증

- 대표 tile 3개 × ragged/strided 및 정렬 shape 2개: forward 단독 검증.
  기록된 최대 relative L2 차이 **2.23e-5 (0.00223%)**.
- 비영 projection의 전체 모듈 5개 shape: Pair L128/L384 D128, token D384/D768,
  ragged `[2,17,96]`. 출력 및 dx/dgamma/dbeta/dWa/dWb/dWs 비교.
- 해당 5개 shape 모두 eager, static fullgraph compile, CUDA graph capture/replay 검증 통과.
  이 실행에서 출력/dx/projection gradients는 동일했고 LN affine gradient만 atomic 합산 순서에 따른
  최대 5.02e-6 (0.000502%)의 상대 L2 차이를 보였다.
- registry/config/grid/driver 관련 검사 34건 통과.
- 새 GPU 회귀 검사 12건: n2/n4, tail/stride/반올림, 모든 parameter gradient 검증 통과.
- 설치본의 기본 runtime config로도 엄격한 relative L2 1e-4 기준의 12건을 통과했다.
- 선택 config를 고정한 Compute Sanitizer 검사: **12건 통과, 메모리 오류 0건**.
  전체 autotune을 sanitizer 아래 실행했을 때는 기존 expand 커널 후보 탐색 중
  `cudaErrorInvalidPc`가 발생했다. 일반 실행에서는 재현되지 않았으나 원인은 확정하지 않았다.
  따라서 이 결과는 선택 config의 메모리 검증이며, 1,296개 전체 후보의 안전성 검증은 아니다.

## 구현 및 재현 자료

- [커널 / autograd 구현](../../runs/transition_triton_audit_20260917/engine/src/miniworld_engine/kernels/transition/triton/residual.py)
- [정확도 로그](../../runs/transition_residual_20260917/verify.log)
- [실제 값과 source SHA-256](residual-evidence.json)
- [수정한 교대 벤치 wrapper](../../runs/transition_residual_20260917/benchmark.py)
- [설치본 기본 config 회귀 테스트 로그](../../runs/transition_residual_20260917/installed-default-tests.log)
- [선택 config 메모리 검사](../../runs/transition_residual_20260917/memcheck-selected-configs.log)
- [Profiler](../../runs/transition_residual_20260917/profile.log)
- [재설치 patch](../../patches/miniworld-engine-transition-residual-fusion.patch)
