# Triton Transition 감사 — 2026-09-17

후속 작업: [Residual 융합 커널을 구현하고 실측했다](RESIDUAL.md). 아래 내용은 융합 전 경로의 감사 기록이다.

범위: 일반 `Transition`, `engine_backend="triton"`, `compile_wrap="custom_op"`.
H100 80GB, BF16, n=4. ConditionedTransition/CuTe/CUDA 전용 경로의 성능 감사는 제외한다.

## 1. Residual: compile 뒤에도 별도 커널

Pair `[1,384,384,128]`에서 eager, 기본 static compile,
`max-autotune-no-cudagraphs`를 실제 실행했다. 모두 `dynamic=False`이며
profiler는 CUDA graph를 끄고 실행했다. 실제 MiniWorld 학습 스크립트의 기본 compile도
`model.compile(dynamic=False)`다.

| 실행 | 추론 FWD residual | 학습 FWD residual | BWD residual |
|---|---|---|---|
| Eager | 별도 ATen add | 별도 ATen add | 별도 ATen add |
| 기본 compile | `triton_poi_fused_add_view_0` | `triton_poi_fused_add_view_0` | `triton_poi_fused_add_1` |
| max-autotune | `triton_poi_fused_add_view_1` | `triton_poi_fused_add_view_1` | `triton_poi_fused_add_3` |

기본 compile의 학습 residual 단일 trace: FWD **35.17us**, BWD **35.20us**.
단일 profiler trace의 커널 시간이며 반복 측정한 전체 모듈 벤치 수치가 아니다.
이 커널들의 `fused`라는 이름은 squeeze/LN에 residual이 융합됐다는 뜻이 아니다.
생성 코드에 GEMM 또는 `layernorm_bwd` 호출 **이후 별도 add launch**가 있다.

원인: 모듈은 `_r(out) = out + x`로 residual을 붙인다. 기본 FWD의 squeeze는
cuBLAS 외부 호출이고, BWD LN은 custom op다. residual은 두 커널 API 모두의 입력이
아니므로 현재 구현이 직접 융합하지 않는다. compiler도 측정한 두 설정에서 이를
없애지 않았다. max-autotune의 추론 squeeze가 Inductor Triton GEMM으로 바뀌어도
residual은 별도로 남았다. 다른 shape나 다른 compiler 옵션 전체를 검증한 것은 아니다.

### `addmm`만 사용하면 해결되는가?

아니다. 이 환경에서 `torch.addmm(x, h, Ws.T)`는 **residual DtoD 복사 → beta=1 GEMM**이었다.
같은 `[M,D]=[147456,128]`의 squeeze+residual CUDA graph 측정:

- 기존 mm+add: 105.91us
- addmm: 107.43us

출력은 이 실험에서 동일했다. 복사 비용이 남고 이득이 없어 이 변경은 채택하지 않았다.
실제 해결 위치는 FWD squeeze epilogue와 BWD LN dx epilogue다.
**이번 감사에서는 residual 융합 알고리즘을 변경하지 않았다.**

## 2. 복사·중간 버퍼 점검

`swiglu_squeeze_backward`의 `torch.cat((Wa,Wb),0)`는 존재한다. 하지만
activation cat이 아닌 **가중치 packing**이다. D128/n4/BF16에서 256KiB다.
반면 dAB activation은 gate backward가 최종 packed 버퍼에 직접 기록하므로
별도 activation cat은 없다. h도 gate backward에서 한 번 재계산해 dWs GEMM에 사용한다.

같은 입력, CUDA graph의 해당 dx 계산만 비교한 결과(ms):

| 행 M | D | weight cat만 | cat + stacked GEMM | 두 GEMM + add |
|---:|---:|---:|---:|---:|
| 16,384 | 128 | 0.00141 | 0.01227 | 0.02478 |
| 147,456 | 128 | 0.00153 | 0.11978 | 0.17418 |
| 589,824 | 128 | 0.00158 | 0.45801 | 0.65522 |
| 384 | 384 | 0.00212 | 0.00910 | 0.01055 |
| 768 | 768 | 0.00428 | 0.02018 | 0.01863 |

Pair D128에서는 작은 복사를 없애려다 GEMM 1회와 activation add를 추가하면 오히려 느려진다.
큰 D·작은 M의 마지막 경우는 별도 튜닝 후보지만, 이 단일 측정만으로 shape 분기를
하드코딩하지 않았다. 양쪽 결과는 BF16 누적/반올림 순서도 다르므로 별도 수치 검증이 필요하다.

## 3. 실제 오류와 수정

### 입력과 gradient flatten

`view(...).contiguous()`는 앞선 차원들이 transpose된 입력에서 **contiguous에 도달하기 전에 실패**했다.
forward/backward 모두 `reshape(...).contiguous()`로 수정했다. 표준 연속 입력에는 복사가 추가되지 않는다.

### 가중치 / 출력 stride

- FWD expand는 Wa/Wb가 항상 row-major라고 가정했다. transposed weight 재현에서
  정상 contiguous 결과 대비 상대 L2 오류가 **1.249**였다.
- BWD gate kernel은 contiguous로 복사한 weight 포인터와 **원래 weight의 stride**를 함께 받았다.
  Wa와 Wb가 같은 stride라는 가정도 있었다.
- `empty_like`가 sliced gradient를 compact하게 할당해도 출력에 입력 gradient의 stride를 사용했다.

Wa/Wb의 stride를 독립적으로 커널에 전달해 원래 버퍼를 직접 읽도록 바꿨다.
잘못된 contiguous weight 복사도 제거했다. h/dA/dB 출력과 incoming gradient의 stride도
분리했다. stacked/non-stacked, saved-xn/recompute 공용 launch 네 곳 모두 수정했다.

### LayerNorm tiled variance

`BLOCK_K < N`의 `E[x²]−E[x]²`는 FP32 large-offset 입력에서 cancellation을 일으켰다.
N128, offset1000, BK64 재현의 상대 L2 오류는 약 **3.93%**였고 covering BK128은 약 0.00294%였다.

각 타일의 중심 분산을 구해 **Welford 방식으로 합산**하도록 수정했다.
기존의 두 input pass와 전체 covering-tile 분기는 유지한다. BF16 일반 학습이 3.93% 틀렸다는
의미가 아니며, 재현된 수치는 FP32 offset 스트레스 입력에 한정된다.

기존 engine main의 FP32 IEEE dot 및 affine 없는 LN 지원 수정도 설치본의 해당 세 파일에 함께 반영했다.

## 4. 타일링 / 설정

FWD expand와 BWD gate 모두 M/N 타일, K loop, tail mask, `tile_order(GROUP_M)`가 연결돼 있다.
기본 grid:

- BM: 32/64/128; BN: 32/64/128/256; BK: 16/32/64
- GROUP_M: 1/4; warps: 1/2/4
- stages: FWD 1..6, BWD 1..5
- FWD 1,296개, BWD 1,080개 후보. 고정 tile 한 개로 바꾸지 않았다.

모든 후보의 성능 최적성을 입증한 것은 아니다. 대표 tile 세 조합과
M/K/ND tail, 서로 다른 weight stride, sliced gradient를 수치 검증했고,
config 축 / grid와 tile 수 / GROUP_M 연결의 기존 정적 검사를 실행했다.

## 5. 검증과 적용

- GPU 수치 회귀 45건 + 관련 registry 검사 15건: **60 passed**.
- 레이아웃 회귀 27건을 설치본에서 Compute Sanitizer로 재실행: **27 passed, 0 memory errors**.
- patch helper 재적용: **Updated 0 engine files**. 수정한 세 파일은 engine 체크아웃과 SHA-256 일치.
- 독립 engine 브랜치 `perf/transition-triton-audit`, 로컬 커밋 `8869c377` (base `b06857c0`). 원격에는 아직 올리지 않았다.
- 레이아웃 비교는 cuBLAS BF16 partial-reduction을 끄고 addressing 오류를 격리했다.
  설정은 fixture 종료 시 복구한다. 실제 모듈 검증은 기본 설정을 사용한다.
- BF16 전체 모듈: Pair `[1,384,384,128]`, token `[1,384,384]`, `[1,768,768]`에서
  eager 대비 static compile 및 CUDA graph 출력/input gradient/모든 parameter gradient 검증 통과.
  별도 FP32 모듈 검사는 PyTorch fallback이므로 Triton 전체 모듈 통과 수에 포함하지 않는다.
- 설치본, 재설치 patch, root 및 `libs/team-gm`의 patch helper에 반영.
- source 변경으로 `transition_expand_swiglu_triton`,
  `transition_bwd_swiglu_recompute_triton`, `layernorm_fwd_saveact_triton`의
  기존 튜닝 cache는 새 source에 대해 재사용할 수 없다. 기존 캐시 빌드 잡은 이전 소스의
  독립 snapshot 작업이므로 이번 수정의 캐시를 만든 것으로 세지 않는다.
- 이번 검증의 cache miss는 heuristic 후보 탐색을 사용했다. 전체 cache 재빌드나
  수정 전후 전체 모듈 속도 개선을 주장하지 않는다.

## 재현 자료

[측정값과 커널 목록의 보존본](runtime-evidence.json).

- [기본 compile profiler](../../runs/transition_triton_audit_20260917/profile-before.log)
- [max-autotune profiler](../../runs/transition_triton_audit_20260917/profile-max.log)
- [cat/LN 재현](../../runs/transition_triton_audit_20260917/cat-ln-probe.log)
- [60건 검증](../../runs/transition_triton_audit_20260917/validation.log)
- [Compute Sanitizer](../../runs/transition_triton_audit_20260917/memcheck.log)
- [전체 모듈 compile/graph 검사](../../runs/transition_triton_audit_20260917/module-check.log)
- [portable patch](../../patches/miniworld-engine-transition-layout-audit.patch)
- [engine 작업 체크아웃](../../runs/transition_triton_audit_20260917/engine)

생성된 Inductor `output_code.py` 및 원시 CUDA trace는 같은 run 디렉터리의
`compile-before`, `compile-max`, `before-profile-pair384-d128`,
`max-profile-pair384-d128`에 보관한다.
