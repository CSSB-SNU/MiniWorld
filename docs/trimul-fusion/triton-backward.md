# Triton 양방향 TriMul backward 융합

2026-09-16. 기존 Triton A의 forward와 저장값을 유지하면서 입력 gradient 계산 두 구간을 구현·비교했다. H100의 출력 projection-aware backward 개발과는 별도의 Triton 변경이다.

## 연산과 커널 연결

| 구간 | 이전 연결 | 신규 연결 |
|---|---|---|
| B9+B10 | cuBLAS `mm` → `dx_gate` → cuBLAS `addmm_` | `_input_dual_bwd_kernel`: 두 GEMM과 합산, `dx_n` 직접 저장 |
| B11+B12 | 입력 LN backward → `dpair_LN` → Inductor `add_view` | `_ln_bwd_residual_kernel`: LN 미분·parameter gradient·residual 합산 |

```text
B9+B10:
  gate_grad  = BF16(dglogit @ Wg.T)
  front_grad = dconc.T @ W_stack
  dx_n       = BF16(front_grad + FP32(gate_grad))

B11+B12:
  dpair_LN, dgamma, dbeta = LN_backward(dx_n)
  dpair = BF16(FP32(BF16(dpair_LN)) + FP32(gy))
```

기존 BF16 중간 반올림 위치를 보존한다. 두 GEMM은 각각 FP32 accumulator를 사용한다. 부동소수점 누적 순서가 달라 backward가 비트 단위로 같다는 뜻은 아니다.

`InputLNResidual` autograd Function이 `(LN(pair), pair의 view)`를 함께 반환한다. 두 출력의 gradient가 같은 backward에 도착하므로, residual gradient를 **LN 미분 후** 더할 수 있다. Residual은 LN의 `dgamma/dbeta` 계산에 들어가지 않는다. Forward의 identity 출력은 view이며 큰 복사가 필요하지 않다.

F1–F7, B1–B8, mask와 dropout scale의 계산 위치는 기존 Triton A 수식을 따른다. `dconc`와 weight-gradient GEMM은 별도로 남는다. BF16의 두 입력 GEMM을 융합하며, 다른 dtype의 GEMM은 기존 `mm/addmm_` 경로를 사용한다.

## 메모리 왕복

각 융합은 논리적으로 `[L²,128]` BF16 중간 결과의 쓰기·읽기 한 번씩을 없앤다.

| L | B9+B10 절감 | B11+B12 절감 | 합계 |
|---|---:|---:|---:|
| 128 | 8 MiB | 8 MiB | 16 MiB |
| 384 | 72 MiB | 72 MiB | 144 MiB |
| 768 | 288 MiB | 288 MiB | 576 MiB |

이는 B1·D128 기준 논리적 전송량이다. L2 cache에 머문 전송과 실제 HBM 전송은 다르며, 이 표는 DRAM counter 실측값이 아니다. 기존 B10은 이미 in-place `addmm_`였으므로 별도 elementwise add 하나가 있던 것으로 해석하지 않는다.

## Config·캐시 관리

두 op를 registry, build driver, 독립 FP32 checker, axes, grid/gmprobe CSV에 등록했다. 타일·warp·stage·GEMM 배치 순서를 CSV로 탐색한다.

| 커널 | 축 | 선언된 조합 | D128에서 유효 조합 |
|---|---|---:|---:|
| B9+B10 | M16/32/64/128, N32/64/128/256, K32/64/128, group1/2/4/8, warps4/8, stages2/3/4 | 1,152 | 648 |
| B11+B12 | M1/2/4/8/16/32/64/128, K64/128/256/512/1024, warps1/2/4/8/16/32, stages1–6 | 1,440 | 270 |

LN은 전체 열을 한 번에 처리하는 경우와 열을 나눠 두 번 읽는 경우를 모두 탐색한다. 전체 열이 한 타일에 들어가면 반복문이 없으므로 stage 중복을 제거한다. GEMM은 N 타일이 하나면 의미가 같은 group 설정을 제거한다. 두 GEMM의 K 경계와 모든 operand stride를 처리한다.

최종 튜닝은 CUDA graph로 수행하고 기존 표준 Triton cache 형식으로 저장한다. L128의 일반 launch 측정은 graph 실행에 나쁜 config를 선택했기 때문에, 아래 최종 표에는 graph 기준으로 다시 튜닝한 결과만 사용한다. `build_trimul_backward_cache.py`에 재현 경로를 보존했다. 일반 `build all`의 기본 측정 방식과 구분해야 한다.

LN cache key는 `both_key(행 수)`다. 초기 드라이버의 `both_key(L)` 오류를 수정한 뒤 올바른 키로 다시 측정했다. Cache key를 바꿔 끼우거나 이전 시간을 새 source identity로 옮기지 않았다. Spawn된 CPU compile worker가 빌드 스크립트 본문을 재실행하지 않도록 main guard도 적용했다.

최종 검증에서는 새 두 op의 cache miss가 발생하면 검증을 실패 처리한다. 비어 있지 않은 fallback 후보 목록을 cache hit로 취급하지 않는다. 기존 forward의 일부 shape는 heuristic fallback을 사용하며 비교하는 네 경로에 같은 조건을 적용한다.

## 최종 측정

**기본 배선: B9+B10과 B11+B12를 모두 융합한다. 세 길이 모두 네 후보 중 가장 빨랐다.**

H100 80GB, BF16, B1, D/hidden128. 정적 compile 뒤 12라운드에서 순서를 번갈아 각 graph를 30회 재생한 중앙값이다. 아래 backward와 F+B에는 gradient reset이 포함되며 optimizer는 포함하지 않는다.

| L | 기존 backward | LN+residual만 | 두 GEMM만 | 둘 다 | 기존 / 둘 다 |
|---|---:|---:|---:|---:|---:|
| 128 | 0.2138 ms | 0.2118 ms | 0.2063 ms | 0.2044 ms | 1.046x |
| 384 | 1.2638 ms | 1.2341 ms | 1.2114 ms | 1.1813 ms | 1.070x |
| 768 | 4.8320 ms | 4.7264 ms | 4.6341 ms | 4.5328 ms | 1.066x |

| L | 기존 F+B | 신규 F+B | 기존 / 신규 |
|---|---:|---:|---:|
| 128 | 0.2998 ms | 0.2912 ms | 1.030x |
| 384 | 1.7924 ms | 1.7068 ms | 1.050x |
| 768 | 6.9420 ms | 6.6363 ms | 1.046x |

### 빌드 winner

| L | B9+B10 (M/N/K, group, warps, stages) | B11+B12 (M/K, warps, stages) |
|---|---|---|
| 128 | 64/128/64, group1, warps4, stages4 | 64/128, warps4, stages1 |
| 384 | 64/128/64, group1, warps4, stages3 | 32/128, warps2, stages1 |
| 768 | 64/128/64, group1, warps4, stages3 | 64/64, warps4, stages4 |

### 수치 차이

| L | 출력 상대 L2 | 입력·parameter gradient 최대 상대 L2 |
|---|---:|---:|
| 128 | 0 | 0.00305856 (0.306%) |
| 384 | 0 | 0.00275746 (0.276%) |
| 768 | 0 | 0.00281879 (0.282%) |

Forward 출력은 동일했다. Gradient 차이는 BF16 GEMM 누적 순서 변경에서 발생한다. 따라서 정확도가 0.01% 이내라거나 bitwise 동일하다는 주장은 하지 않는다.

### 구간별 시간

Eager 연산 경계를 CUDA graph로 재생한 값이다. 위 compiled 모델 전체 backward 시간과는 측정 범위가 다르며 gradient reset은 포함하지 않는다.

| L | B9+B10 기존 → 융합 | 속도비 | B11+B12 기존 → 융합 | 속도비 |
|---|---:|---:|---:|---:|
| 128 | 0.0191 → 0.0153 ms | 1.251x | 0.0146 → 0.0118 ms | 1.240x |
| 384 | 0.1620 → 0.1365 ms | 1.187x | 0.1050 → 0.0762 ms | 1.379x |
| 768 | 0.6110 → 0.5136 ms | 1.190x | 0.3554 → 0.2673 ms | 1.330x |

## 검증 범위

- 단위 검사 **28건 통과**, Compute Sanitizer memcheck **0 errors**. M/N/K 나머지, operand stride, BF16 반올림, FP32 LN, tiled/covering LN, eager/static compile에서 residual의 gradient 위치를 검사했다.
- 각 길이에서 LN 270개, dual GEMM 648개를 탐색했다. GEMM 2개씩은 실행 자원 한도로 제외됐고, 나머지 총 2,748개 측정은 FP32 수식 및 명시적 BF16 반올림 검사를 통과했다.
- 전체 모델은 holed residue mask를 사용하고 출력·입력·모든 parameter gradient를 기존 Triton backward와 비교했다. 정확도 기준은 상대 L2이며 optimizer나 장기 학습 수렴 검증은 포함하지 않는다.
- 별도 production 검사는 고정된 dropout scale과 전부 0인 scale을 사용한다. 후자의 출력은 원래 pair, 입력 gradient는 정확히 `gy`, 모든 parameter gradient는 정확히 0이어야 한다.
- 정적 `torch.compile(fullgraph=True, dynamic=False)`, CUDA graph replay, 실제 CUDA kernel 이름과 새 op cache hit를 확인한다.

## 배포·재현

- [Backward 전체 SVG](../../tmp_kernel/trimul/TRIMUL_BACKWARD.svg) · [L768](../../tmp_kernel/trimul/TRIMUL_BACKWARD_L768.svg) · [확대 뷰어](../../tmp_kernel/trimul/TRIMUL_STATUS.html)
- [엔진 패치](../../patches/miniworld-engine-trimul-backward-fusion.patch) · [SHA-256 manifest](../../patches/trimul-backward-fusion-manifest.json)
- [튜너](../../scripts/build_trimul_backward_cache.py) · [4경로 비교](../../scripts/benchmark_trimul_backward_fusion.py) · [구간별 비교](../../scripts/benchmark_trimul_backward_components.py) · [production 검사](../../scripts/check_trimul_backward_runtime.py)
- [단위 검사](../../tests/test_engine_trimul_backward_fusion.py)

GPU 할당 안에서 설치본을 확인하거나 같은 소스·환경으로 새 캐시를 만드는 명령:

```bash
.pixi/envs/cu128/bin/python scripts/check_trimul_backward_runtime.py --length 384 --output /tmp/trimul-backward-check.json
.pixi/envs/cu128/bin/python scripts/build_trimul_backward_cache.py --op ln --length 384 --output /tmp/trimul-ln-cache.json
```

새 cache의 검증 범위는 H100·BF16·B1·D128·L128/384/768이다. 다른 width·length·장치의 캐시가 모두 완성됐다는 뜻은 아니다. 엔진의 native source identity는 Triton 파일도 포함하므로 이 소스 변경 후 기존 CuTe cache의 identity가 달라진다. CuTe 캐시의 재검증·재빌드는 이 작업에 포함하지 않았다.


### 설치 기록과 원시 결과

설치본과 MiniWorld/team-gm의 패치 목록에 반영했다. [적용 기록](triton-backward/activation.json), [전체 패치 stack·idempotence 검사](triton-backward/patch-validation.json), [memcheck](triton-backward/memcheck.log), [테스트 XML](triton-backward/tests.xml), [build plan](triton-backward/plan.json).

- [L128 모델 비교](triton-backward/graph_benchmark_L128.json) · [L384](triton-backward/graph_benchmark_L384.json) · [L768](triton-backward/graph_benchmark_L768.json)
- [L128 구간별 비교](triton-backward/graph_components_L128.json) · [L384](triton-backward/graph_components_L384.json) · [L768](triton-backward/graph_components_L768.json)
- [고정 dropout 검사 L128](triton-backward/production_L128.json) · [L384](triton-backward/production_L384.json) · [L768](triton-backward/production_L768.json)
- [커널 소스](triton-backward/backward_fused.py) · [모델 배선 소스](triton-backward/bidirectional.py)

GEMM의 `M128/N128/K128/stages4` 두 warp 설정은 shared memory 262,144 bytes를 요구해 H100의 block 한도 232,448 bytes를 넘었다. [실제 OutOfResources 기록](triton-backward/rejected_configs.json). 이 config들은 무한대 시간으로 제외했으며 runtime의 top-5 후보에는 들어가지 않는다.

기존 캐시 빌드 13181은 정상 종료했고, 이전 F567 적용은 중간에 추가된 cache-only 패치의 순서를 검증한 뒤 완료했다. 새 backward 설치가 이전 캐시 writer의 동작 중 소스를 바꾸지는 않았다.

설치 후 새 프로세스에서 L384의 고정 dropout·모든 gradient·실제 cache hit·CUDA graph replay를 다시 확인했다. [설치본 실행 기록](triton-backward/installed_L384.json).
