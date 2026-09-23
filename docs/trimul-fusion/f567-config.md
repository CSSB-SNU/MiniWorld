# A 경로: F4 / Triton F5+F6+F7 설정과 검증

2026-09-16. BF16 양방향 TriMul **학습 forward**에 A를 연결했다. F4 LayerNorm은 별도이며, F5 출력 projection + F6 gate projection + F7 sigmoid·곱·dropout scale·residual을 한 Triton 커널에서 실행한다. backward에 필요한 projection과 gate는 저장한다. dropout 난수 생성은 커널 밖이다.

이전 `scripts/trimul_output_fusion.py`의 Python 설정 목록·`SELECTED`는 A/B 실험 재현용으로 남겼다. 새 엔진 구현은 [output_fused.py 스냅샷](f567-config/output_fused.py)이며, 실제 배포물은 [엔진 패치](../../patches/miniworld-engine-trimul-f567-config.patch)다.

## Config 공간

타일과 실행 스케줄은 [CSV](f567-config/config.csv)에서 읽는다. 커널 내부 고정 타일, 길이별 수동 winner, 고정 fallback config는 없다. `grid`와 `gmprobe` 모두 같은 전체 공간을 제공한다.

| 축 | 후보 |
|---|---|
| `BLOCK_M1` | 16, 32, 64, 128 |
| `BLOCK_N` | 32, 64, 128, 256 |
| `BLOCK_K` | 16, 32, 64, 128 |
| `GROUP_M` | 1, 2, 4, 8 |
| `num_warps` | 1, 2, 4, 8 |
| `num_stages` | 2, 3, 4 |

총 **3,072개**. Shape보다 과도하게 큰 N/K 타일과 중복 grouped-M 스케줄만 사전 제거한다. 레지스터·shared memory 용량은 임의 상수로 예측하지 않고 엔진의 compiler/resource tracker에 맡긴다. 명시적인 단일 config 검사에서는 큰 타일도 실행해 경계 마스크를 검사할 수 있다.

- M/N 모두 타일링한다. 마지막 M/N 타일과 불완전한 M 그룹을 처리한다.
- Projection의 실제 축 `KP`와 gate의 `KG`는 독립적이다. 각 GEMM은 자기 축 길이만 순회하고 따로 K 마스크를 적용한다.
- 두 GEMM의 **물리적 K 타일 크기는 같은 `BLOCK_K`**를 사용한다. 그 값은 위 CSV에서 탐색한다.
- Weight stride를 실제 주소 계산과 캐시 키에 반영한다. Activation/residual/drop scale은 contiguous 입력 계약을 검사한다.
- M 행 주소는 int64로 계산한다. 프로그램 간 atomic이나 split-K는 없다.
- BF16 projection/logit 반올림, forward의 FP32 sigmoid, backward용 BF16 gate 저장을 유지한다.

### 탐색에서 제외한 구현과 이유

독립적인 `BLOCK_K_P`/`BLOCK_K_G` 구현은 ragged shape와 일부 weight stride에서 오답이 발생했다. Compute Sanitizer가 두 번째 `tl.dot`에서 생성 코드의 shared-memory 범위 밖 읽기를 검출했다. 배리어·루프 변경만으로 안전성을 확보하지 못해 채택하지 않았다. 현재 공통 K 타일 구현은 최종 검사를 통과했다.

또한 `num_stages=1`은 L384의 `M64/N32/K64, GROUP_M=1, warps=4`에서 illegal memory access가 재현되어 지원 공간에서 제외했다. **단순히 느려서 제거한 후보가 아니다.** 이는 이번 구현·도구 체인에서 확인한 제한이며, 독립 K 타일이나 1-stage 파이프라인이 모든 구현에서 불가능하다는 뜻은 아니다. 거부한 구현과 원시 로그는 `runs/trimul_f567_config_20260916`에 보존했다.

## 엔진 배선과 캐시

`trimul_output_f567_train_triton`을 registry, driver, accuracy checker, axes에 등록했다. `configs_for()` → 표준 autotuner → `capture.flush()` → 기존 캐시 reader를 사용한다. 캐시 키에는 dtype, KP/KG/N을 담은 shape key, 실제 M/L, 양쪽 weight stride가 포함된다.

BF16 학습에서 `_ln_materialize` 다음 새 F567을 호출한다. 다른 dtype과 inference 경로는 기존 구현을 유지한다. 기존 backward의 saved tensor 순서는 유지한다.

모듈에서 도출한 L128/384/768 학습 계획 모두 새 op에 도달했고, registry 기반 build unit도 생성되었다. 따라서 `build all` 대상에 포함된다. **이번에 실제 생성한 새 op의 캐시는 H100·BF16·B1·pair/hidden128·L384/768 두 shape다.** 모든 길이·dtype·GPU의 캐시를 만들었다는 의미는 아니다.

| L | 선언 config | Shape pruning 후 측정 | 유한 시간 + 수치 검사 통과 | 자원 초과 |
|---|---:|---:|---:|---:|
| 384 | 3,072 | 1,728 | 1,726 | 2 |
| 768 | 3,072 | 1,728 | 1,726 | 2 |

Build winner는 둘 다 `M64/N64/K64, warps8, stages3`이며 GROUP_M은 L384=2, L768=4였다. 이는 **측정 결과**이고 코드에 고정하지 않았다. 새 프로세스에서 표준 캐시의 top-5 후보 사용도 확인했다. Runtime의 최종 선택이 build winner와 같다고 가정하지 않는다.

## 검증과 forward 성능

- **41 tests passed; Compute Sanitizer 0 errors.** M/N/K 나머지, 모든 선언 축 값, 불완전한 그룹, weight stride, 과대 타일, 입력 계약을 검사했다. 모든 가능한 shape×config 조합을 전수 검증한 것은 아니다.
- 두 운영 shape에서는 측정 가능한 config마다 출력·projection·gate를 검사한 뒤 캐시를 발행했다. 반올림을 맞춘 FP32 계산 대비 상대 L2 최대 약 0.161%였다. 이 값에는 BF16 출력 저장 오차가 포함된다.
- 기존 split 경로와 compiled 전체 모듈의 출력·모든 gradient 비교도 통과했다. 최대 상대 L2는 L384 약 8.77e-7, L768 약 7.23e-7이다. 위 FP32 기준 오차와 서로 다른 비교다.
- `torch.compile(dynamic=False, fullgraph=True)` 및 CUDA graph 재생을 검사했다. Graph replay 때 dropout 갱신도 확인했다.

H100 80GB, BF16, B1, pair/hidden128. 시간은 **forward만** 측정했다. Paired CUDA graphs, 12회 측정 × 각 30 replay의 중앙값이다.

| L | 기존 F4–F7 | A: F4 / F567 | 시간 감소 |
|---|---:|---:|---:|
| 384 | 0.2007 ms | **0.1673 ms** | **16.6%** |
| 768 | 0.7312 ms | **0.6268 ms** | **14.3%** |

같은 실행의 전체 TriMul training forward는 L384 0.5810→0.5455 ms, L768 2.1988→2.0931 ms였다. 이 staged package에는 일부 front 캐시가 없어 두 경로 모두 같은 heuristic 후보군을 사용했다. 이 실행 안의 비교에는 사용할 수 있지만, 이전 캐시 상태의 전체 forward 기록과 직접 비교하면 안 된다. L768 전체 시간에는 이상치도 있어 출력 구간 측정이 더 안정적이었다.

## 설치 상태와 재현

패치 9개 파일의 SHA-256, 전체 patch stack dry run, 정확한 이전 버전에 대한 적용, 재적용 idempotence를 검사했다. 루트와 `libs/team-gm`에 배포 파일을 함께 보존했다.

작성 시점 설치본은 기존 7-GPU 캐시 writer **13181**이 사용 중이다. 앞선 CuTe 패치 활성화 **13183** 성공 뒤 이번 패치를 적용하는 **13188 (`afterok:13183`)**을 예약했다. 현재 검증은 staged package에서 완료했으며 **설치본 적용 완료를 뜻하지 않는다.** 적용 결과는 `runs/trimul_f567_config_20260916/activation.json`에 기록된다. 이전 패치/소스 해시가 달라지면 자동 적용을 중단한다.

엔진의 native source identity는 Triton 파일도 포함해 전체 kernels 소스를 해시한다. 따라서 이번 패치도 native identity를 바꾼다. 이전 CuTe 캐시를 새 identity로 재검증한 것처럼 바꾸지 않았다. 새 identity에서 CuTe 캐시의 재검증/재생성은 별도 작업으로 남는다.

GPU 할당 안에서 새 캐시를 재현하는 예:

```bash
export PYTHONPATH="$PWD/runs/trimul_f567_config_20260916/package:$PWD/scripts:$PWD/libs/team-gm/src"
.pixi/envs/cu128/bin/python scripts/build_trimul_f567_cache.py --length 384 --output /tmp/f567-L384.json
.pixi/envs/cu128/bin/python scripts/check_trimul_f567_runtime.py --length 384 --split-source docs/trimul-fusion/f567-config/split_reference.py --output /tmp/f567-runtime-L384.json
```

기존 증거 파일을 덮어쓰지 않는 경로를 사용한다. 설치 활성화 후에는 staged `PYTHONPATH` 없이 설치본을 사용할 수 있다.

### 보존한 증거

- [L384 config 전체 측정](f567-config/safe_build_L384.json), [L768 config 전체 측정](f567-config/safe_build_L768.json)
- [L384 runtime](f567-config/final_runtime_L384.json), [L768 runtime](f567-config/final_runtime_L768.json)
- [41개 테스트 XML](f567-config/final_tests.xml), [memcheck 로그](f567-config/memcheck.log)
- [build 계획](f567-config/final_plan.json), [patch 검증](f567-config/patch-validation.json)
- [테스트 코드](../../tests/test_engine_trimul_f567_config.py), [캐시 빌더](../../scripts/build_trimul_f567_cache.py)
