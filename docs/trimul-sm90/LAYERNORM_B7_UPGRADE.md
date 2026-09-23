# LayerNorm backward 개선과 B7 분리 실험

## 적용 결과

엔진 개발 checkout(`runs/trimul_sm90_parity_20260917/engine`, 기준 `8c7d8b39`)에 적용했다.
**공통 Triton LayerNorm backward를 개선했다. B7은 기존 통합 경로를 유지한다.**
실행 중인 학습 설치본을 교체하거나 원격에 push한 것은 아니다.

1. B4 persistent LayerNorm의 warp 탐색 범위를 `1,2`에서 `1,2,4,8,16,32`로 확대했다.
   기존 BM/BK/stages 축은 유지한다. L768에서는 `BM64/BK256/warps8/stages1`이 선택됐다.
   feature 전체를 덮는 기존 분기가 활성화되면서 feature 분할의 중복 row 읽기·계산을 줄인다.
2. 일반 LayerNorm, strided 출력 LayerNorm, 입력 LayerNorm+residual의 dγ/dβ 누적에
   `sem="relaxed"`를 적용했다. GPU 범위 atomicity, FP32 누적, BF16 residual 이전 반올림은 유지한다.
3. 측정한 H100 shape의 config 캐시를 갱신했다. 전체 grid/shape 재빌드는 아니다.
   확대된 grid와 변경된 source 때문에 다른 shape/다른 GPU의 기존 캐시는 재검증·재빌드가 필요할 수 있다.

## 커널 비교

단위 µs. 실제 dropout 학습 모듈에서 확보한 tensor를 사용해 7회 교대 측정한 중앙값이다.
B4 L768은 main kernel만, atomic B4/B11은 dγ/dβ 초기화 비용을 포함한다.

|연산|L|기존|개선|속도비|
|---|---:|---:|---:|---:|
|B4|384|119.20|102.85|1.159×|
|B11|384|85.02|74.30|1.144×|
|B4|768|539.95|458.27|1.178×|
|B11|768|271.70|239.73|1.133×|

B4 L768의 최종 dγ/dβ reduction 두 개까지 포함한 별도 비교도 559.42→478.14µs로 개선됐다.
가장 빠른 BM64 후보는 spill44가 있으며, spill 없는 BM32 후보보다도 실측이 빨랐다.
레지스터 수/점유율만으로 최적성을 판단하지 않았고, 전체 모듈에서도 같은 config의 이득을 확인했다.
이는 제한된 후보 탐색 결과이며 모든 tile/stage 조합의 전역 최적성을 주장하지 않는다.

### 분석한 LayerNorm backward 경로

|경로|분석·결정|
|---|---|
|B4 small-M strided atomic|각 CTA가 dγ/dβ를 누적하고 후속 kernel에서 소비. acquire/release 불필요. relaxed 채택.|
|B4 large-M canonical persistent|atomic 없음. 좁은 warp grid가 covering tile 선택을 막고 있었음. grid 확대+캐시 갱신.|
|m-major 전용 persistent|VEC_HINT on/off와 BM16/32/64, warp4/8/16 비교. 현재 B4보다 빠르지 않아 dispatch 유지.|
|B11+B12 atomic LN+residual|relaxed 누적과 BM64/BK128/warp4 선택. residual은 기존처럼 LN dx를 BF16으로 반올림한 뒤 더함.|
|B11+B12 persistent 실험|partial-gradient reduction 두 개까지 포함. L384에서는 느리고, L768도 개선한 relaxed atomic보다 느려 채택하지 않음.|
|일반 row-major atomic LN|같은 누적 패턴을 개선. rowscale 유무 모두 수치·성능 검증.|
|partial dγ/dβ 최종 sum|작은 buffer의 reduction. 이번에는 구현 변경 없이 비용에 포함해 확인.|

### atomic 변경의 근거

[Triton atomic_add 문서](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)에 따라
기본값은 acq_rel이다. 여기서는 atomic 결과로 다른 CTA에 activation 저장 완료를 알리거나,
같은 kernel 안에서 dγ/dβ를 소비하지 않는다. 초기화→누적→소비는 stream/graph dependency로 순서가 보장된다.
따라서 unrelated memory를 동기화하는 acquire/release를 제거해도 GPU 전체 atomic 누적은 유지된다.

실제 생성 코드도 확인했다(N128/BM64/warp4):

- 기존 PTX: `atom.global.gpu.acq_rel.add.f32`; SASS: `MEMBAR.ALL.GPU`, `ERRBAR`, `CGAERRBAR`, `ATOMG`, `CCTL.IVALL`.
- 개선 PTX: `atom.global.gpu.relaxed.add.f32`; SASS: `REDG`로 누적하며 위 ordering 명령이 사라졌다.

NCU full-set의 L384/N128, 같은 타일 비교(성능 표의 일반 benchmark 시간과 구분):

|rowscale|NCU 시간 기존→개선|membar stall sampled warps 기존→개선|DRAM bytes 기존→개선|
|---|---:|---:|---:|
|False|69.79→60.61µs|868→0|115.46→115.63MB|
|True|71.04→63.74µs|836→0|116.44→116.36MB|

전송량은 거의 같고 memory-ordering stall이 사라졌다. 줄어든 HBM bytes 때문에 빨라진 것으로 해석하지 않는다.
NCU의 interconnect metric 6개는 이 장비에서 unavailable이었으며 위 표에는 사용하지 않았다.
`--cache-control none`으로 실제 warm-cache 조건을 유지했으므로 profiler 시간만으로 성능 결론을 내리지 않았다.

### 일반 LayerNorm 미분

같은 config에서 acq_rel→relaxed만 비교. BF16 activation, FP32 weight/stats/parameter gradient.
행 수는 L². Atomic 초기화 포함.

|L|N|rowscale|기존 µs|개선 µs|속도비|
|---|---:|---|---:|---:|---:|
|384|128|False|69.06|60.59|1.140×|
|384|128|True|70.18|63.26|1.109×|
|384|256|False|177.82|158.82|1.120×|
|384|256|True|175.68|160.19|1.097×|
|768|128|False|227.81|194.94|1.169×|
|768|128|True|230.43|203.97|1.130×|
|768|256|False|647.33|574.30|1.127×|
|768|256|True|643.57|576.45|1.116×|

## B7 left/right 두 launch 실험

동일 최종 `(4H,M)` buffer에 각 launch가 자기 gradient slice를 직접 쓴다.
추가 allocation/cat/copy가 없으며 mask 곱의 BF16 반올림도 유지한다.
각 방식에서 BLOCK_E128/256/512/1024/2048, warp1/2/4/8/16을 비교하고 상위 후보를 다시 교대 측정했다.
주요 tensor 전송은 `6 reads + 4 writes`로 동일하다.

|L|통합 최선 µs|분리 최선 µs|통합/분리 속도비|
|---|---:|---:|---:|
|384|253.12|254.98|0.9927×|
|768|1006.78|1000.00|1.0068×|

L384에서는 약0.7% 느리고 L768에서는 약0.7% 빠르다. 전체 모듈의 추가 이득은 약0.1%여서 기본 경로를 바꾸지 않는다.
분리는 register pressure를 낮추지만 HBM 전송량을 줄이지 않는다. 단순 분리로 큰 가속을 얻지는 못했다.
실험 구현은 [experimental.py](../../runs/trimul_ln_b7_upgrade_20260917/experimental.py)에 보존했다.

## 전체 양방향 학습 모듈

H10080GB, B1/D=h128/BF16 mixed, **dropout0.25 + 매 replay마다 실제 RNG 갱신**,
static compile + CUDA graph, FWD+BWD(optimizer 제외). 공식 bench 모듈 fixture와 비영 출력 projection 사용.
현재 H100 혼합 경로의 F2/F567/B9+B10 CuTe 선택과 cuBLAS는 고정하고 공통 LN/B7만 바꿨다.
세 graph를 번갈아 12라운드 측정했다. 두 개선은 순수 Triton 경로에도 연결되는 공통 코드지만,
아래 전체 시간은 H100 혼합 경로의 측정치다.

|L|기존 ms|LN 개선 ms|기존/LN 개선|LN 개선+B7 분리 ms|
|---|---:|---:|---:|---:|
|384|1.575112|1.547592|1.0178×|1.545728|
|768|6.118920|5.997288|1.0203×|5.993160|

### 검증

- FP32/BF16, row-major/m-major, width128/137/256/384, row tail263, feature split/covering tile,
  rowscale 유무, residual 반올림 및 strided atomic: **56개 GPU 테스트 통과**.
- config ladder/axis/LayerNorm precision 관련 registry: **8 passed, 1 skipped**.
- B7 분리: BF16/FP32 × mask 없음/전부0/fractional × 세 tile/warp 조합, tail 포함 **18개 bit-exact 검증 통과**.
- 공식 모듈 correctness와 compile evidence, 모든 parameter/input gradient, RNG reset 재현성,
  fresh dropout의 output/dx 변화, graph gradient overwrite 검증 통과.
- 개선 모듈 vs 기존의 최대 gradient 상대L2 오차: L384 `1.35e-6` 미만, L768 `2.20e-4` 미만.
  전체 모델 학습 수렴 검증은 수행하지 않았다.

### 캐시 범위

수정된 source의 과거 timing과 새 timing을 섞지 않고 새 source/env/implementation identity로 기록했다.
각 entry에 **실제로 측정한 후보 subset만** coverage로 기록했다.
현재 갱신 대상은 H100의 atomic strided B4(L384), persistent B4(L768), LN+residual(L384/L768),
일반 atomic LN(L384/L768 × N128/256 × rowscale 유무)이다.
그 외 캐시를 모두 재빌드했다는 뜻이 아니다. 교체 전 파일은 `cache-before/`에 보존했다.

## 재현 자료

- [모듈 실행 스크립트](../../runs/trimul_ln_b7_upgrade_20260917/module.py), [L384](../../runs/trimul_ln_b7_upgrade_20260917/module-L384.json), [L768](../../runs/trimul_ln_b7_upgrade_20260917/module-L768.json)
- [기본 후보 실험](../../runs/trimul_ln_b7_upgrade_20260917/experiments_body.py), [추가 구현 비교](../../runs/trimul_ln_b7_upgrade_20260917/refine_body.py)
- [일반 LN/B7 경계 검증](../../runs/trimul_ln_b7_upgrade_20260917/generic-and-b7.json)
- [NCU 보고서](../../runs/trimul_ln_b7_upgrade_20260917/generic-atomic.ncu-rep), [기본 단위 CSV](../../runs/trimul_ln_b7_upgrade_20260917/generic-atomic.csv)
- [캐시 식별자 검증](../../runs/trimul_ln_b7_upgrade_20260917/cache-validation.json)
