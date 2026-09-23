# 공통 Triton 커널 프로파일: B7과 LayerNorm 비교

후속 구현·검증 결과: [LayerNorm backward 개선과 B7 분리 실험](LAYERNORM_B7_UPGRADE.md).
아래는 개선 적용 전의 프로파일 기록이다.

## 결론

**cuBLAS는 우선 유지한다. 공통 Triton에서는 B7보다 출력 LayerNorm backward(B4)의 개선 근거가 더 확실하다.**
B7의 비용이 크다는 예상은 맞지만, HBM 사용률90~91%이며 전송량도 현재 dense 입력/출력 크기와 대체로 일치한다.
B4는 L768에서 HBM45.9%, 점유율12.3%,235 registers/thread이고, 기존 warp1/2 제한을 넘어선
진단용 config에서 같은 커널·융합으로 **538.16→478.43µs,1.125×**를 확인했다.

![공통 Triton HBM 사용률](../../runs/trimul_common_profile_20260917/COMMON_KERNEL_PROFILE.svg)

## 측정 조건

- H100 80GB HBM3, B1/D=h128, BF16 mixed, dropout=0.25와 실제 RNG, static compile + stochastic CUDA graph.
- 공식 모듈 fixture 및 비영 weight/동일 입력. 현재 선택 경로의 F2/F567/B9+B10은 CuTe 그대로 두고 공통 Triton6개를 계측했다.
- L384/L768 각각 NCU full set, `--profile-from-start off --graph-profiling node --cache-control none`.
- Warmup 후1회 graph replay에 진입해 target kernel들을 profile. NCU의 kernel replay 계측 시간은 성능 벤치 latency와 구분한다.
- NCU CSV를 base units로 export했으며 모든12개 record의 time×bandwidth와 DRAM bytes 일치를 검증했다.
- 일부 cache는 현재 runtime subset을 사용한다. 전체 cache/tuning 완료나 모든 config의 최적성을 주장하지 않는다.

## 커널별 NCU

|구간|L384 HBM %|L768 HBM %|L768 L2 %|L768 점유율 %|L768 registers/thread|판단|
|---|---:|---:|---:|---:|---:|---|
|F1 입력 LayerNorm|77.8|87.7|78.9|59.1|47|대체로 양호; L768 대역폭 한계에 가까움|
|F4 출력 LayerNorm|73.4|78.8|73.9|12.1|193|개선 여지: strided layout·shared 통신·레지스터|
|B1 출력 gate·dropout 미분|84.1|91.1|80.7|74.0|23|양호; 특히 L768 대역폭 포화에 가까움|
|B4 출력 LayerNorm backward|59.2|45.9|54.1|12.3|235|가장 확실한 개선 대상; 좁은 warp 탐색·중복 row 계산|
|B7 입력 gate 미분·mask·gradient packing|90.3|90.9|83.4|17.5|168|시간은 크지만 현 전송량을 효율적으로 처리|
|B11+B12 입력 LayerNorm backward·residual|59.8|69.1|86.4|12.1|251|개선 여지는 있으나 단순 설정 변경의 이득은 작음|

HBM % 하나만으로 효율을 판정하지 않았다. 실제 DRAM bytes, source의 load/store,
shared-memory bank conflict, register/occupancy, local-memory instruction, 설정 A/B를 함께 확인했다.
12개 기본 profile에서 local load/store instruction은0이다. 낮은 점유율만으로 병목을 단정하지 않는다.
[NVIDIA memory 분석 설명](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

### B7: 큰 비용과 비효율은 다르다

B7은 left/right gradient2개와 saved preactivation4개를 읽어4개 gradient를 쓴다.
BF16의 dense load/store 크기는 `20 × H × L²` byte, 양방향 H=256이다. mask도 읽지만 캐시 재사용이 있다.

|L|6입력+4출력 dense bytes|NCU DRAM read+write|대역폭|HBM 사용률|NCU 시간|같은 bytes의 낙관적 HBM 시간|
|---|---:|---:|---:|---:|---:|---:|
|384|0.755GB|0.755GB|3.028TB/s|90.33%|249.31µs|225.20µs|
|768|3.020GB|3.058GB|3.046TB/s|90.88%|1003.84µs|912.29µs|

공유메모리 bank conflict와 local-memory instruction은 모두0이다. L768에서는 registers168/점유율17.5%지만
HBM 사용률은90.9%다. 따라서 레지스터나 점유율 수치 하나만 보고 B7이 나쁘다고 판단하면 틀린다.
이전 B7의 정수 나눗셈 제거 실험도 최선 config에서 가속을 만들지 못했다.

**같은 DRAM bytes를 유지한** 순수 대역폭 여유는 약10~11% 속도비다. 이는 모든 구현의 절대 상한이 아니다.
예를 들어 mask=0인 구간의 입력 읽기를 생략할 수 있는지 검토하면 전송량 자체가 달라질 수 있다.
다만 실제 padding 비율·메모리 sector 단위·NaN/반올림/zero semantics에 좌우되며, 이번에 구현하거나 가속을 측정한 것은 아니다.
현재 고정한 fusion/buffer 계약에서 TMA 또는 명령 정리만으로 큰 이득을 낼 근거는 약하다.

### B4: 실제로 개선 여지를 확인

현재 L768 선택은 `BM32/BK128/warps2`의 persistent kernel이며 단계 수는 런타임 선택에서2 또는3이었다.
입력 feature 폭은256인데 BK128이므로 grid의 feature 축은2다.
각 feature block은 전체256 feature의 row 통계를 계산하고 자기128 feature를 다시 읽는다.
즉 같은 source 안의 covering-tile 경로보다 계산·load instruction이 중복된다.

Persistent grid는 `132 SM × 2 waves`로 고정되고 기본 CSV의 `num_warps`는 **1·2만** 선언돼 있다.
큰 covering tile의 레지스터를 더 많은 warp에 분산하는 후보를 기본 탐색에서 선택할 수 없다.
원래 config 공간 안에서는 B4를 유의미하게 개선하지 못했다. 그래서 원인 확인용으로만 warp4/8을 추가했다.

|L768 B4|BM|BK|warps|stages|교대 측정 중앙값|
|---|---:|---:|---:|---:|---:|
|기준|32|128|2|2|538.160µs|
|진단 후보|32|256|8|1|478.432µs|

- 같은 source와 융합·입출력/partial-gradient 계약, 기존 호출에서 확보한 실제 tensor 사용.
-7라운드 순서 교대. **1.12484×**, 시간11.10% 감소.
- 후보 registers193/thread, spill0. 레지스터/thread 감소를 곧바로 occupancy 증가로 해석하지 않는다.
- 후보 dx 상대 L2 약2.67e-9, dgamma/dbeta는 기준과 동일. 단독 검증이며 전체 모델 동등성 검증 완료를 뜻하지 않는다.
- 이 후보는 **현재 선언 grid 밖**의 진단용 config다. 엔진/CSV/cache/운영 설치본에 적용하지 않았다.
- 두 final parameter reduction 비용은 이 B4 단독 시간에 포함되지 않는다. 전체 모듈 가속은 아직 측정하지 않았다.

### B4 후보의 추가 NCU 확인

같은 실제 학습 그래프에 진단 후보만 강제한 별도 NCU에서 원인을 확인했다.
기준 NCU는 stages3, 위 교대 벤치의 기준은 stages2이므로 profiler 시간과 benchmark 시간을 섞지 않는다.

|항목|기준|후보|
|---|---:|---:|
|NCU 시간|629.28µs|522.37µs|
|L1 global-load sectors|58,990,848|21,242,112|
|SM warp instruction count|223,452,000|114,195,072|
|DRAM read|654.40MB|608.75MB|
|DRAM write|312.84MB|310.86MB|
|HBM 사용률|45.85%|52.52%|
|점유율|12.31%|12.50%|
|registers/thread|235|193|
|local load/store instructions|0/0|0/0|

**점유율을 크게 높여서 빨라진 결과가 아니다.** covering tile로 중복된 global-load 요청과
계산을 줄인 것이 실측 근거다. DRAM read 감소보다 L1 요청 감소가 훨씬 크므로 L2에 적중하던
반복 읽기·명령 비용도 줄었다. Shared bank conflict 총수는 오히려 증가했다. 단일 카운터만으로
원인을 단정하지 않는다. 운영 config/소스는 유지했다.

- [후보 NCU](../../runs/trimul_common_profile_20260917/candidate-L768.ncu-rep)
- [비교 counter](../../runs/trimul_common_profile_20260917/candidate-counter-comparison.json)

### B7을 left/right로 나누는 경우

현재 dL/dR 입력은 별도 버퍼지만 B7 launch는 하나다. 같은 tile에서 양쪽을 계산하고
공통 dconc 출력의 각 slice에 직접 쓴다. 두 kernel로 나누어도 `3입력+2출력` 두 번이므로
총 `6입력+4출력`의 HBM bytes는 그대로다. 중간 cat을 추가할 이유도 없다.

분리하면 한 launch의 working set/레지스터 요구가 줄 수 있지만, 총 저장공간이나 L2 점유가
자동으로 절반/두 배가 되는 것은 아니다. L2 residency는 접근 순서·reuse·eviction에 따라 달라진다.
추가 launch와 mask 재읽기가 생길 수 있다. 이번에는 분리 prototype을 만들지 않았으며
HBM 약91%인 현재 경로보다 빨라질 것이라는 증거도 아직 없다.

### 다른 LayerNorm과 B1

- **F1 입력 LN:** L768 HBM87.7%, 낮은 우선순위. L384는77.8%이지만 절대 시간이 작다.
- **F4 출력 LN:** L768 HBM78.8%, registers193, 점유율12.1%, strided input→contiguous output의 shared 통신에서 bank conflict가 관찰됐다. 후속 구현 점검 대상이다.
- **B11+B12:** L768 HBM69.1%, L2 86.4%, registers251, 점유율12.1%. 타일/atomic/shared 통신을 살필 여지는 있지만, 기존 공간의 bounded probe는 L3841.052× /L7681.011×에 그쳤다.
- **B1 gate/dropout 미분:** L768 HBM91.1%, local/shared conflict0. 현재는 우선순위가 낮다.

## 기존 config 안의 bounded probe

|L|커널|비교 후보 수|기준→후보 µs|속도비|해석|
|---|---|---:|---:|---:|---|
|384|B4|13|119.456→119.264|1.002×|유의미한 개선 없음|
|384|B11|12|85.568→81.344|1.052×|작은 단독 이득; 운영 미반영|
|768|B4|7|535.088→535.520|0.999×|유의미한 개선 없음|
|768|B11|13|271.536→268.480|1.011×|작은 단독 이득; 운영 미반영|

B11/B12와 L384 B4는 동일하게 gradient accumulator 초기화를 포함해서 비교했다.
L768 B4는 kernel이 모든 partial buffer를 덮어쓴다. 모든 passing 후보의 dx/dgamma/dbeta를
기준과 비교했으며 실제 오차는 JSON에 기록했다. 전체 grid 탐색 또는 sanitizer 검증은 수행하지 않았다.

## 우선순위와 cuBLAS 판단

1. **L768 B4:** 좁은 warp 공간의 개선을 실제로 확인했다. 다음 구현 작업에서 config 공간과 persistent grid/row 계산을 먼저 다룰 근거가 있다.
2. **F4, B11+B12:** register/shared layout/atomic 경로를 다음으로 점검. 간단한 설정 교체만으로 큰 가속을 확보했다고 주장하지 않는다.
3. **B7:** 시간이 가장 큰 공통 kernel이지만 현재 메모리 처리 효율이 높다. 추가 시도는 전송 bytes 감소의 근거를 갖고 시작해야 한다.
4. **F1/B1:** 우선 유지.

cuBLAS는 전체 커널 시간의 약34%지만, 이번 관찰에서 이를 대체해야 할 근거는 없다.
현재 범위에서는 contraction/weight-gradient GEMM을 그대로 두고 공통 Triton의 확인된 문제에 집중하는 편이 타당하다.
이는 모든 cuBLAS 호출이 물리적 최적이라는 뜻이나 영구적으로 개선 불가능하다는 주장은 아니다.

## 원본

- [요약·정규화 counter·source SHA](../../runs/trimul_common_profile_20260917/summary.json)
- [L384 NCU](../../runs/trimul_common_profile_20260917/common-L384.ncu-rep) · [L768 NCU](../../runs/trimul_common_profile_20260917/common-L768.ncu-rep)
- [L384 base CSV](../../runs/trimul_common_profile_20260917/common-L384-base.csv) · [L768 base CSV](../../runs/trimul_common_profile_20260917/common-L768-base.csv)
- [L384 config probe](../../runs/trimul_common_profile_20260917/probe-L384.json) · [L768](../../runs/trimul_common_profile_20260917/probe-L768.json)
- [L768 warp4/8 진단](../../runs/trimul_common_profile_20260917/probe-expanded-L768.json)
- [프로파일 스크립트](../../runs/trimul_common_profile_20260917/ncu_module.py) · [config 진단](../../runs/trimul_common_profile_20260917/probe_configs.py)
- [앞선 실제 학습 그래프의 시간 비중](KERNEL_BREAKDOWN.md) · [이전 B7 실험](../../runs/trimul_sm90_bwd_round3_20260917/b7/REPORT.md)
