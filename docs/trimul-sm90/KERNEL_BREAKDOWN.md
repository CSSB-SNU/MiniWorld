# Dropout ON 양방향 TriMul 커널별 시간과 가속 한계

## 결론

**교체한 F2/F567/B9+B10은 현재 H100 경로 GPU 커널 시간의 약26%뿐이다.**
cuBLAS는 약34%, 공통 LayerNorm은 약18%, 공통 gate 미분·packing은 약20%다.
Backward가 전체 커널 시간의 약71%인데, 특화한 B9+B10은 backward의 약12%다.
따라서 일부 커널의15% 가속을 전체 학습15% 가속으로 환산할 수 없다.

![커널 종류별 시간](../../runs/trimul_sm90_breakdown_20260917/KERNEL_BREAKDOWN.svg)

## 조건과 분모

- 개발 엔진 `8c7d8b39`, H100 80GB HBM3, B1/D=h128, BF16 mixed, static compile + 실제 stochastic CUDA graph.
- **dropout=0.25**, 매 replay production RNG 포함. 양방향 모듈 FWD+BWD, optimizer 제외.
- 공식 모듈 fixture, 동일 입력·weight·dy·token mask. F2/F567/B9+B10은 이전에 측정한 backend별 config manifest.
- CUPTI/PyTorch profiler로3회 ×10 replay: 각 GPU node30개 시간의 중앙값을 계산하고, 연산/단계별로 합산했다.
- 표의 `%`는 **해당 forward 또는 backward의 GPU 커널 누적 시간 대비 비중**이다. launch 사이 빈 시간은 제외한다.
- cuBLAS 행에는 해당 호출의 split-K reduction 및 workspace memset도 포함한다. LN 행에는 관련 초기화/reduction도 포함한다.
- 각 replay50개 GPU node(kernel + memset), 단일 stream, GPU node 순서 동일·겹침 없음 확인. 실제 forward 마지막 F567을 경계로 단계를 분리했다.
- 별도 CPU operator/shape trace로 cuBLAS bmm/mm와 F/B 연산을 대응시켰다. 타이밍 측정 중 고정 dropout mask/RNG reset 없음.
- 별도의 profiler OFF 전체 시간은8라운드 교대 측정이다. 프로파일 표의 합과 전체 latency는 측정 방식이 달라 정확히 같지는 않다.

## Forward / backward 합계

|L|경로|Forward 커널 합 ms|Backward 커널 합 ms|GPU 커널 합 ms|Profiler OFF 전체 ms|
|---|---|---:|---:|---:|---:|
|384|Triton|0.501|1.087|1.588|1.626|
|384|H100 선택 경로|0.451|1.079|1.530|1.571|
|768|Triton|1.923|4.289|6.212|6.247|
|768|H100 선택 경로|1.783|4.275|6.058|6.098|

이전 dropout ON 전체 측정은 L3841.035× / L7681.027×였고, 이번 반복은
L3841.035× / L7681.024×다. 실행별 변동을 포함해 약2~4%의 전체 이득으로 본다.

## Forward: 현재 H100 경로

|단계|연산|구현|L384 µs (% 단계)|L768 µs (% 단계)|
|---|---|---|---:|---:|
|F0|난수·dropout scale 생성|ATen / Inductor|6.56 (1.45%)|6.56 (0.37%)|
|F1|입력 LayerNorm|Triton / reduction|24.05 (5.33%)|98.80 (5.54%)|
|F2-prep|입력 weight stack·pair mask 준비|ATen / Inductor|3.42 (0.76%)|3.65 (0.20%)|
|F2|입력 projection + gate + mask|CuTe TMA/WGMMA|176.18 (39.04%)|737.15 (41.34%)|
|F3a|Outgoing contraction|cuBLAS|45.39 (10.06%)|179.74 (10.08%)|
|F3b|Incoming contraction|cuBLAS|42.18 (9.35%)|187.70 (10.53%)|
|F4|출력 LayerNorm|Triton / reduction|57.60 (12.77%)|213.28 (11.96%)|
|F567-prep|출력 weight layout 준비|ATen / Inductor|1.50 (0.33%)|1.54 (0.09%)|
|F567|출력 projection + gate + dropout + residual|CuTe TMA/WGMMA|94.35 (20.91%)|354.64 (19.89%)|

## Backward: 현재 H100 경로

|단계|연산|구현|L384 µs (% 단계)|L768 µs (% 단계)|
|---|---|---|---:|---:|
|B1|출력 gate·dropout 미분|Triton|65.94 (6.11%)|245.63 (5.75%)|
|B2|출력 gate weight gradient|cuBLAS|34.94 (3.24%)|122.77 (2.87%)|
|B3a|출력 projection input gradient|cuBLAS|39.92 (3.70%)|165.54 (3.87%)|
|B3b|출력 projection weight gradient|cuBLAS|56.06 (5.20%)|172.13 (4.03%)|
|B4|출력 LayerNorm 미분·reduction|Triton / reduction|112.26 (10.40%)|544.78 (12.74%)|
|B6a|Outgoing left gradient|cuBLAS|40.11 (3.72%)|193.20 (4.52%)|
|B6b|Outgoing right gradient|cuBLAS|41.55 (3.85%)|188.43 (4.41%)|
|B6c|Incoming left gradient|cuBLAS|41.73 (3.87%)|177.42 (4.15%)|
|B6d|Incoming right gradient|cuBLAS|41.92 (3.89%)|192.02 (4.49%)|
|B7|입력 gate 미분·mask·gradient packing|Triton|247.04 (22.90%)|998.32 (23.35%)|
|B8|입력 projection weight gradient|cuBLAS|131.42 (12.18%)|490.53 (11.48%)|
|B9-prep|입력 weight layout 준비|ATen / Inductor|1.57 (0.15%)|1.57 (0.04%)|
|B9+B10|입력 gradient 두 GEMM + 합산|CuTe TMA/WGMMA|129.49 (12.00%)|503.41 (11.78%)|
|B8-unpack|4개 weight gradient 분리|ATen / Inductor|6.29 (0.58%)|6.34 (0.15%)|
|B11+B12|입력 LayerNorm 미분 + residual|Triton / reduction|77.71 (7.20%)|261.90 (6.13%)|
|B-final|Autograd parameter gradient layout copy|ATen / Inductor|10.93 (1.01%)|10.74 (0.25%)|

## Triton 대비: 같은 학습 그래프 안의 실제 커널 시간

|L|커널|Triton µs|CuTe µs|속도비|
|---|---|---:|---:|---:|
|384|F2|214.096|176.175|1.215×|
|384|F567|105.808|94.352|1.121×|
|384|B9+B10|137.327|129.489|1.061×|
|768|F2|824.321|737.153|1.118×|
|768|F567|406.017|354.641|1.145×|
|768|B9+B10|515.905|503.408|1.025×|

단독 커널 벤치와 실제 모듈 문맥의 캐시/실행 상태가 다르므로 단독 수치를 그대로 대입하지 않는다.
특히 L768에서는 이전 단독 결과보다 F2와 B9+B10의 속도 차이가 작았다. 이번 결과는 실제 학습 그래프 안의 시간이다.

## 전체 가속이 작은 이유: 시간을 가중한 계산

|항목|L384|L768|
|---|---:|---:|
|현재 H100 Specialized 비중|26.14%|26.33%|
|현재 H100 cuBLAS 비중|33.67%|34.16%|
|현재 H100 LayerNorm 비중|17.75%|18.47%|
|현재 H100 Elementwise / packing 비중|20.45%|20.53%|
|현재 H100 Other 비중|1.98%|0.50%|

Triton 기준으로 교체 대상 세 커널의 합은 전체 커널 시간의28.8% /28.1%다.
그 세 커널이 **전부1.15×** 빨라져도 다른 부분이 동일하면 전체 속도비는
`1 / (1 − p + p/1.15)`이므로 **1.039× /1.038×**에 그친다.

현재 세 커널의 시간 가중 속도비는 L3841.143× / L7681.095×다.
이번에 남겨둔 cuBLAS 비용은 약0.515ms /2.069ms다. 이번 조건에서는 그대로 남는다.

전체1.15×를 세 특화 커널만 더 개선해서 달성하려면, 공통 비용이 그대로라는 근사하에
**현재 CuTe보다 추가로 약1.65× /1.72×** 빨라져야 한다. 이는 profiler 커널 시간과 별도 전체 latency를
결합한 대략적인 요구량이며 달성 가능성의 증명은 아니다. 세 커널을 공짜로 만드는 가정의 전체 한계는 약1.39×다.

## 15%가 구현의 상한인가? CUDA C++로 옮기면?

**15%는 상한을 증명한 수치가 아니다. 현재까지 검증한 구현의 성능이다.**
다만 같은 융합·저장 버퍼·수식 아래에서 메모리 병목인 일부 커널은 여유가 작다.

### B9+B10: 같은 DRAM 전송량일 때의 제한

기존과 동일한 최종 소스의 L768 NCU에서는 dense BF16 연산173.946GFLOP, DRAM1.506GB,
시간507.776µs였다. 측정 clock의 HBM peak3.352TB/s에 대해88.49%를 사용했다.
같은 전송량에서 낙관적인 메모리 시간은449.33µs로, **추가 속도비 약1.130×**다.
이는 동기화/지연을 무시한 roof이며 실제 달성 보장은 없다. 캐시 재사용 개선으로 DRAM bytes를 줄이면
이 가정 자체가 달라지므로 모든 가능한 구현의 절대 한계도 아니다.
같은 전송을 유지하고 front GEMM만 제거한 잘못된 수식의 진단에서도 L768은1.48%만 빨라졌다.
현재 스케줄에서는 GEMM 연산 대부분이 메모리 전송과 이미 겹친다는 근거다.

### F2 / F567

- F2는 전송 중첩과 BF16 shuffle/변환 감소로 이미 개선됐다. 최종 NCU L384는 약469MB DRAM traffic 중430MB가 write다. 같은 saved-preactivation/output 버퍼를 유지하면 이 쓰기 비용이 남는다. F2의 정확한 상한을 입증하지는 않았다.
- F567은 이전 동일 소스의 GROUP_M4 NCU에서 L768 DRAM86.23%, L2 throughput92.41%였다. 실제 선택은 GROUP_M1이므로 이 수치를 현재 config의 엄밀한 roof로 사용하지 않는다. output/dropout TMA와 prefetch 개선 이후 메모리·동기화에 민감한 상태라는 근거다.
- B7도 기존 조사에서 HBM 약90.4%였다. 큰 비용이라는 이유만으로 동일 알고리즘에서 크게 빨라질 것이라 단정하지 않는다.

### 언어보다 달라지는 기계어/스케줄이 핵심

현재 CuTe DSL도 Python 계산을 GPU에서 해석하지 않는다. PTX/SASS를 거쳐 GPU binary로 실행하며,
실제 cubin에서 TMA·WGMMA를 확인했다. [NVIDIA의 CuTe code-generation 설명](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html).

CUDA C++/inline PTX에서 더 나은 register allocation, vectorization, barrier/warp scheduling,
buffer 배치 또는 cache 재사용을 만들어내면 빨라질 여지는 있다. 같은 명령·자원·스케줄을 그대로
옮기는 것만으로 가속이 생긴다고 기대할 근거는 없다. 상당수 제어·unroll·TMA descriptor·packing 변경은 이미 시험했다.
그래프 OFF의 C++ 런처는 호스트 호출 비용을 줄일 가능성이 있지만, 이번 graph ON kernel 시간은 별도 문제다.

**권장 순서:** 공통 B4/B11+B12 LayerNorm 및 B1 미분 커널의 실측 병목을 먼저 확인하고,
기존 B7의 메모리 한계와 함께 남은 시간을 평가한다. CUDA C++ 이식은 CuTe가 만드는 특정 비효율
기계어를 지목한 후 한 커널 A/B로 검증하는 편이 합리적이다. 이번에는 구현을 변경하지 않았다.

## 재현·원본

- [후속 공통 Triton NCU·B7/LayerNorm 비교](COMMON_TRITON_PROFILE.md)

- [모든 GPU node별 CSV: 4조건 ×50개](../../runs/trimul_sm90_breakdown_20260917/kernels.csv)
- [집계 JSON](../../runs/trimul_sm90_breakdown_20260917/summary.json)
- [L384 raw profile](../../runs/trimul_sm90_breakdown_20260917/breakdown-L384.json) · [L768](../../runs/trimul_sm90_breakdown_20260917/breakdown-L768.json)
- [프로파일 스크립트](../../runs/trimul_sm90_breakdown_20260917/module_profile_breakdown.py) · [집계/plot](../../runs/trimul_sm90_breakdown_20260917/analyze.py)
- [이전 dropout ON 측정](DROPOUT_TRAINING.md) · [B9+B10 NCU roofline](BACKWARD_ROUND4.md)
- [F2 NCU](../../runs/trimul_sm90_15pct_20260917/front/REPORT.md) · [F567 NCU](../../runs/trimul_sm90_round2_20260917/f567/REPORT.md)
- [NVIDIA roofline 해석](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

CSV에는 실제 커널 이름·순번·단계·연산 대응·30회 min/max/median과 비중을 모두 남겼다.
Trace JSON과 CPU operator shape trace는 같은 실험 폴더에 있다. 대형 activation cat을 새로 도입하지 않았으며,
표의 weight stack/unpack은 작은 parameter layout 작업이다. 공통 LayerNorm cache fallback을 포함한 측정이고
전체 config 공간의 최적값 또는 전체 native cache 완성을 주장하지 않는다. 운영 학습 설치본은 변경하지 않았다.
