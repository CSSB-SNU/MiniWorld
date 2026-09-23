# Triton F4–F7 융합 A/B 구현 및 실측

2026-09-16 · H100 80GB HBM3 한 개 · BF16 · B=1 · pair/output=128 · 방향별 hidden=128.

**두 후보 모두 구현해 전체 양방향 TriMul 학습에 연결했고, 두 길이에서 A가 조금 더 빨랐다.** 이번 변경 구간은 F4–F7이며 F1–F3b와 backward는 기존 Triton 경로 그대로다. 실험 코드와 결과를 보관했으며 설치본의 기본 dispatch 및 공용 캐시에는 변경을 적용하지 않았다.

## 실제 커널 경계

| 경로 | GPU 호출 1 | 호출 2 | 호출 3 | 호출 4 |
|---|---|---|---|---|
| 기존 | F4: Triton LN | F5: cuBLAS projection | F6: cuBLAS gate GEMM | F7: Triton sigmoid·곱·dropout·residual |
| **A** | 기존 F4 | **Triton F5+F6+F7**, `_f567(PROJECT=True)` | — | — |
| **B** | **Triton F4+F5**, `_f45` | **Triton F6+F7**, `_f567(PROJECT=False)` | — | — |

- 정규화값, projection, sigmoid gate 및 LN 통계를 기존 dtype/layout으로 저장한다. backward 함수 자체를 그대로 공유한다.
- norm/projection/gate logits의 BF16 중간 반올림을 유지한다. sigmoid는 FP32로 epilogue에 사용하고 backward 저장만 BF16이다.
- B는 정규화값을 한 번 저장하면서 dot에는 레지스터 값을 직접 넘긴다. 출력 N 타일이 여러 개인 경우 저장은 첫 타일만 수행한다.
- 실제 전체 학습 profiler 호출 수는 두 길이 모두 **62→60**으로 감소했다. 전체 모듈이 두 커널이라는 뜻은 아니다.

## 속도: 동일 GPU에서 교대 측정

`torch.compile(dynamic=False, fullgraph=True)`, compiler CUDA graphs 비활성화 후 수동 CUDA graph capture. 전체 학습은 p_drop=0.25, 일부 token mask, gradient 초기화 포함. 각 replay가 학습 1회다. 30회 replay 평균을 12라운드 교대 측정한 중앙값이며 컴파일·튜닝은 시간에서 제외했다. 출력 구간은 training 저장값과 고정된 dropout scale을 포함한다.

| 범위 | L | 기존 (ms) | A (ms) | B (ms) | A 시간 감소 | B 시간 감소 |
|---|---:|---:|---:|---:|---:|---:|
| 출력 F4–F7 forward | 384 | 0.2028 | 0.1739 | 0.1764 | 14.24% | 13.02% |
| 출력 F4–F7 forward | 768 | 0.7289 | 0.6398 | 0.6511 | 12.22% | 10.67% |
| 출력 구간 forward+backward | 384 | 0.5697 | 0.5389 | 0.5413 | 5.41% | 4.99% |
| 출력 구간 forward+backward | 768 | 2.0993 | 2.0016 | 2.0208 | 4.66% | 3.74% |
| **전체 TriMul forward+backward** | 384 | 1.8115 | 1.7829 | 1.7848 | 1.58% | 1.47% |
| **전체 TriMul forward+backward** | 768 | 6.8806 | 6.7933 | 6.8097 | 1.27% | 1.03% |

출력 구간의 개선이 전체 학습에선 약 1–2%가 되는 것은 이번에 바꾼 구간의 비중 때문이다. F1–F3b와 나머지 backward 시간을 줄인 결과가 아니다. 이 결과로 다른 shape나 dtype의 우열을 일반화하지 않는다.

## HBM: 계산뿐 아니라 DRAM counter 확인

L768의 F4–F7만 Nsight Compute `dram__bytes_read.sum` / `dram__bytes_write.sum`으로 수집했다. 단위는 MiB.

| 경로 | 논리 모델 | 실제 read+write | 기존 대비 실제 감소 |
|---|---:|---:|---:|
| baseline | 2016 | 2017.3 | 0.0 |
| A | 1584 | 1588.6 | 428.7 |
| B | 1440 | 1445.2 | 572.1 |

계산한 절감량 A=432 MiB / B=576 MiB와 실측은 가깝다. **B가 더 적게 전송하지만 A가 조금 더 빠르다.** 메모리 절감이 latency로 전부 바뀌지는 않는다. B의 LN reduction·layout 변환·dot을 함께 처리하는 비용이 후보의 차이이며, 개별 원인의 기여도까지 분리 측정한 것은 아니다.

Counter는 warmup 후 각 후보 1회, kernel replay 및 `--cache-control none`으로 수집한 스냅샷이다. L2 상태를 강제로 같게 만들지 않았으며 counter 수집 중의 latency는 성능표에 사용하지 않았다. 작은 차이를 정밀한 보장값으로 읽으면 안 된다.

## 설정 탐색

길이마다 F45 63개, F67 126개, F567 126개: **315개씩, 총 630개**. 모든 설정의 출력 전체를 baseline과 비교했고, 상위 6개는 교대 재측정했다. runtime의 shape·stride·dtype 키로 선택하며 최종 검증에서 cache miss가 있으면 실패하도록 했다.

| L | 커널 | BM | BN | BK | warps | stages | registers/thread | spill | shared bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | f45 | 64 | 128 | 전체 K=256 | 4 | 3 | 198 | 0 | 98304 |
| 384 | f67 | 32 | 128 | 64 | 8 | 3 | 74 | 0 | 40960 |
| 384 | f567 | 64 | 128 | 64 | 8 | 3 | 128 | 0 | 73728 |
| 768 | f45 | 64 | 128 | 전체 K=256 | 4 | 3 | 198 | 0 | 98304 |
| 768 | f67 | 16 | 128 | 64 | 4 | 2 | 71 | 0 | 18432 |
| 768 | f567 | 64 | 128 | 64 | 8 | 3 | 128 | 0 | 73728 |

이 설정 기록은 이번 실험용이다. 엔진의 공용 autotune cache/build-all 등록까지 끝났다는 뜻은 아니다.

## 정확도와 검증

- FP32 PyTorch reference에 대해 출력, 입력 gradient, 모든 파라미터 gradient 비교. mask 적용, residual 경로, dropout scale의 backward 적용을 확인했다.
- 작은 M=289, 비정렬 M=283/K=250/N=125, 0 가중치 및 실제 L384/L768 검증.
- 전체 compiled CUDA graph에서 replay마다 dropout 결과가 달라지는지 검사. 두 후보 모두 통과.
- Compute Sanitizer 초기 비정렬/0 가중치 검사: **0 errors**. 선택된 설정의 L384/L768 검사도 **0 errors** ([L384](ab-results/memcheck_L384.log) · [L768](ab-results/memcheck_L768.log)).
- 상대 L2 허용 기준은 FP32 gradient 2.5%, 전체 baseline 대비 1.5%, 출력 구간 baseline 대비 1.2%다. 이는 원소별 0.01% 보장이 아니다. 실제 오차는 아래와 같다.

| L | 경로 | 전체 출력·gradient 중 최대 FP32 상대 L2 | 최대 baseline 상대 L2 |
|---|---|---:|---:|
| 384 | baseline | 0.5953% | 0.0000% |
| 384 | A | 0.5953% | 0.0001% |
| 384 | B | 0.5952% | 0.0319% |
| 768 | baseline | 0.5979% | 0.0000% |
| 768 | A | 0.5979% | 0.0001% |
| 768 | B | 0.5979% | 0.0001% |

최초 전체 벤치는 정확도 검증 출력이 default-stream autograd 참조를 유지한 상태여서 baseline부터 capture에 실패했다. 참조를 해제하고 warmup/capture 스트림을 통일한 최종 실행이 위 표의 근거다. 실패 실행의 전체 학습 값은 사용하지 않았다.

## 구현과 재현

- [Triton 커널 및 launch](../../scripts/trimul_output_fusion.py)
- [전체 TriMul 연결: 기존 backward 공유](../../scripts/trimul_fusion_training.py)
- [정확도·튜닝·교대 벤치](../../scripts/benchmark_trimul_fusion_ab.py)
- [DRAM counter 수집 대상](../../scripts/profile_trimul_fusion_memory.py)
- [L384 전체 기록](ab-results/L384.json) · [L768 전체 기록](ab-results/L768.json)
- [실측 DRAM CSV](ab-results/dram_L768.csv) · [집계](ab-results/dram-summary.json)
- [초기 정확도](ab-results/smoke.json) · [메모리 검사](ab-results/memcheck.log)

GPU 할당 안에서 저장된 설정을 재사용하는 예:

```bash
PYTHONPATH="$PWD/scripts:$PWD/libs/team-gm/src" .pixi/envs/cu128/bin/python scripts/benchmark_trimul_fusion_ab.py \
  --length 768 --load-tuning docs/trimul-fusion/ab-results/L768.json \
  --output runs/trimul_fusion_ab_recheck_L768.json
```

`--load-tuning`을 빼면 전체 설정 공간을 다시 탐색한다. 기본 설치 경로를 바꾸지 않고 벤치 프로세스에서만 후보를 연결한다.

**판단: 이번 두 shape에서는 A를 다음 통합 후보로 선택할 근거가 있다. B도 동작·메모리 절감은 검증됐지만 현재 구현의 성능 우위는 없었다.**
