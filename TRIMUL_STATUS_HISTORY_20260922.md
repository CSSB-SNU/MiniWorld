# 최신 B1+B7 전체 재측정 · 2026-09-22

L384, BF16 C128/H256, dropout25%/mask/residual, node01 H100, job15526.
같은 실행에서 이전 개선 B7 조합1065.824µs → 최신 B1+단일 B7 후보1009.808µs (5.26% 단축).
구형 B7 검사 코드는1222.288µs. 예전1048µs 또는1185µs를 현재의 다른 실행과 직접 비교하지 않는다.
최신 forward294.384µs / backward712.624µs이며 full은 별도로 직접 측정했다.
**최신 조합은 입력 LN gamma/beta gradient 상대L2 최대9.535e-6로 기존 한도5e-6를 초과했다.**
후보의 성능 진단값이며 정확도 통과·기본 경로 승격으로 표시하지 않는다.
[전체 표·배선·검증·구성 대조](runs/trimul_full_latest_20260922/index.html).

---

# B1–B4 후속 실험 · 2026-09-22

7개 방향·새 후보70개(대조군 포함83개 shape/config) 검사. 현재 선택은 아래 캐시 개선 구현을 유지한다.
동기화 후보 L384 B1은182.624→182.112µs(-0.28%)였지만, 예전 B7 구현을 사용하는 별도 검사 코드의 전체 차이는0.07%다.
그 코드의1,185µs는 앞서1,048~1,076µs였던 개선 B7 조합의 최신 시간이 아니며 직접 비교하면 안 된다.
변경 없는 forward에서도 비슷한 변동이 있어 전체 학습의 확정 이득으로 채택하지 않았다. SoL90 미달.
[후속 실험·HBM 시각화·NCU 근거](runs/trimul_b1_cache_followup_20260922/index.html).

---

# 현재 B1–B4 · 2026-09-22

dWproj partial 저장을 dWgate 계산 뒤로 미루고, dGate 보존·소비 후 캐시 우선순위 조절을 적용했다.
현재 개발 `runs/trimul_training_current.py`에 연결했고 기존 B7는 유지했다.

| L | 이전 v51 | 새 B1 | 시간 단축 |
|---|---:|---:|---:|
| 384 | 190.53 µs | 182.18 µs | 4.38% |
| 768 | 640.96 µs | 636.13 µs | 0.75% |

동일 실행 5×200 교차 측정. 전체 11 gradients·graph/eager bit-exact, 두 길이 memcheck/racecheck 통과.
전체 학습 변화는 이 B1 개선율보다 작다. SoL90 미달이며 생산용 dispatch 승격은 아니다.
[새 배선·전체 비교·NCU·검증](runs/trimul_b1_gate_demote_20260922/index.html).
아래는 이전 날짜의 개발 기록이다.

---

# B1–B4 대기 축소: 2026-09-21 개발 경로

2026-09-21 · node01 H100. 비교 기준은 직전 v50 자체 학습 경로이며 Anthropic 추론이나 cuEquivariance가 아니다.

양방향 C128/H256 BF16, dropout25%/mask/residual, L384/L768. 구간별600회 교대 CUDA graph 중앙값. 전체 수치는 live packing/Wp 전치/cuBLAS/B7/11 gradients를 포함하며 optimizer/RNG 생성/compile/CPU dispatch를 제외한다. FWD 구현은 동일하므로 그 차이는 측정 변동이다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | 직전 v50 | 0.2924 | 0.1904 | 0.9012 | 1.1883 |
| 384 | 대기 축소 | 0.2919 | 0.1885 | 0.8992 | 1.1857 |
| 768 | 직전 v50 | 1.1665 | 0.6467 | 3.5883 | 4.7890 |
| 768 | 대기 축소 | 1.1699 | 0.6359 | 3.5766 | 4.7740 |

## 실제 변경

1. dGate TMA 저장 완료를 dNorm WGMMA 뒤로 옮겨 기존 CTA barrier와 합쳤다. dGate가 있던 shared scratch를 LN parameter 합산이 재사용하기 전에 저장 완료를 반드시 기다린다.
2. dTri TMA 저장 뒤의 warp-group barrier는 바로 다음 호출자의 CTA barrier가 포괄하므로 제거했다. 저장 완료 대기와 호출자의 CTA barrier는 유지한다.

**다음 raw TMA 전에 두 warp-group의 WGMMA 완료를 확인하는 CTA barrier를 유지한다.** 이전 경쟁 조건을 재도입하지 않았다. FP32 덧셈 순서, BF16 반올림, 입력 affine x_n 저장, 원본 tri BF16와 출력 mean/rstd FP32 저장 정책은 그대로다. 출력 LN activation은 저장하지 않는다. B7/cuBLAS는 변경하지 않았다.

## SoL90 진행 상황

| L | NCU B1 μs | DRAM peak | 실측 트래픽 roofline | 고유 payload 모델 |
|---|---|---|---|---|
| 384 | 188.768 | 61.19% | 61.22% | 41.97% |
| 768 | 624.320 | 67.38% | 67.42% | 50.76% |

**SoL90 미달이다.** DRAM 처리율을 전체 알고리즘 SoL로 표시하지 않는다. 이전과 같은 낙관적 모델 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`은 scalar/shared/instruction/의존성 및 작은 scratch 비용을 생략한다. 실측 트래픽 모델에는 dWgate의 x_n/dGate 재읽기가 포함된다. 정확한 도달 가능한 최소시간을 증명한 수치는 아니다.

[H100 공식 사양](https://www.nvidia.com/en-us/data-center/h100/)의 SXM3.35TB/s, dense BF16 989.5TF/s를 사용한다. 공식1979TF/s는 sparsity 수치다. 별도 node01 streaming 교정은3.105TB/s였다.

### dWgate만 분리한 진단

| L | 단독 CUDA event μs | 단독 NCU μs | 단독 DRAM peak |
|---|---|---|---|
| 384 | 37.024 | 35.136 | 73.71% |
| 768 | 109.472 | 109.984 | 85.00% |

원래 CTA 소유 행/입력/부분합을 그대로 쓰고 shared reservation도 동일하게 유지한 진단이다. 최종 CTA 간 합산은 제외한다. 단독 부분합은 원래 B1과 bit-exact. 단독 측정은 전체 실행 중 해당 구간의 직접 타이밍과 다르며, 단독85%를 전체 SoL85%로 해석하면 안 된다.

## 검증

- 두 길이 출력·전체11 gradients: 기준과 bit-exact.
- 입력/가중치/dy/dropout/mask 변경 및 gamma_out=0: bit-exact.
- CUDA graph 재실행/eager, 원본tri·stats 저장 정책 검사 통과.
- memcheck 두 길이0 errors, racecheck L3840 hazards.
- CTA 일부 및 WG1을 의도적으로 지연: 두 길이×3seed×20replay의 B1 여섯 출력 bit-exact.

개발 경로이며 production 승격이 아니다. 기존 B7 독립 기준 L768 dWL 상대L2 0.055569%가 한도0.05%를 넘는 문제는 그대로 남아 있다.

## 추가 시도

- dWproj와 dNorm WGMMA를 연속 발행: 유효하지만 추가 이득이 작아 제외.
- 출력 LN affine과 gate 재계산, sigmoid와 projection 겹치기: 추가 이득이 작아 제외.
- LN parameter 합산 동기화를 warp-group으로 축소: 추가 이득이 없어 CTA 동기화 유지.
- 마지막 partial의 L1 우회/일반 읽기: 추가 이득 없어 기존 volatile 읽기 유지.
- 마지막 CTA partial 합산 unroll1~132: 기존 컴파일러 설정보다 유의한 개선 없음. 낮은 unroll은 오히려 느리다.


<!-- B1_STAGE_DIAGNOSTIC -->
# B1–B4 내부 구간 측정과 추가 후보 검증

2026-09-21 · node01 H100 · C128/H256 BF16 · dropout25%/mask/residual · v51 기준.

**이번 후보에서 추가 성능 향상을 얻지 못했다. 검증된 v51 구현과 저장 정책을 유지한다. SoL90은 미달이다.**

## 커널 내부 시간

각 CTA의 thread0이 globaltimer를 기록했다. 10회 replay의 CTA별 구간 중앙값이며, 서로 다른 CTA는 겹쳐 실행된다. 표의 중앙값을 합해 전체 GPU critical path라고 해석하면 안 된다. 진단용 timestamp 출력도 전체 B1 여섯 출력과 bit-exact였다. 일반 CUDA event450회 교대 측정에서 계측 오버헤드는 L384약0.25%, L768약0.66%였다.

| CTA 내부 구간 | L384 μs | L768 μs |
|---|---|---|
| 주 계산 전체 | 133.632 | 513.792 |
| dWgate | 33.280 | 104.704 |
| 최종 grid 대기 | 6.144 | 7.424 |
| 최종 parameter 합산 | 5.888 | 6.144 |

한 CTA의 반복 횟수는 L38417~18회, L76869~70회다. 반복 타일당:

| 타일 내부 구간 | L384 μs | L768 μs |
|---|---|---|
| 다음 raw 입력 대기 | 0.000 | 0.000 |
| LN affine + projection/gate 재계산 | 2.816 | 2.816 |
| dWproj | 0.768 | 0.768 |
| dNorm + LN 미분/parameter 합산 | 3.584 | 3.584 |

globaltimer 관측값은256ns 단위로 나타났다. raw 대기 중앙값0은 비용이 전혀 없다는 뜻이 아니라 선행 TMA와 타이머 해상도의 결과다. 주 계산 구간에는 반복 제어와 partial 저장도 포함된다. dNorm+LN 구간은 GEMM만을 뜻하지 않는다.

## 실제 구현하고 제외한 후보

각 후보는 부모v51과450회 교대 CUDA graph 중앙값으로 비교했다. 아래는 각 후보군에서 가장 좋은 컨피그이며, 양수는 오히려 느려졌다는 뜻이다. 일반 B1 여섯 출력은 bit-exact. 성능에서 탈락했으므로 새 전체 모듈 sanitizer 승격 검증은 수행하지 않았고 기본값에도 연결하지 않았다.

| L | 후보 | 지연 증가 | B1 registers | spill store/load bytes |
|---|---|---|---|---|
| 384 | 레지스터 x̂ 보존 | +12.79% | 255 | 128/136 |
| 384 | 누산값 shared 이동 + x̂ 보존 | +8.27% | 237 | 0/0 |
| 384 | shared x̂ 보존 | +5.24% | 253 | 0/0 |
| 384 | γ shared / β warp 합산 | +2.36% | 255 | 0/0 |
| 384 | γ·β 동시 scratch | +0.64% | 254 | 0/0 |
| 384 | 다중 lane 합산 + swizzle | +1.88% | 255 | 0/0 |
| 768 | 레지스터 x̂ 보존 | +13.85% | 255 | 132/140 |
| 768 | 누산값 shared 이동 + x̂ 보존 | +11.81% | 237 | 0/0 |
| 768 | shared x̂ 보존 | +7.31% | 255 | 0/0 |
| 768 | γ shared / β warp 합산 | +3.99% | 255 | 0/0 |
| 768 | γ·β 동시 scratch | +1.82% | 255 | 0/0 |
| 768 | 다중 lane 합산 + swizzle | +2.59% | 255 | 0/0 |

단순 x̂ register 보존은 ptxas에서 spill이 발생했다. dWproj 누산값 절반을 shared로 옮기면237 registers/0spill이 되지만 shared 왕복과 Wp 재로딩 비용으로 여전히 느리다. shared x̂ 보존도 재계산 절약보다 이동 비용이 컸다. γ·β 합산을 병렬로 바꾸거나 lane을 나누는 후보 역시 기존 스케줄보다 빨라지지 않았다. ptxas 수치는 보조 b1_reduce가 아니라 **b1_fused 항목**에서 추출했다.

## 별도 B4 커널의 이동 비용 검토

전체 B1 다음에 B4를 별도로 실행하면 dNorm BF16의 추가 write/read와 원본 tri 재읽기가 생긴다. 최소 논리적 추가량은 `3 × 2bytes × 256 × L²`: L384226.49MB, L768905.97MB. L768에서는3.35TB/s로도270.44μs 분량이다. 이는 실제 전체 지연 증가를 증명하는 값은 아니다. L2 적중 및 계산과의 중첩, register 감소 효과를 별도 실험해야 한다. 다만 전체 activation을 HBM으로 내리는 단순 분리에는 상당한 비용이 예상된다.

다음 큰 변경은 이런 왕복을 새로 만들지 않고 dW 누산값의 긴 register 점유를 줄이거나, on-chip/L2에서 생산·소비를 겹치는 설계여야 한다. 현행 SoL90 달성을 주장할 근거는 없다.


<!-- B1_SOL90_FOLLOWUP -->
# B1–B4 SoL90: 추가 설계 실험

2026-09-21 · node01 H100 · L384/768 · C128 / 양방향 H256 · BF16 · dropout25% / mask / residual.

**SoL90 미달. 이번 실험에서 채택할 만한 추가 성능 향상을 확인하지 못했다. 현재 배선은 검증된 v51을 유지한다.**

현재 저장 정책: 입력 affine BF16 x_n, 원본 BF16 tri, 출력 LN의 FP32 mean/rstd. 출력 normalized activation은 forward에서 저장하지 않는다. B7과 기존 cuBLAS 경로는 변경하지 않았다.

## 후보별 최선 컨피그

각 후보와 v51을 같은 GPU에서 교대로 CUDA graph450회 측정했다. **음수는 지연 감소, 양수는 지연 증가**다. 서로 다른 행의 절대 시간은 직접 비교하지 않는다. 후보를 고르는 탐색 결과이며 1% 미만 변화가 반복 가능한 개선이라고 증명한 표는 아니다.

| 구현한 후보 | L384 지연 변화 | L768 지연 변화 | 일반 B1 검증 |
|---|---|---|---|
| dWproj를 dWgate 단계로 이동 | +18.08% | +21.41% | 6개 출력 정확히 동일 |
| dW 단계 이동 + FP32 x̂ 보존 | +14.72% | +18.45% | 6개 출력 정확히 동일 |
| dTri 채널별 조기 TMA 저장 | +0.76% | +1.88% | 6개 출력 정확히 동일 |
| dTri 저장과 다음 gate 중첩 | +1.98% | +3.05% | 6개 출력 정확히 동일 |
| 행 단위 LN + shared 전치 | +9.35% | +13.45% | 반올림 순서 차이 있음 |
| 행 단위 LN + bank swizzle | +4.66% | +7.45% | 반올림 순서 차이 있음 |
| 인접 행 두 개 / warp | +1.70% | +2.95% | 반올림 순서 차이 있음 |
| 행 두 개 + γ·β 동시 합산 | +1.73% | +3.10% | 반올림 순서 차이 있음 |
| 기존 LN + γ·β vector 저장 | +0.65% | +0.82% | 6개 출력 정확히 동일 |
| dW / dNorm을 4개 WG로 분리 | +24.86% | +29.14% | 6개 출력 정확히 동일 |
| 4개 WG + shared dNorm | +30.93% | +38.13% | 6개 출력 정확히 동일 |
| dProj 저장으로 gate 재계산 제거 | +16.21% | +21.16% | 6개 출력 정확히 동일 |
| 원시 moment에서 LN 통계 복원 | -0.91% | -0.34% | 반올림 순서 차이 있음 |
| 중심화 합산 뒤 rstd 곱셈 | -0.05% | +0.11% | 반올림 순서 차이 있음 |
| dNorm WGMMA 두 그룹 순차 소비 | +1.53% | -0.82% | 6개 출력 정확히 동일 |
| dNorm N128 + Wp TMA 배치 변경 | -0.60% | -0.09% | 6개 출력 정확히 동일 |
| gate / LN·projection WG 분리 | +23.60% | +25.99% | 6개 출력 정확히 동일 |
| 위 구조의 동적 gate 배열 제거 | +7.95% | +11.45% | 6개 출력 정확히 동일 |
| 두 WG의 gate·projection 절반 교환 | +30.39% | +37.06% | 6개 출력 정확히 동일 |

일반 설정 검사 216건 중 216건이 각 실험에 명시된 검증 기준을 통과했다. 행 단위 LN 및 moment 실험은 FP32 합산 순서가 달라지므로 bit-exact라고 표시하지 않았다. 그 외 표에서 정확히 동일한 후보들은 B1 여섯 출력을 비교했다. 탈락 후보에 대한 전체 sanitizer 검증은 수행하지 않았다.

## 작은 단독 이득을 전체 학습에서 재검사

L384 N128, L768 두 WGMMA 그룹 변형은 일반 입력, 변경 입력, gamma_out[0]=0, graph/eager에서 출력과 **11개 gradient 모두 v51과 bit-exact**였다. 아래는 새로600회 교대 측정한 직접 구간 시간이다.

| L | 후보 | 측정 범위 | v51 μs | 후보 μs | 지연 변화 |
|---|---|---|---|---|---|
| 384 | N128 | b1 | 190.016 | 188.960 | -0.556% |
| 384 | N128 | backward | 896.304 | 895.456 | -0.095% |
| 384 | N128 | forward_backward | 1188.496 | 1187.744 | -0.063% |
| 768 | two WGMMA groups | b1 | 636.320 | 641.264 | +0.777% |
| 768 | two WGMMA groups | backward | 3602.592 | 3618.112 | +0.431% |
| 768 | two WGMMA groups | forward_backward | 4835.488 | 4846.736 | +0.233% |

L768에서는 처음 관측한 단독 이득이 재현되지 않았다. L384 전체 이득은 약0.06%로 채택 근거가 충분하지 않다. 따라서 선택 파일과 기본 진입점을 변경하지 않았다. 독립 기준에서 이미 알려진 B7 L768 dWL 상대L2 0.055569% 문제 역시 이 작업으로 해결한 것이 아니다.

## 무엇을 확인했나

- 4 warp-group으로 dW와 dNorm을 분리해도 SM의 register pool은 공유된다. dW에 충분히 할당하면 dNorm에 spill이 생겼고, dW 할당을 줄이면 ptxas가 WGMMA 직렬화를 보고했다. 136/120부터160/96까지 검사했다. 공유 dNorm과 loop unroll 변경으로도 개선되지 않았다.
- dW 계산을 뒤로 옮기고 FP32 x̂를 보존하면 재계산은 줄지만, weight 단계의 추가 입력 읽기와 재계산이 이득을 상쇄했다. dProj까지 backward에서 저장하는 변형은 L38437.75MB / L768150.99MB의 scratch와 각각 한 번의 write/read를 추가하고도 느렸다.
- gate/LN·projection을 두 WG에 배정한 변형에서 `0 spill`이어도 동적 배열 때문에176B local stack이 생겼다. 완전 unroll로 이 배열을 제거하자 개선됐지만 여전히 v51보다 느렸다. stack/local memory와 spill 통계를 따로 봐야 한다.
- dNorm N128은 N64 두 개와 같은 결과를 내도록 Wp의 TMA tile 및 projection 재계산 주소도 함께 수정했다. WGMMA 명령 수 감소만으로 큰 이득을 얻지는 못했다.
- LN의 rstd 곱셈을 합산 뒤로 이동하는 수식도 검사했다. 큰 offset, 작은 분산, 상수, 큰 scale을 포함한14개 shape/input 조합에서 dTri 상대L2 최대는 raw moment1.39e-5, centered moment4.25e-6였다. 전체 오차 보장이나 production 승격을 뜻하지 않는다. 속도 이득도 작아 적용하지 않았다.

## SoL 해석

선택된 v51의 기존 NCU 측정: 고정 스케줄의 실측 traffic roofline61.22% /67.42%, 고유 입력·출력만 세는 낙관적 payload 모델41.97% /50.76% (L384/L768). **둘을 전체 알고리즘 SoL90으로 부르지 않는다.** 후자는 scalar/shared/instruction/의존성 비용을 생략한다. 이번 후보 중 어느 것도 SoL90 도달을 입증하지 못했다.

register 재분배의 플랫폼 제약은 [NVIDIA PTX setmaxnreg 문서](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)를 참고했다. 실제 채택 판단은 보존된 CUDA source, cubin, ptxas 로그, 교대 벤치와 gradient 검사에 근거한다.

## 새 NCU 실측 (job13718)

| L | 경로 | NCU μs | DRAM MB | warp 명령 M | shared bank conflict M | scheduler당 eligible warp |
|---|---|---|---|---|---|---|
| 384 | 현재 v51 | 187.648 | 387.20 | 63.49 | 1.162 | 0.555 |
| 384 | N128 | 188.576 | 387.15 | 61.69 | 1.167 | 0.538 |
| 768 | 현재 v51 | 626.720 | 1410.62 | 252.45 | 3.997 | 0.647 |
| 768 | 두 WGMMA 그룹 | 640.128 | 1410.96 | 254.55 | 6.380 | 0.654 |

NCU full, cache/clock control none, profiler 범위 안 B1 한 회를 측정했다. 위 CUDA event 중앙값과 별도 측정이다. L384 N128은 warp 명령 수가 약2.8% 감소해도 시간은 줄지 않았다. L768 두 WGMMA 그룹은 shared bank conflict와 지연이 늘었다. 이 상관관계만으로 특정 소스 줄의 원인을 확정하지 않는다.

새로 측정한 **현재 v51**의 traffic roofline은 L38461.59% / L76867.19%, 낙관적 고유 payload 모델은 42.22% / 50.57%다. 이전61.22%/67.42%와의 작은 차이는 새 측정값이며, 커널 변경에 따른 향상이 아니다. **SoL90 미달**이다.


<!-- B1_CLUSTER_FOLLOWUP -->
# B1–B4 SoL90: CTA 간 전송과 누산 레지스터 분리 실험

2026-09-21 · node01 H100 80GB HBM3 · L384/768 · BF16 C128/H256 · dropout25%, mask, residual.

**SoL90 미달. 현재 구현은 v51 유지. 새 후보를 기본 경로로 채택하지 않았다.**

저장 정책은 입력 affine BF16 x_n, 원본 BF16 tri, 출력 LN FP32 mean/rstd다. 출력 normalized activation을 forward에서 저장하지 않는다. B7과 cuBLAS 경로는 변경하지 않았다.

## 실제 B1 후보: 같은 실행의 v51과 교대 측정

단위 μs. 음수는 지연 감소다. 각 행은 후보군에서 가장 좋은 설정이며, 서로 다른 행의 절대 시간 비교는 피한다. TMA 설정과 tail reduction은 후보당 CUDA graph1000회, 나머지는450회 교대 측정했다. 본 표는 B1 단독6개 출력 검증이며 전체11 gradient 재검증이나 production 승격을 뜻하지 않는다.

| 후보 | L384: v51 → 후보 | L768: v51 → 후보 | 검증 |
|---|---|---|---|
| TMA L2 promotion 설정 | 188.480 → 188.480 (+0.00%) | 638.544 → 635.808 (-0.43%) | 6개 출력 bit-exact |
| 3 WG: core 1 + dWproj 2 | 188.544 → 325.232 (+72.50%) | 633.072 → 1176.768 (+85.88%) | 6개 출력 bit-exact |
| 3 WG + shared dNorm | 188.320 → 250.432 (+32.98%) | 635.200 → 834.048 (+31.30%) | 6개 출력 bit-exact |
| 마지막 partial 합산 별도 커널 | 188.384 → 187.040 (-0.71%) | 639.424 → 640.768 (+0.21%) | 6개 출력 bit-exact |
| dWproj 전체를 기존 global partial에 임시 저장 | 188.256 → 262.352 (+39.36%) | 634.336 → 965.360 (+52.18%) | 6개 출력 bit-exact |
| dWproj 절반을 shared 64KiB에 임시 저장 | 188.640 → 219.664 (+16.45%) | 633.872 → 754.976 (+19.11%) | 6개 출력 bit-exact |

- 3 WG에서는 core1개와 dWproj2개로 레지스터 소유를 나눴다. 공유 SM의 register pool 제약 때문에 일반 구현은 spill이 컸다. dNorm을 사용이 끝난 Wg shared 영역에 저장하는 변형도 여전히 느렸다.
- 마지막 합산 분리는 global barrier를 없애지만 별도 launch와 추가 스케줄 비용이 있다. L384의 1% 미만 탐색 이득은 반복 가능한 전체 학습 개선을 입증한 결과가 아니다. L768은 느려졌다.
- dWproj를 기존 global partial 버퍼에 임시 저장하는 변형은 forward 저장 정책과 수치를 유지한다. 부분 unroll은 512B local stack을 만들었다. 완전 unroll로 이를 없애도 임시 저장·읽기 비용을 상쇄하지 못했다. `0 spill`만 보고 local memory가 없다고 판단하지 않았다.
- shared 임시 보관은 dWproj 절반을 norm32KiB+Wg32KiB에 보관한다. 다음 raw prefetch를 늦추고 Wg를 다시 TMA로 읽어야 한다. HBM activation 추가 저장은 없지만 무료인 교환은 아니다.

## Hopper cluster: 80KiB/tile 전달 비용

이는 **TriMul 커널 성능이 아닌 전송 microbenchmark**다. 모든 CTA256threads, dynamic shared228352B로 현재 B1과 같은 shared footprint다. cluster의 마지막 CTA가 consumer이고 나머지가 producer다. 80KiB는 output-affine32 + dProj16 + dGate16 + input-x_n16이다.

mode0=전송+handshake, mode1=consumer가 전부 읽음, mode2=mode1+producer3μs 합성 지연, mode3=mode2+consumer1μs 합성 지연. 합성 지연은 실제 GEMM/LN의 tensor-core/shared/issue 경합을 모델링하지 않는다. **이 시간을 실제 B1 시간에 단순 합산하거나 겹쳐서 전체 속도를 예측하지 않는다.**

| 전송 | L | cluster CTA | 활성 CTA | 계산 producer CTA | mode0 μs | mode1 μs | mode2 μs | mode3 μs |
|---|---|---|---|---|---|---|---|---|
| TMA | 384 | 2 | 132 | 66 | 88.4 | 107.8 | 144.5 | 145.9 |
| TMA | 384 | 4 | 120 | 90 | 158.0 | 218.1 | 221.0 | 298.6 |
| TMA | 384 | 6 | 102 | 85 | 312.7 | 374.0 | 377.0 | 514.3 |
| TMA | 384 | 8 | 120 | 105 | 288.7 | 418.3 | 421.3 | 576.5 |
| TMA | 768 | 2 | 132 | 66 | 309.1 | 382.9 | 524.2 | 526.9 |
| TMA | 768 | 4 | 120 | 90 | 594.6 | 821.6 | 824.6 | 1134.4 |
| TMA | 768 | 6 | 102 | 85 | 1195.5 | 1438.3 | 1441.3 | 1990.2 |
| TMA | 768 | 8 | 120 | 105 | 1116.4 | 1621.3 | 1624.4 | 2243.6 |
| SIMT | 384 | 2 | 132 | 66 | 101.7 | 108.9 | 207.4 | 208.4 |
| SIMT | 384 | 4 | 120 | 90 | 155.8 | 220.4 | 223.3 | 300.3 |
| SIMT | 384 | 6 | 102 | 85 | 314.9 | 378.9 | 381.8 | 518.1 |
| SIMT | 384 | 8 | 120 | 105 | 283.5 | 424.2 | 427.1 | 580.9 |
| SIMT | 768 | 2 | 132 | 66 | 361.6 | 388.2 | 781.2 | 782.4 |
| SIMT | 768 | 4 | 120 | 90 | 582.5 | 833.5 | 836.5 | 1145.5 |
| SIMT | 768 | 6 | 102 | 85 | 1194.2 | 1459.1 | 1462.2 | 2004.5 |
| SIMT | 768 | 8 | 120 | 105 | 1101.2 | 1645.2 | 1648.1 | 2265.2 |

GPC 배치 제약으로 cluster4/6/8에서 활성 CTA가132보다 작았다. 계산용 producer도66/90/85/105개로 줄었다. 80KiB를 한 consumer로 집중시키는 현재 설계는 전송비와 계산 SM 감소를 동시에 부담한다. DSMEM 전체가 무의미하다는 결론은 아니다.

### 동기화 검증

초기 구현에서 producer 하나의 empty barrier를 두 slot이 공유해 parity가 두 번 돌아가는 ABA 문제가 있었다. 지연을 grant 대기 전에 넣자 교착이 재현됐다. 해당 잡만 종료하고 **producer별·slot별 독립 mbarrier**로 수정했다. 이 표는 수정 후 결과만 사용한다.

TMA/SIMT 각각 L384/768, cluster2/4/6/8, payload16/48/80KiB에서 모든 단어를 검증했다. producer3μs/consumer1μs 지연 stress, 일부러 한 단어를 깨뜨린 negative control도 검사했다. L384에서 두 방식 모두 memcheck0errors 및 racecheck0hazards를 통과했다. **전송 프로토콜 검증이며 새 TriMul gradient 검증은 아니다.**

## 계산용 CTA가 줄어드는 손해를 별도 확인

아래는 실제 B1에서 **모든 weight gradient 계산을 제외**한 진단용 코드다. dGate/dTri는 bit-exact이며 LN parameter는 실험 허용치로 검사했다. FP32 xhat을 register에 잠시 보관하는 설정이다. 정상 B1 대체물이나 알고리즘 전체 하한으로 표시하지 않는다.

| L | 계산 CTA 수 | dW 제외 진단 μs |
|---|---|---|
| 384 | 66 | 219.888 |
| 384 | 85 | 216.992 |
| 384 | 90 | 166.656 |
| 384 | 105 | 181.408 |
| 384 | 132 | 127.984 |
| 768 | 66 | 1044.032 |
| 768 | 85 | 804.144 |
| 768 | 90 | 772.848 |
| 768 | 105 | 657.280 |
| 768 | 132 | 449.760 |

L768에서는 cluster 배치에 해당하는 producer CTA 수로 줄이면 dW를 완전히 생략해도 현재 B1보다 느렸다. 현재 스케줄에서 계산 SM을 빼는 방식에 불리한 증거다. 다른 계산 배치까지 불가능하다는 증명은 아니다.

## SoL 수치와 판정

이전 job13718에서 선택된 v51을 NCU로 측정한 값은 L384187.648μs / L768626.720μs다. 측정 DRAM traffic과 명목3.35TB/s로 계산한 traffic roofline은 **61.59% /67.19%**, 고유 입력·출력 payload만 세는 낙관적 모델은 **42.22% /50.57%**다. 이번에는 선택 구현이 바뀌지 않았고 이 NCU 값을 새로 측정하지 않았다. 두 지표 모두 전체 알고리즘 SoL90 도달을 입증하지 않는다.

현재 선택은 개발 adapter다. 기존 B7 L768 dWL 상대L2 0.055569% 이슈는 별개로 남아 있고, 이 실험에서 수정하지 않았다.

Hopper 명령과 cluster launch의 출처: [NVIDIA PTX 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html), [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf). 성능 판단은 보존된 source/cubin/ptxas 및 실측 로그에 근거한다.


<!-- B1_BANDWIDTH_CEILING -->
# B1–B4: 메모리 상한 실측 보강

2026-09-21 · node01 H100 80GB HBM3 · job13768 streaming / job13771 NCU.

**SoL90 미달, 선택 구현은 v51 유지. 아래 비율은 메모리 대역폭 참고치이며 전체 알고리즘 SoL이 아니다.**

독립 CUDA streaming 커널에서 1/2/3회 읽기와1회 쓰기, CTA132/264/528/1056, unroll1/4/8을 비교했다. 출력128/512MiB별36개 설정, 총72개 설정이다. 모든 출력 원소를 검증했다. 입력·출력 합계는 L2 용량보다 크다. 각 설정을 CUDA graph20회 실행씩11개 구간 측정했고 구간별 평균 지연의 중앙값을 사용했다. 그래프에 묶은20회는 서로 다른 단일 실행 샘플로 취급하지 않았다.

선택된2회 읽기·1회 쓰기 설정은 두 크기 모두264CTA×256threads, unroll8이다. 명목3.35TB/s 기준과 실측 streaming 기준을 함께 제시한다. B1의 읽기/쓰기 비율은 정확히2:1이 아니므로 이는 대표적인 혼합 비율을 이용한 참고 모델이다.

| B1 L | streaming 출력 크기 | event 논리 TB/s | NCU 실제 DRAM TB/s | B1/명목3.35 | B1/event streaming | B1/NCU streaming |
|---|---|---|---|---|---|---|
| 384 | 128 MiB | 2.973 | 2.930 | 61.59% | 69.39% | 70.41% |
| 768 | 512 MiB | 3.007 | 2.996 | 67.19% | 74.85% | 75.12% |

## 해석의 한계

- **B1을 새로 더 빠르게 만든 수치가 아니다.** B1 시간과 DRAM 바이트는 이전 job13718의 선택된v51 NCU 측정에서 가져왔다. 이번 실험에서 새로 측정한 것은 독립 streaming 커널이다. 서로 다른 실행의 계측값을 이용한 참고 비교다.
- event 논리 대역폭은 요청한 입력+출력 바이트를 센다. NCU 실제 바이트는 cache와 측정 경계의 영향을 받는다. 출력128MiB에서는 일부 쓰기가 kernel 범위 내 DRAM counter에 포함되지 않아 두 방식의 값이 다르다. 실제 DRAM 값으로 바꿔 더 큰 비율만 강조하지 않았다.
- NCU는 cache/clock control none, profiler 범위의 streaming 커널 한 회를9pass 수집했다. event 중앙값과 별도 측정이다. 두 독립 streaming 설정의 NCU DRAM peak 비율은 약87.44%/89.39%다. **이 streaming 커널의 비율을 B1의 SoL로 옮겨 쓰지 않는다.**
- NCU report 생성 뒤 CLI 출력 단계에서 Python user-site encoding 오류가 났다. 생성된 두 report를 PYTHONPATH 해제 및 PYTHONNOUSERSITE=1 환경에서 정상 import해 위 metrics를 확인했다. 계측을 재실행한 것처럼 표시하지 않았다.
- LN, gate 재계산, dW, scalar 명령, shared memory, 동기화 비용을 모두 반영한 알고리즘 하한은 아직 증명하지 못했다. 실측 streaming 기준으로도 B1은 약70%/75% 수준이다. 90% 달성을 주장할 근거가 없다.

현재 저장 정책과 dropout/mask/residual, B7/cuBLAS 경로는 바뀌지 않았다.


<!-- B1_INSTRUCTION_FOLLOWUP -->
# B1–B4: 명령 공급과 BF16 sigmoid 조회표

2026-09-21 · node01 H100 · L384/768 C128/H256 BF16 · dropout25%, mask, residual.

**SoL90 미달. 채택할 추가 개선이 없어 현재 v51을 유지했다.** 입력 affine x_n, 원본 tri, 출력 FP32 mean/rstd 저장 정책과 B7/cuBLAS는 변경하지 않았다.

## 실제 B1 후보 비교

단위 μs. 각 설정을 같은 GPU의 v51과 CUDA graph450회 교대 측정했다. 서로 다른 행의 절대 시간은 직접 비교하지 않는다. 모든 후보의 B1 여섯 출력은 bit-exact다. 성능상 탈락했으므로 새 전체11-gradient/sanitizer 검증이나 production 승격은 하지 않았다.

| 후보 | L384 v51 → 후보 | L768 v51 → 후보 |
|---|---|---|
| dNorm/LN의 warp-group 주소 상수화 | 188.480 → 202.480 (+7.43%) | 637.344 → 659.264 (+3.44%) |
| 재계산까지 warp-group 주소 상수화 | 187.904 → 208.240 (+10.82%) | 636.096 → 705.312 (+10.88%) |
| sigmoid shared LUT | 187.600 → 196.480 (+4.73%) | 634.944 → 664.400 (+4.64%) |
| sigmoid global LUT | 191.168 → 200.544 (+4.90%) | 634.752 → 678.256 (+6.85%) |
| 분기 없는 shared LUT | 190.240 → 199.872 (+5.06%) | 636.400 → 676.416 (+6.29%) |
| 분기 없는 global LUT | 188.016 → 197.728 (+5.17%) | 635.936 → 670.112 (+5.37%) |
| 양수·음수 packed shared LUT | 187.456 → 198.592 (+5.94%) | 636.224 → 677.456 (+6.48%) |
| 양수·음수 packed global LUT | 188.160 → 199.616 (+6.09%) | 636.560 → 676.608 (+6.29%) |

## Warp-group 상수화: 코드가 커지고 명령 공급이 악화

mode0은 control, mode1은 dNorm/LN만, mode2는 gate·projection 재계산까지 채널 절반을 template 인자로 고정했다. 두 warp-group의 barrier 참여 수와 raw prefetch 전 CTA 동기화를 보존했다. 여섯 출력은 exact였으나 지연이 증가했다.

정적 SASS 명령 수는 L3845375→7187→8507, L7685687→7355→8747개다. 이 수는 코드 크기의 지표이며 실제 실행 명령 수와 다르다. 아래는 새 NCU job13789에서 측정한 실제 실행 및 stall 지표다.

| L | mode | NCU μs | 실행 warp 명령 M | no-instruction stall | scheduler당 eligible warp |
|---|---|---|---|---|---|
| 384 | 0 | 189.536 | 63.488 | 1.66% | 0.552 |
| 384 | 1 | 207.168 | 64.958 | 7.87% | 0.506 |
| 384 | 2 | 210.336 | 64.194 | 18.55% | 0.423 |
| 768 | 0 | 627.008 | 252.449 | 2.21% | 0.649 |
| 768 | 1 | 664.736 | 254.161 | 5.95% | 0.606 |
| 768 | 2 | 707.552 | 247.625 | 22.29% | 0.458 |

mode2 L768은 실행 명령 수가 감소했는데도 시간은 증가했다. no-instruction stall이 약2.2%에서22.3%로 증가한 점은 명령 공급 문제가 악화됐다는 증거다. 코드 크기 증가만으로 특정 cache miss를 모두 설명했다고 단정하지 않는다. NCU는 clock/cache control none이며 event 중앙값과 별도 측정이다.

이전 v51 full NCU report의 source-level warp samples도 확인했다. L768 long-scoreboard samples11402개 중8803개가 gate phase TMA try-wait 이후 분기에 모였다. 이는 샘플 분포이며 전체 지연의77%가 해당 명령이라는 뜻은 아니다. gate phase의 deep buffering과 K grouping은 기존 실험에서 이미 검사했고 추가 이득이 작아서 동일한 sweep을 반복하지 않았다.

## BF16 sigmoid: 전수검증되는8KiB 조회표

기존 B1은 gate GEMM 결과를 BF16으로 반올림한 뒤 sigmoid를 계산하고 결과를 BF16으로 반올림한다. 그래서 BF16 입력별 최종 결과를 GPU에서 기존 함수로 미리 생성할 수 있다. 이는 tanh 근사나 오차 허용치를 완화한 경로가 아니다.

- 양·음수 각각 BF16 magnitude0x3b00..0x42ff의2048개 결과를 저장한다. 범위 밖은 양 끝 값으로 clamp하고 NaN은 원래 결과를 보존한다. 총4096×2byte=8KiB다.
- **모든65,536개 BF16 bit pattern**을 기존 exp/rcp 함수와 비교했다. 두 부호의0, subnormal, 유한수, ±Inf, 모든 NaN payload를 포함한다. 각 테스트에서 차이0개였다. 이 결과는 검사한 H100/compiler/cubin에 대한 근거다.
- shared 방식은 현재 parameter-reduction 경로에서 사용하지 않는 shared tmp8KiB를 재사용한다. global 방식은 읽기 cache를 사용한다. forward activation 저장은 늘리지 않았다.
- 후속 구현은 NaN 결과 선택을 branch-free로 만들고, 같은 magnitude의 양·음수 결과를32bit에 묶는 배치도 비교했다. 이 경우도 전수검증과 B1 여섯 출력이 exact다.
- 그래도 조회 주소 계산과 읽기 비용을 상쇄하지 못해 약5~7% 느렸다. 정확한 LUT라는 사실과 빠른 kernel이라는 사실은 별개였다. 모든 후보를 비활성 상태로 남겼다.

## 추가 코드 생성·중첩 검사

| 후보군 최선 탐색 설정 | L384 v51 → 후보 | L768 v51 → 후보 |
|---|---|---|
| PTXAS 코드 생성 설정 | 187.200 → 187.232 (+0.017%) | 637.008 → 636.512 (-0.078%) |
| BF16 gate 압축 후 sigmoid/projection 중첩 | 190.528 → 190.320 (-0.109%) | 635.392 → 639.024 (+0.572%) |

PTXAS register-usage-level0/3/5/7/10, opt-level2/3, expensive optimization 해제, extra-device-vectorization을 비교했다. 중첩 후보는 gate를 먼저16개 BF16 pair로 압축해 이전32개 FP32 gate accumulator보다 적은 register를 유지하고 projection WGMMA 실행 중 sigmoid를 계산한다. affine unroll2/4/8을 검사했다.

이 두 검사24건을 포함해 이번 일반 설정 검사는46건이며 모두 B1 여섯 출력이 bit-exact다. 작은 음수 변화는 0.1% 안팎의 탐색 결과여서 반복 가능한 개선으로 표시하지 않았다. 새 전체-module 승격 검증을 하지 않았고 현재 선택은 그대로다.

## 현재 상한 상태

이전 v51 NCU의 명목3.35TB/s 대비 memory roofline61.59%/67.19%, 별도 streaming NCU 대비 참고 비율70.41%/75.12%는 그대로다. 이번 구현들은 선택되지 않았고 상한 수치를 높이지 않았다. 전체 알고리즘 SoL90은 입증되지 않았다. 기존 독립 B7 L768 dWL 상대L2 0.055569% 문제도 별개로 남는다.
