# TriMul 개발 마무리 · 2026-09-22

이번 최적화 실험은 종료한다. **성능 실험 완료와 전체 정확도 통과를 구분한다.**
최신 L384 조합은 1.010ms지만 입력 LN gradient 검증이 남아 기본 경로로 승격하지 않았다.
SoL90·B7 252µs 목표 달성을 주장하지 않는다.

## 최종 성능 — 동일 실행 job15526

H100/node01, L384, C128/양방향 H256, BF16, dropout25%, pair mask/residual.

| 구성 | forward | backward | 전체 fwd+bwd |
|---|---:|---:|---:|
| 이전 개선 B7 조합 | 294.400µs | 767.296µs | 1065.824µs |
| 구형 B7을 둔 검사 코드 | 294.592µs | 923.712µs | 1222.288µs |
| 최신 B1 + 최신 단일 B7 후보 | 294.384µs | 712.624µs | **1009.808µs** |

이전 개선 조합 대비 **5.26% 시간 단축 / 1.0555배**. 전체6×300회, 구간5×300회 교차 graph 측정.
full은 직접 측정했으며 분리 구간의 합계가 아니다. optimizer/RNG 생성/CPU dispatch/compile은 제외한다.
과거1048µs·직전1185µs와 직접 나누지 않는다. 이전 개선 구성의 B1/B7 cubin SHA 일치를 확인했다.
L768 최신 B7 조합의 전체 시간은 측정하지 않았다.

## 선택과 적용 상태

| 부분 | 구현 / 위치 | 상태 |
|---|---|---|
| Forward | Anthropic 유래 infer_k1 + save_k3, cuBLAS | 동일한 학습 forward 유지 |
| B1–B4 | runs/trimul_b1_gate_demote_20260922/policy.py | dWproj 저장 지연 + TMA 캐시 정책 선택; L384/768 bit-exact·memcheck·racecheck 통과 |
| B7–B12 | runs/trimul_b7_weight_batch128_20260922/plan.py | K128/ring12/producer32/cluster2 단일 CUDA; L384 개별 검사·sanitizer 통과, 새 전체 검사에서 입력 LN gradient 한도초과 |
| 최종 측정 조합 | runs/trimul_full_latest_20260922/policy.py, Latest | 위 B1/B7을 결합한 진단 후보; 기본 배선 아님 |
| 기존 개발 진입점 | runs/trimul_training_current.py | 기존 joint_lncolumns B7 유지; 최종 측정 조합과 다름을 docstring에 명시 |
| 생산 dispatch | miniworld-engine | 이번 마무리에서 변경 없음 |

B1의 dWproj 누산값은 gate 단계까지 레지스터에 유지한다. dGate TMA 쓰기는 evict_last,
소비할 때는 evict_first를 적용한다. L384는 소비한 x_n도 evict_first다.
B7은 명시적 TMA/WGMMA, 260 CTA/256 threads, shared112KiB, K128 weight stages와 global ring을 쓴다.
단일 호출이어도 global ring/partial scratch 왕복이 존재한다.

## 저장 정책

forward에서 BF16 ab(left/right), 원본 BF16 tri, affine 입력 x_n BF16, 출력 LN mean/rstd FP32를 유지한다.
출력 정규화 activation·projection·gate는 별도 저장하지 않고 backward 안에서 다시 계산한다.
cuBLAS contraction과 기존 수학/반올림 정책을 유지한다.

## 검증과 미해결 항목

최신 조합의3case에서 forward는 정확히 일치하고 모든 경로의 graph/eager 결과는 bit-exact다.
dX와 가중치 gradient는 기존 허용치 내지만 아래 입력 LN 파라미터 gradient는 한도를 넘었다.

| case | gradient | 상대L2 | 기존 한도 |
|---|---|---:|---:|
| 1 | dgamma_in | 8.370424e-6 | 5e-6 |
| 1 | dbeta_in | 6.891240e-6 | 5e-6 |
| 2 | dgamma_in | 9.534856e-6 | 5e-6 |

한도를 완화하지 않았다. 개별 커널 sanitizer 통과가 전체 수치 정확도를 보장하지 않는다.
재개할 때 우선 이 오차를 독립 참조와 비교해 원인을 분리하고, 해결한 뒤 전체3자 벤치와 검증을 다시 수행한다.
L768 최신 단일 B7 적용, 생산 dispatch 승격, SoL90 목표는 완료 항목이 아니다.

## 추가 실험 정리

B1 후속7개 방향·새 후보70개(대조군 포함83개 shape/config)를 기록했다.
TMA 묶음, L2 보존 비율, 타일 분할·부분합 캐시, gate/LN 중첩, 합산 vector load, 동기화 변경을 검사했다.
추가 채택은 없다. 작은 동기화 후보는 B1 0.28% 감소였지만 전체0.07% 차이는 변동과 구분하기 어려웠다.
실패 후보와 cubin/NCU/samples는 재현을 위해 보존한다. 성능을 반복 주장하는 낡은 헤드라인은 첫 화면에서 제외했다.

## 출처와 기조

Anthropic의 biomolecular inference 구현과 TMA/WGMMA primitives를 적극 차용한 학습 확장이다.
자체 추론 구현보다 우수했던 upstream 결과를 계승한다는 기존 기조와 라이선스 표기를 유지한다.
관련 통합 기록: [Anthropic inference](../../../tmp_kernel/ANTHROPIC_INFERENCE.md).

## 재현·증거

- [최종 벤치/실패 수치/telemetry](../../../runs/trimul_full_latest_20260922/README.md)
- [B1 선택과 검증](../../../runs/trimul_b1_gate_demote_20260922/README.md)
- [B7 선택과 전체 검사 제한](../../../runs/trimul_b7_weight_batch128_20260922/README.md)
- [후속 제외 실험과 NCU](../../../runs/trimul_b1_cache_followup_20260922/README.md)
- [고정 증거 목록·SHA-256](manifest.json)
- [누적 상태 보관본](../../../TRIMUL_STATUS_HISTORY_20260922.md)

동일 조건 재현: 저장소 루트에서 `sbatch --job-name=trimul-full-now runs/trimul_full_latest_20260922/bench.sbatch`.
새 실험은 실행하지 않았다. 관련 마지막 실험·검증 잡들은 종료된 상태이며 다른 학습 잡은 건드리지 않았다.
이번 마무리에서 commit/push는 수행하지 않았다.
