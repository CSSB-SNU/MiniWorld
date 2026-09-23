# Transition 추가 개발: b2b에서 xn 저장 — 2026-09-18

**hand-CUDA b2b 본체를 수정했다.** 이미 계산한 BF16 xn을 128-bit 벡터 store로 함께 저장한다.
Backward는 이를 재사용하며, 별도 정규화 커널이나 xn 재계산이 필요 없다.
Forward와 backward의 residual 융합은 유지한다.

## 최종 default auto (node02)

| L | D | Triton 학습 | 새 H100 학습 | 속도비 |
|---|---|---:|---:|---:|
| 384 | 128 | 1.047ms | 0.940ms | 1.11배 |
| 768 | 128 | 3.975ms | 3.525ms | 1.13배 |
| 384 | 256 | 2.437ms | 2.263ms | 1.08배 |

별도 같은 조건의 old/new 직접 비교에서는 기존 H100보다 약 6.6~6.8% 빨랐다.
모듈 벤치, BF16, B1/n4, nonzero squeeze, static compile, manual CUDA graph, 두 capture 평균.
학습은 forward+backward이고 optimizer는 제외한다. 상세 수치는 아래 원본에 있다.

## 프로파일에서 무엇이 바뀌었나

L768/D128의 같은 입력, torch.profiler 단일 compiled 실행:

| 커널 | 기존 재계산 | 새 저장·재사용 |
|---|---:|---:|
| CUDA b2b forward | 499.23µs | 504.89µs |
| Triton gate-backward | 1480.64µs | 1229.40µs |

Forward 추가 저장 약 5.7µs로 backward 약 251µs를 절약했다.
이전 약 1.05배에 그쳤던 원인인 xn 재계산 비용을 줄인 결과다.
D128/D256 saved 경로의 gate-backward는 Triton이며 cuBLAS 4회도 유지된다.
즉 전체 backward가 CUDA로 교체된 것은 아니다.

## Forward: hand-CUDA와 현재 CuTe 직접 비교

| L | D | b2b | CuTe expand + squeeze/residual | b2b 속도비 |
|---|---|---:|---:|---:|
| 384 | 128 | 0.160ms | 0.309ms | 1.93배 |
| 768 | 128 | 0.571ms | 1.170ms | 2.05배 |
| 384 | 256 | 0.525ms | 0.652ms | 1.24배 |

이는 동일 shape의 모듈 비교다. b2b와 CuTe split의 융합 경계가 다르므로 언어 자체의 우열로 해석하면 안 된다.
CuTe 전체 config 재탐색은 하지 않았다. b2b의 기존 유효 공간(D128 4개, D256 2개)은 모두 확인했다.

## 범위와 비용

- `transition_h100_save_xn=True`: native D128/256, M≥16384, gradient가 필요한 학습에서 기본 ON.
- 추론/작은 M/설정 False는 이전 경로. Wide CuTe는 변경하지 않았다.
- 보관 tensor 크기: L384/D128 36MiB, L768/D128 144MiB, L384/D256 72MiB. 전체 모델 peak memory 증가량을 측정한 수치는 아니다.
- 회귀 검증 **52건 통과**, compute-sanitizer **2건 통과 / 오류 0**.
- b2b 6개 shape·저장모드 캐시 bucket을 기록했다. 다른 native 커널 전체 캐시 재빌드는 별도다.
- 개발 checkout에 적용했으며 실행 중 MiniWorld 학습/설치본/원격에는 배포하지 않았다.

[구현·검증 기록](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-b2b-savedxn-20260918/README.md) · [원본 시간표](../../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-b2b-savedxn-20260918/timings.md) · [수정 전 분석](BEFORE_SAVEDXN.md)
