# 양방향 TriMul H100 구현

**2026-09-18 개발 마무리:** [9월 11일 커밋과 동일 조건 재측정](../../runs/trimul_sm90_parity_20260917/engine/docs/records/trimul-weekly-closeout-20260918/README.md).
양방향 학습(dropout 0.25, static compile + CUDA graph)은 당시 Triton 대비 현재 H100 혼합 경로가
L384 **1.271배**, L768 **1.303배** 빨라졌다(시간 **21.3% / 23.2% 감소**).
현재 Triton만 비교하면 각각 **1.230배 / 1.254배**다. 새 mapped B4 실험은 최종 수치에서 제외했다.
이 비교에는 당시 커밋의 H100 캐시 부재와 현재 튜닝의 차이도 포함된다.

**2026-09-18 최신 정정:** [B4 경로 재비교·L384 개선 실험](../../runs/trimul_sm90_parity_20260917/engine/docs/records/trimul-b4-l384-20260918/README.md).
이전 L768 B4 1.187배는 선택된 Triton persistent 대비였다. 현재 Triton atomic이 그 TMA 구현보다 빠르다.
새 실험 후보는 가장 빠르게 측정된 Triton atomic 대비 B4가 L384 약 1.023배, L768 약 1.080배다.
전체 학습은 L384에서 일관된 개선이 없고, L768에서 추가 약 0.44–0.49% 개선했다(dropout=0.25).
후보는 아직 production 미반영이며, 기존 배선도 유지 중이다. 아래는 이전 단계의 기록이다.

최신 변경은 [공통 LayerNorm backward 개선·B7 분리 실험](LAYERNORM_B7_UPGRADE.md)이다.
공통 Triton LN 개선 후 현재 H100 혼합 경로의 학습 시간은 L384 **1.548ms**, L768 **5.997ms**다.
이 변경 전 Triton/CuTe 비교는 [Dropout ON 재측정](DROPOUT_TRAINING.md)에 있다.
CuTe 커널 구현은 [TMA 추가 최적화 2차](TMA_ROUND2.md) 이후 유지했다.
이후 [Backward 추가 실험 3차](BACKWARD_ROUND3.md)는 기존보다 빠른 후보가 없어
구현을 유지했다. B9+B10 재측정은 L3841.067× / L7681.051×다.
[Backward 추가 실험 4차](BACKWARD_ROUND4.md)에서도 큰 개선은 없었다.
dense BF16 roofline을 새로 수집했으며, B9+B10 L768은 HBM roof의88.49%다.
Triton의 융합 경계·수식·CSV config 공간을 유지하며 F2, F567, B9+B10을
명시적 TMA/WGMMA 구현으로 개선했다.

- F2는 L384 **1.203×**, L768 **1.187×**로 커널 15% 목표에 도달했다.
- F567은 L3841.132×, L768 약1.150×다. L768은15% 경계에 걸쳐 안정적인 달성으로 보지 않는다.
- B9+B10과 전체 모듈은 아직15% 미달이다. L128 성능은 목표에서 제외했다.
- 전체 모듈 p=0.25/RNG 포함, compile + CUDA graph: L384 **1.627→1.572ms** (1.035×), L768 **6.286→6.119ms** (1.027×).
- Graph OFF에서는 L384 CuTe가 약24.9% 느리다. 실행 모드별 표는 최신 보고서를 참조한다.
- 두 shape의 전체 gradient 검증과 선택 config memcheck/racecheck를 통과했다.
- 개발 checkout에서 검증했으며 운영 학습 설치본에는 적용하지 않았다.
- 전체 native cache와 전체 새 후보 공간의 튜닝은 아직 완료하지 않았다.

## 자료

- [LayerNorm backward 개선·B7 분리 비교·NCU 및 전체 학습 검증](LAYERNORM_B7_UPGRADE.md)

- [공통 Triton NCU: B7·LayerNorm 및 B4 config 진단](COMMON_TRITON_PROFILE.md)

- [Dropout ON 커널별 FWD/BWD 시간·cuBLAS 비중·CUDA 이식 판단](KERNEL_BREAKDOWN.md)

- [Dropout ON 전체 학습·실제 RNG graph 검증](DROPOUT_TRAINING.md)

- [Backward 4차·dense BF16 roofline·3D TMA](BACKWARD_ROUND4.md)
- [Backward 추가 실험·제외 이유·NCU](BACKWARD_ROUND3.md)
- [현재 구현·성능·검증](TMA_ROUND2.md)
- [이전600c8c4c 구현·결과](OPTIMIZATION.md)
- [엔진의 사용 옵션·config/cache 계약](../../runs/trimul_sm90_parity_20260917/engine/docs/kernels/trimul-sm90-parity.md)
- [실제 cubin/SASS 분석의 초기 기록](CUBIN_AUDIT.md)
- [첫 구현 당시 결과](BASELINE.md) — 현재 성능 순위가 아닌 과거 기록

개발 브랜치 `perf/trimul-sm90-parity`. 추가 최적화의 실험 원본은
`runs/trimul_sm90_15pct_20260917`에 보관한다.

## 2026-09-17 overnight normalization work

[Normalization, TMA B4 qualification and cache coverage report](../../runs/trimul_sm90_parity_20260917/engine/docs/records/normalization-h100-20260917/README.md).
