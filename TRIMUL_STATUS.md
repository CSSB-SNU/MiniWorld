# TriMul 개발 현황 · 2026-09-23

**L384 입력 LN gradient 엄격 검증 문제를 해결했다.** 기존 상대L2 한도5e-6를 유지했고, 실패3case의 최대 오차는9.535e-6 → **3.550e-7**로 감소했다.

- 원인: gate 미분 `((da*g)*p)*(1-g)`가 원래 `((da*p)*g)*(1-g)`와 FP32 중간 반올림 순서가 달랐다. 기존 순서를 복원했다.
- 검증: 기존 실패3case, 추가12조건 × TriMul/전체 블록, graph/eager bit-exact, 같은 cubin memcheck/racecheck 통과.
- 적용: `runs/trimul_training_current.py`의 L384 개발 경로. 엔진 production auto-dispatch는 변경하지 않았다.
- 성능: job15612 같은 실행 TriMul 전체1006.800 →1009.168µs(+0.24%). MiniPairformer는 job15616에서1575.856 →1579.664µs.
- 남은 범위: 큰 폭의 추가 최적화, production 승격, SoL90/252µs 목표. 이번 정확도 수정으로 완료됐다고 주장하지 않는다.

[수정·검증 상세](runs/trimul_ln_gradient_20260922/index.html) · [선택·SHA-256](runs/trimul_ln_gradient_20260922/selected.json) · [수정 전 마무리 기록](docs/trimul-fusion/closeout-20260922/README.md)

## MiniPairformer 1블록 · 새 CUDA Transition 포함

L384/C128 H100, job15581: 최신 추론 **0.406ms**, 학습 forward **0.431ms**, 학습 fwd+bwd **1.582ms**. 동일 실행의 이전 H100 TriMul + 기존 Triton Transition은 각각 0.578/0.626/2.470ms. 수정 전 측정 기록이며, 이후 L384 LN gradient 엄격 검증을 통과했다.

[비교표·배선·검증](runs/minipairformer_block_cuda_20260922/index.html)

## MiniPairformer 1블록 · PyTorch / cuEquivariance

job15592, 같은 실행 최신 추론 0.386ms / 학습 forward 0.412ms / 학습 fwd+bwd 1.520ms. 학습은 PyTorch compile 대비 3.43배, cuEq primitives + PyTorch Transition 대비 2.28배, cuEq + 동일 CUDA Transition 대비 1.81배. 수정 전 측정 기록이며, 이후 L384 엄격한 LN gradient 검증을 통과했다.

[조건·전체 표·검증](runs/minipairformer_block_baselines_20260922/index.html)

## Transition backward 추가 개선 · 2026-09-22

| 범위 | 이전 µs | 변경 µs | 시간 감소 |
|---|---:|---:|---:|
| Transition backward · L384 | 443.840 | 429.312 | 3.27% |
| Transition backward · L768 | 1714.896 | 1703.360 | 0.67% |
| MiniPairformer fwd+bwd · L384 | 1565.920 | 1548.688 | 1.10% |

기존 hand-CUDA 대비. 엄격 검증·memcheck/racecheck 통과, Transition 개발 작업 트리에 반영. [구조·결과](runs/transition_bwd_upgrade_20260922/index.html).

## TriMul D별 최적화 · 2026-09-23

[최신 표·배선·검증](runs/trimul_cuda_widths_opt_20260923/index.html). D128 L768은 단일 B7로 전환했다. D64/256/384/512도 이전 CUDA 대비 개선했지만 Triton보다 느리다. 10shape autograd와 수정 커널 sanitizer 검증 통과.

## 큰 D forward 개선 · 2026-09-23

우리 K1/K3 기반의 forward 전용 경로를 엔진 release 작업 트리에 추가했다.
H100, B1, BF16, 양방향, 저장값·dropout25%·residual 포함, 같은 실행에서 이전 CUDA와 교대 측정했다.

| D | L384 이전 → 신규 ms | L768 이전 → 신규 ms |
|---:|---:|---:|
|256|1.150 → 0.936|4.654 → 3.810|
|384|2.016 → 1.816|8.328 → 7.359|
|512|3.261 → 2.940|13.575 → 12.020|

전체 forward 시간 9.9–18.6% 감소. K3는 출력 LN 타일을 shared memory에서 재사용하고,
D512 K1은 shared operand와 가중치 조각별 공급을 사용한다.
6shape PyTorch/Triton 출력, 저장값, 변경된 입력의 CUDA graph 검증 통과.
명시적 `h100_wide_forward.Forward` 진입점이며 기존 자동 dispatch와 backward 연결은 변경하지 않았다.
[구조·사용법·측정·sanitizer 결과](runs/trimul_forward_wide_20260923/README.md).

## 큰 D backward 착수: B1 첫 후보 · 2026-09-23

우리 forward의 shared LN 재사용을 B1 projection/gate 재계산에 적용했다.
D256·384에서 B1 4.5–8.9%, 전체 backward 1.5–4.1%, fwd+bwd 2.0–3.5% 단축.
비교 양쪽 모두 새 forward를 사용하므로 backward 변경만의 효과다.
D512 후보는 전체 backward가 2.1% 느려져 채택하지 않았다.
검사한 5shape의 11개 gradient가 기존 relative-L2 < 0.01 기준을 통과했고,
D256·384 L384의 변경 커널 memcheck/racecheck는 오류·hazard가 없다.
실험 진입점만 추가했으며 엔진 자동 dispatch는 그대로다. B7 구조 변경은 아직 남아 있다.
[D128 비교·B1 결과·남은 tensor 소비 관계](runs/trimul_backward_wide_20260923/README.md).

## D256 backward: D128 구조 적용 · 2026-09-23

D128의 producer-local dW를 D256에 적용하여 미분 타일에서 dW를 즉시 누적한다.
L384/L768의 B7 24.1–25.3%, 전체 backward 12.0–12.2%, fwd+bwd 9.8–9.9% 단축.
양쪽 모두 새 forward와 직전 B1을 사용하는 동일 실행 내 비교다.
L384 backward 4.173→3.671ms, L768 16.989→14.911ms.
L768 dW split16으로 엄격한 오차 기준을 유지했다. 10개 변경 조건, graph,
두 길이 memcheck/racecheck 및 명시적 선택 진입점 검증 통과.
**SOL90 미달**: 낙관적 재계산 roofline 기준 20.7%/23.6%. B1 재설계와 B7
미분 버퍼 축소가 남아 있으며, spill로 느려진 완전 융합/ring 후보는 채택하지 않았다.
자동 engine dispatch는 변경하지 않았다.
[구조·선택 진입점·정확도·SOL 계산·실패 후보](runs/trimul_d256_bwd_sol90_20260923/README.md).
