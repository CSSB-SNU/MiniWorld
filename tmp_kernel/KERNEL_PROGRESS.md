# Trimul / Transition 개발 현황

정리: 2026-09-19. 로컬 개발 checkout 기준이며 실행 중 MiniWorld 학습 설치본과 구분한다.

## 개발 방향 전환 — Anthropic의 추론 최적화를 계승

우리도 생체분자 모델의 추론 커널을 자체 개발해 왔지만,
[Anthropic이 공개한 추론 최적화](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling)의
성과가 우리의 자체 개발보다 훨씬 뛰어났음을 인정한다. 그 성과를 존중하며,
이제는 해당 개발을 계승하고 **그 위에 고성능 학습 지원을 확장하는 것**을
miniworld-engine의 개발 방향으로 삼는다.

엔진의 인터페이스·튜닝·캐시·검증 골격은 유지한다. upstream 커널 전체를 조사해
통합하고 H100에서 재현·NCU 프로파일링한 뒤, 해당 구현과 아이디어를 바탕으로
학습 forward/backward를 개발한다. 원본 구현·우리의 수정·새 학습 구현의 기여를
명확히 구분하고 출처를 보존한다.

이는 방향 전환의 선언이며, 전체 이식이나 H100 검증이 완료되었다는 뜻은 아니다.
아래 기록은 전환 이전 자체 개발의 결과이며, 당시의 유지·개발 계획보다 이 방향이
우선한다. 수치 비교는 동일 조건의 실측으로 별도 증명한다.

[엔진의 공식 개발 방향과 기여 표기 원칙](../runs/trimul_sm90_parity_20260917/engine/docs/project-direction.md).

### 9/19 원본 연결 및 H100 분석

[Anthropic 추론 이식·분석 보고서](ANTHROPIC_INFERENCE.md): 공통 추론 연산 16개 분류,
후보/shape 178개 검증 통과, 72개 NCU profile. TriMul·Transition·TriangleAttention의
명시적 backend와 기타 primitive/operation 접근 경로를 연결했다.
학습 커널 개발, 전체 모델 검증, 모델별 genomics/Pallas 경로 실행은 미완료 범위다.

## 9/19 CUDA 구현 최적화

융합 경계는 유지하고 shared-memory IO를 명시적 PTX로 바꿨다. Backward의 h/dA/dB 출력은 TMA store로 옮겼다. 선택한 16개 커널/config의 실제 SASS에서 local spill load/store가 없어졌다. 명시적 CUDA variant에 반영했다.

node02 H100, L384, static compile + manual CUDA Graph, 단위 ms. CUDA 열은 두 형식 중 해당 모드에서 빠른 값이다.

| D | 새 CUDA 추론 | 최선 Triton 추론 | 이전 CUDA 학습 | 새 CUDA 학습 | 최선 Triton 학습 |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.2030 | 0.1651 | 1.0360 | 0.9552 | 0.9382 |
| 256 | 0.5341 | 0.5639 | 2.4492 | 2.1956 | 2.2715 |
| 384 | 1.5376 | 1.4074 | 5.4726 | 4.9734 | 4.7506 |
| 512 | 2.8190 | 2.3541 | 9.4058 | 8.1341 | 7.5051 |

기존 CUDA 대비 전체 학습은 1.08~1.16배 빨라졌다. 최선 Triton 대비로는 D256만 약 1.04배 빠르며, 전체 15% 목표는 미달이다. 경계 shape 24건, graph 8건, memcheck/racecheck 각 16설정 통과. M129의 8개 조합에서 이전 CUDA와 출력·6개 gradient가 bitwise 일치했다.

[모든 L384/768·형식별 측정](../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-opt-20260919/RESULTS.md) · [구현·검증·cubin/ptxas/NCU 근거](../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-opt-20260919/README.md).

## Transition forward — 9/18 기준, 이번 CUDA 최적화 이전

node02 H100, B1 pair, expansion4, BF16 activation/weight + FP32 LN affine, nonzero squeeze. 모두 static compile + manual CUDA Graph. 단위 ms. 이전 Triton은 현재 코드에서 다시 실행한 이전 split 알고리즘이며 역사적 커밋 복원 비교는 아니다. 현재 Triton은 D128/256 full-K b2b, D384/512 split이다. 새 CUDA는 두 형식 중 해당 shape에서 빠른 값이며 기존 H100 auto 경로와 별개다.

| D | L | PyTorch | 이전 Triton split | 현재 Triton | 새 CUDA 최선 | 현재 Triton/PyTorch 속도비 | 현재 Triton/이전 속도비 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 0.4343 | 0.2755 | 0.1659 | 0.2108 | 2.618× | 1.660× |
| 128 | 768 | 1.6520 | 1.0324 | 0.5939 | 0.7734 | 2.781× | 1.738× |
| 256 | 384 | 0.8447 | 0.7078 | 0.5651 | 0.6465 | 1.495× | 1.252× |
| 256 | 768 | 3.2776 | 2.8965 | 2.1154 | 2.4305 | 1.549× | 1.369× |
| 384 | 384 | 1.4372 | 1.4128 | 1.4128 | 1.6431 | 1.017× | 1.000× |
| 384 | 768 | 5.7091 | 5.7812 | 5.7812 | 6.3429 | 0.988× | 1.000× |
| 512 | 384 | 2.1787 | 2.3445 | 2.3445 | 3.3561 | 0.929× | 1.000× |
| 512 | 768 | 8.7894 | 9.2645 | 9.2645 | 12.8118 | 0.949× | 1.000× |

[새 CUDA 자체의 PyTorch/이전 Triton 대비 속도비까지 포함한 표](../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918/FORWARD_COMPARISON.md) · [학습/방향별 시간](../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918/RESULTS.md) · [설정](../runs/trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918/CONFIGS.md).

아래 9/18 측정에서 새 CUDA 두 형식은 당시 최선 Triton을 이기지 못했다. D384/512에서는 PyTorch보다도 느리다. 작은 D의 Triton 개선과 새 CUDA의 성능을 혼동하지 않는다.

## 개발 상태

| 대상 | 완료 | 유지 / 남은 일 |
|---|---|---|
| Trimul Triton | F5+F6+F7, B9+B10, B11+B12 융합; 단방향 연결; 양방향 추론 activation cat 제거; 공통 LN backward 개선 | B7 분리는 실익 작아 통합 유지; contraction/weight gradient cuBLAS 유지 |
| Trimul H100 | 동일 융합의 F2/F567/B9+B10 TMA/WGMMA 선택 경로, B4 native 선택 경로 | B4 최선 Triton 대비 우위 미달, 새 mapped B4 미승격; 전체 native 캐시/shape 검증 미완 |
| Transition Triton | fwd/bwd residual 융합; D128/256 full-K b2b 기본 연결; D384/512 두 형식 실험 및 설정 관리 | 큰 D 기본은 split. 작은 D 추가 개선과 큰 D b2b 성능 문제는 별도 |
| Transition 새 CUDA | streamed-K/full-K 두 형식의 fwd + saved-xn gate bwd + LN/residual bwd; TMA/WGMMA; 명시적 Transition 모듈 선택 | 9/19: 선택 커널 spill 제거·TMA store 개선 완료. 큰 D 추가 개선, build-all/cache 등록, 자동 승격 미완 |

Transition은 GPU 18건 및 memcheck 10건(오류 0건), 관련 import/dispatch 47건을 통과했다. Forward에서는 큰 h를 HBM에 저장하지 않는다. Backward의 cuBLAS 네 번과 작은 weight cat 비용은 측정에 포함한다. Native TMA/WGMMA는 SASS에서 확인했으며, NCU는 큰 D의 낮은 issue 효율과 local-memory traffic을 보여준다. 이 결과를 성능 상한이라고 판단하지 않는다.

기존 Transition H100 auto(CUDA b2b/CuTe) 배선은 유지했다. 새 CUDA는 `Transition(implementation="cuda", cuda_variant=..., cuda_forward_config=..., cuda_backward_config=...)`로 명시적으로 선택한다. Triton 강제 모드에서는 명시 CUDA를 거부하므로 engine_backend="auto"로 설정한다.

## Trimul 누적 학습 개선

9월18일 마무리 당시 실제 9월11일 커밋을 동일 런타임으로 복원해 재측정한 기록이다. 이번 Transition 실험 중 다시 실행한 수치는 아니다. H100, B1/D=hidden128, dropout0.25 및 replay마다 실제 RNG 갱신, static compile+CUDA Graph, 양방향 모듈 fwd+bwd, optimizer 제외.

| L | 9/11 Triton ms | 현재 Triton ms | H100 선택 경로 ms | 누적 H100 속도비 |
|---:|---:|---:|---:|---:|
|384|1.950904|1.586288|1.535416|1.271×|
|768|7.727528|6.162152|5.932240|1.303×|

현재 Triton 대비 H100의 추가 이득은 각각 1.033× / 1.039×다. 누적 27~30% 속도비를 H100 이식만의 이득으로 해석하지 않는다. 당시 baseline의 H100 캐시 부재와 현재 튜닝 차이도 포함한다. H100 열은 front/f567/dual_bwd/out_ln_bwd를 명시 선택한 결과이며 untouched default가 아니다. 전체 MiniWorld 학습 속도 측정은 아니다.

[Trimul 마무리 원본](../runs/trimul_sm90_parity_20260917/engine/docs/records/trimul-weekly-closeout-20260918/README.md) · [B4 기준 정정 및 미승격 실험](../runs/trimul_sm90_parity_20260917/engine/docs/records/trimul-b4-l384-20260918/README.md).

[Transition 두 버전 배선 그림](transition/TRANSITION_VARIANTS.svg) · [Trimul 배선 뷰어](trimul/TRIMUL_STATUS.html).

## Trimul PyTorch 비교 추가 실측 (2026-09-19)

node02 H100, B1/D=hidden128, BF16 + FP32 norm affine. PyTorch 포함 모두 정적 compile + CUDA Graph. 학습은 dropout0.25/RNG 갱신 포함 fwd+bwd, 추론은 dropout0. H100은 위와 같은 명시적 선택 경로다. 두 독립 capture로 반복했다.

| 구분 | L | PyTorch ms | Triton ms | H100 ms | Triton/PyTorch | H100/PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| training | 384 | 4.0121 | 1.5914 | 1.5380 | 2.521× | 2.609× |
| training | 768 | 35.3398 | 6.1435 | 5.9168 | 5.752× | 5.973× |
| inference | 384 | 1.3027 | 0.4252 | 0.4048 | 3.064× | 3.218× |
| inference | 768 | 11.6595 | 1.6290 | 1.5128 | 7.157× | 7.707× |

[비교 조건·검증·원본](../runs/trimul_sm90_parity_20260917/engine/docs/records/trimul-pytorch-compare-20260919/README.md).
