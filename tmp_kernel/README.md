# 커널 그림 · 연산별

한 장 = 한 연산의 한 방향. 세로로 세 칸이며 왼쪽부터 **HBM READ → KERNEL(수식 · PyTorch 대응) → HBM WRITE**,
행 하나가 커널 하나다. 표시된 크기는 버퍼 payload이고 실제 HBM 전송량은 L2 hit·CTA 재읽기에 따라 달라진다.

| 연산 | 그림 | 상태 |
|---|---|---|
| **TriMul** | [FORWARD](trimul/TRIMUL_FORWARD.svg) · [BACKWARD](trimul/TRIMUL_BACKWARD.svg) · [L768 FWD](trimul/TRIMUL_FORWARD_L768.svg) · [L768 BWD](trimul/TRIMUL_BACKWARD_L768.svg) · [INFERENCE](trimul/TRIMUL_INFERENCE.svg) | 측정 |
| | [Anthropic 커널 지도](trimul/ANTHROPIC_TRIMUL_KERNELS.svg) · [B1–B4 backward](trimul/ANTHROPIC_B1_B4_BACKWARD.svg) | 측정 |
| **Transition** | [전체](transition/TRANSITION.svg) · [FORWARD](transition/TRANSITION_FORWARD.svg) · [BACKWARD](transition/TRANSITION_BACKWARD.svg) · [residual](transition/TRANSITION_RESIDUAL.svg) · [변형 비교](transition/TRANSITION_VARIANTS.svg) | 측정 |
| **OuterProductMean** | [FORWARD](msa_opm/MSA_OPM_FORWARD.svg) | 측정 (H100 0.906 ms) |
| | [BACKWARD](msa_opm/MSA_OPM_BACKWARD.svg) | **설계 (P2)** · 시간 미측정 |
| **MSAPairWeightedAveraging** | [FORWARD](msa_pwa/MSA_PWA_FORWARD.svg) | 측정 (H100 0.736 ms) |
| | [BACKWARD](msa_pwa/MSA_PWA_BACKWARD.svg) | **설계 (P3)** · 시간 미측정 |
| **Cropping** (커널 아님, 데이터 경로) | [전체 그림](cropping/CROPPING.svg) | — |

backward 두 장은 머리띠가 자주색이다. **측정값이 아니라 설계**라는 뜻이고, FLOP과 트래픽만 근거로 적었다.

## 문서

| | |
|---|---|
| [KERNEL_PROGRESS.md](KERNEL_PROGRESS.md) | 커널 작업 전체 진행 상황 |
| [ANTHROPIC_INFERENCE.md](ANTHROPIC_INFERENCE.md) · [ANTHROPIC_STATUS.html](ANTHROPIC_STATUS.html) | Anthropic 추론 커널 도입 현황 |
| [trimul/TRIMUL_STATUS.md](trimul/TRIMUL_STATUS.md) · [.html](trimul/TRIMUL_STATUS.html) | TriMul 현황판 |
| [trimul/ANTHROPIC_TRIMUL_TRAINING.md](trimul/ANTHROPIC_TRIMUL_TRAINING.md) · [ANTHROPIC_TRIMUL.html](trimul/ANTHROPIC_TRIMUL.html) | TriMul K1/K3 학습용 확장 |
| [trimul/CLAUDE_HANDOFF_B1B4.md](trimul/CLAUDE_HANDOFF_B1B4.md) | B1–B4 인수인계 |
| [transition/TRANSITION_STATUS.html](transition/TRANSITION_STATUS.html) | Transition 현황판 |

## 다시 만들기

    python tmp_kernel/generate_msa_svg.py          # msa_opm/, msa_pwa/ 네 장
    dot -Tsvg docs/cropping-current.dot -o cropping/CROPPING.svg

TriMul·Transition 그림은 각 fusion 문서(`docs/trimul-fusion/`, `docs/transition-fusion/`)가 출처다.
