# TriMul · 최신 학습 커널 배선

학습 SVG는 **개발 커밋 `8c7d8b39`** 기준으로 갱신했다. 왼쪽은 Triton,
오른쪽은 **동일한 융합 알고리즘을 유지하는 CuTe TMA/WGMMA 선택 경로**다.
운영 설치본 적용 여부와 구분하며, 아래 이전 설치본 기록은 과거 상태다.

- [확대 뷰어](TRIMUL_STATUS.html)
- Forward: [L384](TRIMUL_FORWARD.svg) · [L768](TRIMUL_FORWARD_L768.svg)
- Backward: [L384](TRIMUL_BACKWARD.svg) · [L768](TRIMUL_BACKWARD_L768.svg)
- [Dropout ON 전체 학습 성능·검증](docs/trimul-sm90/DROPOUT_TRAINING.md)
- [커널 성능·NCU·검증 근거](docs/trimul-sm90/TMA_ROUND2.md)

| 연산 / 융합 경계 | Triton | 최신 H100 선택 경로 |
|---|---|---|
| F2: 4 projection + gate + mask + saved preactivation | `_bidir_front_kernel` | `FrontSingleWarpgroupSm90` (측정한 4-warps config) |
| F567: 독립 projection/gate GEMM + dropout + residual | `_output_f567_kernel` | `ParityF567Sm90` |
| B9+B10: 입력 gradient의 두 GEMM + 합산 | `_input_dual_bwd_kernel` | `DualBackwardSm90` |
| F1/F4, B1/B4/B7, B11+B12 | Triton | 같은 Triton 경로 |
| Contraction / weight-gradient GEMM | cuBLAS | 같은 cuBLAS 경로 |

설정은 모델 구성·compile 전에 적용한다.

```python
settings.configure(
    engine_backend="triton",
    trimul_sm90_kernels={"front", "f567", "dual_bwd"},
)
```

H100 overrides의 기본값은 빈 집합이다. 그림의 tile/warp/stage는 명시적으로
측정한 후보이며, runtime의 shape별 하드코딩이나 전체 cache 완성을 뜻하지 않는다.
NCU 관찰에는 해당 checkpoint와 shape를 명시했다. 커널 시간은 별도 CUDA graph
반복 측정이며, 그림 상단 시간은 forward+backward 전체 모듈 시간이다.
L128은 성능 목표에서 제외했다.

재생성: `python3 scripts/render_trimul_fusion.py`와
`python3 scripts/render_kernel_viewers.py --trimul-only`.

---

## 이전 설치본 기록 — 아래 내용은 최신 개발 경로가 아님

# TriMul 배선도 — 현재 구현 / 실험 분리

**[확대 가능한 뷰어](TRIMUL_STATUS.html)** · [Transition 배선도](TRANSITION_STATUS.html)

2026-09-17 cu128 설치본 기준. 기본 학습 그림은 **현재 Triton / 현재 H100 두 열**이다. 기존 3열 그림의 이전 H100 경로와 미적용 CuTe F567 실험을 기본 비교에서 분리했다.

| 구분 | 그림 | 범위 |
|---|---|---|
| 추론 | [SVG](TRIMUL_INFERENCE.svg) | Triton / H100, 단방향 / 양방향 네 경로 |
| 학습 forward | [L384](TRIMUL_FORWARD.svg) · [L768](TRIMUL_FORWARD_L768.svg) | 양방향, 현재 Triton / 현재 H100 |
| 학습 backward | [L384](TRIMUL_BACKWARD.svg) · [L768](TRIMUL_BACKWARD_L768.svg) | 양방향, 현재 Triton / 현재 H100 |

## 현재 연결

- **Triton 학습:** F4는 별도, F5+F6+F7은 `_output_f567_kernel` 하나. Backward는 B9+B10과 B11+B12가 각각 한 커널이다. 이 융합은 단방향 outgoing/incoming 학습에도 연결되어 있다. 단방향은 hidden 폭 h와 contraction 1회, 양방향은 2h와 contraction 2회이므로 양방향 그림의 tensor 폭과 GEMM 수를 그대로 단방향에 적용하면 안 된다.
- **H100 양방향 학습, 그림의 L384/L768:** forward는 `output_training.forward`의 xhat LN → weight fold → cuBLAS projection → 별도 gate GEMM → gate/dropout/residual kernel이다. **현재 forward에서 CuTe F567을 호출하지 않는다.** Backward의 `_DgradLNRowsSm90`은 dgrad GEMM과 LN dx 보정을 융합하며, row correction/parameter gradient는 별도다.
- **Triton 양방향 추론:** `packed_forward`의 두 bmm이 최종 tri slice에 직접 기록한다. activation `cat`이 제거됐으며, 출력 F4~F7은 `_back_kernel` 하나다.
- **H100 CuTe 양방향 추론:** activation `cat`과 분리된 출력 경로가 남는다. H100 단방향 추론 출력은 Triton `_back_kernel`을 사용한다.

H100 학습 출력 특화는 BF16, H100 80GB HBM3, B1, D=h=128, L384/L768 및 지원 layout 조건의 소스 분기다. 모든 shape의 H100 경로를 뜻하지 않는다. 상세 추론 조건은 [설명](docs/trimul-fusion/inference.md)에 있다.

## 설정·캐시·실측

- 타일링과 융합 경계는 구분한다. 그림의 M/N/K, GROUP_M, warps, stages는 설정 탐색 축이며 모든 shape의 캐시 완성을 뜻하지 않는다.
- 9/17 F567 수정으로 이전 F567 source identity가 무효화됐다. 단방향 L128/384/768의 새 세 커널 캐시는 만들었지만 양방향 KP256까지 모두 재튜닝됐다는 뜻은 아니다.
- [현재 양방향 vs 단방향 2개 실측](runs/trimul_bi_vs_uni_20260917/REPORT.md): 전체 모듈 기준. 일부 shape는 기본 24개 후보 runtime autotune을 사용했다.
- [단방향 학습 최적화](runs/trimul_unidirectional_20260917/REPORT.md) · [양방향 추론 cat 제거](runs/trimul_inference_nocat_20260917/REPORT.md).

## 실험 / 이전 그림

- [CuTe F567 실험 forward](docs/trimul-fusion/forward-new.svg): 현재 설치본 forward와 분리. [실험 결과](docs/trimul-fusion/f567-cute.md).
- [9/16 이전 3열 SVG 보관](docs/trimul-fusion/archive-20260916/README.md): 당시 비교 경로의 기록이며 현재 기본 배선을 나타내지 않는다.
- [v6 Git 계보](docs/trimul-fusion/history/README.md) · [Triton backward 융합](docs/trimul-fusion/triton-backward.md) · [F567 설정 점검](docs/trimul-fusion/f567-config.md).

실선은 커널 융합 경계, 점선 회색은 compiler의 복수 연산, 노랑은 HBM buffer다. cuBLAS 호출의 내부 launch 수와 weight 준비·cast 등 보조 연산 전체를 세는 그림은 아니다. 이번 정리는 소스 배선 확인이며 GPU profiler 재실행은 아니다. [학습 소스 SHA-256](docs/trimul-fusion/training-sources-20260917.json).

재생성: `python scripts/render_trimul_fusion.py` / `python scripts/render_trimul_inference.py`.
