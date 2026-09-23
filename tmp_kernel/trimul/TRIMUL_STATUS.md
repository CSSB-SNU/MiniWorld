# TriMul · 현재 Forward / Backward 배선

2026-09-21 소스와 실측 기록을 대조한 **Anthropic 파생 CUDA 학습 개발 경로**다.
최근 전체 연결 실험 `FP32 x̂/rstd 저장 + raw tri 없는 B1 + split_xn_pc1 B7`을 표시한다. 엔진의 production 기본값을 뜻하지 않는다.

**학습 커널 개발 설정:** K3에서 입력 affine BF16 `x_n`, 출력 pre-affine 정규화 값 FP32 `x̂`, 출력 FP32 `rstd`를 저장한다. B1은 `tri`나 평균을 받지 않는다. 평균·분산 및 `(tri−mean)*rstd` 재계산 없이 `x̂*gamma+beta`만 적용한다.
[현재 구현·측정·검증](../../runs/trimul_xhat_native2_20260921/README.md). 이전 FP32 정규화 저장 경로 대비 B1 지연 −1.4/−2.8%, 전체 −0.55/−0.43%. 아직 raw tri 재계산 기준보다 전체 약6.4/6.8% 느리다. 현재 방향은 정규화 값 저장으로 유지한다.
이전 통계만 저장하는 경로는 비교 이력으로 남긴다. 현재 개발 어댑터는 [training.py](runs/trimul_xhat_native2_20260921/training.py)다.

- [확대 뷰어](TRIMUL_STATUS.html) · [온라인 현황판](https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#current-wiring)
- Forward: [L384](TRIMUL_FORWARD.svg) · [L768](TRIMUL_FORWARD_L768.svg)
- Backward: [L384](TRIMUL_BACKWARD.svg) · [L768](TRIMUL_BACKWARD_L768.svg)
- [Anthropic 도입 전 그림 보관](../../docs/trimul-fusion/archive-pre-anthropic-20260921/README.md)

## 그림 읽는 법: HBM 왕복

각 행은 **HBM READ → 커널 내부 계산 → HBM WRITE**다. B1–B4의 Phase A/B는 두 행으로 나눠 표시했지만 **동일 CUDA 호출 1회**의 순차 단계다.
가운데 계산 박스를 좌우로 나누어 **왼쪽에는 설명·수식, 오른쪽에는 PyTorch 대응 코드**를 표시했다. LN backward helper도 그림 하단에 수식과 코드로 전개했다.
코드는 FP32 수학적 대응이며 CUDA의 BF16 중간 반올림·타일/CTA 실행 순서는 생략했다.
입출력마다 이름, shape, dtype, 버퍼 크기와 생산자/소비자를 표시했다. 자주색은 같은 호출 안의 global scratch 쓰기·재읽기다.

- B1 scratch: dW 25.95 MB + LN 270.34 kB. 합계 26.22 MB를 쓰고 다시 읽는다.
- B7 dW scratch: 16.78 MB 쓰기 + 16.78 MB 읽기. B7 dX LN scratch는 135.17 kB씩이다.
- B1 Phase A는 `dy/x_n/x̂/rstd`를 각 한 번 읽고, Phase B는 `x_n/dGate`를 한 번 더 읽는다. B7 dW는 16개 채널 그룹이 `x_n`을 각각 읽으며, dX도 독립적으로 읽는다.
- F.pack은 입력 WT 4개(262.14 kB)도 만들지만 현재 `DX_RESIDENT=1` 후보는 W1을 사용해 이 네 WT를 소비하지 않는다. 출력 gate의 Wg_T는 소비한다.
- 현재 K3는 정적 ptxas spill 0 B다. B1에는 16 B spill stores/loads가 남는다. 명시적 버퍼 크기와 실제 DRAM 트래픽은 구분한다.

**표시 크기는 버퍼 payload이며 실제 HBM 전송량이 아니다.** L2 hit와 반복 load, compiler spill은 NCU 실측이 필요하다. 이번 그림을 SoL 측정치로 사용하면 안 된다.

## Forward

| 순서 | 실제 호출 | 융합 연산 / HBM 저장 |
|---|---|---|
| 준비 | `F.pack`, torch.compile | 매 호출 가중치 packing. 측정 시간에 포함 |
| 1 | `infer_k1` | 입력 LN + 네 projection/gate + sigmoid/mask → `left`, `right` |
| 2–3 | cuBLAS 2회 | outgoing/incoming contraction → 최종 `tri` 버퍼에 직접 기록 |
| 4 | `save_k3` | 출력 LN/projection + 입력 LN 재계산/output gate + dropout/residual → `y`. 이 커널에서 입력 `x_n`과 출력 정규화 `x̂`/rstd 저장 |

입력 LN은 K1과 K3에서 각각 계산한다. **K1에서 저장한 값을 K3가 읽는 구조가 아니다.**
Backward까지 `left/right`, FP32 정규화 `x̂`/rstd, BF16 입력 affine LN 출력 `x_n`과 원래 입력·가중치·mask/dropout 참조를 유지한다. `tri`는 K3 이후 backward 저장 목록에서 제거한다.
`x_n` 저장량은 L384 37.75 MB / L768 150.99 MB이며, 출력 `x̂+rstd`는 151.58 MB /606.34 MB다(십진 MB).

## Backward

| 순서 | 실제 호출 | 융합 / 재계산 |
|---|---|---|
| 준비 | `Wproj.t().contiguous()` | 가중치 준비 복사 1회. 전체 시간에 포함 |
| B1–B4 | `b1_fused` 1회 | Phase A에서 저장 정규화 값으로 affine·projection·gate 각1회, dGate/dProj 공유 → dWproj·dTri/LN 미분. Grid barrier 후 Phase B에서 x_n/dGate 재읽기 → dWgate. 최종 reduction 포함 |
| B5–B6 | cuBLAS 4회 | contraction 미분. `dLeft/dRight` 최종 버퍼에 직접 기록 |
| B7–B12, dW | `front_b7b12`, role 1 | 입력 projection/gate 재계산 + sigmoid/mask 미분 + 네 입력 가중치 gradient + reduction |
| B7–B12, dX | `front_b7b12`, role 2 | 입력 projection/gate 재계산 + 입력 네 경로/출력 gate의 입력 gradient 합산 + 입력 LN 미분 + residual gradient |

B7의 두 호출은 같은 stream에서 dW → dX 순으로 실행된다. 서로 재계산한 P/G를 HBM으로 전달하지 않으며, 같은 입력을 각각 읽는다.
Gradient 결과와 FP32 partial reduction scratch는 HBM을 사용한다. dX의 입력 LN 통계는 raw `x`에서 재계산한다.
Backward 호출은 **CUDA 3회 + cuBLAS 4회 + 준비용 복사 1회**이며, cuBLAS 내부 launch 수와는 구분한다.

## 이전 별도 forward 저장 실험 (당시 기록)

평균·역표준편차, 출력 `xn_out`, 입력/출력 projection·gate 저장 옵션은
[runs/trimul_save_cost_20260921](runs/trimul_save_cost_20260921)에서 forward 비용만 측정했다.
**이 값들은 위 backward 후보에 아직 연결하지 않았다.** SVG에서 회색 점선으로 구분했다.

입력 `x_n` 기준에 입력·출력 통계를 추가하면 약 0–1%, 통계 저장 기준에 양쪽 projection/gate까지 추가하면 약 38–41% forward 시간이 늘었다.
이는 별도 A/B 측정이며 아래 전체 학습 결과와 임의로 합산하지 않는다.

## 전체 연결 후보의 동일 실행 실측

[최신 구현·NCU·검증 보고서](../../runs/trimul_b1_shared_20260921/README.md). H100 node02, BF16 C128/H256 양방향, mask/dropout25%/residual. Live packing과 모든11개 gradient, CUDA graph600회 교대 측정.

| L | B1 이전 → 신규 | B1 속도 | 전체 fwd+bwd 이전 → 신규 | 전체 속도 |
|---|---:|---:|---:|---:|
| 384 | 316.384 → 262.336 μs | 1.206× | 1.311136 → 1.259424 ms | 1.041× |
| 768 | 1227.776 → 904.112 μs | 1.358× | 5.387984 → 5.017744 ms | 1.074× |

전체 시간은 독립된 전체 graph 직접 측정값이다. 기준은 원본 Anthropic 추론이 아니라 직전 saved-x_n 학습 경로다.
**1.7× / SoL90 목표 미달.** 로드 전용 producer warpgroup도 비교했지만 더 느려 제외했다.
**정확도 검증 미완료:** L768 변경 입력 dWL 독립 기준 상대L2 0.055569%가 한도0.05%를 넘는다. 기존과 신규가 같은 값으로 실패하며 이 gradient는 둘이 bit-exact다. B1 자체 출력은 기존 한도 내이며 memcheck/racecheck를 통과했다. Production 기본값으로 승격하지 않았다.

## 근거와 재생성

- [전체 연결과 호출 순서](runs/trimul_b1_shared_20260921/training.py)
- [Forward K1/K3 연결](runs/trimul_ln_only_save_20260921/ln_save_core.py)
- [B1–B4 CUDA](runs/trimul_b1_shared_20260921/b1_fused.cu) · [B7 역할별 CUDA](runs/trimul_split_bwd_20260921/b7_roles.cu)
- [선택 / 검증 상태](../../runs/trimul_b1_shared_20260921/README.md)
- [소스·결과·SVG SHA-256](runs/trimul_diagrams_20260921/manifest.json)

```sh
python3 scripts/render_trimul_current.py
python3 scripts/render_kernel_viewers.py --trimul-only
```

기존 `render_trimul_fusion.py`는 Anthropic 도입 전 그림 생성기다. 현재 root SVG의 재생성에는 위 명령을 사용한다.
이번 B1–B4 변경은 node02에서 전체 연결 성능·변경 입력 graph replay·NCU·메모리 검사를 수행한 개발 후보다.

## 출력 LN 저장 실험 (2026-09-21)

입력 x_n 저장은 유지한다. 출력 LN을 추가 저장하고 B1에서 재사용해도 전체 학습은 L384 +1.4~1.8%, L768 +0.1~0.2%로 이득이 없었다. [실험 결과](../../runs/trimul_output_ln_save_20260921/README.md). 당시 SVG 정책은 변경하지 않았다. 현재 선택은 상단의 native FP32 x̂/rstd 경로다.

## tri 대체 저장 검증 (2026-09-21)

이전 출력 LN 추가 저장 실험은 tri도 읽는 경로였다. 후속 실험에서 tri를 B1 인자 및 backward 저장 목록에서 제거하고 NaN poison 검사를 통과했다. 정규화 값 FP32+rstd는 기존과 bit-exact, BF16은 약0.35% gradient 오차로 실패. FP32 첫 구현의 전체 시간은 약8.6% 증가했으나 재튜닝 전이며 정책 자체의 최적성 결론은 아니다. [상세 결과](../../runs/trimul_replace_tri_20260921/README.md). 당시 기본값/SVG는 기존 경로였다. 현재 선택은 상단의 native FP32 x̂/rstd 경로다.

## 출력 LN 통계 TMA 최종 선택

[개발 어댑터](runs/trimul_ln_policy_v4_20260921/training.py)와 [측정·검증 보고서](../../runs/trimul_ln_policy_v4_20260921/README.md). 원본 tri 유지 + 통계만 저장한다. 최종 두 길이 출력/11개 gradient bit-exact, memcheck0 errors, L384 B1/K3 racecheck0 hazards. NCU DRAM·레지스터·tensor 지표도 보고서에 기록했다.
