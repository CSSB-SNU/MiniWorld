# EDM 학습 발산 원인 분석: 트렁크 `token_single` rank 붕괴

- **대상 run**: `exp-msa3_24_3-edm-revisit420` (3/24/3 AF3-like EDM, frozen distogram trunk `revisit_epoch420` 위에서 학습)
- **증상**: 학습이 잘 되다가 특정 시점부터 loss explosion
- **결론**: 새로 학습되는 **pairformer single-track**이 발산. 트렁크 `token_single`이 학습 내내 저랭크(eff_rank ≈ 2)였고, **epoch ~472에서 rank-1로 붕괴**하며 magnitude가 동시에 폭발 → diffusion conditioning을 무력화 → loss explosion.
- 분석 일자: 2026-06-13

---

## 1. 방법

`epoch=04xx.pt` 체크포인트를 로드하고, 실제 학습과 동일하게 트렁크(=`revisit_epoch420`에 존재하는 2074개 param)를 freeze한 뒤, **고정 배치 1개**(seed=0)에 대해:

- fwd + bwd 1회 → 전 모듈 forward activation / backward grad / param grad 통계 (`/tmp/debug_edm_activations.py`)
- `condition_forward`만 호출해 epoch별 `token_single` **effective rank** 궤적 (`/tmp/rank_trajectory.py`)

effective rank = participation ratio of singular values, `(Σσ²)² / Σσ⁴`. 입력 임베딩(`token_single_input`)을 대조군으로 사용 (데이터/입력 효과 분리).

> 프로브 스크립트는 repo 미포함(`/tmp`). 트렁크 forward는 bf16, precision 32, compile off.

---

## 2. 핵심 발견

### 2.1 single-track은 frozen이 아니라 trainable이며 학습 중 발산

distogram 트렁크는 **pair-only**였다. single-track은 EDM용으로 새로 init + 학습된다.

| param | in frozen trunk? | L2 norm ep450 → ep500 |
|---|---|---|
| `pair_to_single` (528 keys) | **No (trainable)** | 312 → 388 (×1.24) |
| `transition_single` (240) | **No (trainable)** | 306 → 502 (×1.64) |
| `add_single_recycle` (4) | **No (trainable)** | 20.2 → 27.3 (×1.36) |
| `to_token_init` (1) | **No (trainable)** | 27.0 → 32.3 (×1.19) |
| `add_pair_recycle`, `tri_multi_*`, `distogram_head` | Yes (frozen) | ep450/500 비트 단위 동일 (diff 0.000) |

가중치는 ×1.2~1.6 정도만 증가하지만, single residual stream에 **블록 간 정규화가 전혀 없고**(`pairformer.py:100-101`) `add_single_recycle`이 recycle마다 피드백되는 **재귀 증폭기**라, 활성값은 훨씬 크게 증폭된다.

### 2.2 `token_single` effective rank 궤적 (384차원, 유효잔기 381, 동일 배치)

```
epoch | INPUT(raw) eff_rank rms | SINGLE(trunk) eff_rank top1   cos      rms
  425 |   1.34          0.23   |   2.14      0.646  0.590       8.2
  445 |   1.34          0.23   |   2.59      0.593  0.541      69.3
  450 |   1.34          0.23   |   2.84      0.557  0.507      80.1
  465 |   1.34          0.23   |   2.04      0.684  0.662      86.4
  470 |   1.34          0.23   |   1.54      0.802  0.786     114.6
  475 |   1.34          0.23   |   1.00      1.000  1.0000   4172.6   ← 붕괴
  480 |   1.34          0.23   |   1.00      1.000  1.0000  18145.4
  500 |   1.34          0.23   |   1.00      1.000  1.0000  18087.5
```

관찰:
1. **single은 학습 내내 저랭크** — 시작 직후(425)부터 eff_rank ≈ 2/384. 384차원 표현이 사실상 2차원 부분공간에만 존재 → **풍부한 per-residue 정보를 학습한 적이 없음.**
2. **epoch 470→475에서 날카로운 상전이**: eff_rank **1.54 → 1.00**, RMS **115 → 4,173 (×36)**. rank 붕괴와 magnitude 폭발이 **같은 사건**. 한 방향으로 붕괴 → 무정규화 재귀 증폭기가 그 단일 방향을 RMS ~2만까지 증폭. = loss explosion 시점.
3. **대조군 입력 임베딩은 전 epoch 고정**(eff_rank 1.34, rms 0.23) → 데이터/입력이 아니라 **single-track 학습 동역학**이 원인.

### 2.3 붕괴 후(epoch 500) 단면

- `token_single` (384잔기×384채널): eff_rank **1.00**, top1 energy **1.000**, **모든 잔기 pairwise cos = 1.0000** (raw·per-token LN 모두). → 모든 잔기가 동일 벡터. per-residue 정보 0.
- 블록별 single RMS: block0 5→1080, 이후 cos(증분, stream)이 0.06→0.97로 상승하며 block47에서 RMS 18,100, absmax ~72,000. bf16(크기 7e4에서 간격 ~256)이라 잔기별 미세차이는 양자화 바닥으로 소실.
- `token_pair`는 정상(absmax ~2e3, std ~32; distogram_head로 supervise됨).
- param grad 1순위: `diffusion_transformer.blocks.21.attention_pair_bias.ada_ln_in.ln_cond.weight` absmax 28 (ep450엔 0.03) — 거대 single을 정규화하는 conditioning LN이 grad hotspot.

---

## 3. 메커니즘 요약

1. distogram 사전학습은 `token_pair`만 supervise → single track은 무감독.
2. EDM 학습에서 새로 얹은 single track(pair_to_single + transition_single + add_single_recycle)은 **블록 간 정규화 없음 + recycle 피드백** = 재귀 증폭기.
3. single은 처음부터 저랭크(≈2)로 의미 있는 per-residue 표현을 못 만든 채, 가중치가 서서히 인플레.
4. epoch ~472에서 rank-1로 bifurcation → 단일 방향이 폭발적으로 증폭(RMS 115→4000+).
5. diffusion conditioning이 이 발산·rank1 single을 떠안아 정규화 → conditioning LN grad 폭발 → 양의 피드백 → loss explosion.

---

## 4. 수정 후보 (미적용)

- **single track 정규화 추가** (근본책): 블록 간 또는 `add_single_recycle` 피드백 직전 RMSNorm/LayerNorm으로 stream 크기·방향 폭주 차단.
- **single-track param에 weight decay** (또는 AdamW): 가중치 인플레 억제.
- **재시작은 epoch ≤465** (붕괴 전)에서. 단, 미수정 시 rank가 이미 ≈2라 재붕괴 가능성 높음.
- 보조: LR 하향, single-track grad clip, attention `use_qk_norm=True`.
- 근본: 트렁크 재학습 시 single에 정규화/약한 감독 추가.
