# Diffusion 학습 안정화 계획 — EDM2 magnitude preservation 적용

> **한 줄 요약**: train loss는 정상인데 학습 후반(epoch ~526)에 샘플링 품질(lDDT 0.71→0.33)이 무너지는 문제의 원인은 **weight magnitude(‖w‖)의 무한 증가**로 확인됨. 이를 막는 EDM2 계열 기법들을 **비용·효과 순으로** 정리한다. rotation modulation은 [나중에 추가](#부록-나중에-추가-rotation-modulation)로 분리.

---

## 0. 배경 — 무엇이/왜 문제인가 (이 문서를 처음 보는 사람용)

### 0.1 증상
- **train loss는 정상**: epoch 527에서 diffusion_loss ≈ 0.027, 안정적, 스파이크 없음.
- **그런데 샘플링이 붕괴**: 같은 타깃에서 lDDT가 epoch 467 **0.71** → epoch 526 **0.33** 으로 반토막.
- activation 자체엔 이상 없어 보임 → "loss·activation 멀쩡한데 샘플만 죽는" 전형.

### 0.2 진단 (체크포인트 오프라인 분석으로 확인)
학습 시간축으로 layer별 `‖w‖`과 effective LR(`‖Δw‖/(‖w‖·Δstep)`)을 측정한 결과:

| run | max ‖w‖ (epoch 425→끝) | eff_lr spread | 결과 |
|---|---|---|---|
| no-single (qknorm 없음) | 55 → 103 | 30 → **9093 (폭발)** | 발산 (loss에도 보임) |
| **현재 biasnorm run** | 55 → **124 (무한 증가)** | 18~50 (통제됨) | **loss 정상, 샘플 붕괴** |

- **‖w‖이 학습 내내 단조 증가, 멈춤 없음** (EDM2가 말한 *"grow without bound"*).
- 증가 진원지: `diffusion_module.*.attention_pair_bias.to_bias`, `*.ada_ln_in.to_bias` (AdaLN conditioning / pair-bias projection) — 25~28× (no-single에선 126×).
- qknorm/biasnorm(출력 RMSNorm)은 **eff_lr 폭발(loss에 보이는 붕괴)은 막았지만 ‖w‖ 증가는 못 막음**. 오히려 출력을 정규화하면 weight가 scale-invariant가 되어 ‖w‖이 더 자유롭게 자람.

### 0.3 근본 원인 (EDM2 §2.2–2.3)
- 출력이 정규화된 weight는 **scale-invariant** → loss gradient가 `wᵢ`에 **수직** → 매 step `‖w+Δ‖² = ‖w‖²+‖Δ‖² > ‖w‖²` → **‖w‖ 단조 증가, 복원력 0**.
- ‖w‖이 커질수록 effective LR(`‖Δw‖/‖w‖`)이 **layer마다 제각각 줄어듦** → 네트워크가 자기 교정 능력을 잃음 → 회복 불가.
- loss(σ별 denoising MSE 평균)는 scale에 둔감해 멀쩡해 보이지만, 샘플링(ODE 궤적 전체에 score 오차 누적)은 민감 → **loss는 정상, 샘플만 붕괴**.

> 출처: Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion Models* (EDM2), arXiv:2312.02696, §2.2–2.3, Fig. 3.

---

## 1. 핵심 원리 — "모든 연산이 활성값 분산을 1로 유지"

EDM2의 magnitude preservation은 **입력이 단위분산(RMS≈1)이면 출력도 단위분산이 되도록**, 활성값을 보지 않고 각 연산에 **고정 보정 스케일**을 심는 것. 분산을 바꾸는 연산마다 그 역수를 곱해둔다. 아래 모든 기법이 이 원리의 사례다.

**큰 좌표(xyz) 스케일은 문제 안 됨**: MP는 네트워크 *내부* 활성값을 단위분산으로 유지하고, 큰 출력 스케일은 EDM preconditioning(`c_in`/`c_out`/`σ_data`)이 *밖에서* 처리한다. → **`σ_data`를 좌표 std에 맞추는 것만 확인**하면 MP를 그대로 쓸 수 있다 (EDM2는 EDM preconditioning 위에 세워짐).

---

## 2. 추가하면 좋을 것들 — 우선순위 순

> 표기: 🟢 1순위(근본 해결, EDM2 정석) · 🔵 보조(싸고 일반적) · ⚪ 대안/나중에

### 🟢 P0. 모니터링 + σ_data 확인 (선행 작업, 거의 0 비용)
- **WDYN_MONITOR** (이미 추가됨): 학습 중 layer별 `‖w‖`·`eff_lr_spread`를 step 단위로 로깅. 환경변수 `WDYN_MONITOR=1 WDYN_EVERY=50`.
  - 무엇을 볼까: `wdyn/wnorm_max`가 단조 증가하는지, `wdyn/eff_lr_spread`가 튀는지.
- **σ_data 점검**: EDM diffuser config의 `sigma_data`가 실제 좌표 std와 맞는지 확인 (AF3 관례 ~16Å). 이게 틀리면 c_in/c_out 정규화가 깨져 MP 효과가 반감.

### 🟢 P1. Forced weight normalization (+ on-use WN) + inverse-sqrt LR — EDM2 정석 (1순위)
- **무엇**: 각 출력채널 weight `wᵢ`를 **매 step `‖wᵢ‖=√fan_in`으로 강제** + forward에서 정규화된 weight 사용.

  EDM2 원본 코드 (Algorithm 1, arXiv:2312.02696):
  ```python
  def normalize(x, eps=1e-4):                       # 출력채널(dim0)별 단위노름
      dim = list(range(1, x.ndim))
      n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
      alpha = np.sqrt(n.numel() / x.numel())        # ≈ 1/√fan_in
      return x / torch.add(eps, n, alpha=alpha)

  class MPLinear(nn.Linear):                        # 우리 team_gm Linear 상속해도 됨
      def forward(self, x):
          if self.training:
              with torch.no_grad():
                  self.weight.copy_(normalize(self.weight))   # (A) forced WN
          w = normalize(self.weight) / math.sqrt(self.in_features)  # (B) on-use WN + MP scale
          return F.linear(x, w, self.bias)
  ```
  - **(A) forced WN**: 저장된 weight를 매 step 구면으로 끌어다 놓음 → ‖w‖ 고정.
  - **(B) on-use WN**: gradient를 접평면(⊥ wᵢ)에 투영 → Adam 분산추정이 실제 step 기준으로 계산됨. **(B)를 같이 안 쓰면 step이 의도보다 작아져 학습이 느려진다** (EDM2 Fig. 23). 둘은 세트.
- **inverse-sqrt LR** (forced-WN을 켜면 ‖w‖ 증가가 주던 암묵적 LR 감소가 사라지므로 명시적으로 넣어야 함):
  ```
  α(t) = α_ref / sqrt(max(t / t_ref, 1))     # EDM2 Eq. 67
  ```
  현재 step-decay 스케줄을 이걸로 교체.
- **⚠️ zero-init 비호환 함정**: 우리 `to_bias`/`ada_ln_in.to_bias`는 `init="zero"`인데, `normalize(0)=0`이라 forced-WN을 걸면 **영원히 0에 갇힘**. EDM2 해법은 *"weight를 0으로 시작"이 아니라 "weight는 정상 init(forced-WN) + 그 뒤에 zero-init learned scalar gain"*:
  - 주력 matmul(`to_query/key/value`, `to_out`, transition Linear 등, **zero-init 아닌 것**) → MPLinear로 교체.
  - zero-init projection은 → unit-Gaussian init MPLinear + **뒤에 zero-init scalar gain `g`** (초기 conditioning off = identity 시작). 아래 P2-④ 참고.
- **내 모델 적용 범위**: 학습되는 **diffusion module의 conv/linear weight만** (trunk frozen이라 무관). **LayerNorm/RMSNorm affine, scalar gain에는 적용하지 않음**.
- **비용/주의**: DiT 논문 측정 **런타임 +8.5%** (대부분 forced-WN). **LR 재튜닝 필수** — forced-WN은 effective LR 정의를 바꾸고 *더 큰 stable LR을 허용*하므로, LR을 안 올리면 이득이 안 나거나 손해(아래 §3 주의).

### 🔵 P2. Magnitude-preserving fixed-function layers — 분산 보정 (싸고 일반적)
모두 "이 연산이 분산을 X배 하니 1/√X 곱한다"의 사례. 학습 파라미터 없음(④ 제외).

- **③ MP residual sum** (DiT 논문 Lemma 4, EDM2 mp_sum): 일반 residual `x + f(x)`는 분산이 2배(블록마다 √2배 증폭) → 가중합으로 보존:
  ```python
  def mp_sum(a, b, t):                        # t = residual branch 비중
      return torch.lerp(a, b, t) / math.sqrt((1 - t)**2 + t**2)
  # out = mp_sum(x, f(x), t),  t≈0.3 (블록), 0.5 (embedding)
  ```
  근거: `M[√α·x + √(1−α)·y] = σ` (uncorrelated, M[x]=M[y]=σ일 때). — DiT 논문 Lemma 4.
- **② MP SiLU** (EDM2 §2.5 / DiT 논문 Lemma 5): SiLU가 단위분산을 std 0.596로 줄임 → 되돌림:
  ```python
  def mp_silu(x):
      return F.silu(x) / 0.596
  ```
- **① Fourier ×√2** (EDM2 §2.5): noise embedding의 sin/cos는 std 1/√2 → √2 곱해 단위분산:
  ```python
  fourier = math.sqrt(2) * torch.cat([torch.sin(f), torch.cos(f)], dim=-1)
  ```
  *(DiT 논문엔 이 항목이 별도로 없음 — EDM2 전용. 우리 σ Fourier embedding에 적용 가능.)*
- **④ 두 learned scalar gain** (EDM2 §2.5): MP로 크기를 다 묶은 뒤, **크기 조절이 꼭 필요한 두 곳**에만 학습 스칼라 1개씩 (forced-WN 대상에서 제외):
  - **출력 끝단 gain (zero-init)**: 단위분산 내부 ↔ 실제 타깃 스케일을 잇는 다리. xyz처럼 큰 출력에 필수.
  - **블록별 conditioning gain (zero-init)**: 초기엔 conditioning off(identity 시작), 학습하며 세기 결정. **P1의 zero-init 함정을 푸는 장치.**

### 🔵 P3. Post-hoc EMA (언제든, 고가치)
- **무엇**: 학습 중 power-function EMA profile을 기록해두고, **학습 후에 EMA length를 골라** 샘플 품질을 스윕 (EDM2 §3).
- **왜**: 샘플 품질이 EMA decay에 매우 민감. 우리는 이미 `ModelEMA` 콜백이 있으니, post-hoc 스윕으로 최적 EMA를 사후 선택 가능.
- **비용/주의**: 추론/평가 단계 작업. 학습 비용 거의 없음.

### ⚪ (대안) Weight decay — 비-EDM2, forced-WN과 택일
- **무엇**: forced-WN(P1) 대신 AdamW `weight_decay`를 `0 → 0.01~0.05`.
- **왜 통하나**: qknorm/biasnorm으로 해당 weight가 **scale-invariant**라, weight decay가 함수를 안 망치고 **‖w‖만 끌어내림**. "정규화된 net에서 weight decay ≡ effective-LR 조절"은 알려진 결과(EDM2가 인용한 van Laarhoven 등이 그 등가성 근거).
- **⚠️ EDM2가 쓴 방법은 아님**: **EDM2 = 순수 Adam (β2=0.99) + forced-WN, weight decay 없음, AdamW 아님.** forced-WN이 ‖w‖을 이미 √fan_in으로 못 박으므로 **둘은 중복 → 택일.**
- **언제 쓰나**: forced-WN의 +8.5% 비용 / LR 재튜닝이 부담스러울 때, 같은 목표(‖w‖ 억제)를 비용 ≈0으로 얻는 **싼 대체재**로만. 가능하면 정규화 뒤 weight(`to_bias`, `ada_ln_in.to_bias`)에만 param group 분리 적용.

---

## 3. 솔직한 주의사항 (논문 수치 그대로)

- **forced-WN은 "항상 FID를 올리는" 기법이 아니다.** DiT 논문 Table 1에서 forced-WN 추가(Config C→D)는 FID를 **오히려 약간 낮췄다** (DiT-XS/2 86.49→89.38). 이유:
  1. 짧은·작은 런(400K, ImageNet-128)이라 ‖w‖ drift가 아직 파국을 안 만든 구간 → 예방 효과가 FID에 안 잡힘.
  2. forced-WN은 **더 큰 LR을 허용**하는데, LR을 안 올리면 이득이 없거나 손해.
  - 반면 **EDM2 본가(대형·장기·튜닝됨)에선 forced-WN이 FID 개선**(3.75→3.02). → 스케일/길이/LR튜닝에 따라 결과가 갈린다.
- **우리 관건은 "peak FID"가 아니라 "후반 ‖w‖ 붕괴 예방"** 이다. forced-WN/weight decay는 정의상 ‖w‖을 묶으므로 이 문제를 직접 해결한다. 짧은 런 FID는 이걸 측정하지 못한다.
- **부분 적용 = 부분 효과**: qknorm만 넣었을 때 발산은 막았지만 ‖w‖은 못 잡은 게 그 예.
- **EDM2는 weight decay/AdamW를 쓰지 않았다.** optimizer는 **순수 Adam (β2=0.99)** + forced weight normalization. weight decay는 EDM2가 *인용만* 한 대안(§2 ⚪)이며, forced-WN과 함께 쓰면 중복이다.

---

## 4. 권장 실행 순서

1. **P0** — WDYN_MONITOR로 baseline 곡선 확보 + σ_data 점검.
2. **P1 (forced-WN + on-use WN + inverse-sqrt LR + zero-init gain)** — EDM2 정석. ‖w‖을 √fan_in으로 못 박아 근본 해결. **LR 재튜닝 동반**, +8.5% 비용 감수.
3. **P2 (MP residual/SiLU/Fourier)** — 싸고 일반적인 분산 보정. P1과 독립적으로 더해도 됨.
4. **P3 (post-hoc EMA)** — 평가 단계에서 EMA length 스윕.
5. **⚪ rotation modulation** — 아래 부록, 나중에.

> **(대안)** forced-WN의 비용/튜닝이 부담되면 **weight decay**(§2 ⚪)로 ‖w‖만 싸게 억제 — 단 EDM2 비표준이고 forced-WN과 **택일**.

각 단계 후 **반드시 동일 타깃으로 추론(lDDT) + `wdyn/eff_lr_spread`/`wnorm_max`** 를 재확인해 인과를 검증.

---

## 5. 구현 계획 (전체 적용 + 테스트) — rotation 포함

### 5.0 전제 / 설계 원칙
- **브랜치(worktree)에서 작업.** 침습적이고 phase별 독립 커밋 → 회귀 시 롤백.
- **diffusion module에만 적용.** trunk는 frozen(warm-start). **frozen trunk 레이어의 forward를 바꾸면 안 됨**(normalize-on-use를 trunk에 걸면 학습된 trunk가 깨짐). → MP/rotation은 **diffusion 서브config 플래그**로만 켜고, trunk는 기존 레이어 유지.
- **warm-start 가능 여부**:
  - P1(forced-WN)·P2(MP fixed)·P3(EMA): weight **shape 동일** → 기존 ckpt warm-start 가능 (단 forced-WN은 effective-LR 정의가 바뀌어 사실상 LR 재튜닝).
  - **Rotation(Phase 4): param shape 변경**(`to_bias [d_cond,d_hidden]` → `to_angle [d_cond,d_hidden//2]`) → 해당 AdaLN 모듈은 **fresh init 필요**(param_policy로 reinit, 나머지 warm-start).
- 단계마다 **WDYN_MONITOR(‖w‖, eff_lr_spread) + 고정 타깃 lDDT**로 검증.

### 5.1 먼저 알고 갈 구현 난점 (코드 조사로 확인됨)
1. **Triton 커널이 raw `.weight`를 직접 소비** ([transition.py](../libs/team-gm/src/team_gm/modules/layers/transition.py), layernorm). `forward` 오버라이드만으론 Triton 경로에 MP가 안 먹힘. → **방침: MP 레이어는 PyTorch forward로 둔다**(사용자 승인). 즉 MP가 켜진 레이어는 `implementation=PYTORCH` 강제 + `forward`에서 EDM2 Algorithm 1(in-training 시 raw weight를 `normalize`로 in-place 투영 = forced-WN, on-use 정규화)을 직접 수행. (성능 최적화가 필요하면 추후 Triton 커널에 normalize를 흡수.)
2. **Transition은 SiLU가 아니라 SwiGLU(`swish_gate`)** ([transition.py:50](../libs/team-gm/src/team_gm/modules/layers/transition.py#L50)). EDM2의 `÷0.596`(SiLU)이 그대로 안 맞음. → SwiGLU의 분산보존 상수를 **측정/유도**하거나, 1차엔 transition의 LayerNorm을 **유지**(norm 제거는 후순위).
3. **norm 제거(EDM2 Config E/F)는 Triton norm 커널과 얽힘** → **1차 적용에선 기존 RMSNorm/LayerNorm 유지**. MP weight-norm + forced-WN + MP residual + 출력 gain만. norm 제거는 선택/후순위.
4. **frozen trunk 보호** → 전역 클래스 교체 금지, diffusion 서브config 플래그로만 instantiate.

### 5.2 Phase별 계획

**Phase 0 — 준비**
- worktree 생성, baseline 고정: 현재 `last.pt`로 고정 타깃 N개 lDDT(기준선) + WDYN_MONITOR baseline.
- σ_data 확인: 학습 config의 `diffuser.sigma_data == 16.0`(좌표 std와 일치) 점검. *(이미 16.0 확인됨.)*

**Phase 1 — Forced weight normalization (+on-use WN) + inverse-sqrt LR**  ← §2 P1
- [primitives.py](../libs/team-gm/src/team_gm/modules/primitives.py): `normalize()` + weight parametrization 헬퍼(또는 `MPLinear`).
- diffusion module 레이어 생성부([augmented_attention.py](../libs/team-gm/src/team_gm/modules/layers/augmented_attention.py), [conditioned_transition.py](../libs/team-gm/src/team_gm/modules/layers/conditioned_transition.py), atom/token transformer block): MP 플래그 받아 **비-zero-init Linear**(`to_query/key/value/to_out`, `expand_a/b`)에 parametrization. **zero-init Linear(`squeeze`, `to_bias`, `ada_ln_in.to_bias`)는 제외** → P2-④ gain으로 identity 보장.
- `team_gm/core/callbacks/`에 **`ForcedWeightNorm(Callback)`** 추가: `on_train_step_end`에서 MP 파라미터의 raw weight를 `normalize`로 in-place 투영(trainable=diffusion module만 자동 대상).
- [utils.py](../src/miniworld/utils/utils.py): `get_inverse_sqrt_scheduler_with_warmup` 추가 → run script에서 `get_step_decay_scheduler_with_warmup` 교체. **α_ref / t_ref 소규모 스윕**(forced-WN은 더 큰 LR 허용).
- 검증: `wnorm_max` 평탄, `eff_lr_spread`<~50 유지, lDDT ≥ 기준선.

**Phase 2 — MP fixed-function + 출력/conditioning gain**  ← §2 P2
- `ops.py`: `mp_sum(a,b,t)=lerp(a,b,t)/√((1−t)²+t²)`. transformer block의 residual `x = x + f(x)` → `mp_sum(x, f(x), 0.3)`.
- SwiGLU MP: `swish_gate` 출력 스케일 상수 측정(z~N(0,1)에서 `E[swish_gate²]`) 적용, **또는 transition LayerNorm 유지로 우회**.
- Fourier ×√2: [embeddings.fourier_embedding](../src/miniworld/modules/embeddings.py) 정규화 상태 확인 후 √2 보정.
- **출력 gain**: diffusion model 최종 출력 projection 뒤 **zero-init scalar gain**. **conditioning gain**: `conditioned_transition.to_scale`(sigmoid gate, bias −2.0)가 이미 그 역할 → 유지/조정.
- 검증: 활성값 RMS가 깊이에 따라 ~일정, lDDT 유지/개선.

**Phase 3 — Post-hoc EMA**  ← §2 P3
- EMA profile 기록 + 추론 시 EMA length 스윕 (`ModelEMA` 확장 또는 별도 분석 스크립트).

**Phase 4 — Rotation modulation** (param shape 변경 → AdaLN fresh init)  ← 부록
- [adaln.py](../libs/team-gm/src/team_gm/modules/layers/adaln.py) `AdaptiveLayerNorm`: `use_rotation` 플래그. `to_bias`(zero) 제거 → `to_angle = Linear(d_cond, d_hidden//2)`. forward: `h = sigmoid(scale)*ln_in(x); return apply_pairwise_rotation(h, to_angle(ln_cond(cond)))`.
- `ops.py`: `apply_pairwise_rotation` (RoPE식 half-split로 효율화). (forward는 순수 PyTorch라 Triton 무관.)
- `param_policy`: AdaLN 모듈만 reinit, 나머지 warm-start.
- 검증: `ada_ln_in.to_*` ‖w‖ 폭증 소멸, lDDT 유지.

**Phase 5 — 통합 장기 테스트**
- 전체 켜고 **epoch 530+(위험구간) 통과**까지 학습. `eff_lr_spread`/`wnorm_max` 평탄 + lDDT 비붕괴 확인. **현재 biasnorm run의 epoch 526 붕괴와 직접 대조.**

### 5.3 테스트 매트릭스 (각 phase 공통)
| 지표 | 도구 | 합격 기준 |
|---|---|---|
| ‖w‖ 증가 | `wdyn/wnorm_max` | 평탄 (비단조증가) |
| effective LR 불균등 | `wdyn/eff_lr_spread` | <~50 유지, 후반 미폭발 |
| 샘플 품질 | 고정 타깃 lDDT/RMSD | 기준선 이상, 후반 비붕괴 |
| 학습 정상성 | train loss | 정상 하강 |
| 비용 | step time | +~8.5% 내외 허용 |

### 5.4 위험 & 롤백
- phase 독립 커밋, 회귀 시 직전 phase로.
- 최대 위험: ① SwiGLU MP 상수(측정 필요) ② rotation ckpt 비호환(fresh init 계획됨) ③ Triton 경로 MP 적용(parametrization으로 해결) ④ frozen trunk 오염(서브config 플래그로 차단).
- **norm 제거는 이번 범위에서 제외**(Triton norm 커널 리스크) — 안정화 후 별도 검토.

---

## 6. 구현 현황 (브랜치 `edm2-magnitude-preservation`)

모든 신규 동작은 **플래그 OFF가 기본** → 기존 학습/warm-start에 영향 없음. `tests/test_mp_edm2.py` 8개 통과(CPU). 전체 모델 import·빌드 정상 확인.

### 구현됨 ✅
| 항목 | 위치 | 비고 |
|---|---|---|
| `magnitude_normalize` + **`MPLinear`** (forced + on-use WN) | [primitives.py](../libs/team-gm/src/team_gm/modules/primitives.py) | EDM2 Algorithm 1. 행 노름을 √fan_in으로 고정. zero-init은 거부 |
| **`mp_sum`**, **`apply_pairwise_rotation`** | [ops.py](../libs/team-gm/src/team_gm/modules/layers/ops.py) | 분산보존 residual / 회전 |
| AdaLN **rotation modulation** (`use_rotation`) | [adaln.py](../libs/team-gm/src/team_gm/modules/layers/adaln.py) | shift→회전, to_angle zero-init=identity |
| 어텐션 q/k/v + **pair-bias projection** MP + rotation 배선 | [augmented_attention.py](../libs/team-gm/src/team_gm/modules/layers/augmented_attention.py) | q/k/v→MPLinear(Triton attn 커널과 호환). **`to_bias`(pair→logit bias, ‖w‖ 폭증 최상위)도 MPLinear**(zero-init→normal-init, norm_bias와 상보) |
| transition MP + rotation + PYTORCH 강제 | [conditioned_transition.py](../libs/team-gm/src/team_gm/modules/layers/conditioned_transition.py) | expand_a/b→MPLinear (Triton transition은 raw weight 소비→PyTorch 강제) |
| Config→Block 플래그 + mp_sum residual | [diffusion_transformer.py](../libs/team-gm/src/team_gm/modules/blocks/diffusion_transformer.py) | `magnitude_preserving/use_rotation/mp_residual/residual_t` |
| **inverse-sqrt LR** 스케줄러 | [utils.py](../src/miniworld/utils/utils.py) + run script | env `EDM2_INV_SQRT_LR=1`, `EDM2_LR_TREF` |

### 아직 안 됨 (2차 분산보정 — 영향 작음)
- **SwiGLU용 MP 상수**: transition은 SiLU가 아니라 SwiGLU → EDM2의 `÷0.596`이 안 맞음. 상수 측정/유도 필요. (현재 transition LayerNorm 유지로 우회 중.)
- **Fourier ×√2**: [embeddings.fourier_embedding](../src/miniworld/modules/embeddings.py) 정규화 확인 후 적용.
- **전용 출력 끝단 gain**: 현재 zero-init(`to_out`/`squeeze`)과 `to_scale` 게이트가 identity-start/conditioning-gain 역할을 대체 중.
- **Post-hoc EMA** (P3).

### 활성화 방법
model config의 diffusion 서브config(`atom_dit`, `token_dit`)에 추가:
```yaml
model:
  diffusion:
    atom_dit: { magnitude_preserving: true, use_rotation: true, mp_residual: true }
    token_dit: { magnitude_preserving: true, use_rotation: true, mp_residual: true }
```
학습 실행 시: `EDM2_INV_SQRT_LR=1 EDM2_LR_TREF=70000 WDYN_MONITOR=1 torchrun ...`

### ⚠️ warm-start 주의
- `magnitude_preserving`·`mp_residual`: weight shape 동일 → **warm-start 가능**(MPLinear가 첫 forward에서 기존 weight를 정규화).
- **`use_rotation`: param shape 변경**(`to_bias[d_cond,d_hidden]`→`to_angle[d_cond,d_hidden//2]`) → 해당 AdaLN은 **ckpt 로드 불가**. `--no-ckpt-strict` + param_policy reinit 또는 fresh 학습 필요.
- 권장 검증 순서: ① `magnitude_preserving+mp_residual`만 켜서 warm-start로 `wnorm_max` 평탄/lDDT 확인 → ② 안정되면 `use_rotation` 추가(해당 모듈 reinit).

---

## 부록: Rotation modulation (Phase 4 — 수식/코드 참조)

> 출처: *Exploring Magnitude Preservation and Rotation Modulation in Diffusion Transformers*, arXiv:2505.19122 (2025). **지금은 안 함**, P1~P3 안정화 후 옵션으로 검토.

- **동기**: AdaLN의 **additive shift `+b`** 가 magnitude를 깨는 주범 (우리 `ada_ln_in.to_bias` 폭증 지점과 정확히 일치). shift를 **norm-보존 회전**으로 대체.
- **기존 AdaLN (Eq. 4)**: `x ← g ⊙ layer(s ⊙ x + b)` — scale `s`, shift `b`, gate `g`.
- **Rotation modulation (Eq. 6)**: 토큰 `x∈ℝ^d`를 `d/2`개 2D 쌍으로 나눠, conditioning에서 각도 `θ∈ℝ^{d/2}`(= `Linear(d_cond, d/2)`)를 예측 후 각 쌍에 Givens 회전:
  ```
  [x_2i ; x_2i+1] ← [[cosθ_i, -sinθ_i], [sinθ_i, cosθ_i]] · [x_2i ; x_2i+1]
  ```
  회전이라 ‖x‖ 불변 (Lemma 2: `M[Rx]=M[x]`). RoPE와 동일 연산, 각도만 학습됨.
  ```python
  def apply_pairwise_rotation(x, theta):       # x:[...,d], theta:[...,d/2]
      c, s = torch.cos(theta), torch.sin(theta)
      xe, xo = x[..., 0::2], x[..., 1::2]
      out = torch.stack([c*xe - s*xo, s*xe + c*xo], dim=-1).flatten(-2)
      return out
  # AdaLN 교체: bias 제거, theta = to_angle(cond); x = apply_pairwise_rotation(sigmoid(scale)*ln(x), theta)
  ```
- **결과/비용**: scale+rotation이 AdaLN(scale+shift)과 **경쟁력 + 파라미터 ~5.4%↓**(31.0M vs 32.8M). FID는 **약간 높음(소폭 손해)** — 이득은 magnitude 보존 + 파라미터 절감. 연산은 RoPE급(무시 가능, shift보다 파라미터 적음). rotation 단독은 scale 단독보다 못함 → **scale 유지 + shift만 rotation**.
- **검증 한계**: DiT-XS/S, ImageNet-128, 400K step의 소형 실험만. 대규모 미검증.

---

## 참고문헌
- **EDM2**: T. Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion Models*, CVPR 2024, [arXiv:2312.02696](https://arxiv.org/abs/2312.02696). — §2.2–2.3(magnitude drift, effective LR), §2.5(MP fixed-function), §3(post-hoc EMA), Algorithm 1(forced WN), Eq. 67(inverse-sqrt LR), Fig. 3/23.
- **EDM2 + DiT**: *Exploring Magnitude Preservation and Rotation Modulation in Diffusion Transformers*, [arXiv:2505.19122](https://arxiv.org/abs/2505.19122) (2025). — Table 1(Config A–E FID), Lemma 4(residual), Lemma 5(SiLU 0.596), Eq. 4/6(modulation), +8.5% 비용, modulation ablation(Table 2).
- **본 조사 산출물**: `WeightDynamicsMonitor` 콜백([run_miniworld_no_single_edm_train.py](../scripts/run_miniworld_no_single_edm_train.py)), 오프라인 ‖w‖/eff_lr 분석 스크립트.
