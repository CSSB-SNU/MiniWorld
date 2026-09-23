# Claude 인계: MiniWorld 양방향 TriMul 학습 B1–B4 CUDA 최적화

작성 기준: 2026-09-20의 최신 실측 및 로컬 코드. 아래 내용을 이어받아 작업해줘.

## 1. 내가 원하는 것

MiniWorld 양방향 TriMul 학습의 **B1–B4 융합 CUDA 커널을 H100에서 더 최적화**하고 싶다.
이전 목표인 Triton/cuBLAS 대비 1.7배는 달성했다. 이제는 실행 가능한 최적 구현의 상한(SoL)에 얼마나 가까운지 분석하고, 근거를 가지고 **상한의 90% 수준에 근접했다면 마무리**해도 된다.
현재는 아직 그 수준에 도달했다고 판단하지 않았다. 특히 L384에 개선 여지가 더 남아 있다.

Anthropic의 inference 커널 성과가 우리의 이전 개발보다 우수했다는 점을 인정한다.
우리는 그 구현과 아이디어를 적극 차용하여 **학습용 커널을 확장하는 입장**이다.
Anthropic native v5 CUDA primitives를 읽고 활용해줘. Triton으로 돌아가는 작업이 아니라 H100용 CUDA 구현을 개선하는 작업이다.

우선 B1–B4에 집중한다. Forward 저장 정책과 B1–B4 경계, BF16 반올림 시점, 출력 계약은 유지한다.
내부 타일링·데이터 이동·warp/CTA 역할·파이프라인은 근거에 따라 개선해도 된다.
학습 핵심 길이는 L384/768, C128/H256이다. L128 성능은 신경 쓰지 않는다.

## 2. 먼저 읽을 파일과 정확한 작업 위치

아래 경로는 모두 `/home/psk6950/MiniWorld` 기준이다.

```text
P = runs/anthropic_b1b4_pipeline_20260919
E = runs/trimul_sm90_parity_20260917/engine
A = runs/anthropic_adoption_20260919
S = runs/anthropic_ln_equal_saves_20260919
```

읽을 순서:

1. `E/.agents/AGENTS.md`: 로그인 노드에서 컴파일·벤치·프로파일 금지.
2. `P/SOL_AUDIT_20260920.md`: 현재 선택안의 9항목 상세 보고서.
3. `P/dual_ln_prefetch.cu`, `P/dual_ln_prefetch.py`: **현재 선택안**.
4. `P/dual_experiment.py`: 빌드/zero-spill 검사/launch/Plan 구현.
5. `P/core.py`, `P/dual.py`: 기준 수식, 입력, descriptor, workspace.
6. `P/integrate.py`: `backward_cuda`로 전체 backward에 연결하는 실험용 harness.
7. `P/measure_sol_final.py`, `P/check_experiment.py`, `P/sol_prefetch_audit.sh`.
8. `E/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/`의 CUDA primitives.

실제 이번 작업에 사용한 엔진은 위 **E checkout**이다. `/home/psk6950/miniworld-engine` 등의 다른 checkout과 혼동하지 말 것.
기존 변경과 다른 작업자의 파일을 먼저 확인하고 보존해줘.

원래 작업 지시서:
`/home/psk6950/.codex/attachments/5ddb56bc-0884-46b9-9a23-356a2db6b5c7/pasted-text.txt`

그 문서의 수식·정확성 요구는 유효하지만, **현재 커널 이름·CTA/WG 배정·병목·성능 숫자는 오래됐다.**
`dual_dspref`를 최신으로 착각하지 말고 현재 보고서와 소스를 우선해줘.

## 3. 현재 알고리즘과 구현

### 고정된 연산 계약

M=L², C=128, H=256. Forward saves: `xn, norm, tri, mean, rs, gate, proj`.
`ds[L,128]`는 이미 dropout scale이 반영된 BF16 마스크이며 row r는 `ds[r % L]`를 사용한다.

```text
y     = dy * ds[row % L]
dp    = bf16(y * gate)
dg    = bf16(((y * proj) * gate) * (1 - gate))
dWg   = bf16(sum_fp32(xn.T @ dg))
dWp   = bf16(sum_fp32(dp.T @ norm))
dnorm = bf16(dp @ Wp)
xhat  = (tri - mean) * rs
h     = dnorm * gamma
dtri  = bf16(rs * ((h - mean(h)) - xhat * mean(h*xhat)))
dgamma = sum_fp32(dnorm*xhat)
dbeta  = sum_fp32(dnorm)
```

출력은 `dg, dWg, dtri, dgamma, dbeta, dWp` 6개. B5 이후는 기존 Triton 경로이며 `dg/dtri`를 소비한다.
BF16인 `dp/dg/dnorm`의 반올림 위치와 곱셈 결합 순서를 함부로 바꾸지 말 것.

### 현재 CUDA 구조

- 한 cooperative launch, 132 CTAs, CTA당 256 threads = 2 warpgroups.
- **40 DW CTAs**: B1 + B2 + B3b, 즉 dp/dg와 두 weight gradient를 계산.
- **92 DX/LN CTAs**: B1 + B3a + B4, 즉 dp 재계산, dnorm GEMM, LN backward.
- count66에서는 같은 비율인 20/46. 64-row tile을 역할별로 순환 배정한다.
- 두 역할이 dy/gate를 각각 읽고 dp를 각각 계산한다. dp/dnorm은 global에 저장하지 않는다.
- 명시적 TMA/WGMMA, 두 입력 stage, register에 tri/dnorm 유지.
- DW shared: 96KiB × 2 stages. DX shared: 64KiB × 2 stages + resident Wp 64KiB.
- Dynamic shared 231,424B + static 1,024B, **252 registers, spill 0**, 1 CTA/SM.
- DW는 thread당 FP32 accumulator 192개를 유지한다.
- PART2: 같은 launch에서 partial publication → grid ticket barrier → final reduction → counters reset.
- PART1: 별도 `unified_reduce`를 사용하는 fallback. 계속 유지해야 한다.

최근 선택한 변경은 `dual_balanced` → `dual_ln_prefetch`다.
DX에서 saved mean/rstd를 **TMA wait 뒤, B1 앞**으로 옮겨 vector preload한다.
gamma는 launch마다 한 번 shared에 저장한다. B1의 기존 CTA barrier로 통계까지 publish하여 late load와 통계용 barrier 1회/tile을 줄였다.

새 shared 배치:

| byte 범위 | 용도 |
|---|---|
| 65536–73728 | 기존 warp별 dgamma/dbeta partials |
| 73728–74240 | slot0 mean/rstd |
| 74240–74752 | slot1 mean/rstd |
| 74752–75776 | launch마다 갱신하는 gamma |
| 163840–229376 | resident Wp |
| 229376–231424 | running dgamma/dbeta sums |

**전체 backward harness 연결과 검증은 끝났지만, production 기본 dispatch/autotune에는 승격하지 않았다.**

## 4. 최신 성능과 비교 대상

H100 80GB, BF16, dropout 0.25. 같은 process에서 순서를 교대하여 explicit CUDA Graph로 측정.
20 warmup + 200 samples를 3회 반복한 pooled 600개 median/p90이다. 아래 시간 단위는 µs.
핵심/전체 backward는 별도 측정 구간으로 나눴다. 클록을 고정한 실험은 아니다.

### B1–B4

| L | 정확한 `core.baseline()` Triton/cuBLAS | 이전 `dual_balanced` | 현재 `dual_ln_prefetch` | 기준 대비 |
|---|---:|---:|---:|---:|
|384|300.000 / 301.984|175.584 / 178.688|**172.640 / 175.136**|**1.738×**|
|768|1182.256 / 1186.240|622.656 / 636.672|**611.232 / 621.888**|**1.934×**|

이번 prefetch 변경만의 시간 감소는 **1.68% / 1.83%**다. 누적 배속과 구분할 것.

### 전체 backward, B5+ 그대로

| L | Triton/cuBLAS 기준 | 현재 CUDA B1–B4 + Triton B5+ | 배속 |
|---|---:|---:|---:|
|384|1080.928|957.584|1.129×|
|768|4332.368|3779.136|1.146×|

원자료: `P/sol-final-paired-results.json`.
전체 Triton 기준의 input-dual-backward에는 heuristic cache 경고가 있으므로, 모든 baseline을 완전 튜닝했다는 주장은 하지 않는다.
모델 전체 학습 step의 배속은 측정하지 않았다.

### cuEquivariance 비교 — 별도 실험

cuEq ops-cu12 0.10.0 / torch wrapper 0.9.1. ONDEMAND autotuning + static fullgraph `torch.compile`, 양쪽 explicit CUDA Graph.

| L | cuEq 구성 전체 backward | 현재 전체 backward | 배속 |
|---|---:|---:|---:|
|384|1498.624|963.312|**1.556×**|
|768|5679.440|3728.448|**1.523×**|

공개 cuEq TMU는 단방향이다. 우리 양방향은 output LN을 공유256채널에 적용하므로,
vendor LN/gated GEMM + 두 contraction + shared output LN + Torch gate/projection으로 **동일 수식의 양방향 구성**을 만들어 비교했다.
공개 단방향 API 자체와 직접 비교한 숫자가 아니다. 전체 배속에는 B5+의 차이도 포함된다.
Vendor 내부 저장/반올림 차이로 gradient 상대L2는 약0.1–0.6%; strict same-saves 비교와 다르다.

별도의 B1–B4 실험인 **2.462× /2.408×**는 공통 B1+cuBLAS에 **B4만 cuEq LN backward로 교체한 hybrid** 대비다.
이를 cuEq 전체 TriMul 대비 배속으로 보고하지 말 것.

근거: `P/compare_cueq.py`, `P/cueq-tuned-comparison.json`, `P/compare_cueq_core.py`, `P/cueq-core-comparison.json`.
Vendor 비교 재실행 시 BF16 0/1 mask를 사용한다. bool mask는 vendor tuner의 `randn_like(bool)` 오류가 있었다.
각 forward/capture stream 안에서 별도의 leaf tensors를 만들어야 AccumulateGrad의 legacy-stream 의존으로 graph capture가 깨지지 않는다.

## 5. SoL 판단과 남은 병목

**아직 SoL 90% 도달을 입증하지 못했다.** 서로 다른 분모를 섞어 90%를 만들지 말 것.

| 지표 | L384 | L768 |
|---|---:|---:|
|Warm NCU duration|172.672µs|612.768µs|
|Warm NCU HBM 처리율|74.14%|81.98%|
|Tensor elapsed 처리율|14.53%|16.35%|
|별도 3-read/1-write 대역폭 probe|2.992TB/s|3.062TB/s|
|그 probe 대비 유효 데이터 처리율|약80.6%|약89.0%|

유효 row traffic 모델: `(2056 read + 768 write) × L²` = 416.416 /1665.663MB.
3.35TB/s 기준 bandwidth-only 최소 시간은124.30/497.21µs.
그 처리율의90%에 해당하는 시간은138.11/552.45µs: 현재보다 약20%/9.6% 더 줄여야 한다.
이 단순 모델은 실제 달성 가능성을 보장하지 않는다. 모델 자체의 가정도 확인하고 instruction/barrier/shared traffic을 함께 평가해줘.
메모리 probe 대비89%를 곧바로 전체 커널 SoL89%라고 부르면 안 된다.

계측한 completion-frontier 시간:

| L | Body | Partial dump/publish | Grid wait | Reduce/publish | Reset |
|---|---:|---:|---:|---:|---:|
|384|165.776|2.816|0.576|2.800|0.256|
|768|605.632|0.560|0.640|2.592|0.288|

이는20회 계측 launch의 진단값이다. CUDA-event headline과 동일한 측정이 아니며 단계별 overlap이 존재한다.
**Body가 지배적**이다. 마지막 grid wait/reduction만 제거해서 큰 이득이 날 것으로 가정하지 말 것.

부분합 global traffic은 현재8.052736MB one-way, read+write16.105MB이다.
이전52MB 설명은 옛 구조의 global traffic이며 전부 HBM이라고 단정할 수 없다.
Workspace 할당 용량26.223MB는 그대로다. 양 역할의 중복 dy/gate 요청512B/row도 L2 hit를 고려해야 한다.

Warm NCU: `--clock-control none --cache-control none --replay-mode application`,20 warmups 뒤 profile.
Default full replay는204.8/738.112µs 등으로 다르다. cold/replay traffic과 warm event latency를 섞어 roofline을 계산하지 말 것.
Long-scoreboard ratio는0.467→0.370 /0.349→0.239로 감소했지만 barrier 비용은 남아 있다.
NCU local load/store0, SASS LDL/STL0. Shared bank conflicts 전체가 줄어든 것은 아니다.

근거 파일:

- `P/sol-prefetch-warm-L384/768.ncu-rep`와 같은 이름의 `.csv`
- `P/sol-prefetch-full-L384/768.ncu-rep`와 `.csv`
- `P/sol-prefetch-source.csv`, `P/dual_ln_prefetch.sass`
- `P/sol-prefetch-phases.json`, `P/sol-bandwidth.json`
- `P/sol-audit.json`에 정확한 cubin 경로, compiler command, 원본 SHA-256.

## 6. 이미 시도한 것 — 원인 없이 반복하지 말 것

| 시도 | 결론 / 자료 |
|---|---|
|LN statistics SoA / direct global|SoA 이득 없음, direct global L768 악화. `sol-ln-candidates.json`|
|gamma/beta register persistent accumulation|약192/740µs로 악화. `sol-lnacc.json`|
|warp별 shared accumulation / xhat 캐시|일관된 전체 이득 없음. `sol-epilogue.json`|
|TMA read-only completion wait / streaming hint|일관된 이득 없음. `sol-tma-read.json`, `sol-tma-stream.json`|
|독립 full-channel WG tile|284B spill store/260B spill load → **GPU 실행 전 제외**|
|위 설계의 shared reload variant|spill0이나 약197/711µs. `sol-wgreload.json`|
|WGMMA fence/commit 횟수 감소|이득 없음. `sol-mma.json`|
|mean/rstd를 B1 전에 preload, gamma launch당1회|현재 채택. `sol-stats-prefetch.json`|
|두 tile 앞 통계 prefetch / cp.async 통계|더 느림. `sol-async-stats.json` 등|
|CTA role ratio / physical placement|40/92 유지. place13 core 소폭 개선이나 전체 L768 악화. `sol-combinations.json`, `sol-final-paired-results.json`|
|mask float decode / packed ballot|float 악화, packed 변동 작아 미채택. `sol-mask-float.json`, `sol-mask-pack.json`|

소스별 분석에서 옛 shared excessive wavefront36,864는 row-statistics의 두 STS64에 귀속됐다.
8KiB gamma/beta partial buffer가 원인이라는 초기 추정은 틀렸다. SoA로 바꿔도 실측 성능은 좋아지지 않았다.

## 7. 앞으로의 작업 순서

1. 현재 선택안과 실제 소스/host 경로를 읽고 정확한 수식·SMEM·ownership·동기화를 확인해줘.
2. 현재 설정에서 baseline과 selected를 같은 process로 한 번 재현하여 비교 기준을 확보해줘.
3. 기존 NCU/source/SASS와 phase probe를 활용하여 **body의 노출된 비용**을 찾고, 예상 이득과 추가 resource/traffic을 계산해줘.
4. 가장 근거 있는 후보부터 새 `dual_<name>.cu/.py`로 실험해줘. 원본과 현재 선택안은 보존해줘.
5. 새 코드의 ptxas spill이0인지 먼저 확인한 뒤, 정확성 → 동기화 검증 → 반복 paired benchmark → NCU 순으로 평가해줘.
6. B1–B4와 전체 backward를 둘 다 측정하고, SoL의 정의·분모·한계를 명시해줘. 추정 이득을 실측처럼 보고하지 말 것.
7. 채택/기각 이유와 실제 배선 SVG, 기존 현황 HTML을 갱신해줘. Production 승격 여부는 실험 성공과 구분하여 명시해줘.

마지막 제안 후보는 DW/DX 역할 사이의 **tail work sharing**이었다. 아직 구현하지 않았다.
다만 L384의 DW→DX body frontier 여유가 약2.6µs뿐이라, helper의 Wp load나 초기화가 이득을 지울 수 있다.
이 제안을 정답으로 고집할 필요는 없다. resource accounting과 NCU 근거로 더 좋은 후보를 판단해줘.

## 8. 유지해야 할 정확성 및 실험 조건

- `dg` bit-exact, signed zero 포함.
- Relative L2: dtri≤2e-5, dgamma/dbeta≤5e-6, dWg/dWp≤5e-4.
- 일반 L≥64, L²%64=0 및 ragged tile 분배를 유지. 벤치 입력값/마스크 값으로 계산을 생략하지 말 것.
- L64/384/768 × dropout0/.25 × count66/132 × PART1/2 =24개 필수 조합.
- 일반 실행 + dy/ds 변경 CUDA Graph replay2회 =72번 비교, 매번 counters0.
- mean/rstd/gamma를 바꾼 추가 replay도 유지: launch 사이 stale cache 금지.
- memcheck/racecheck/synccheck L64/384, PART1 standalone reducer memcheck.
- 변경 후 전체 backward11 gradients도 확인.
- 현재 위 필수 검증은 통과했다. 기존 extra L72 한 seed에서 원본과 동일한 dtri2.1726e-5가 있어, 모든 shape/all-seed 통과로 일반화하면 안 된다.
- mbarrier phase, proxy fence, WGMMA fence/commit/wait, shared buffer 재사용과 cross-WG barrier의 근거를 코드에 남길 것.
- 최종 선택은20 warmup+200 samples×3, median/p90. 컴파일/할당/descriptor 준비는 타이밍 밖.

## 9. 실행 환경과 재현 명령

**node02만 사용. node01 사용 금지.** 컴파일·벤치·NCU·설치는 로그인 노드에서 실행하지 말 것.
기존 학습 job을 침범하지 말고, 새로 확보한 GPU 범위만 사용해줘. 여러 GPU가 배정되면 독립 후보 실험을 병렬로 진행하면 된다.
직전 벤치용2 GPU allocation은 반납했다. 당시 학습 job13228은 node02에서 RUNNING이었지만, 현재 점유는 다시 확인해야 한다.

CUDA12.9 `/usr/local/cuda-12.9`, sm_90a 사용. 새 toolkit으로 바꾸는 데 시간을 쓰지 말 것.
실행 환경은 `A/env.sh`가 설정한다. Torch2.10.0+cu128.
Anthropic 배포 cubin의 CUDA13/driver 호환 문제 때문에12.x로 재빌드한 경로를 사용 중이다.
파일 이름을 `cutlass*`로 만들지 말 것.

아래는 **node02에 GPU를 배정받은 shell에서** 실행한다.

```bash
cd /home/psk6950/MiniWorld
P=runs/anthropic_b1b4_pipeline_20260919
ENV=runs/anthropic_adoption_20260919/env.sh

# 기존 원자료를 보존하며 기준 재현 (새 파일명이 비어 있는지 먼저 확인)
sed 's/sol-final-paired-results.json/claude-start-paired-results.json/g' \
  "$P/measure_sol_final.py" > "$P/measure_claude_start.py"
bash "$ENV" python -u -B "$P/measure_claude_start.py"

# 새 커널은 별도 source 이름으로 검증
# bash "$ENV" python -u -B "$P/check_experiment.py" --source dual_<new_name>
# saved-updates / sanitizer / NCU는 아래 스크립트를 복제하여
# source와 모든 JSON/log/profile output 이름을 바꾼 후 실행:
#   check_sol_saved_updates.py
#   sol_prefetch_audit.sh
```

이 스크립트들은 현재 선택안을 기본으로 한다. 새 실험은 별도 source/output 이름으로 만들고 이전 JSON/log를 덮어쓰지 말 것.

## 10. 시각화와 인계 주의점

현재 비공개 현황 URL:
https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#b1b4-audit

실제 wiring SVG: `P/b1b4-shared.svg`.
새 결과도 이 HTML에 누적하되, 단순 시간표뿐 아니라 어떤 연산을 어떤 CUDA kernel/CTA/WG가 수행하고 어디에 저장하는지 보여줘.
기존 training-shapes 섹션과 다른 개발 기록은 보존해줘.

현재 게시 버전은18, source checkout은 **`P/site-visuals`**, 게시 commit은
`d7fd3ac33ed23f1290fc758c2b7cf9e6a64c095c`.
정확한 site/version/deployment 메타데이터는 `P/site-publication-sol.json`에 있다.

이전 자동 승인 검토에서 **CUDA 소스·raw JSON·source archive까지 포함한 넓은 push가 거부**됐다.
범위를 줄여 **HTML 결과표와 SVG만 게시**했고 나머지는 로컬에 보관했다.
`A/site`의 로컬 commit `31094ec`는 해당 전체 자료를 포함한 **미게시 커밋**이다.
이를 실수로 push하지 말고, 승인된 시각화 범위의 checkout인 `P/site-visuals`를 기준으로 작업해줘.
외부 게시 권한이 없다면 로컬 결과물까지 완성하고 필요한 범위를 명확히 설명해줘.

별도 완료 사항: 학습 token bucket은384/768, atom bucket은4096/8192로 정리했다.
Inference는 기존 여러 bucket을 유지한다. 관련 기록은 `runs/training_shape_policy_20260920/`에 있으며 이번 작업에서 되돌리지 말 것.

## 최종 보고에서 받고 싶은 것

1. 현재 병목을 source/SASS/NCU와 유효 traffic 모델로 설명.
2. 각 후보의 구현 차이, 예상 이득, 실제 채택/기각 이유.
3. 같은 process의 baseline/기존 selected/새 selected B1–B4 및 전체 backward median/p90.
4. 정확성·sanitizer·spills·graph replay 검증 결과와 재현 명령.
5. SoL90%에 근접했는지, 판단에 쓴 분모와 아직 남은 제한.
6. 갱신한 실제 배선 SVG/HTML 및 production 연결 상태.

**단순히 작은 개선을 누적했다고 마무리하지 말고, 어디까지가 측정된 상한인지 근거로 판단해줘.**
