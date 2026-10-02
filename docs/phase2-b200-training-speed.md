# Phase 2 학습 속도: recycle 깊이, 체크포인팅, CUDA 그래프, 커널 배선 (B200, 2026-10-02)

phase 2(frozen trunk + diffusion head) 학습의 마이크로 스텝을 B200 한 장에서 재고, 속도를 깎던 항목을 고친 기록이다.
이 문서의 코드는 `benchmarks/phase2_step/`와 `tests/test_phase2_*.py`에 있다. 3.3절과 3.5절의 비교는 **같은 프로세스 안에서** 한 것이다.
요약 표는 각 단계의 값을 서로 다른 프로세스의 측정에서 이어 붙인 것이라 ±3% 잡음이 있다(전력 상한과 메모리 배치 때문에 같은 코드도
프로세스마다 조금씩 다르게 나온다).

## 요약

phase 2a v200(crop 384 토큰 / 4096 원자, MSA 2048, augment 48, diffusion bf16), 마이크로 스텝 ms.

| 단계 | ms | 비고 |
|---|---:|---|
| 원래 (체크포인팅 켬, recycle 4 고정) | 121.9 | |
| + 체크포인팅 끔 | 108.0 | 피크 메모리 5.1 → 20.2 GiB |
| + 랜덤 recycle 1~4 (균등 평균) | 84.3 | 학습 분포가 바뀐다(아래 1절) |
| + CUDA 그래프 (recycle별 전체 스텝, 풀 공유) | **75.8** | 원래 대비 −38% |

- 랜덤 recycle은 속도 최적화가 아니라 학습 분포의 변경이다. phase 1과 같은 규칙으로 되돌린 것이다.
- v200 phase 2a/2b 설정은 체크포인팅을 끄고 랜덤 recycle로 바꿨다(`train_recycle: random`).
- 커널 배선은 전부 이어져 있다(4절). 엔진 커널 62%, cuBLAS 33%, 나머지 PyTorch 연결부 약 4%.

## 측정 조건

B200 1장(sm_100a, 전력 상한 1000 W), torch 2.13.0+cu129, triton 3.7.1, miniworld-engine `5d8bb030`.
설정 `configs/miniworld/phase2a_diffusion_v200.yaml`, 합성 배치(`_build_precompile_batch`), 전체 모델 `torch.compile(dynamic=False)`.
시간은 마이크로 스텝 하나(순전파 + 손실 + 역전파)이며 옵티마이저와 EMA는 제외한다(옵티마이저 Adam+clip 약 10 ms,
EMA 약 9 ms를 `grad_accum_steps`번의 마이크로 스텝마다 한 번 쓴다). inductor 캐시가 따뜻한 상태다.

## 1. recycle 깊이

**이전**: `DiffusionModel._condition_impl`이 학습에서도 항상 `n_recycle_max`(4)를 돌렸다. docstring의 이유는 "frozen trunk라
가장 좋고 결정적인 conditioning을 준다"였다. phase 1(`MiniSWAModel._run_trunk`)과 `docs/foldbench-evaluation.md`의 학습은 recycle을
균등 랜덤(1~4)으로 뽑는다.

**변경**:
- `DiffusionModel.Config.train_recycle: "max" | "random"` (기본 `max`: ConfidenceModel은 그대로). `random`이면 학습에서 마이크로 스텝마다
  `rng`에서 1..`n_recycle_max`를 균등 추첨한다. 추론은 항상 전체 깊이, `_forced_n_recycle`이 둘 다 이긴다.
- 추첨은 `@torch.compiler.disable` 메서드(`_draw_train_recycle`)에서 한다. 컴파일 안에서 `int(rng.integers(...))`를 하면 inductor가
  `aten._local_scalar_dense`를 만들지 못해 variant마다 컴파일 실패 후 그래프 브레이크로 넘어간다.
  이 메서드는 의도한 그래프 브레이크가 되고, 반환된 정수로 recycle 횟수마다 컴파일 그래프가 하나씩 생긴다.
- warm-up(`_warmup_bucket_shapes`)이 recycle 1~4를 각각 실제 학습 경로(강제하지 않은 경로)로 컴파일한다.
  원래 warm-up은 `_forced_n_recycle = 2`로 컴파일했다. 실측: 강제 recycle 2 컴파일 80 s 뒤 같은 recycle 2를 강제 없이 돌리면 그래프가
  1개 더 생기고 첫 스텝이 11 s 더 걸렸다(캐시가 따뜻한 상태). 강제 경로와 실제 경로는 컴파일을 공유하지 않으므로 원래 warm-up은
  실제 학습(recycle 4)의 컴파일을 하나도 데워 주지 못했다.

**효과**(체크포인팅 끔): recycle 1~4가 60.5 / 76.4 / 92.2 / 108.0 ms, 균등 평균 84.3 ms. recycle를 하나 늘릴 때마다 약 15.8 ms.

**검증**(`bench_step.py --script-warmup`, 체크포인팅 끔): script warm-up 114 s(recycle 4개 컴파일, 따뜻한 캐시) 뒤 recycle 1~4 측정과
실제 난수 생성기 40스텝에서 새로 컴파일된 그래프 0개. 400스텝 중 중앙값의 3배를 넘는 스텝 0개, 파이썬 GC gen2 0회. 스텝당 정확히 한 번 추첨.
체크포인팅을 켠 실행에서만 약 1초짜리 정체 3건(880 / 1138 / 765 ms)이 있었고 원인은 확인하지 못했다.

## 2. 체크포인팅

v200 설정은 token DiT(24 블록)와 atom SWA 인코더/디코더(3+3 블록)의 블록별 체크포인팅을 껐다. 켜면 역전파에서 모든 블록의 순전파를 다시
돌린다. fp32 학습의 메모리 대책이었고 bf16 diffusion에서는 필요 없다.

| | 체크포인팅 켬 | 끔 |
|---|---:|---:|
| 2a L384, recycle 4 | 121.9 ms, 5.1 GiB | 108.0 ms, 20.2 GiB |
| 2b L768 / 8192 원자, recycle 4 | 336.9 ms, 10.7 GiB | 317.8 ms, 40.9 GiB |

## 3. CUDA 그래프

### 3.1 구조

- **전체 스텝 그래프** (`graph_full.py`): recycle 횟수마다 그래프 1개. 그래프 안에 GPU 샘플링(회전, 노이즈 수준), frozen trunk, diffusion
  head 순전파, EDM 손실, 역전파가 모두 있고 gradient는 미리 만든 `.grad`에 제자리로 누적된다. recycle 횟수는 캡처 때 박히고, 학습에서는
  호스트가 횟수를 뽑아 해당 그래프를 replay한다. 옵티마이저, EMA, 데이터 로딩은 그래프 밖이다.
- **풀 공유** (`--share-pool`): 그래프는 한 번에 하나씩만 replay되고 풀 안의 값이 스텝 사이에 살아남지 않는다(grad는 풀 밖의 `.grad`,
  loss는 replay 직후에 읽는다). 그래서 모든 recycle 그래프를 한 메모리 풀에 캡처할 수 있다. 4개를 따로 두면 그래프당 풀이 따로 상주한다.
- **분리 그래프** (`graph_split.py`): trunk와 head는 frozen trunk가 `no_grad`라 경계(`token_single_input`, `token_pair_trunk`)에 autograd가
  걸치지 않는다. 거기서 잘라 E(임베더, 1회), T(trunk recycle 1회분, r번 replay), H(샘플링 + head + 손실 + 역전파)로 나눈다.
  recycle 횟수가 캡처와 컴파일 키에서 빠진다.

### 3.2 캡처되도록 바꾼 것 (`benchmarks/phase2_step/graph_safe.py`)

원본 학습 스텝은 그대로는 캡처되지 않는다. 아래는 확인한 사실이다.

| 원본 | 캡처 | 대체 |
|---|---|---|
| `torch.linalg.svd`(`weighted_align`) | 실패(호스트 복사) | Horn 쿼터니언 정렬, 제곱 24번, 순수 fp32 원소 연산 |
| `torch.tensor(±1.0, device=cuda)` | 실패 | 사용하지 않음 |
| `.item()` 5개, `if torch.isnan` | 호스트 읽기(캡처 불가) | 손실에서 제거(`cal_loss_gs`) |
| scipy 회전, CPU `torch.rand` 노이즈 | 상수로 굳음(replay마다 같은 값) | GPU 샘플링(replay마다 새 값) |
| 엔진 weight-pack 캐시(`_PACKS`) | 캡처 때 캐시가 맞으면 pack 커널이 기록되지 않아 replay가 옛 가중치를 씀 | 캡처 직전과 직후에 비움 |

Horn 정렬은 행렬곱이나 einsum을 쓰지 않는다. 학습 스크립트가 `torch.set_float32_matmul_precision("medium")`를 쓰므로 fp32 행렬곱은
bf16으로 돌고, 늘어진 구조에서 정렬이 틀어진다(제곱 12번 + 행렬곱 첫 버전은 합성 배치에서 5.5~6.2 Å 어긋났다).
fp64 SVD 기준 해와의 좌표 최대 오차(합성 배치): Horn 0.02 Å, 원본 fp32 SVD(`medium`) 0.38 Å. 구형, 타원형 구조에서 Horn은 기준 해와 일치하고
(RMS < 1e-5 Å) SVD(`medium`)는 0.005~0.011 Å, 300/5/5 Å 막대형에서 Horn 0.0001~0.0002 Å, SVD(`medium`) 0.06~0.07 Å다.

### 3.3 결과 (체크포인팅 끔, ms)

| 구성 | recycle 1 / 2 / 3 / 4 | 균등 평균 | 예약 메모리 |
|---|---|---:|---:|
| 현재 방식 (같은 프로세스) | 60.3 / 77.0 / 92.9 / 108.7 | 84.7 | |
| 호스트 대기 제거만(eager) | 58.7 / 75.8 / 91.8 / 107.5 | 83.5 | |
| 전체 스텝 그래프, 풀 따로 | 53.0 / 68.0 / 83.3 / 98.7 | 75.8 | 82.4 GiB |
| **전체 스텝 그래프, 풀 공유** | 53.2 / 68.2 / 83.3 / 98.7 | **75.8** | **22.4 GiB** |
| 분리 그래프 (E + T×r + H) | 53.2 / 68.5 / 83.8 / 99.7 | 76.3 | 24.4 GiB |
| head 그래프 1개 (trunk는 그래프 없이) | 56.2 / 72.0 / 87.9 / 104.1 | 80.0 | |

체크포인팅을 켠 설정: 현재 방식 76.8 / 92.6 / 108.4 / 120.8 (평균 99.7), 전체 스텝 그래프 63.1 / 78.2 / 93.4 / 108.9 (85.9, 22.6 GiB),
분리 그래프 63.3 / 78.6 / 93.9 / 109.5 (86.3, 9.3 GiB).

- 그래프가 줄이는 시간은 recycle 횟수에 크게 달라지지 않는다(체크포인팅 끔 7~10 ms, 켬 12~15 ms). trunk의 큰 커널은 이미 GPU 병목이고, 발사 병목은 작은 커널이 많은
  head와 손실 쪽이다. 프로파일의 recycle 4 GPU 실행시간은 체크포인팅 끈 설정에서 98.8 ms(eager)와 97.1 ms(그래프)로 같고, 그래프 replay
  벽시계가 98.7 ms다.
- 입력은 고정 배치 버퍼다. 새 배치를 `copy_`로 덮어쓰는 시간은 16 MiB에 0.4 ms(고정 메모리)이고 스텝 시간에는 넣지 않았다.
- 그래프 안에서 매 replay마다 weight pack이 다시 만들어진다(추정 0.3~0.5 ms, 위 수치에 포함).

### 3.4 정확성

- replay와 일반 실행이 같은 입력에서 비트 단위로 같다(loss, grad).
- 가중치를 제자리에서 바꾼 뒤 replay도 새 가중치를 쓴 일반 실행과 일치한다(grad 상대 L2 1.2e-9 ~ 2.9e-9). 바꾸기 전 replay와는 다르다.
- 풀을 공유한 채 recycle 2, 4 그래프를 4 → 2 → 4 → 2 순서로 섞어 재생해도 일반 실행과 비트 단위로 같다.
- 분리 그래프(E + T×2 + H)도 일반 실행과 비트 단위로 같다(pair, loss, grad).
- 손실의 정렬 대체(SVD → Horn): 상대오차 `highest`에서 6e-6 이하, `medium`에서 5.4e-4 이하(원본 SVD가 행렬곱 bf16 때문에 갖는 오차 포함).

### 3.5 저장소의 trunk 전용 옵션 (`train.trunk_compile_mode=reduce-overhead`)

frozen trunk만 inductor cudagraph-trees로 돌리는 기존 옵션을 같은 구조의 대조군(`trunk_compile_mode=default`)과 재었다.

| | recycle 1 / 2 / 3 / 4 | 평균 |
|---|---|---:|
| 체크포인팅 끔: 대조군 → 옵션 | 60.8 / 76.1 / 92.1 / 107.7 → 58.0 / 73.1 / 88.2 / 103.8 | 84.2 → 80.8 (−4.0%) |
| 체크포인팅 켬: 대조군 → 옵션 | 75.3 / 90.1 / 105.9 / 123.5 → 75.5 / 88.2 / 101.0 / 116.4 | 98.7 → 95.3 (−3.4%) |

이득은 대부분 head에서 나오므로 trunk만 그래프로 만드는 옵션은 효과가 작다.

### 3.6 한계와 남은 일

- 그래프 트레이너는 아직 저장소 학습 스크립트에 들어가 있지 않다. `benchmarks/phase2_step/`는 측정과 검증 도구다.
- **결정**(트레이너에 넣을 때): B200은 recycle별 전체 스텝 그래프(풀 공유, 22.4 GiB), 그 외 GPU는 분리 그래프(E + T×r + H, 24.4 GiB).
  풀을 공유하면 전체 스텝 그래프의 메모리가 분리 그래프와 비슷해져서 메모리는 더 이상 이유가 아니다. 분리 그래프가 남기는 이점은
  recycle 횟수가 캡처와 컴파일 키에서 빠진다는 것(trunk 한 단계를 한 번만 컴파일)이다.
- 템플릿이 0개인 배치는 정적 shape가 달라진다. phase 1 트레이너의 `prepare_template_graph`처럼 padding과 presence 게이트가 필요하다.
- `bond_loss`와 smooth-lDDT는 phase 2a/2b에서 가중치가 0이라 캡처 대상이 아니다. 켜려면 bond 쌍을 고정 길이로 padding해야 한다.
- 옵티마이저, grad clip, EMA는 그래프 밖이고 `.grad` 초기화도 그래프 밖에서 해야 한다.
- 수치는 합성 배치 기준이다.

## 4. 커널 배선 점검 (`wiring_audit.py`)

체크포인팅 끈 설정, 컴파일 recycle 4 마이크로 스텝, GPU 95.4 ms, 커널 3535개.

| 계열 | GPU 시간 | 비율 | 커널 수 |
|---|---:|---:|---:|
| 엔진 커널 | 59.5 ms | 62.4% | 1625 |
| cuBLAS / cutlass GEMM | 31.4 ms | 32.9% | 1263 |
| inductor 융합 | 2.2 ms | 2.3% | 343 |
| aten 나머지 | 2.0 ms | 2.1% | 269 |
| FA4 | 0.14 ms | 0.2% | 3 |
| memcpy / memset | 0.06 ms | 0.1% | 32 |

(aten 나머지에는 엔진의 Triton LayerNorm/RMSNorm 커널 약 1 ms가 분류 규칙 때문에 섞여 있다.) 비엔진 커널 상위 목록은 전부 cuBLAS GEMM이다.

eager recycle 4에서 모듈 호출 수와 실행된 엔진 연산 수가 구조와 정확히 일치한다.

| 구성 | 모듈 호출(recycle당) | 실행된 엔진 연산 |
|---|---|---|
| TriMul (pairformer 48 + MSA 4 + template 8) | 60 | `trimul_b200_inference` ×60 |
| Transition (pairformer 48 + MSA 7 + template 8) | 63 | `transition_fused_fwd_sm100a` ×52 + `transition_wide_fwd_sm100a` ×11 |
| OPM (MSA 4) | 4 | `opm_h100_infer` ×4 (커널 `opm_epilogue_sm100`) |
| PWA (MSA 3) | 3 | `pwa_h100_infer` ×3 (커널 `pwa_gate_out_sm100`) |
| token DiT 24블록 | 학습 1회 | `bias_only_dit_train_sm100_fwd/bwd` ×24 |
| atom SWA 인코더 3 + 디코더 3 | 학습 1회 | `swa_dit_block_fwd/bwd` ×6 |
| conditioning Transition 4 | 학습 1회 | `transition_wide_fwd/bwd_sm100a` ×4 |

- 융합 연산이 블록을 통째로 대체하므로 token DiT와 atom SWA 블록 안의 LayerNorm, AdaLN, Linear 모듈은 실행되지 않는다.
- 정적 목록의 `KernelBackend.*` 값은 모듈 기본값이지 실제 커널이 아니다. 증거는 호출된 연산 이름이다.
- 안 이어진 것: 입력 임베더의 SWA 블록 3개는 융합 블록 연산이 아니라 모듈 경로(Triton RMSNorm + FA4 + cuBLAS)로 돈다. A=1에서는 융합 블록이
  더 느린 것을 측정해서 의도적으로 쓰지 않는다. 임베더 최적화는 `docs/gpus/b200/token_pair_init`(엔진)을 본다.

## 5. 코드 점검 항목과 실측 크기

| 항목 | 위치 | 실측 |
|---|---|---|
| 체크포인팅 켬 | model yaml | 마이크로 스텝 −14 ms (L384) |
| warm-up이 recycle 2만 컴파일 | `run_miniworld_diffusion_train.py` | 첫 실제 스텝에서 재컴파일(위 1절) |
| 호스트 대기(SVD, `isnan`, `.item()` 5개, scipy 회전, CPU 노이즈) | `team_gm` align / edm, `client.py` | 합쳐 평균 −3.8% (현재 방식 99.7 → 호스트 대기 없는 eager 95.9 ms, 체크포인팅 켠 설정) |
| trunk CUDA 그래프 없음 | `enable_trunk_cudagraph` 기본 꺼짐 | trunk 전용 −3.4~−4.0%, 전체 스텝 그래프 평균 −10% 추가 |
| `_scatter_atom_to_token`의 밀집 one-hot GEMM | `diffusion_module.py` | 리뷰 추정 0.3~0.5 ms, 미측정 |
| smooth-lDDT 손실의 augment별 체크포인트 루프 | `client.py` | 0 가중치(v200)라 해당 없음 |
| 증강 회전 `bmm`과 Kabsch 공분산이 `medium` 행렬곱 정밀도 | team-gm `diffuser` / `align` | 회전 최대 오차 0.28 Å (좌표 크기 1149 Å, 상대 2.5e-4), SVD 정렬 오차는 3.2절 |

## 6. 재현

저장소 루트에서(GPU 1장, 같은 프로세스 비교):

```sh
# recycle별 시간, 랜덤 추첨 40스텝, script warm-up 검증, 정체 진단
python -m benchmarks.phase2_step.bench_step --config configs/miniworld/phase2a_diffusion_v200.yaml \
    --script-warmup --recycles 1,2,3,4 --steps 8 --random-steps 40
python -m benchmarks.phase2_step.bench_step --config configs/miniworld/phase2a_diffusion_v200.yaml --recycles 1,2,3,4 --stall-probe 400
# trunk 전용 그래프(저장소 옵션)와 대조군
python -m benchmarks.phase2_step.bench_step --config ... --recycles 1,2,3,4 --forced --trunk-graph default
python -m benchmarks.phase2_step.bench_step --config ... --recycles 1,2,3,4 --forced --trunk-graph reduce-overhead
# CUDA 그래프: 전체 스텝(풀 공유), 분리
python -m benchmarks.phase2_step.graph_full --config ... --steps 10 --share-pool
python -m benchmarks.phase2_step.graph_split --config ... --steps 10
# 커널 배선 점검 (eager 인벤토리 + 컴파일 계열별 시간)
python -m benchmarks.phase2_step.wiring_audit --config ...
```

체크포인팅을 다시 켜려면 `model.diffusion.token_dit.n_checkpoint_segments=24 model.diffusion.atom_swa.n_checkpoint_segments=3`.
`tests/test_phase2_train_recycle.py`와 `tests/test_phase2_graph_safe.py`가 recycle 정책, warm-up, 캡처 안전 대체(SVD와 같은 정렬, 같은 손실)를 CPU에서 검증한다.
