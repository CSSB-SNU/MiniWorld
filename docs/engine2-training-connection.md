# Engine 2 연결 및 별도 학습 프로파일

2026-09-23 사용자 요청으로 기존 잡 `13228`을 epoch467 / step46700 체크포인트에서 종료하고, 같은 W&B run과 로그 경로를 유지하는 Engine2 재개 잡 `16681`을 제출했습니다. 아래 프로파일 기록의 “13228 유지”는 전환 전 상태입니다. 기존 스냅샷은 수정하지 않았습니다.

## 새 실행의 엔진 선택

`scripts/with_engine2.sh`는 해당 명령의 `PYTHONPATH`에만 최신 엔진을 먼저 지정합니다.

```bash
scripts/with_engine2.sh .pixi/envs/cu128/bin/python \
  scripts/run_miniworld_distogram_train.py train \
  --config configs/miniworld/phase1a_distogram_medium_v120.yaml \
  train.engine_backend=auto
```

이 문서의 명령은 별도의 새 학습을 시작하는 예시이며 자동 실행하지 않았습니다. 실제 시작 시 고유한 run directory를 사용해야 합니다. `MINIWORLD_ENGINE2_ROOT`로 선택할 엔진의 Python package 부모 경로를 지정할 수도 있습니다.

- Phase1 체크포인트 재개 시 Adam 상태의 stride를 새 TriMul 파라미터와 일치시킵니다. 값은 그대로 보존합니다.
- `fused_msa_train=false`는 Engine 2에서도 꺼지도록 환경변수에 명시적으로 `0`을 설정합니다.
- 현재 학습 구성의 `fused_msa_train=true`는 유지합니다.
- 모델/optimizer 상태 검증과 MSA flag 검증: `tests/test_engine2_training_resume.py`.

## 프로파일 범위

`runs/engine2-training-profile-20260923/`에 실행 스크립트, 고정 체크포인트 provenance, dispatch 기록, 시간 측정 및 CUDA trace를 저장합니다.

기존 학습과 같은 MiniWorld/team-gm 스냅샷에서 엔진만 교체합니다. 기존 엔진+triton과 Engine2 `20f0c9a5`+auto를 비교합니다. 실제 v1.2 배치와 epoch445 모델 및 optimizer를 양쪽에 동일하게 로드합니다.

Pairformer16, MSA4, L384/atom4096, MSA pool8192/subsample1024, dropout/residual, compile을 포함합니다. recycle1/4를 각각 고정해 측정합니다. 단일 GPU이며 DDP, 데이터 로딩/H2D는 제외합니다. microbatch와 64 microbatch마다 실행하는 optimizer/clip/EMA 비용을 분리합니다.

[HTML 결과](../ENGINE_TRAINING_PROFILE.html).

## 확인된 기존 잡의 MSA 경로

기존 설정에는 `fused_msa_train=true`가 있지만 9/17 엔진 스냅샷에는 `integrations/pwa_train.py`와 `opm_train.py` 및 이를 호출하는 모듈 분기가 없습니다. 따라서 비교 baseline의 MSA는 일반 경로이며, 최신 엔진에서 native MSA가 실제 연결됩니다.

## 전체 모델 compile에서 발견한 수정

D64 템플릿 TriMul은 `[1,L,L]` 마스크를 전달합니다. 최신 엔진의 wide 학습 경로가 저장 마스크를 `[L,L]`로 선언하면서 실제로는 3D로 반환해 Inductor stride assertion이 발생했습니다. `h100_width.Training`에서 저장 마스크를 `(L,L)` view로 정규화했습니다. 추가 복사는 없으며, 2D/3D 마스크와 네 폭을 포함한 8개 회귀 테스트를 통과했습니다. 프로파일은 이 수정이 포함된 엔진을 사용합니다.

프로파일의 실제 유효 크기는 token360, atom2877, MSA row467입니다. 버킷은 L384/atom4096/MSA1024이며, 이 단일 배치 결과를 전체 데이터 분포의 처리량으로 간주하지 않습니다.

## Recycle 4 배선 수정 (2026-09-23)

- `no_grad`이지만 training dropout이 켜진 양방향 TriMul은 전용 native forward로 보냅니다. D128에서는 x_n·출력 LN 통계 저장을 생략하며 dropout/residual 수식은 유지합니다.
- wide forward는 backward workspace·descriptor·GP 계획을 생성하지 않습니다. wide 커널이 forward 계산에 사용하는 x_n과 LN 작업 공간은 유지합니다.
- TMA descriptor는 CUDA context·주소·dtype·레이아웃·타일·옵션이 같을 때 인코딩된 바이트만 재사용합니다. 최대8192개이며 GPU tensor를 캐시에 붙잡지 않습니다. 파라미터 값이나 activation을 재사용하는 캐시가 아닙니다.
- 별도 H100 전체 모델 trace에서 no-grad 전용84회 + 학습 저장형28회를 확인했습니다. 이전은 저장형112회였습니다.
- 실행 중인 학습 snapshot은 변경하지 않습니다. 새 프로세스는 `scripts/with_engine2.sh`로 로컬 수정 엔진을 선택할 수 있습니다.
- 측정 근거: `runs/engine2-training-profile-20260923/`; 최종 표는 `ENGINE_TRAINING_PROFILE.html`.

최종 재측정(15회): 기존 fwd121.77/bwd67.35/합계189.14ms → 수정 엔진 fwd91.54/bwd56.30/합계147.85ms. 합계 기준1.279배. node01 잡16425·16427 완료, node02 학습13228 유지. 수정은 로컬이며 아직 추가 커밋/푸시는 하지 않았습니다.

## 남은 CPU 호출 비용 분석

추가 검증(job16449·16450): 일반 compile 기존189.22/최신147.51ms, 같은 recycle4 CUDA graph 진단 기존178.27/최신110.14ms. 일반-graph 차이는10.95→37.36ms. 순수 CPU 시간이 아닌 준비·호출·스케줄링 효과를 포함합니다. GPU kernel 합계도178.17→109.92ms로1.62배이므로 비교 범위보다 호출부 비용이 주원인입니다. B7 Plan의 매회 컴파일 캐시 파일 접근이 확인됐으며, worker cProfile에서20회 약15ms(계측 오버헤드 포함)입니다. 분석 근거는 runs/engine2-training-profile-20260923/gap-summary.json과 B7_plan-worker.txt에 있습니다. 이 분석 단계에서는 커널/호출부 구현을 추가 변경하지 않았습니다.

## 호출부 정리 구현 (2026-09-23)

- B7의 kernel 선택·컴파일 캐시 확인·shared-memory 설정·occupancy 확인은 장치/length/clusters/mode별 정적 계획으로 분리했습니다. 매 backward에서는 현재 입력과 출력 포인터를 바인딩합니다.
- packaged JSON 설정은 프로세스당 한 번 읽습니다. wide forward/backward occupancy도 장치와 설정별로 재사용합니다.
- C 구조체의 정렬·패딩 포맷을 재사용하고, 여러 필드를 한 번의 struct.pack으로 포장합니다. 값과 tensor 주소는 매 호출 읽습니다.
- 인자용 호스트 저장 공간은 스레드별로 재사용합니다. 살아 있는 두 포장 객체는 서로 다른 공간을 임대하며, CUDA driver가 인자를 복사하고 호출이 반환된 후 반환합니다. 풀은 tensor나 CUDA stream을 보관하지 않고 약4MiB로 제한합니다.
- GPU activation·gradient·작업 버퍼는 각 호출이 소유합니다. GPU 메모리 재사용은 PyTorch allocator가 담당합니다. 동시에 살아 있는 forward나 graph의 결과를 공유 버퍼로 덮어쓰지 않습니다.
- 정상 CUDA context가 현재 스레드에 있으면 context 확인 과정에서 임시 GPU tensor를 만들지 않습니다.

검증: CPU 구조체 ABI/새 값 반영/다중 인자/동시 포장/스레드 분리/설정 단일 읽기 검사, GPU no-grad dropout·wide mask·compile gradient·독립 forward 소유권·graph live-weight 검사. Warm 경로에서 compiler-cache 함수와 설정 파일 접근을 강제로 금지한 검사도 포함합니다. D64의 기존 FP32 LN atomicAdd 합산은 비트 단위 결정적이지 않으므로 이 추가 반복 검사는 해당 gradient에만 상대 L2 오차1e-4를 적용합니다. CUDA 계산 소스와 기존 수치 검증 기준은 변경하지 않았습니다.

최종 검증(job16472/node02): CPU8건·GPU17건 통과. 같은 node02에서 기존/수정 엔진을15회 비교했습니다. 결과 JSON: `runs/engine2-training-profile-20260923/host-final-comparison.json`. B7 계획 worker profile에서 compile-cache 호출0회를 확인했습니다. 고정 ABI 변환기와 TMA 메타데이터 검사를 캐시하고, 검증한 CUDA context를 native op 단위로 공유·복구합니다. 모든 동적 포인터와 값은 호출마다 갱신합니다.

## 직접 weight pack 및 graph 유휴율 측정 (2026-09-23)

- TriMul 내부의 중첩 `torch.compile` weight pack을 직접 호출하는 Triton 복사로 교체했습니다. 32행 단위 gate/projection 배치와 현재 가중치를 읽는 의미를 유지합니다. D64/128/256/384/512, 두 stride 배치, graph live-weight 검증 및 기존 hot-path/no-grad 검사 14건 통과. 전체 모델 교대 측정은 `pack-comparison.json`에 있습니다.
- `profile_matrix.py`는 PF16/PF48 × recycle1~4에서 일반 compile과 전체 fwd+bwd graph를 비교합니다. Graph는 미리 존재하는 parameter gradient 버퍼에 누적합니다. `zero_grad(set_to_none=True)`로 해당 버퍼를 교체하면 graph와 parameter.grad 연결이 깨지므로 캡처 이후에는 버퍼 주소를 유지해야 합니다.
- `profile_graph_validation.py`는 PF48/R1의 1·2·64회 누적을 같은 RNG 상태에서 일반 실행과 비교하고, 실제 Adam 갱신 후 다시 확인합니다. Loss는 동일해야 하고 각 gradient의 상대 L2 오차는 1e-3 미만이어야 합니다. 단순 finite 검사만으로 완료 처리하지 않습니다.
- PF48 성능 모델은 PF16 체크포인트의 block i%16을 독립 parameter를 가진 block i에 복사합니다. 학습된 PF48 체크포인트가 아니며, PF16 optimizer state를 48블록 모델에 억지로 로드하지 않습니다.
- GPU 유휴율은 replay를 추적한 CUDA event 전체 구간에서 kernel/memcpy/memset 실행 구간의 합집합을 빼서 계산합니다. 비계측 시간에서 계측 kernel 시간을 빼는 추정과 구분합니다. kernel 내부 stall/occupancy는 유휴율에 포함하지 않습니다.
- 이 실행은 단일 GPU의 고정 입력 buffer 계산 경로입니다. 실제 DDP 학습 job에 적용하지 않았으며, 데이터/H2D·optimizer/EMA·DDP 통신 및 recycle별 graph 선택은 시간에서 제외됩니다. `scripts/with_engine2.sh`는 엔진 선택만 하며 이 graph 실행을 자동 활성화하지 않습니다.
- 최종 수치: `runs/engine2-training-profile-20260923/matrix-summary.json`, 검증: `graph-accum-pf48-r1.json`, 시각화: `ENGINE_TRAINING_PROFILE.html`. 측정 과정의 초기 gradient-overwrite graph 자료는 `matrix-before-accumulation/`에 보존했습니다.

### 상위 모듈 경계와 랜덤 recycle 검증

- `profile_module_boundaries.py`는 PF48/R1의 `temp_embedder`, `msa_module`, `pairformer_blocks`에 forward/backward CUDA event를 직접 기록합니다. Hook은 Dynamo에서 제외하고 전체를 CUDA graph에 캡처합니다. 계측으로 compile 분할이 달라질 수 있으며 전체 시간은 113.76ms로 이전 113.53ms 대비 약0.2% 차이였습니다. 결과는 `module-boundary-summary.json`과 HTML 최상단에 있습니다.
- `profile_random_recycle.py`는 PF16에서 R1~R4 전체 fwd+bwd graph를 독립 pool로 캡처하고, 공통 gradient 버퍼에 `2,3,4,1,3,4,1,2` 순서로 누적합니다. RNG 상태를 맞춘 일반 실행과 loss가 일치하고, Adam 갱신 전후 최대 gradient 상대 L2 오차는0.000106입니다. 기존 별도-R 검증을 넘어 graph 간 전환과 누적까지 검증한 결과이며 DDP/변하는 배치의 입력 복사는 여전히 별도 범위입니다.
- Recycle 선택은 graph 밖에서 합니다. 단일 전체-model graph를 한 번 캡처한 뒤 Python의 랜덤 recycle 분기가 매 replay 달라진다고 가정하면 안 됩니다. Shape bucket·recycle별 capture와 고정 grad 주소 관리가 필요합니다.

### 최신 경로 배선 재감사

PF48/R4 compile trace에서 TriMul no-grad180/train-fwd60/train-bwd60, Transition D128 fwd208/bwd52 및 wide fwd44/bwd11, OPM no-grad12/train4/bwd4, PWA fwd12/bwd3를 확인했습니다. 예상 호출 수와 모두 일치합니다. 실제 모델의 Transition 입력은 모두 native eligible이며 TriMul도 모두 native eligible입니다. Embedding의 FlashAttention/Triton 경로는 별도 연산으로 유지됩니다.

감사 중 기존 Transition 수치/배선 테스트가 fused=True에서도 `engine_backend=triton`을 강제하는 오류를 찾았습니다. D128/wide 테스트를 native arm은 auto, 비교 arm은 triton으로 수정했습니다. 수정 후 D1288건/D647건이 통과했고 OPM/PWA/TriMul dropout·residual·compile 검사8건도 통과했습니다. Production 커널이나 dispatch 수식 변경은 없습니다. 감사 결과 및 소스 해시는 `wiring-audit-summary.json`, 실제 trace는 `wiring-pf48-r4-trace.json`에 있습니다.

이 결론은 별도 최신 엔진 프로파일 프로세스에 대한 것입니다. 기존 학습13228은 이전 snapshot + Triton 정책을 유지하며, 최신 CUDA graph DDP 학습까지 연결됐다고 의미하지 않습니다.

### Anthropic original TriMul control (2026-09-23)

Jobs 16640/16641 completed. Full Phase1 inference, recycle1, L384/atom4096/MSA1024, BF16, eval/no_grad. Replace only bidirectional TriMul in Template/MSA/Pairformer with upstream v5 K1/K3 and Python runtime; other modules stay on latest engine. Two half-channel cuBLAS contractions preserve MiniWorld bidirectional semantics. This is not an all-Anthropic implementation of the full model. Source SHA-256 checks cover both used CUDA units and ten Python files; all match local upstream.

| Blocks | Mode | Original TriMul ms | Latest ms | Speedup |
|---|---|---:|---:|---:|
|16|compile|25.733|22.465|1.145x|
|16|compile + graph|15.951|14.671|1.087x|
|48|compile|42.686|37.036|1.153x|
|48|compile + graph|28.952|26.763|1.082x|

Same GPU per block count, latest/original/original/latest. Pooled medians of 30 ordinary samples and 10 groups of 10 graph replays per process. Original adapter is compiler-disabled, so ordinary results include host integration differences; prioritize graph numbers for GPU path comparison. All 22/54 distinct TriMul modules were checked against latest on the same inputs, max output relative L2 0.00038078; all outputs finite. Existing training13228 unchanged. Raw results: `runs/engine2-training-profile-20260923/inference-original-comparison.json`; visualization: `ENGINE_TRAINING_PROFILE.html#inference-original`.

### Corrected scope: original TriMul + Transition + OPM + PWA

The preceding TriMul-only control did not answer a full upstream-kernel comparison. Jobs16646/16647 now replace all four module families in the full Phase1 model: upstream v5 TriMul, Transition v2, msa_opm.forward_mask_norm with shipped configuration, and msa_pwa.forward_masked (g_fo4p). The engine's anthropic_msa adapter is deliberately bypassed: it contains an engine-derived OPM CUDA epilogue and a substituted PWA LayerNorm. Original PWA uses stock torch LayerNorm/Linear surrounds. All668 opt_core source files checked against UPSTREAM.json hashes match. Other embedding/attention/head operations remain common.

| PF blocks | Mode | Original four families ms | Latest ms | Speedup |
|---|---|---:|---:|---:|
|16|compile|33.615|22.676|1.482x|
|16|compile + graph|22.882|14.683|1.558x|
|48|compile|48.985|36.455|1.344x|
|48|compile + graph|36.795|26.754|1.375x|

Same ABBA procedure/shape as above. The original Transition cell table names v2 as H100 C128/N<=400 forward winner; C64 graph also names v2. This comparison uses shipped configurations, not a new exhaustive upstream tuning sweep. Strict provider selection asserts v2; no fallback is accepted. Audited unique modules: PF16 TriMul22/Transition25/OPM4/PWA3, PF48 TriMul54/Transition57/OPM4/PWA3. Max same-input output relative L2: TriMul0.0374%, Transition0.4114%, OPM0.1403%, PWA0.2178%. BF16 arithmetic differs (notably upstream OPM BF16 mask counts versus engine FP32); this is not bitwise parity or task-level accuracy certification. Source/measurements at `runs/engine2-training-profile-20260923/{anthropic-all-source-check,inference-original-all-comparison}.json`. Existing training13228 remains unchanged.

### Training numerical validation, 2026-09-23

Jobs16677/16679/16680 completed the focused validation. The initial broad-width test job16678 was stopped while building widths unrelated to this model; the final suite completed with **56 passed, zero failed/skipped**. Tests include MSA1024 OPM/PWA, dropout/residual gradients, D64/D128 Transition, TriMul width gradients, compile, saved tensor ownership and live-weight graph replay. Benchmark engine remains the isolated `engine2-cache` copy; production training13228 unchanged.

Additional bidirectional TriMul oracle: PyTorch FP32 using identical quantized inputs/weights/upstream derivative and dropout25% mask, L384 D64/128 and L768 D128, plus L384 D128 zero-gamma case. Maximum relative L2: output0.30794%, input gradient0.44552%, parameter gradient0.59422%; all finite.

Actual checkpoint445, restored Adam, PF16/recycle4, L384/atom4096/MSA1024, three optimizer steps compared latest versus current engine forced Triton with native OPM/PWA disabled. Dropout OFF to isolate arithmetic: loss relative differences0.05179/0.07148/0.02211%; global gradient relative L2 2.95099/1.21971/1.01310%, cosine0.999622/0.999938/0.999953; actual parameter-update relative L2 3.56924/4.72848/5.92374%. All436 gradients finite. Later steps include independent optimizer trajectory differences; these measurements are not bitwise equivalence or a <0.01% error claim.

Separate actual-model dropout ON graph versus ordinary compiled execution: 1/2/64 accumulated backwards, before/after optimizer; every loss exactly equal; max per-tensor gradient relative L2 0.00627555%. No long-term convergence, held-out quality, DDP or all training buckets verified by this full-model run. Evidence: `runs/engine2-training-profile-20260923/accuracy-summary.json`, `accuracy-kernels.xml`, and HTML accuracy section.


### Same-run production continuation (2026-09-23)

- Old job13228 -> new job16681, node02 H100x4; frozen verified checkpoint epoch467/global_step46700.
- W&B `team_gm/MiniWorld/tapgki9e`, `resume="must"`; run name unchanged. SDK log confirms `Resuming run`.
- Same original training subdirectory, `train.log`, checkpoint directory, and `slurm-13228.log` (Slurm append mode); no new experiment run. W&B SDK creates a local resumed-session directory under the existing wandb directory while retaining the same remote run ID.
- Original model/data/config/Fabric plain compile retained; resolved config diff exactly `train.engine_backend: triton -> auto`. Native OPM/PWA enabled, verified Engine2 source copied into immutable continuation tree.
- Strict model/Adam/scheduler/EMA restore; 436 optimizer states and EMA tensors. Adam layout aligned before/after synthetic warmup without changing values; first load realigned160 tensors.
- Continuation source and provenance: `runs/v1.2.0/phase1a/medium_launch_20260917_130505/engine2_resume_20260923/`.
- Original experiment snapshot remains untouched. Frozen resume checkpoint SHA256 `7ce45f1722215cbe266aae54ec3bf7a93bb52d4e4dd66e8961430919de9aea5c`.

Production continuation confirmed: all four recycle warmups completed in1108.8s; epoch467 real-data optimizer steps46701 onward logged to the original train.log. W&B API reports the same tapgki9e run RUNNING, step46706 with fresh loss1.6290. Slurm16681 RUNNING on node02 H100x4.
