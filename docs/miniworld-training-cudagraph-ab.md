# Phase 1a: torch.compile + CUDA graph 실측

## 결과

**양쪽 모두 `torch.compile(dynamic=False)`를 켰다.** Inductor 자체 CUDA graph는
끄고, 같은 compiled 모델의 전체 forward/loss/backward에 수동 CUDA graph만
OFF/ON 했다. H100 80GB 한 장, v1.1 Phase 1a, 전체 Pairformer 48층,
L384 / MSA2048 / atom4096, 21,781,312 parameter, Triton 엔진 모듈 296개다.

1 optimizer step = microbatch 32개 누적이며 backward·Adam·EMA·gradient clipping을
포함한다. 준비용 step 1회 이후 측정한 3 step의 중앙값이다.

| 고정 recycle | compile + graph OFF | compile + graph ON | 속도 향상 | 시간 감소 | OFF / ON peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.822 s | 6.444 s | 1.059× | 5.54% | 54.25 / 54.33 GiB |
| 4 | 12.681 s | 12.156 s | 1.043× | 4.14% | 54.22 / 54.35 GiB |

이 설정에서는 compile 위에 CUDA graph를 추가했을 때 **약 4~6% 처리량 증가**를
확인했다. `compile + CUDA graph`가 1.8배 빠르다는 결과는 아니다.
그래프 capture 자체는 recycle 1회 약 0.37초, 4회 약 0.47초였으며 측정에서 제외했다.

초기 가중치·입력 배치와 선택된 Triton config 38개의 SHA-256은 각 OFF/ON에서
일치했다. 각 round의 scalar loss와 gradient norm도 일치했다. 전체 gradient
원소별 비교는 아니며, FA4 단위 출력·Q/K/V gradient 정확도는 별도로 검증했다.

[원본 JSON·로그·패치 manifest](phase1-compile-graph-checks/)에 결과를 보관했다.

## Phase 1에서 FlashAttention backward가 실행되는 이유와 수정

Phase 1은 입력 atom embedder의 SWA 3층까지 학습한다. 고정된 reference 좌표를
입력으로 받아도 학습하는 Q/K/V projection에는 backward가 필요하다.
실제 경로는 `MiniSWAModel._embed` → `InputFeatureEmbedderESMFold2Style` →
`SWAAtomTransformer` → 엔진 `SWAAtomAttention` → FA4다.

첫 compile 시도(job 13207)는 등록된 FA4 backward에서 AOTAutograd가 DLPack
변환까지 따라가 FakeTensor의 data pointer를 읽으려 하면서 실패했다. FA4의
기존 재계산·native backward를 opaque custom op로 감싸 수정했다. Attention 수식,
padding, gradient, 기존 추가 forward 재계산 1회를 유지한다.
[수정 내용과 검증](phase1-flash-backward-compile.md)을 참고한다.

H100 단위 검증 5건, 같은 5건의 memcheck 오류 0건, CPU 배선 검사 8건 통과 후
job 13210에서 위 네 조건의 전체 모델 학습을 완료했다. FA2 경로는 이번 수정과
compile 검증 범위에 포함하지 않는다.

## 측정 범위와 재현

- BF16 autocast, 기본 dropout, interchain distogram weight 2.0을 유지했다.
- seed 17로 같은 초기 가중치를 새로 생성했다. checkpoint 수렴 성능 실험은 아니다.
- 현재 데이터 catalog의 지문이 달라 기존 catalog를 명시적인 진단용 snapshot으로
  읽었다. 공유 catalog를 수정하지 않고 현재 source weight로 재균형했다.
- 실제 전처리·crop·bucket collate로 준비한 동일한 두 배치를 GPU에서 번갈아
  재사용했다. GPU static buffer 복사는 양쪽 시간에 포함한다.
- 데이터 로딩, CPU→GPU 전송, DDP 통신, wandb, checkpoint 저장은 제외한다.
  따라서 8-GPU end-to-end 처리량은 이 결과로 확정할 수 없다.
- 기본 Phase 1a는 recycle 수를 1~4회 무작위로 고른다. 실험은 각각 1회·4회로
  고정했다. 실제 무작위 recycle에는 recycle별 graph 선택이 별도로 필요하다.
- **측정 runner에서 명시적으로 model.compile 후 capture했다.** 기존 production
  `cudagraph_trainer.py`는 여전히 eager capture이며 이 실험에서 배선을 바꾸지 않았다.

```bash
python scripts/benchmark_miniworld_training_graph.py \
  --compile --graph manual --recycles 1 \
  --batch runs/miniworld_graph_ab/real_batches.pt \
  --output /tmp/phase1-compile-graph.json
```

`--graph off`로 compiled baseline을 실행한다. 각 조건을 동일 H100 allocation의
새 프로세스에서 실행한다. 실제 실행 스크립트는
`runs/phase1_compile_graph_fix/compiled.sbatch`다. 최초 Triton autotune, compile,
capture, Adam state 초기 생성은 timing 전에 끝낸다. Runtime 튜닝 결과는
`TRITON_CACHE_AUTOTUNING=1`로 공유한다.

## 메모리 수치

표는 allocator peak **reserved** GiB다. CUDA graph는 capture 때 중간 버퍼를
확보하고 replay에서 재사용하므로, capture 후 reset한 peak allocated만 보면
실제 replay 메모리 요구량을 과소평가한다. 초기화 포함 peak allocated, reserved,
GPU 전체 사용량은 원본 JSON에 구분해 기록했다.

## 이전 compile OFF 실험

앞서 보고한 1.04배는 compile OFF의 eager ↔ manual graph 비교였다.

| 고정 recycle | eager | eager + graph | 속도 향상 |
| --- | ---: | ---: | ---: |
| 1 | 7.500 s | 7.226 s | 1.038× |
| 4 | 14.359 s | 13.844 s | 1.037× |

이전 결과와 최초 compile 실패 로그는
[miniworld-training-graph-checks](miniworld-training-graph-checks/)에 보존한다.
이전 실험과 이번 실험 사이에 FA4 backward 경계 수정이 있으므로 두 표를 이용해
compile 단독 효과를 엄밀하게 분리했다고 주장하지 않는다.
