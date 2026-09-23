# Phase 1 입력 atom attention의 backward compile 수정

Phase 1a `MiniSWAModel`은 distogram loss로 입력 atom embedder까지 학습한다.
`model_mini_swa.py:_embed` → `input_feature_embedder_esmfold2_style.py` →
`SWAAtomTransformer`의 3개 블록 → 엔진 `SWAAtomAttention` → FlashAttention 경로다.
Reference 좌표가 입력으로 고정되어 있어도 Q/K/V projection 등 학습 파라미터는
미분해야 한다. Phase 1 설정에는 이 입력 embedder를 freeze하는 정책이 없다.

## 원인과 변경

기존 forward는 compiler에 opaque custom op로 등록됐지만, 등록한 backward는
FA4 forward를 재계산하고 `torch.autograd.grad`를 호출했다. AOTAutograd가 그
backward를 추적하면서 FA4의 DLPack 변환까지 들어가 FakeTensor의 data pointer를
읽으려 해 실패했다. 첫 실패는 **compile ON + CUDA graph OFF**에서 발생했다.

FA4의 재계산과 native backward 호출을 새 opaque backward op 안으로 옮겼다.
FA4 `FlashAttnVarlenFunc`와 같은 `_flash_attn_fwd` / `_flash_attn_bwd`를 호출한다.
Padding 정리, output gradient masking, 원래 입력 dtype으로의 gradient 복원,
Q/K/V 중 필요한 gradient만 반환하는 동작을 유지한다. 기존의 추가 forward
재계산 1회도 유지하므로 attention 알고리즘과 재계산 횟수는 바뀌지 않는다.

커스텀 op 내부에서 단순히 `enable_grad`로 기존 reentrant autograd를 감싸는
방법은 사용하지 않았다. Custom op backend는 autograd dispatch 아래에서 실행돼
그것만으로는 내부 autograd graph를 만들 수 없기 때문이다.

FA2는 기존 경로를 유지하며 이번 compile 수정·GPU 검증 범위는 FA4다.
검증 환경은 PyTorch 2.10.0+cu128, flash-attn-4 4.0.0b19,
CUTLASS DSL 4.5.2, H100 80GB다. FA4 내부 launcher API를 사용하므로 FA4 버전을
변경할 때는 이 회귀 검사를 다시 실행해야 한다.

## 검증

`tests/test_phase1_flash_backward_compile.py`:

- BF16 / FP32, global / local window, noncontiguous Q/K/V
- Padding 입력의 NaN 차단과 padding output/gradient 0
- 원래 FA4 autograd 계산과 출력·Q/K/V gradient 상대 L2 오차 <0.5%
- `fullgraph=True, dynamic=False` compile + manual CUDA graph
- Q/K가 frozen이고 V만 gradient를 요구하는 경우
- custom-op schema, autograd 등록, fake tensor, AOT dispatch opcheck

H100 job 13208: 5건 통과. Job 13209: 동일 5건 통과,
compute-sanitizer memcheck 오류 0건. CPU 설정·배선 검사 8건도 통과했다.
전체 Phase 1a compile + CUDA graph 성능 결과는
[학습 A/B 보고서](miniworld-training-cudagraph-ab.md)에 기록한다.

패치는 `miniworld-engine-swa-fa4-backward-compile.patch`로 관리한다.
검증 원본은 `runs/phase1_compile_graph_fix/`에 있다.

설치본과 root/team-gm의 재설치 patch stack에 반영했으며 파일 SHA-256을
검증했다. [배포·검증 기록](phase1-compile-graph-checks/)을 함께 보관한다.
