# 학습에서 엔진 Triton 경로 고정

학습 명령 끝에 다음 override를 추가한다.

```bash
train.engine_backend=triton
```

예:

```bash
torchrun --nproc_per_node=8 scripts/run_miniworld_distogram_train.py train \
  --config configs/miniworld/phase1a_distogram_v110.yaml \
  train.engine_backend=triton
```

GPU 개수에 따른 gradient accumulation 등 나머지 학습 설정은 별도로 지정한다.
YAML로 저장할 때는 다음과 같다.

```yaml
train:
  engine_backend: triton  # auto | triton
```

기본값 `auto`는 기존 GPU/shape별 엔진 선택을 유지한다. 옵션은 Phase 1·2·3의
Client와 Phase 1의 별도 manual CUDA-graph trainer에 연결되어 있다. 모델 생성과
compile/capture 전에 적용되고 실행 config에 저장된다. 잘못된 값은 거부한다.

## 실제 변경되는 배선

| 부분 | `auto` | `triton` |
| --- | --- | --- |
| 양방향 TriMul | H100에서 CuTe를 선택할 수 있음 | 현재 Triton F567·backward 융합 경로 |
| Transition | 내부에서 H100 CUDA b2b/CuTe로 재분기 | LN + Triton SwiGLU/gradient 경로 + residual |
| 엔진 LayerNorm | 자동 backend | Triton; row-scale/옵션에 의한 CUDA backward도 차단 |
| 기타 엔진 module | 기존 정책 | `miniworld` 요청을 Triton으로 해석 |

단순히 모듈 enum을 `TRITON`으로 바꾸는 것과 다르다. 기존 Transition의
`TRITON` family도 내부에서 H100 전용 커널을 선택하므로, 새 정책은 그 분기보다
먼저 shape-general split 경로를 고른다. 따라서 이전에 측정한 “Triton family”
Transition 수치를 이 옵션의 성능으로 그대로 사용하면 안 된다. 이 옵션의 전체
모델 학습 속도는 별도 측정이 필요하다.

PyTorch를 명시한 모듈은 PyTorch를 유지한다. Strict 정책과 충돌하는 엔진
`CUTE`/`CUDA` 명시 요청은 오류를 낸다. 엔진 내부 native 배선을 피하면서 모델의
수식, residual, dropout, 파라미터 이름과 checkpoint shape는 유지한다.

## 범위

- **엔진의 전용 CuTe/hand-CUDA 선택을 제어하는 옵션**이다.
- 외부 **FlashAttention(FA4의 CuTe 구현 포함)**, PyTorch/cuBLAS GEMM, Inductor가
  생성하는 코드는 기존 경로를 사용한다. 모든 GPU launch가 Triton이라는 뜻은 아니다.
- 정책은 엔진 settings의 **프로세스 단위 상태**다. 실행 도중 또는 기존 모델/graph가
  살아 있는 상태에서 전환하지 않는다. `auto`와 `triton` 실험은 별도 프로세스로 실행한다.
- `miniworld_engine` 모델 구성은 그대로 둔다. `model.shared.implementation=triton`은
  이 옵션의 사용법이 아니다.

## 설치 및 검증

엔진 변경은 `miniworld-engine-triton-backend-policy.patch`로 관리하며 root와
team-gm의 재설치용 patch stack에 함께 반영한다. 환경을 재설치하면 기존과 같이
`pixi run -e cu128 engine-setup`을 실행한다.

테스트는 `tests/test_triton_backend_option.py`에 있다. CPU에서는 세 단계의 Hydra
설정과 모델 전체 module 배선/checkpoint shape를 검사한다. H100에서는 D128/D512
Transition과 양방향 TriMul의 출력·입력/파라미터 gradient를 PyTorch와 비교하고,
compile·CUDA graph·inference를 실행한다. Native selector와 H100 전용 진입점을
호출하면 실패하도록 하여 숨은 재분기를 검사한다.

### 이번 검증 결과

- CPU 설정·배선 검사 8건 통과. Phase 1·2·3 override와 checkpoint 구조를 확인했다.
  설치본 반영 후 기존 distogram 회귀 검사와 함께 27건 통과, GPU 전용 5건 skip.
  설치 파일 SHA-256 및 root/team-gm patch stack 일치도 확인했다.
- H100 job 13201: Transition D128/D512 및 양방향 TriMul L128,D128의
  BF16 출력·입력/전체 파라미터 gradient, compile, CUDA graph, eval 검사 통과.
  출력 상대 L2 오차 <2%, 입력 gradient <3%, 파라미터 gradient <4% 조건을 검사했다.
- D128 Transition과 TriMul은 compute-sanitizer memcheck 오류 0건.
- H100 job 13203: 별도 LayerNorm D128/D512도 CUDA override를 차단하고
  출력·모든 gradient 상대 L2 오차 <2%, memcheck 오류 0건을 확인했다.
- **D512는 일반 실행 검증만 통과했다.** memcheck를 켜고 autotune 후보를 탐색하면
  `transition_fwd_kernel`에서 `cudaErrorInvalidPc`가 재현된다(job 13200).
  당시 후보는 BLOCK_K=32, BLOCK_M1=64, BLOCK_N=256, GROUP_M=4,
  num_warps=4, num_stages=2였다. 일반 실행은 동일 검사에 통과하므로 원인은 아직
  확정하지 않았다. 해당 후보를 제거하거나 캐시를 조작하지 않았다.
- 전체 모델 학습·처리량 벤치마크와 모든 shape의 캐시 완성 검증은 포함하지 않는다.
  이번 소형 검증 shape 중에는 cache miss에 따른 현장 autotune도 있었다.

검증 원본은 `runs/triton_backend_option/`, 공유할 결과 사본은
[triton-backend-checks](triton-backend-checks/)에 둔다.


### 전체 모델 후속 검사

[Phase 1a 전체 학습 A/B](miniworld-training-cudagraph-ab.md)에서 compile ON 상태의
수동 CUDA graph OFF/ON 측정을 완료했다. 최초에 발견한 SWA FA4 backward의
FakeTensor 오류는 `miniworld-engine-swa-fa4-backward-compile.patch`로 수정했다.
전체 모델의 고정 recycle 1회·4회에서 모두 compile + CUDA graph 학습이 동작한다.
무작위 recycle·DDP end-to-end 처리량이나 FA2 compile까지 검증한 것은 아니다.
