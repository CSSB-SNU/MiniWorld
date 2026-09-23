# Triton bidirectional TriMul의 v6 계보 확인

2026-09-16 확인. 현재 설치 코드의 주석만 읽지 않고 `/home/psk6950/miniworld-engine`의 과거 Git 객체를 `git show`로 직접 읽었다. 아래 다섯 커밋은 모두 설치 패키지의 원본 pin `1bc0803e3b2fef3b963fdc383e090c0adcccdb43`의 조상임을 `git merge-base --is-ancestor`로 확인했다.

## 결론

**양방향 학습의 Triton 구현은 2026-07-12 `a5b58314`에서 이미 CuTe의 v6 계열 `BidirBackHalf` 구조로 바뀌었다.** 이 세션에서 처음 그 구조를 만든 것이 아니다. 단, 모든 시점과 추론까지 융합 경계가 같았다는 주장은 틀린다.

| 시점 | 커밋 | 실제 변경 |
|---|---|---|
| 7/12 12:15 | `a476bbad` | 최초 Triton bidirectional. 방향별 front를 각각 호출하고 `einsum`, `F.layer_norm`, autograd를 조합. 이때는 v6 merged-back 구조가 아님. |
| 7/12 13:21 | `a5b58314` | `_BidirBackHalfTriton`으로 다시 작성. CuTe `BidirBackHalf`의 학습 순서, 공유 helper, 수동 backward, dxn 누적 구조를 옮김. |
| 7/31 | `a519b907` | CuTe bidirectional 학습 gate에 residual/dropout 융합. 당시 Triton은 외부 `_r()`에서 처리하여 차이가 있었음. |
| 8/2 | `e196f903` | Triton도 동일 gate helper에 residual/dropout 전달. 별도 dropout·residual 덧셈을 gate store로 옮김. |
| 9/8 | `a1551fed` | Triton **추론** `_bidir_infer`를 LN + projection + gate GEMM + residual의 fused back으로 변경. 이 diff는 학습 `_BidirBackHalfTriton`을 해당 커널로 바꾸지 않음. |

## 제목 외에 실제 코드로 대조한 사항

`a5b58314`의 두 파일을 그대로 추출했다:

- [Triton 원본](mirror-triton.py): `_BidirBackHalfTriton`, lines 141–203.
- [CuTe 원본](mirror-cute.py): `BidirBackHalf`, lines 35–115, `BidirV6TriMul` line 129.

두 구현 모두:

1. `triton_layernorm` 입력 정규화.
2. gated front로 좌우 결과와 backward용 preactivation 생성.
3. outgoing/incoming contraction 두 번 및 `cat`.
4. 같은 `_te_forward`로 출력 LN + projection.
5. 같은 `gate_elem_triton`으로 출력 gate.
6. backward는 `gate_elem_bwd_ew` → `_te_backward` → contraction 미분 → `front_bwd_dW`.
7. 입력 gradient의 두 분기를 GEMM 누적에 합침. Triton은 `mm` + in-place `addmm_`, CuTe는 CuTe/CuBLAS 후보 dispatch.

따라서 **학습 구조 및 공유 연산은 같았지만, front의 GPU 구현과 GEMM backend 선택은 달랐다.** 또한 autograd.Function이 하나라는 것은 GPU kernel 하나라는 뜻이 아니다.

당시에도 [공유 gate helper](shared-gate.py)의 line 101은 `glogit = xn_flat @ Wg`이고, line 107에서 `_gate_mul_kernel`을 별도 실행한다. 출력 gate GEMM과 elementwise를 한 커널이라고 설명할 근거는 과거에도 없다.

## 현재 그림과 과거 상태를 구분할 것

- 7/12 원본은 forward `cat` 1회 + backward `cat` 2회가 있었다.
- 현재 그림의 contraction 최종 buffer 직접 기록은 로컬 `miniworld-engine-triton-packed-training.patch`의 후속 변경이다. 해당 패치는 현재 Git 미추적 파일이므로 원래부터 Git 커밋에 들어 있었다고 서술하면 안 된다.
- 신규 `_DgradLNRowsSm90` 출력 backward는 이번 최적화에서 추가한 것이다. 과거 v6와 공유하던 `_te_backward`에서 새로 갈라진다.
- 9/8 변경의 [실제 diff](inference-only.diff)는 추론 함수만 바꾼다. 추론의 더 큰 융합을 학습에 그대로 적용해 설명하면 안 된다.

## 재현 명령

```bash
git -C /home/psk6950/miniworld-engine show a5b58314 -- src/miniworld_kernels/kernels/trimul_inproj/triton/bidirectional.py
git -C /home/psk6950/miniworld-engine show a5b58314:src/miniworld_kernels/kernels/trimul_inproj/cute/bidir_training.py
git -C /home/psk6950/miniworld-engine show e196f903
git -C /home/psk6950/miniworld-engine show a1551fed -- src/miniworld_engine/kernels/trimul_inproj/triton/bidirectional.py
```

커밋 전문/변경 통계는 이 폴더의 해시별 `.txt`, 전체 해시 및 ancestry 결과는 [manifest.json](manifest.json)에 보관했다. 이번 확인에서는 과거 테스트를 재실행하지 않았으며, 과거 커밋 메시지의 수치와 이번 실제 측정을 혼동하지 않았다.
