# TriMul 추론 배선 — 2026-09-17 설치본

[전체 SVG](../../TRIMUL_INFERENCE.svg)

기준은 MiniWorld cu128 환경에 **실제로 설치된 miniworld_engine 소스**다. 이 그림은 추론 배선을 다룬다. H100, BF16, B=1, d_pair=d_hidden=128, mask 전달, 기본 layout을 그렸다. 그림은 소스 배선을 나타낸다. 아래 Triton contraction 검증은 별도로 수행했으며, 전체 모듈의 GPU 커널 프로파일은 아니다.

## 진입 조건

`model.eval()`과 `torch.no_grad()` / `torch.inference_mode()`를 함께 사용하는 경우다. 코드의 실제 분기는 `not torch.is_grad_enabled()`와 `dropscale is None`이다. `eval()`만 호출해 grad가 켜져 있으면 학습용 autograd 경로로 들어간다. `train()` 상태에서 dropout scale이 만들어지면 no_grad에서도 학습용 커널을 사용한다.

## 실제 연결

| 단계 | Triton 단방향 | Triton 양방향 | H100 CuTe 단방향 | H100 CuTe 양방향 |
|---|---|---|---|---|
| F1 입력 LN | Triton `layer_norm_fwd_fused` | 동일 | 동일 | 동일 |
| F2 입력 projection/gate/mask | `_bidir_front_kernel` 1회, hidden=128 | 같은 커널 1회, hidden=256 | `MaskedGatedSm90` 2회: left/right | `MaskedGatedSm90` 4회: left/right × out/in |
| F3 contraction | `torch.bmm` 1회 | `torch.bmm` 2회 | `torch.einsum` 1회 | `torch.einsum` 2회 |
| contraction 결과 결합 | 불필요 | `packed_forward`: 최종 `tri` slice에 직접 기록, cat 없음 | 불필요 | `torch.cat` 있음 |
| F4~F7 출력 | Triton `_back_kernel` 하나로 융합 | 같은 커널, LN/proj 폭=256, gate 폭=128 | Triton `_back_kernel` 재사용 | F4 Triton LN/layout → F5 cuBLAS proj → F6 cuBLAS gate GEMM → F7 Triton sigmoid/mul/residual |

- `_back_kernel`은 **출력 LN + projection GEMM + gate GEMM + sigmoid + 곱 + residual**을 한 커널에서 실행한다. 출력 norm/proj/glogit/gate를 별도의 HBM tensor로 만들지 않는다. norm은 BF16으로 변환해 dot에 사용하고, proj/gate accumulator는 FP32로 유지하다 최종 y를 저장한다.
- 학습용 `_output_f567_kernel`은 F4 LN이 별도이고 proj/gate를 backward용으로 저장한다. **추론 `_back_kernel`과 다른 커널**이다.
- H100 CuTe 단방향은 출력부도 CuTe인 것이 아니다. 실제로 Triton `_back_kernel`을 호출한다.
- H100 CuTe 양방향 추론은 개발된 CuTe F567을 호출하지 않는다. `gate_elem_infer()` 역시 단일 GPU 커널이 아니라 **cuBLAS GEMM + Triton elementwise**다.
- Triton 양방향 추론은 학습과 같은 `packed_forward`를 사용한다. `tri`를 한 번 할당하고 두 cuBLAS GEMM이 `tri[:h]`, `tri[h:]`에 직접 기록한다. 별도 `O_out`/`O_in` buffer와 그 뒤의 `cat`이 제거되었다. H100 CuTe 양방향의 `torch.cat`은 그대로이며, 해당 경로의 compile 시 복사 제거 여부는 프로파일하지 않았다.
- H100 CuTe 양방향 F2의 결과는 4개의 `[128,L,L]` tensor다. Triton 양방향은 left/right 각각 `[256,L,L]`로 직접 기록한다.

## mask가 없을 때

H100 기본 `trimul_out_layout()`은 `bdll_direct_wide`다. `mask=None`이면 masked-front 분기를 건너뛰고, 각 left/right에 wide GEMM을 실행한 뒤 `_glu_wide_kernel`로 sigmoid와 곱을 수행한다. 따라서 단방향은 **CuTe/quack GEMM 2회 + Triton GLU 2회**, 양방향은 **GEMM 4회 + GLU 4회**다. mask를 모두 True인 tensor로 전달하면 위 그림의 masked-front 경로가 선택된다. 명시적인 layout override는 별도다.

## 그림에서 생략한 것

Weight interleave/transpose/cast, mask 준비, 소규모 메타데이터 및 할당은 생략했다. 입력 LayerNorm은 추론에서도 mean/rstd buffer를 생성한다. F2의 preactivation 저장은 꺼져 있다. 따라서 이 그림을 전체 GPU launch 수나 모든 HBM 쓰기의 완전한 목록으로 해석하면 안 된다. cuBLAS 한 호출의 내부 launch 개수도 단정하지 않는다.

## 소스 근거

기준 경로: `/home/psk6950/MiniWorld/.pixi/envs/cu128/lib/python3.10/site-packages/miniworld_engine`

- 단방향 Triton: `kernels/trimul_inproj/triton/unidirectional.py`, `_uni_infer`, `trimul_triton`
- 양방향 Triton: `kernels/trimul_inproj/triton/bidirectional.py`, `_bidir_infer`; `kernels/trimul_inproj/triton/contract.py`, `packed_forward`
- 단방향 H100: `modules/triangle_multiplication/module.py`, `_forward_cute_free`
- 양방향 H100: `modules/triangle_multiplication/bidirectional.py`, `_forward_cute`
- CuTe front 분기: `kernels/tm1/cute/launch.py`, `tm1_cute_forward`
- Masked front: `kernels/trimul_inproj/cute/masked_front.py`, `MaskedGatedSm90`
- 융합 출력: `kernels/trimul_inproj/triton/back.py`, `_back_kernel`
- 분리 출력: `kernels/layernorm/triton/transpose.py`, `_ln_transpose_dbn_kernel`; `kernels/trimul_inproj/triton/gate_elem.py`, `gate_elem_infer`
- [원본 경로·SHA-256 기록](inference-sources.json)

재생성: `python scripts/render_trimul_inference.py`. 이 스크립트는 설치본을 import하지 않고 SVG 및 소스 해시를 기록한다. 배선 설명은 위 소스를 감사해 작성한 정적 스냅샷이므로 코드가 바뀌면 설명도 함께 갱신해야 한다.

## Triton 양방향 cat 제거 검증

설치본 검사 7건 통과: 기존 출력과 bitwise 일치, static fullgraph compile, CUDA graph 재생, contraction의 ATen `bmm` 2회 / `cat` 0회. [검증 및 구간 벤치 결과](https://github.com/SanggeunParrk/miniworld-engine/blob/b06857c0/docs/records/trimul-inference-nocat-20260917/REPORT.md).
