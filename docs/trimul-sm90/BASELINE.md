# 양방향 TriMul SM90 구현 — 2026-09-17

> 최신 구현·공정한 후보 비교·전체 모듈 checkpoint는 [최적화 후속 보고서](OPTIMIZATION.md)에 있다.
> 이 문서는 첫 구현 당시 기록이다. 아래 수치는 제한된 Triton heuristic 및 이전 CuTe 구현의
> 결과이며 현재 성능 순위가 아니다. 실제 D128 backward KG는128이다.

## 기준

Triton의 학습 경로를 기준으로 F2, F567, B9+B10 커널 내부를 TMA/WGMMA로 교체한다.
F1/F4 LayerNorm, contraction의 두 forward/네 backward GEMM, B1/B7 elementwise,
weight-gradient GEMM, B11+B12 LN/residual의 경계는 유지한다.

- 입력 mask는 front 출력에 적용하며 입력 LN/output gate에는 적용하지 않는다.
- Front는 FP32 GEMM 결과로 sigmoid/gate를 계산하고, BF16 gate-product 반올림 후 mask를 곱한다.
- F567은 실제 LN affine 적용 결과를 입력으로 받는다. weight folding이나 별도 bias 준비를 추가하지 않는다.
- F567의 projection과 gate logit BF16 반올림, 출력 계산용 FP32 sigmoid와 저장용 BF16 gate를 구분한다.
- B9+B10은 gate gradient를 BF16으로 반올림한 뒤 FP32 front gradient에 합산한다.
- 기존 backward 저장값, dropout scale 및 residual gradient 위치를 유지한다.
- Front의 left/right는 하나의 packed allocation의 겹치지 않는 view로 반환한다. 큰 복사와 추가 커널은 없다.

## 기존 구현 재사용 판단

| 기존 구현 | 판단 | 이유 |
|---|---|---|
| Quack SM90 GEMM/TMA/WGMMA machinery | 재사용 | descriptor, operand layout, warp specialization 등 검증된 하드웨어 구현 |
| `MaskedGatedSm90` front 전체 | epilogue/config 수정 후 활용 | 기존 mask 앞 BF16 반올림 위치가 Triton과 다르고 stages가 명시적으로 제어되지 않음 |
| 이전 CuTe F567 | 참고 구현 | LN affine weight folding을 제거해야 현재 알고리즘과 일치; 전체 K staging 대신 실제 stage ring 필요 |
| 이전 projection-aware output backward | 이번 경로에 미사용 | B3/B4 경계를 바꾸고 별도 correction/weight-fold 계산을 도입함 |
| packed CuTe contraction | 재사용 가능 | 두 forward/네 backward GEMM과 최종 버퍼 직접 저장 구조가 일치. 이번 세 커널 비교에서는 cuBLAS 기준을 고정 |
| 기존 LN/elementwise | 유지 | 이 연산들에는 WGMMA 행렬곱이 없음. 연산 경계를 바꾸지 않는 이번 GEMM 구현과 분리 |

## Config 계약

별도 축을 복사해 유지하지 않고 **현재 Triton CSV를 직접 읽는다**.

| 대상 | Triton 원본 CSV | 선언 후보 수 |
|---|---|---:|
| F2 | `trimul_gemm_gate_mmajor_triton.csv` | 864 |
| F567 | `trimul_output_f567_train_triton.csv` | 3,072 |
| B9+B10 | `trimul_input_dual_bwd_triton.csv` | 1,152 |

`partition_configs`는 각 제외 후보의 config와 이유를 반환한다.
선언 공간이 같다는 것과 실제 실행 가능한 공간이 같다는 것은 구분한다.
WGMMA는 4-warp 단위이며 현재 mainloop의 물리 tile/warp 관계, TMA alignment 및 shared memory 제약이 있다.
지원하지 않는 후보를 다른 tile/warp 수로 몰래 실행하지 않는다.
구현 제한과 실제 하드웨어 제한도 검증 보고서에서 구분한다.

`num_stages`는 실제 TMA stage buffer/동기화를 바꾼다. K 반복 횟수가 적은 경우 사용되는 stage가 제한될 수 있다.
GROUP_M은 해당 GEMM의 tile 방문 순서를 바꾼다. Front 원본 CSV에는 GROUP_M 축이 없다.

## 연결

구현은 `runs/trimul_sm90_parity_20260917/engine`의 개발 브랜치 `perf/trimul-sm90-parity`, 커밋 `ae97bf26`에 저장했다.
이번 검증은 이 checkout을 PYTHONPATH로 지정해 수행했으며 운영 설치본에 자동 적용하지 않았다.

```python
from miniworld_engine import settings
settings.configure(trimul_sm90_kernels={"front", "f567", "dual_bwd"})
```

모델 생성/compile 전에 설정한다. 지정한 커널만 바꾸며 legacy CuTe 학습 알고리즘으로 들어가지 않는다.
기본은 빈 집합으로, 성능 비교 전 production 기본 경로를 바꾸지 않는다.
명시적으로 PyTorch implementation을 선택한 모델은 reference로 유지한다.
추론에서 front는 공유하지만 F4~F7의 기존 단일 커널은 유지한다. F567 교체는 학습 경계용이다.

Native cache는 shape/stride/dtype과 source identity를 구분한다. Cache miss 기본 후보를 전체 튜닝 완료로 표시하지 않는다.
전체 후보를 탐색하려면 기존 native build 경로를 사용한다. 새 op는 registry/driver/checker에 등록한다.
현재 새 CuTe op의 compile은 GPU allocation에서 launch 전에 수행하며, legacy Quack CPU precompile ABI를 사용하지 않는다.

## 검증 결과

커널별 결과와 전체 모듈 검증은
[runs/trimul_sm90_parity_20260917](../../runs/trimul_sm90_parity_20260917)에 저장한다.
일반 실행, 같은 Triton 커널과의 비교, selected-config sanitizer, fullgraph compile/CUDA graph를 구분해 기록한다.


### 검증 완료 범위

- 최종 통합 GPU 회귀 44건 통과. 이후 F567 native selector가 출력 버퍼를 재사용하는 검사 1건 추가 통과.
- 선택 config Compute Sanitizer: F2 6건, F567 24건, B9+B10 14건 모두 오류 0건.
- 모듈 L128 eager 및 static fullgraph compile, L384 static fullgraph compile: 같은 nonzero 가중치·mask·dropout scale에서 출력 및 모든 gradient 비교 통과.
- 최대 상대 L2: L128 약 4.08e-7, L384 약 9.83e-6. Zero dropout scale에서 output=input, dx=dy, parameter gradients=0 확인.
- Registry/명명/launch 연결 검사 185건 통과. 실제 native driver/checker와 cache 후보 재구성 일치 확인.
- F2/F567/B9+B10의 생성 PTX에서 TMA 및 WGMMA 확인.
- **기존 Triton도 대표 config 3개 모두 WGMMA를 사용한다.** 해당 config에서는 TMA instruction은 없었다.
  이번 비교는 Tensor Core 미사용 대 사용 비교가 아니다. TMA와 파이프라인/배치 구현의 비교다.

### 커널 단독 성능

H100 BF16. CUDA graph. F2는 L384의 명시한 고정 Triton config와 비교했고,
F567/B9+B10은 runtime heuristic 후보로 선택한 Triton과 비교했다.
CuTe 또한 아래 값은 탐색한 후보 중 결과이며 전체 선언 공간의 최적값은 아니다.

| 커널 | L | Triton (ms) | CuTe (ms) | Triton/CuTe |
|---|---:|---:|---:|---:|
| F2 | 384 | 0.266530 | 0.283832 | 0.939× |
| F567 | 128 | 0.011343 | 0.015817 | 0.717× |
| F567 | 384 | 0.106290 | 0.151682 | 0.701× |
| F567 | 768 | 0.414277 | 0.542944 | 0.763× |
| B9+B10 | 128 | 0.019580 | 0.023948 | 0.818× |
| B9+B10 | 384 | 0.180554 | 0.182084 | 0.992× |
| B9+B10 | 768 | 0.715165 | 0.652220 | 1.097× |

F2는 864개 선언 후보 중 현재 구현의 물리 배치에 맞는 144개를 시도했다.
140개는 정확도/벤치를 통과했고 4개는 shared-memory 용량 초과로 거절됐다.
M32 및 warp1/2/4는 이 front 구현이 지원하지 않는다. 이 중 warp4는 WGMMA 자체가 불가능해서가 아니라
현재 producer/consumer 구성의 제약이다. M128을8warp로 처리하도록 구현을 확장했다.
F567/B9+B10은 M64/M128에서 warp4/8을 각각 지원하며, M16/M32에 대한 논리 tile padding은 구현하지 않았다.

### 전체 모듈 성능

공식 bidirectional TriMul 모듈 셋업, B1/L384/D128, 비영 가중치, BF16,
static compile + manual CUDA graph, FWD+BWD, optimizer 제외, 12라운드 교대 측정 중앙값.
하네스는 nonzero dropout의 graph 측정을 거부하므로 **성능 벤치는 dropout=0**으로 수행했다.
Nonzero dropout 정확도는 위 별도 검사에 포함된다. 새로운 CuTe cache가 완성된 것처럼 처리하지 않고
[측정한 config manifest](../../runs/trimul_sm90_parity_20260917/module/measured-configs-L384.json)를 명시했다.

| 교체 범위 | 학습 시간 (ms) | Triton 대비 속도비 |
|---|---:|---:|
| 기존 Triton | 1.648600 | 1.000× |
| F2만 | 1.703424 | 0.968× |
| F567만 | 1.702560 | 0.968× |
| B9+B10만 | 1.653832 | 0.997× |
| 세 커널 모두 | 1.764840 | 0.934× |

**현재 세 커널 전체 교체는 약7.1% 느리므로 기본 활성화하지 않는다.**
B9+B10의 L768 단독 이득은 해당 shape 전체 학습의 이득을 뜻하지 않는다.
전체 shape/native cache 빌드와 모든 선언 config의 sanitizer 검증은 완료하지 않았다.

[원시 결과·소스 SHA-256](evidence.json)
