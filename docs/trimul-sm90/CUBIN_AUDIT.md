# TriMul Triton / CuTe 실제 cubin 비교

> 후속 [최적화 결과](OPTIMIZATION.md)에는 더 빠른 Triton 4-warp F2 기준과 실제 KG128 backward 비교가 있다. 이 문서의 이전 고정 후보 비교만으로 현재 backend 우열을 판단하지 않는다.

2026-09-17. H100 80GB, BF16. 개발 checkout `ae97bf263a035b3971eab2730b6332d510899db3`.
실제로 실행한 Triton `CompiledKernel.asm['cubin']`와 CuTe compiler가 보존한 cubin을
`cuobjdump --dump-sass/--dump-resource-usage/--dump-elf`로 읽었다.
PTX를 별도로 재조립한 결과가 아니다. 바이너리 SHA-256 및 SASS 원문을 보관한다.

원본 자료: [trimul_cubin_audit_20260917](../../runs/trimul_cubin_audit_20260917).

## 비교 기준

- 같은 fusion/math, 실제 production stride. 컴파일/튜닝 비용을 제외한 CUDA graph 실행 시간.
- config를 같게 고정한 비교와 각 구현에서 선택한 config 비교를 분리한다.
- runtime heuristic 선택값은 전체 config 공간의 최적값을 뜻하지 않는다.
- Nsight Compute 수치는 별도 kernel replay 결과다. profiler의 시간은 CUDA graph 시간과 섞지 않는다.
- SASS의 정적 명령 개수는 loop/unroll/분기 때문에 실제 실행 횟수와 다르다.
- `cuobjdump`의 SHARED 값만으로 동적 shared memory 사용량을 판단하지 않는다.
  아래 shared memory는 Nsight의 실제 launch dynamic shared memory이다.

## F567: 동일 config에서도 구현 차이가 확인됨

L384, M147456, KP256, KG128, N128. 두 구현 모두 4608 CTA, 128 threads/CTA.
공통 config: BM64, BN64, BK64, GROUP_M4, 4 warps, 2 stages.

| 항목 | Triton | CuTe |
|---|---:|---:|
| CUDA graph, 5회 중앙값 | 113.243 us | 151.056 us |
| Registers/thread | 128 | 162 |
| Dynamic shared memory/CTA | 32 KiB | 73 KiB |
| 레지스터 제한 동시 CTA/SM | 4 | 3 |
| 실제 점유율 | 23.85% | 18.06% |
| DRAM 처리량 | 2.009 TB/s | 1.472 TB/s |
| Tensor pipe 사용률 | 15.70% | 10.83% |
| 실행된 warp instructions | 35.961 M | 34.007 M |
| Barrier stall cycles/issued instruction | 0.847 | 1.256 |
| Fixed latency wait cycles/issued instruction | 1.009 | 2.416 |
| Cubin LOCAL / STACK | 0 / 0 | 0 / 0 |

Triton runtime이 선택한 BM64/BN64/BK64/G8/4w/3s는 106.592 us.
따라서 이번 CuTe는 동일 config 대비 33.4%, 선택된 Triton 대비 41.7% 느리다.
출력/저장 projection/저장 gate 상대 L2는 각각 3.28e-6 / 0 / 0.

### SASS에서 확인한 구현 차이

- 양쪽 모두 `HGMMA.64`를 사용한다. 기존 Triton도 이미 WGMMA이다.
- Triton의 입력은 `LDGSTS.E.BYPASS.128`, CuTe는 `UTMALDG.2D`.
- Triton의 출력은 vector `STG.E.128`. CuTe는 shared tile에 쓴 뒤
  `UTMASTG.2D` + commit + wait를 세 출력마다 반복한다.
- CuTe의 dropout scale 읽기는 `LDG.E.U16` scalar load 32곳이며,
  Triton은 residual/dropout 읽기에 `LDG.E.128` vector load를 사용한다.
- CuTe sigmoid reciprocal은 PTX `rcp.rn.f32`로 내려가며 SASS에
  범위 검사/보정 연산 및 예외 경로가 남는다. 이 예외 경로가 정상 입력에서도
  항상 실행된다는 뜻은 아니다.
- CuTe는 두 GEMM의 operand stage rings를 동시에 유지한다.
  더 큰 shared memory와 register 사용, 낮은 occupancy, 긴 대기가 실제 계측된다.

즉 TMA/WGMMA 명령 누락이나 spill 문제가 아니라, 현재 CuTe의 자원 사용과
epilogue/메모리 접근/대기 구조가 개선 대상이다. 개별 항목의 시간 기여율은
독립적인 변경 실험 없이는 확정하지 않는다.

자료: [timings](../../runs/trimul_cubin_audit_20260917/f567/report.json),
[SASS 요약과 SHA-256](../../runs/trimul_cubin_audit_20260917/f567/sass-summary.json),
[Nsight 상세](../../runs/trimul_cubin_audit_20260917/f567/counters.txt).

### F567 reciprocal 한 항목만 변경한 실험

엔진 source를 바꾸지 않고 복사본에서 sigmoid reciprocal만
`cute.arch.rcp_approx`로 바꿨다. 타일, stage, fusion, 메모리 배치는 그대로다.
동일 세션의 5회 CUDA graph 중앙값:

| Triton 동일 config | 기존 CuTe | reciprocal 변경 CuTe |
|---:|---:|---:|
| 113.847 us | 151.499 us | 129.599 us |

CuTe 시간이 14.46% 줄고 register 162→105, SASS `CALL` 32→0,
`BSSY/BSYNC` 각각 53→4, `BRA` 117→33으로 줄었다.
이 실험은 reciprocal 처리 자체가 실제 손실의 일부였음을 보여준다.
동시에 scalar load 및 shared memory 배치는 그대로이고 Triton보다 13.84% 느려,
이것만으로 개선이 끝난 것은 아니다.

일반 입력(seed123)의 y/projection/gate는 Triton과 비트까지 일치했다.
큰 logit [-100,100]에서는 FTZ(매우 작은 값의 0 처리) 때문에 gate 최대 절대 차이
1.001e-38이 남는다. 해당 시험의 상대 L2 0은 norm underflow를 포함하므로
모든 입력의 비트 일치로 해석하지 않는다. 운영 source에는 적용하지 않았다.

자료: [단일 변경 실험](../../runs/trimul_cubin_audit_20260917/f567/rcp-ablation/report.json).

## F2: 입력 재사용과 타일 순회 순서

L384, K128, H2=256. 원래 비교한 Triton M64/BK64/BH64/8w/4s는 264.880 us,
CuTe M128/BK64/BH64/8w/2s는 284.519 us였다.
같은 M128 config의 Triton은 353.9 us이므로 F2는 config 선택 차이도 크다.

| 원래 선택 config의 Nsight 결과 | Triton | CuTe |
|---|---:|---:|
| DRAM read | 40.48 MB | 251.28 MB |
| DRAM write | 약 431 MB | 약 431 MB |
| L2 hit | 92.44% | 70.16% |
| 실제 점유율 | 24.25% | 7.83% |
| 실행된 warp instructions | 82.43 M | 46.49 M |

CuTe는 명령 수가 더 적지만 DRAM 읽기가 6.21배이다.
Triton은 한 M tile에서 channel chunks를 순회한다. CuTe의 persistent scheduler는
group_size=1로 모든 M tile을 먼저 지나간 뒤 다음 N tile로 넘어가므로,
다음 channel chunk가 같은 입력을 읽을 때 L2 재사용을 놓치는 구조다.

### 같은 cubin에서 scheduler group만 바꾼 실험

원래 엔진 파일을 수정하지 않는 monkeypatch로 runtime scheduler group만 변경했다.
Config와 cubin은 같다. 8회 교대 CUDA graph 측정.

| Scheduler group | 1 (기존) | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| CuTe 시간 (us) | 284.550 | 266.290 | 249.958 | 237.645 |

Group8은 기존 CuTe보다 시간이 16.48% 줄고, 위 Triton 선택 config보다
1.115배 빠르다. 출력과 raw preactivation 저장값은 네 경우 모두 비트까지 같다.
이는 원래 순회 순서가 실제 성능 손실을 만들었다는 직접적인 변경 실험이다.
같은 변경의 Nsight DRAM read도 251,278,336→38,401,280 bytes로 84.7% 줄었다.
쓰기량은 431.52→430.40 MB로 비슷하다. 입력 재사용 개선이 실제로 확인된다.
이 shape에서 유효한 결과이며, group8을 모든 shape의 상수로 정한 것은 아니다.
운영 source에는 적용하지 않았다.

자료: [F2 binary/counters](../../runs/trimul_cubin_audit_20260917/front/evidence.json),
[순회 순서 실험](../../runs/trimul_cubin_audit_20260917/front/swizzle.log).

## B9+B10: 이전 성능 비교 정정

이전 보고의 Triton 기준은 runtime heuristic 24개 후보 중 선택값이었다.
CuTe에서 고른 config를 Triton에도 직접 적용하니 더 빨랐다.
따라서 이전 L768의 CuTe 1.097배 우위는 구현의 우위로 해석할 수 없다.

공통 config: BM64, BN128, BK64, GROUP_M1, 4 warps, 3 stages.
이 표의 단독 시험은 KG256, KP1024, N128이며 front-gradient/weight는 column-major다.
**후속 모듈 배선 점검에서 D128 모델의 실제 gate K는 128임을 확인했다.**
따라서 이 KG256 결과는 범용 shape 시험으로 분류하고, D128 모델의 대표 성능은
[후속 최적화](OPTIMIZATION.md)의 KG128 재측정 및 전체 모듈 결과를 사용한다.

| L | Triton runtime heuristic | Triton 공통 config | CuTe 공통 config | CuTe 지연 증가 |
|---|---:|---:|---:|---:|
| 384 | 180.757 us | 154.395 us | 181.976 us | 17.9% |
| 768 | 716.920 us | 585.045 us | 652.276 us | 11.5% |

출력은 이 비교에서 Triton과 일치한다. 공통 config에서 Triton/CuTe는
각각 162/154 registers이며 양쪽 모두 LOCAL/STACK 0, HGMMA를 사용한다.
Triton은 stage 진행 중 `wgmma.wait_group 1`, 마지막에 `0`을 사용하지만,
CuTe는 stage마다 `wait_group 0`과 CTA barrier로 모든 MMA 완료를 기다린다.
CuTe 출력도 scalar `STG.E.U16`이며 Triton의 vector `STG.E.128`과 다르다.
이 차이는 TMA 사용 여부만 비교해서 놓쳤던 pipeline/출력 구현의 개선 지점이다.

Nsight L384 공통 config에서 점유율은 18.28% / 18.31%로 거의 같다.
그러나 Triton/CuTe DRAM 처리량은 2.650 / 2.200 TB/s,
long-scoreboard stall은 7.255 / 9.585 cycles/issued instruction이다.
Fixed-latency wait도 1.137 / 1.737로 늘어난다.
GMMA stall도 0.0378 / 0.501로 늘어, CuTe의 매 stage MMA 완료 대기 구조와 일관된다.
반면 barrier stall 자체는 2.725 / 2.605로 CuTe가 작다.
따라서 F567의 낮은 점유율 설명을 B9+B10에 그대로 적용하거나,
정적 barrier 명령 개수만으로 병목을 확정해서는 안 된다.

자료: [실행 cubin 및 측정 결과](../../runs/trimul_cubin_audit_20260917/dual_bwd/results.json).
