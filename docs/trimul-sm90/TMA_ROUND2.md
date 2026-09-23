# H100 TriMul 추가 최적화 2차

검증된 커밋은 `8c7d8b39`다. 기준은 `0f2d455b`이며, F2와 B9+B10은 유지하고 F567을 추가 개선했다.
Triton과 같은 융합·수식·BF16 저장값·dropout·residual·CSV 설정 축을 유지한다.
L128은 성능 목표에서 제외했다. 개발 checkout에서 검증한 결과다.

## F567 결과

BF16, H100, CUDA graph, 독립 입력 할당2회 × 교대 측정5라운드의 중앙값.
이전에 GROUP_M4로 입력 할당3회도 확인했고, GROUP_M1/2/4/8을 별도로 비교했다.

| L | 이전 CuTe µs | 새 CuTe µs | 추가 가속 | 가장 빠른 측정 Triton µs | Triton 대비 |
|---|---:|---:|---:|---:|---:|
|384|95.236|93.231|1.022×|105.510|1.132×|
|768|362.116|355.706|1.018×|409.107|약1.150×|

**L384는 15% 목표 미달이다. L768은 개별 할당에서1.1496~1.1506×로,
15%를 안정적으로 넘겼다고 판단하지 않는다.** 더 느린 동일-config Triton을
분모로 사용하지 않았다. F2의 이전 검증 결과는 L3841.203× / L7681.187×다.

측정 후보는 두 길이 모두 `M64 N64 K64 GROUP_M1 warps4 stages2`다.
이 값은 benchmark manifest에 있으며 production의 길이별 하드코딩이 아니다.
전체 native cache 또는 전체 후보 공간의 튜닝 완료를 뜻하지 않는다.

## 무엇을 바꿨나

1. **Dropout scale을 TMA로 적재.** 사용이 끝난 operand shared ring에 네 번째
   출력 크기 타일이 들어갈 때만 그 공간을 사용한다. 기존 P/G/Y와 겹치지 않으며
   shared allocation은 늘지 않는다. 부족한 공간, row wrap, N tail은 기존
   vector/scalar 경로를 사용한다. 종료한 gate barrier의 다음 위상을 재사용한다.
2. **Projection 초기 입력을 L2에 미리 가져오기.** Gate 계산 중 기존 stage 수만큼
   `cute.prefetch` 힌트를 발행한다. 실제 TMA 전송·완료 확인은 그대로 수행한다.

### NCU 확인

최종 GROUP_M4 source 분석에서 dropout의 LSU global-load sectors는0으로 줄었다.
TMA로 읽는 데이터 자체가 없어졌다는 뜻은 아니다. DRAM read/write bytes는 거의 같다.

L768에서는 이전 N128 후보 대신 N64를 선택하면서 registers163→92,
shared66,560→41,984B, 점유율18.50→30.66%로 변했다. 이는 구현과 config 변경이
함께 반영된 결과다. L2 처리량은 peak의92.41%, DRAM은86.23%였다.
NCU 계측 시간은 위 성능 표의 CUDA graph 측정 시간과 구분했다.

## 채택하지 않은 실험

- F567에서 계속 앞서가는 projection prefetch와 작은 ring의 GROUP_M 반복은 느렸다.
- B9+B10의8-warp producer/consumer 분리는 가장 나은 후보도139.12µs로 기존132.38µs보다 느렸다.
- B9+B10의 세 가지 L2 prefetch는 느리거나 차이가 측정 변동 수준이었다.
- Front weight의 nonswizzled layout은 WGMMA descriptor 생성에서 거부됐다.

따라서 **B9+B10 production은 변경하지 않았다.** NCU의 source-PC 분석에서
주된 대기는 front TMA readiness였지만, 시도한 변경이 개선으로 이어지지는 않았다.

## 이전 전체 양방향 학습과 검증 — dropout OFF

아래 성능 표는 dropout=0 진단 기록이다. 실제 dropout=0.25 학습 비교는
[Dropout ON 재측정](DROPOUT_TRAINING.md)을 기준으로 한다.

공식 벤치 셋업 B1/D128/BF16, static compile + manual CUDA graph,
FWD+BWD, optimizer 제외, dropout0,12라운드 교대 측정.

| L | Triton ms | H100 세 커널 ms | 속도비 |
|---|---:|---:|---:|
|384|1.608384|1.557472|1.033×|
|768|6.281248|6.106904|1.029×|

**전체15% 목표도 미달이다.** 이전 전체 모듈 측정과의 차이가 작으므로,
이번 F567 추가 개선이 전체 모듈에 큰 가속을 만들었다고 해석하지 않는다.

- Production 회귀 검사35건 통과. 최종 커널35건의 memcheck/racecheck: 오류·경합0.
- 두 길이에서 output과 모든 gradient 검증 통과. 구멍 있는 mask, 비영 weight,
  고정 비영 dropout 포함. 최대 상대L2 `1.415e-6`.
- Zero dropout scale에서는 residual 출력/입력 gradient identity 및 parameter gradient0 확인.
- F2·B9+B10 source는 기준 커밋과 같다. 운영 학습 설치본은 변경하지 않았다.

## 정리 및 근거

사용자가 지정한 루트의 `cutlass`로 시작하는 PTX/cubin66개를 삭제했다.
새 덤프는 실험 디렉터리에 생성한다.

- [F567 상세·NCU·검증](../../runs/trimul_sm90_round2_20260917/f567/REPORT.md)
- [최종 후보·입력 할당별 수치](../../runs/trimul_sm90_round2_20260917/f567/final-results.json)
- [B9+B10 실험](../../runs/trimul_sm90_round2_20260917/dual_bwd/REPORT.md)
- [L384 모듈](../../runs/trimul_sm90_round2_20260917/module/benchmark-L384.json) · [L768 모듈](../../runs/trimul_sm90_round2_20260917/module/benchmark-L768.json)
- [최종 source·검증 기록](../../runs/trimul_sm90_round2_20260917/module/final-evidence.json)
- [삭제 목록](../../runs/trimul_sm90_round2_20260917/cleanup/root-dumps.json)
- [이전 최적화](TMA_PIPELINE.md) · [SVG 뷰어](../../tmp_kernel/trimul/TRIMUL_STATUS.html)
