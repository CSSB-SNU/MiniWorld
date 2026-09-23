# H100 TriMul 추가 최적화 — TMA 중첩과 레지스터 수명

Triton과 같은 융합 경계·수식·저장값·CSV config 공간을 유지한 추가 구현이다.
명시적인 TMA/WGMMA를 사용하며, 개발 checkout은
`runs/trimul_sm90_parity_20260917/engine`이다. 검증된 커밋은 `0f2d455b`다.
운영 학습 설치본은 변경하지 않았다.

목표는 먼저 커널별 Triton 대비 **1.15×**, 최종적으로 전체 모듈 **1.15×**다.
사용자 지시에 따라 L128 성능은 목표에서 제외했다.

## 검증된 체크포인트 결과

BF16, H100 80GB, CUDA graph 반복 측정. 양쪽 모두 탐색에서 빠르게 측정된 config를
사용했다. 아래는 전체 선언 공간의 전역 최적을 주장하는 표가 아니다.

| 커널 | L | Triton µs | CuTe µs | 속도비 | 1.15× 목표 |
|---|---:|---:|---:|---:|---|
| F2 입력 projection + gate |384|220.238|183.078|**1.203×**|도달|
| F2 입력 projection + gate |768|876.068|737.801|**1.187×**|도달|
| F567 출력 projection + gate + dropout + residual |384|105.586|95.373|1.107×|미달|
| F567 출력 projection + gate + dropout + residual |768|410.061|360.106|1.139×|미달|
| B9+B10 입력 gradient의 두 GEMM |384|141.449|132.717|1.066×|미달|
| B9+B10 입력 gradient의 두 GEMM |768|534.529|510.572|1.047×|미달|

F567 L768의 이전 교대 측정은 Triton402.596µs / CuTe362.478µs였다.
반복 실행 간 차이를 고려하면 **약1.11–1.14×**이며, 15% 달성으로 취급하지 않는다.
F2의20.3% 속도 향상은 시간 감소로는16.9%다.

### 전체 양방향 TriMul 학습

공식 벤치 모듈 셋업, B1/D128, 비영 가중치, BF16, static compile + manual CUDA graph,
FWD+BWD, optimizer 제외, dropout=0, 12라운드 교대 측정.
비영 dropout과 residual 정확도는 별도로 검증했다.

| L | Triton ms | H100 세 커널 ms | 속도비 |
|---|---:|---:|---:|
|384|1.608288|1.557488|**1.033×**|
|768|6.285792|6.113424|**1.028×**|

**전체 15% 목표에는 아직 도달하지 않았다.** F2 단독의20% 개선을 전체 학습20%로
바꾸어 표현하지 않는다. 공통 LayerNorm·gradient packing·contraction/weight-gradient
GEMM은 같은 경로를 사용한다. 세 비교 대상 외의 공통 커널은 동일 cache/heuristic을
사용했으며, 전체 native cache의 튜닝 완료를 뜻하지 않는다.

## 채택한 변경

### F2: TMA 전송과 연산을 겹치고 BF16 교환 비용 감소

- Operand A/B와 output의 shared-memory 영역을 분리했다. 현재 output 계산 중 다음
  channel의 B를 읽고, 다음 GEMM 중 이전 output을 저장한다.
- K가 stage ring에 들어가는 경우 A를 재사용한다. 그렇지 않은 경우에도 기존 ring과
  barrier phase를 유지하며 다시 채운다.
- Mask를 CTA당 한 번 읽는다. BF16 값 두 개를 Uint32 하나로 묶어 warp 내부에서
  교환하므로 불필요한 FP32 변환과 shuffle이 줄었다.
- 저장 완료·WGMMA 완료를 확인하는 동기화는 유지했다. Shared-memory 한도는 별도의
  A/B/output 공간을 모두 포함하며, 학습·추론의 saved-preactivation 차이도 반영한다.

같은 config의 구현 비교는211.823→190.641→182.556µs였다
(기존 → 전송 중첩·mask 재사용 → packed shuffle).
실제 cubin의 SHFL64→32, F2F64→32이며128 registers, spill0이다.

NCU에서 기존 CuTe와 새 CuTe의 DRAM 트래픽과 점유율은 거의 같았다.
Long-scoreboard stall fraction은33.73%→22.89%였다. 즉, HBM 복사량 감소보다는
전송 중첩과 명령 감소가 이번 개선의 근거다. Triton과 비교한 L2 트래픽은
1,433.5→967.2MB였다. 이 카운터를 DRAM bytes와 혼동하지 않는다.

### F567: 작은 묶음으로 출력 계산

전체 residual/dropout/출력 fragment를 동시에 레지스터에 유지하지 않고,
matrix-copy 단위로 읽고 계산한다. 사용을 마친 A/B shared ring에 projection·gate·출력을
서로 겹치지 않게 배치한다. Residual tile은 읽기 전용으로 유지한다.
공간이 부족한 config는 기존 epilogue를 사용한다.

추가 global 중간 버퍼나 shared allocation은 없다. L384 winning config의 registers는
128→92, 점유율은23.98%→30.53%다. BF16 projection·gate 저장값, full-range sigmoid,
dropout·residual 수식은 유지했다.

Projection TMA를 기다리는 동안 gate accumulator 자체를 FP32 sigmoid(BF16(logit))로
덮어써 sigmoid 계산을 앞당겼다. 추가 fragment 없이 같은 rounding을 유지한다.
N128 config의 registers는168→163이며 spill은 없다.

### B9+B10: gate stage를 사용한 직후 다음 입력을 미리 읽기

Gate reduction이 ring에 들어가면 비어 있는 stage와 사용을 마친 gate stage에
front GEMM 입력을 미리 읽는다. Front는 그에 맞춘 ring 위치에서 시작한다.
긴 gate reduction은 일반 refill 경로를 유지한다. 추가 shared/HBM 공간은 없다.

같은 L384 config의 기존CuTe135.384→132.717µs였다.
Cold-cache 비교에서도141.920→139.008µs로 개선됐다.

## 채택하지 않은 실험

- F2의 추가 CTA barrier 제거는 유의미한 이득이 없어 유지하지 않았다.
- F567 output의 단순 동시 저장, 직접 global store, 조기 gate 변환, 여러 register
  packing·tile 변형은 최선 설정보다 느리거나 이득이 불분명했다.
- B9+B10의 gate shared staging은 registers를 낮췄지만 추가 shared 사용·복사 비용으로
  느려졌다. 명시적인 packed-gate assembly는137→105 registers로 줄었지만, 열 개 주변
  config와7라운드 교대 측정 후132–133µs 범위에서 약0.15% 차이여서 채택하지 않았다.
- Register cap은 spill을 발생시켜 제외했다. Warm-cache에만 의존하는 cache policy도
  채택하지 않았다.
- F567의 gate-only 및 resident-weight TMA multicast는 collective/leader-only 양쪽을
  시험했으나 더 느렸다. 측정한 aggregate L2 트래픽도680.5→746.6MB로 증가했다.
  단순히 multicast를 사용하면 L2/HBM traffic이 줄어든다고 가정하지 않았다.

- F567은 GROUP_M에 해당하는 여러 M tile에서 두 weight를 공유 메모리에 유지하는
  구현도 시험했다. L2 bytes는680.49→530.01MB로 줄었지만 shared 사용량이
 41,984→74,752B, 점유율이30.48%→18.06%로 변했다. L384 실행시간은
  약95→110µs로 늘어 채택하지 않았다. 트래픽 감소 자체가 가속을 보장하지 않는다.

## 검증과 남은 범위

- F2:41 kernel checks 및 native storage checks2건 통과.
- B9+B10:36건 통과. F567 최신 production:27건 통과.
- 선택 config sanitizer: F2 36건, B9+B10 36건, F567 27건에서 memcheck/racecheck 오류0.
- L384/L768 static compile에서 output과 모든 gradient를 Triton과 비교했다.
  구멍 있는 mask, 고정 비영 dropout, 비영 가중치 사용. 최대 상대L2 **1.44e-6**.
  Zero scale에서 residual 출력·입력 gradient identity와 parameter gradient0을 확인했다.
- Registry/native 검사에서1145건 통과. 두 문서 집계 오류를 수정하고 관련13건을
  다시 통과했다. 별도 cache-freshness 검사는 기존 LayerNorm/Transition cache10개에서
  실패하며, 변경 전600c8c4c의 독립 checkout에서도 같은 실패를 재현했다.
  이 작업에서 cache 파일을 삭제하거나 실패 검사를 무력화하지 않았다.
- Config 축은 F2 864, F567 3072, B9+B10 1152개 선언을 그대로 읽는다.
  Hardware/layout 불가능 후보는 명시적으로 제외한다. 전체 새 grid의 cache 빌드 및
  모든 config의 sanitizer 실행 완료를 주장하지 않는다.

## 근거

- [최종 source SHA-256·검증·모듈 결과](../../runs/trimul_sm90_15pct_20260917/module/final-evidence.json)

- [F2 상세·96 CuTe / 72 Triton 탐색·cubin·NCU](../../runs/trimul_sm90_15pct_20260917/front/REPORT.md)
- [F567 실험·검증](../../runs/trimul_sm90_15pct_20260917/f567/REPORT.md)
- [B9+B10 실험·검증](../../runs/trimul_sm90_15pct_20260917/dual_bwd/REPORT.md)
- [L384 전체 모듈·config·12라운드](../../runs/trimul_sm90_15pct_20260917/module/benchmark-final-L384.json)
- [L768 전체 모듈·config·12라운드](../../runs/trimul_sm90_15pct_20260917/module/benchmark-final-L768.json)
- [L384 전체 gradient](../../runs/trimul_sm90_15pct_20260917/module/validation-final-L384.json),
  [L768 전체 gradient](../../runs/trimul_sm90_15pct_20260917/module/validation-final-L768.json)
- [기존 cache 실패의 baseline 재현](../../runs/trimul_sm90_15pct_20260917/module/baseline-cache-check.log)
- [이전600c8c4c 보고서](OPTIMIZATION.md)
