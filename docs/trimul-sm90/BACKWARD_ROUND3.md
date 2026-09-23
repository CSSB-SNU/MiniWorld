# TriMul backward H100 추가 최적화 3차

**이번 후보에서는 기존 구현을 이기는 결과가 없어 production을 변경하지 않았다.**
개발 checkout은 `8c7d8b39`를 유지한다. H100 2장에서 병렬로 실험했고,
Triton의 융합·연산·BF16 저장 경계·중간 버퍼·CSV 컨피그 공간을 유지했다.
성능 비교는 L384/L768이며 L128은 제외했다.

## 현재 B9+B10 재측정

실제 양방향 D128 shape: M=L², KG128, KP1024, N128, BF16.
CUDA graph 교대 측정 5라운드 중앙값이며 컴파일 시간은 제외했다.
이 표는 **융합된 B9+B10 한 커널**의 시간이다. 전체 backward 시간은 아니다.
Triton 분모는 이전 탐색에서 가장 빨랐던 측정 컨피그를 다시 실행한 값이며,
이번에 전체 CSV 탐색을 완료했다는 뜻은 아니다.

| L | Triton µs | 유지한 CuTe µs | Triton 대비 | 새 벡터 저장 후보 µs |
|---|---:|---:|---:|---:|
|384|141.712|132.813|1.067×|134.325|
|768|535.432|509.581|1.051×|512.794|

**커널 1.15× 목표는 여전히 미달이다.** 후보가 기존보다 느려 반영하지 않았다.

## 시험한 변경과 제외 이유

### 1. 입력·가중치 TMA 버퍼의 깊이를 분리

큰 입력은 기존 단계 수만큼 미리 읽고, 주로 L2에서 읽는 가중치의 버퍼만
줄였다. 별도 완료 barrier, 공유 완료 barrier, BF16 gate register packing을
각각 또는 함께 시험했다. 새 컨피그 축이나 전역 중간 버퍼는 추가하지 않았다.

- 대표 L384 후보는 약142–144µs로 기존 약133µs보다 느렸다.
- Packed gate는 실제 register를137→115로 줄였지만 shared-memory 제한 때문에
  기대했던 네 블록 동시 실행을 얻지 못했다. 초기 자원 계산이 드라이버의
  블록별 shared-memory 예약량을 빠뜨렸음을 cubin/NCU로 확인했다.
- 가중치 버퍼를 한 단계로 줄인 후보는 측정 점유율이18.14→22.50%로 늘었지만
  L384가161.00µs로 악화했다. 가중치를 미리 읽는 간격이 짧아진 영향으로
  추정하지만, 이 counter만으로 전체 원인이 입증된 것은 아니다.
- L2 traffic은 줄었어도 DRAM read는 약341MB로 유지됐다. 점유율이나 한 계층의
  전송량 감소만으로 성능 개선을 판단할 수 없는 사례다.

### 2. 두 N 타일에 입력을 TMA multicast

N64 타일을 처리하는 두 CTA가 같은 입력을 multicast로 받도록 구현했다.
가중치는 각자 읽으며, 같은 cluster·동기화를 쓰는 일반 로드도 대조했다.

L384에서 기존 N128은131.89µs, 일반 N64는151.70µs였다. 매 단계 cluster 전체를
동기화하는 control은245.70µs였다. 재사용할 슬롯의 remote-empty barrier만
기다리는 multicast 후보로223.17µs까지 줄였지만 기존보다 크게 느렸다.
L384에서 제외했으므로 L768이나 production 안전성 검증을 완료한 것으로
취급하지 않는다. 이 결과는 시험한 구현에 대한 판단이며 모든 multicast
설계가 느리다는 결론이 아니다.

### 3. 출력 TMA store를128-bit vector store로 대체

기존 STSM 출력 재배치를 유지하고, 완전한 타일만 shared→register→global의
벡터 저장으로 바꿨다. 입력 TMA/WGMMA와 tail 경로는 유지했다.
실제 cubin에128-bit LDS/STG가 생성됐지만 register가137→154로 늘었고,
위 표처럼 두 길이 모두 느렸다. Spill은 없었다.

### 4. B7의 좌표 계산 제거

B7은 GLU 미분과 gradient packing을 수행하는 elementwise 커널이다.
채널×행 타일 grid로 바꿔 원소별64비트 나눗셈을 없앴다. 같은 컨피그 공간에서
기존·후보 각각 빠른 컨피그를 골라 비교했다.

| L | 기존 Triton µs | 후보 µs |
|---|---:|---:|
|384|247.727|247.699|
|768|1002.427|1002.469|

차이는 측정 변동 수준이다. 명령 수는 줄었지만 NCU에서 기존 경로가 이미
약754MB를3.03TB/s, 보고된 DRAM peak의90.42%로 처리했다. 같은 측정 전송량과
peak를 가정하면 낙관적인 시간 하한도 약225µs다. 1.15× 목표인215µs는 이
가정 아래에서 도달할 수 없다. 융합과 필수 버퍼를 유지하는 이번 조건에서는
TMA로 명령만 바꾸는 작업을 추가하지 않았다.

## 검증과 적용 상태

- 기록된 production-size 후보 출력은 Triton과 bitwise 일치했다.
- CPU 모델로 비대칭 ring23,760개, remote-empty 위상2,200개 조합을 확인했다.
  이는 GPU sanitizer 검증을 대신하지 않는다.
- 느린 후보를 제외했으므로 새 production sanitizer/전체 모듈 재측정은 하지 않았다.
- 엔진 작업 트리는 깨끗하며 B7·B9+B10 SHA-256이 시작 시점과 같다.
- 새로운 배선, runtime 옵션, cache 변경은 없다. 학습 설치본도 유지했다.
- 실험용 Slurm13236의 GPU2장을 반납했다. 루트 `cutlass*` 파일은0개다.

## 근거

- [최종 source·시간·적용 상태](../../runs/trimul_sm90_bwd_round3_20260917/final-evidence.json)
- [B9+B10 상세·NCU·cubin](../../runs/trimul_sm90_bwd_round3_20260917/dual/REPORT.md)
- [Multicast 상세](../../runs/trimul_sm90_bwd_round3_20260917/dual_multicast/REPORT.md)
- [B7 대역폭·기존/후보 비교](../../runs/trimul_sm90_bwd_round3_20260917/b7/REPORT.md)
- [현재 적용된 F2/F567/전체 모듈 결과](TMA_ROUND2.md)
