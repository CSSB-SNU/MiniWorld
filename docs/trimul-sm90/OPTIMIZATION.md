# H100 양방향 TriMul 최적화 — 2026-09-17

Triton의 fusion 경계와 CSV config 축·값을 유지하며 F2, F567, B9+B10을
명시적인 TMA/WGMMA 구현으로 개선했다. 개발 checkout은
`runs/trimul_sm90_parity_20260917/engine`이며 운영 학습 설치본에는 적용하지 않았다.

## 결과

BF16, H100 80GB. 커널은 CUDA graph 측정이다. 각 행의 실제 입력과 후보 탐색 범위는
아래에 명시한다. 전체 CSV의 전역 최적이나 전체 native cache 빌드 완료를 뜻하지 않는다.

| 커널 | L | Triton (us) | CuTe (us) | Triton/CuTe |
|---|---:|---:|---:|---:|
| F2, 8-warp | 128 | 26.715 | 26.385 | 1.013× |
| F2, 4-warp | 384 | 221.739 | 213.979 | 1.036× |
| F2, 4-warp | 768 | 875.087 | 828.302 | 1.056× |
| F567 | 128 | 9.554 | 9.729 | 0.982× |
| F567 | 384 | 107.467 | 100.188 | 1.073× |
| F567 | 768 | 417.103 | 388.902 | 1.073× |
| B9+B10, 실제 모델 KG128 | 128 | 14.362 | 13.130 | 1.094× |
| B9+B10, 실제 모델 KG128 | 384 | 141.964 | 135.110 | 1.051× |
| B9+B10, 실제 모델 KG128 | 768 | 535.993 | 517.291 | 1.036× |

L128 F2는 거의 동률이며 F567은 소폭 느리다. F2 4-warp는 L128에서33.551us로
느렸고, 8-warp는 L384/L768에서214.927/853.572us였다. 두 경로를 config로 유지하며
sequence length별 선택을 코드에 하드코딩하지 않는다.

### 전체 모듈

공식 bidirectional TriMul 벤치 셋업, B1/L384/D128, 비영 가중치, BF16,
static compile + manual CUDA graph, FWD+BWD, optimizer 제외, dropout=0.
12라운드 교대 측정 중앙값. 비영 dropout 정확도는 별도로 검증한다.

| 교체 범위 | 학습 시간 (ms) | Triton 대비 속도비 |
|---|---:|---:|
| Triton | 1.615280 | 1.000× |
| F2 4-warp만 | 1.610768 | 1.003× |
| F567만 | 1.612624 | 1.002× |
| B9+B10만 | 1.610576 | 1.003× |
| 세 커널, F2 4-warp | 1.602736 | 1.008× |
| F2 8-warp만 | 1.611520 | 1.002× |
| 세 커널, F2 8-warp | 1.603888 | 1.007× |

**전체 학습 개선은 약0.8%다.** 커널 단독 개선을 전체 모듈의 큰 개선으로 바꾸어
표현하지 않는다. 12개 라운드 각각의 Triton/전체4-warp 시간비는1.0069~1.0096이었다.
비교한 세 Triton 커널도 최선 측정 config로 고정했고, 나머지 공통 커널은 동일한
cache/heuristic 경로를 사용했다. Native cache가 모두 튜닝됐다는 뜻은 아니다.

## F2: 실제 바이너리에서 찾은 원인과 구현

가장 빠르게 측정된 Triton 4-warp cubin도 WGMMA를 사용한다. 따라서 단순한
Tensor Core 사용 여부가 원인이 아니다. 이전 CuTe 8-warp는 입력 재읽기를 줄인 뒤에도
레지스터232개, dynamic shared memory229376B를 사용했고 warp 점유율은7.83%였다.
Triton은168개/73728B/18.25%였다. CuTe의 HBM 입력 읽기는 오히려 더 적었다.

- **8-warp:** M64 consumer 레지스터 예산을128로 줄이고 실제 tile/stage의 보수적인
  shared-memory 상한과 register/thread 한도에서 CTA 수를 계산한다. 실측 dynamic
  shared81920B, 점유율15.52%로 개선했다. 동일 바이너리에서 grid132→264 ablation이
  개선을 확인했다. 숫자264를 코드에 고정하지 않는다.
- **4-warp:** 한 CTA가 한 M tile에서 모든 channel tile을 처리한다. 기존 stage ring에
  K가 모두 들어가면 A를 channel 간 재사용한다. 그렇지 않으면 같은 ring을 재순환한다.
  B 저장소를 사용한 뒤 raw/gated output의 공유메모리로 재사용하고 STSM/TMA로
  최종 버퍼에 쓴다. gated fragment의 lane 배치는 shuffle로 변환한다.
- K stage별 사용 횟수로 barrier phase를 유지한다. 불필요한 중간 WGMMA 대기는
  stage를 덮어쓰지 않는 경우에만 제거하며, 하드웨어의 outstanding group 한도를 지킨다.
- 기존 Quack reciprocal의 FTZ 차이를 수정했다. 극소 sigmoid 및 큰 projection에서도
  Triton과 같은 full-range division을 사용한다. 마스크 앞 BF16 반올림과 saved preact는 유지한다.

실제4-warp cubin에 TMA load/store, HGMMA, STSM이 있으며 register128, stack/local0이다.
NCU에서 dynamic shared50176B,128threads,2304CTAs, warp 점유율23.52%를 확인했다.
Triton 대비 후보 탐색은 L384의18개 Triton 후보와8개 CuTe 4-warp 후보, 별도의8-warp
자원 ablation을 거쳤다. 최종 F2 표는 동일 입력의 Triton 및4/8-warp 후보를8라운드
순환 측정했다. 변경 전140개 CuTe 통과 기록을 새 구현 전체 탐색 완료로 재사용하지 않는다.

## F567: 같은 수식의 연산·메모리 비용 감소

- 비싼 rounded reciprocal 대신 Triton과 동일한 `div.full.f32` 사용.
- Residual을 LDSM으로 읽고, 정렬된 dropout scale을 BF16 두 값씩 읽는다.
  tail에는 마스킹된 scalar 경로를 유지한다.
- 두 독립 GEMM의 A/B stage ring을 WGMMA 완료와 CTA barrier 뒤에 재사용한다.

L384 matched config의 register162→128, dynamic shared74752→41984B,
점유율18.06→24.16%. 초기 CuTe 약151us가 약100us로 줄었다.
공통13개 config와 기존 Triton 후보를 양쪽에 적용하고 각자의 최선 측정값을 비교했다.
Projection/gate BF16 반올림, dropout, residual, backward 저장값은 유지한다.

## B9+B10: 중간값 수명과 최종 출력 저장

첫 GEMM 결과를 원래 수식대로 BF16으로 유지하여 FP32 accumulator 수명을 줄인다.
최종 출력은 사용을 끝낸 operand shared storage에 STSM으로 모아 TMA로 저장한다.
추가 global 중간 버퍼는 없다. 출력 tile이 더 크면 storage/feasibility를 함께 키운다.
TMA 정렬이 맞지 않는 N은 masked scalar store를 사용한다.

실제 D128/H256 모듈의 shape는 **KG128/KP1024/N128**이다. L384 공통30후보,
L128/768은 양쪽 상위 후보의 합집합5개를 비교했다. 같은 config의 실제 KG128에서
CuTe 구현 전후는169.047→135.426us였고 같은 실행의 Triton은141.430us였다.
이전 KG256 실험을 실제 모델 대표값과 섞지 않는다.

`wait_group1` overlap 및 K-loop unroll의 단독 실험은 느려져 채택하지 않았다.

## 검증과 적용 범위

- 최종 통합 GPU 회귀 **95건 통과**: F2 37건, F56726건, B9+B10 32건.
- Registry/축/launch/명명 검사414건 통과.
- 선택 config memcheck: F2 32건, F56726건, B9+B10 32건 오류0.
  동일한 F2 32건, F56726건, B9+B10 32건의 racecheck도 경합/오류/경고0.
- 모듈 L128 eager/static compile, L384 static compile에서 output과 모든 gradient 비교 통과.
  고정 비영 가중치, 구멍 있는 mask, 비영 dropout scale 사용. 최대 상대 L2 **7.98e-7**.
  Zero-scale에서 residual 출력/입력 gradient identity 및 parameter gradient0 확인.
- 8-warp 자원 정책은 cold/warm cache와 manual CUDA graph16조건에서 검증했다.
- 운영 설치본·진행 중인 학습은 변경하지 않았다. 전체 native cache 빌드 및 전체 선언
  config의 sanitizer 검증은 완료하지 않았다. 기능은 기존 명시적 SM90 선택 옵션으로 연결된다.

## 근거

- [최종 F2 시간](../../runs/trimul_sm90_opt_20260917/front/final_bench.json)
- [F2 fair cubin/SASS 및 NCU](../../runs/trimul_sm90_opt_20260917/front/fair-binary/REPORT.md)
- [8-warp 자원 ablation](../../runs/trimul_sm90_opt_20260917/front/resource-ablation/REPORT.md)
- [F567](../../runs/trimul_sm90_opt_20260917/f567/README.md), [B9+B10](../../runs/trimul_sm90_opt_20260917/dual_bwd/README.md)
- [최종 모듈 시간·config·전체 samples](../../runs/trimul_sm90_opt_20260917/module/benchmark-L384.json)
- [통합 회귀 로그](../../runs/trimul_sm90_opt_20260917/integrated-production.log)
- [소스의 연결·cache 계약](../../runs/trimul_sm90_parity_20260917/engine/docs/kernels/trimul-sm90-parity.md)

개발 브랜치 `perf/trimul-sm90-parity`, 커밋 `600c8c4c`.
[최종 소스 SHA-256·검증 요약](optimization-evidence.json).
