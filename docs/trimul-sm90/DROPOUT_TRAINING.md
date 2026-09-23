# Dropout ON 양방향 TriMul 학습 재측정

이전 전체 모듈 답변은 dropout=0 조건이었다. 실제 학습 설정의 **p=0.25**와
매 호출의 난수 생성 비용을 포함해 다시 비교했다. 엔진 개발 커밋 `8c7d8b39`는 그대로다.

## 결과

H100 80GB HBM3, B1, D=h=128, BF16 mixed, 고정 shape compile,
양방향 TriMul 모듈 1개 FWD+BWD, optimizer 제외.
H100 열은 기존 Triton 경로의 F2/F567/B9+B10만 CuTe로 교체한 결과다.
각 backend는 이전에 측정한 가장 빠른 config manifest를 사용한다.

| L | 실행 | Triton ms | H100 CuTe ms | 속도비 | 시간 감소 |
|---|---|---:|---:|---:|---:|
|384|compile + CUDA graph|1.626976|1.572272|1.0348×|+3.36%|
|384|compile, graph OFF|1.802504|2.250592|0.8009×|-24.86%|
|768|compile + CUDA graph|6.286352|6.118560|1.0274×|+2.67%|
|768|compile, graph OFF|6.290064|6.141488|1.0242×|+2.36%|

**CUDA graph에서 전체 이득은 약 3.5% / 2.7%이며, 목표 15%에는 미달한다.**
Graph OFF에서는 L384가 약 24.9% 느려졌고, L768은 약 2.4% 가속이다.
호출 경로의 CPU 처리도 포함되는 실행 결과이며, graph ON/OFF 차이를
단독 커널의 GPU 시간 차이로 해석하지 않는다. 전체 MiniWorld 학습 속도 측정은 아니다.

## 측정과 검증

- 공식 `bench_module_triangle_multiplication_bidirectional` 셋업을 재사용했다.
  6개 projection 가중치 모두 비영 초기화, 동일 state/input/dy/token mask 사용.
- Graph ON: 독립 fixture/graph 생성 2회 × 각각 12라운드 순서 교대 측정.
  표는 총 24개 라운드 중앙값이다. Graph OFF는 12라운드 교대 측정이다.
- Graph OFF는 각 backend 설정을 **warmup/측정 직전** 복원한다. 설정 변경은 시간 밖이다.
  CUDA profiler로 실제 Triton 3개 커널과 CuTe 3개 커널이 각각 실행됨을 확인했다.
  Graph도 동일 fixture/config로 재캡처한 별도 replay profiler에서 이를 확인했다.
  전역 설정을 복원하지 않았던 최초 임시 결과는 폐기했다.
- 공식 하네스는 stochastic dropout graph를 거부하고 replay 출력 동일성을 요구한다.
  따라서 엔진 하네스를 바꾸지 않고 실험용 `measured_result` adapter에서
  실제 CUDA graph를 캡처했다. fixture 설정의 `cudagraph=disabled`와 실제 측정 모드는 구분한다.
- 캡처 전 warmup/forward/backward는 같은 side stream에서 실행한다.
  캡처 때 `.grad=None`인 새 gradient 생성 경로를 기록하며, replay는 정적 gradient buffer를 덮어쓴다.
- **시간 측정 중 RNG reset이나 고정 dropout mask를 사용하지 않았다.**
  그래프 재생 간 CUDA RNG state, 출력, 입력 gradient가 모두 변했다.
- 검증 단계에서만 RNG를 되돌렸을 때 출력/입력 gradient는 정확히 재현됐고,
  모든 parameter gradient도 상대 L2 1e-5 이내였다. atomic LN reduction의 작은 변동만 있다.
  모든 11개 gradient 존재/유한값, buffer pointer 유지, 측정 종료 후 유한값을 확인했다.
- FP32 기준과의 공식 paired-dropout 정확도 검사는 별도 고정 마스크로 수행한 뒤,
  원래 production RNG 함수를 복원했음을 검증했다.

현재 native cache 전체와 전체 config grid를 재빌드한 것은 아니다.
공통 LayerNorm은 stale-cache 경고 후 runtime 후보 fallback을 사용했다.
선택한 F2/F567/B9+B10 config는 두 backend 모두 명시적으로 고정했고,
새 GPU 런타임/캐시 상태 전체에서의 최적성을 주장하지 않는다.
운영 MiniWorld 설치본과 실행 중인 학습 설정은 변경하지 않았다.

## 원본

- [후속 커널별 프로파일·가속 한계 분석](KERNEL_BREAKDOWN.md)

- [Graph L384](../../runs/trimul_sm90_dropout_20260917/graph/graph-L384.json) · [Graph L768](../../runs/trimul_sm90_dropout_20260917/graph/graph-L768.json)
- [Graph OFF L384](../../runs/trimul_sm90_dropout_20260917/module/benchmark-L384.json) · [Graph OFF L768](../../runs/trimul_sm90_dropout_20260917/module/benchmark-L768.json)
- [Graph replay 커널 확인 L384](../../runs/trimul_sm90_dropout_20260917/graph/provenance-L384.json) · [L768](../../runs/trimul_sm90_dropout_20260917/graph/provenance-L768.json)
- [Stochastic graph adapter](../../runs/trimul_sm90_dropout_20260917/graph/stochastic_graph.py)
- [Graph OFF benchmark](../../runs/trimul_sm90_dropout_20260917/module/benchmark.py)
- [이전 dropout OFF 결과](TMA_ROUND2.md) · [SVG 뷰어](../../tmp_kernel/trimul/TRIMUL_STATUS.html)
