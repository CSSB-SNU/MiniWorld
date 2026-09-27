기존 Triton 대비 검증된 비교 결과. 시간 단위는 ms이며, 배율은 기존 / 우리 커널이다. F+B는 직접 측정했다.

| D | L | 기존 BWD → 우리 BWD | BWD 배율 | 기존 F+B → 우리 F+B | F+B 배율 | 1.5배 충족 |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 384 | 1.089 → 0.715 | 1.5232× | 1.634 → 1.018 | 1.6061× | 충족 |
| 128 | 768 | 4.342 → 2.874 | 1.5106× | 6.539 → 4.075 | 1.6047× | 충족 |
| 256 | 384 | 2.220 → 1.751 | 1.2683× | 3.546 → 2.621 | 1.3531× | 미달 |
| 256 | 768 | 12.112 → 7.128 | 1.6992× | 17.463 → 10.506 | 1.6621× | 충족 |
| 384 | 384 | 4.016 → 2.886 | 1.3914× | 6.370 → 4.585 | 1.3895× | 미달 |
| 384 | 768 | 18.892 → 12.333 | 1.5318× | 28.820 → 19.096 | 1.5092× | 충족 |
| 512 | 384 | 5.900 → 4.474 | 1.3187× | 9.506 → 7.048 | 1.3487× | 미달 |
| 512 | 768 | 28.003 → 18.621 | 1.5038× | 43.382 → 28.665 | 1.5134× | 충족 |

전체 목표: 미달. 최적화 진행 중.

D128은 기존 검증 결과를 비교 기준으로 포함했다. 엔진 dispatch는 변경하지 않았다.

- D128 L384: [baseline-D128-D256-L384-17324.json](baseline-D128-D256-L384-17324.json)
- D128 L768: [baseline-D128-D256-L768-17323.json](baseline-D128-D256-L768-17323.json)
- D256 L384: [result-d256-pool-checkpoint-vs-triton-D256-L384-19254.json](result-d256-pool-checkpoint-vs-triton-D256-L384-19254.json)
- D256 L768: [result-d256-pool-checkpoint-vs-triton-D256-L768-19249.json](result-d256-pool-checkpoint-vs-triton-D256-L768-19249.json)
- D384 L384: [result-wide-checkpoint23-vs-triton-D384-L384-18910.json](result-wide-checkpoint23-vs-triton-D384-L384-18910.json)
- D384 L768: [result-wide-checkpoint23-vs-triton-D384-L768-18911.json](result-wide-checkpoint23-vs-triton-D384-L768-18911.json)
- D512 L384: [result-wide-checkpoint24-vs-triton-D512-L384-19391.json](result-wide-checkpoint24-vs-triton-D512-L384-19391.json)
- D512 L768: [result-wide-checkpoint24-vs-triton-D512-L768-19353.json](result-wide-checkpoint24-vs-triton-D512-L768-19353.json)
