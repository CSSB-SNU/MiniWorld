# MiniWorld graphical abstract — review draft

2026-09-27. GPU 실행 없이 기존 코드와 측정 기록으로 만든 편집 가능한 초안이다.

## 연산별 독립 그림

`operators/index.html`에서 TriMul, Transition, OPM, PWA, LayerNormLinear,
TriangleAttention, LayerNorm, RMSNorm의 그림을 각각 볼 수 있다.
각 그림은 SVG / PNG / PDF로 제공하며 `operators/all-operators.pdf`는 8페이지 묶음이다.
CPU compute node에서 `python docs/graphical-abstract/render_operators.py`로 재생성한다.
TriMul은 기존 알고리즘 확대도를 재사용한다. LayerNormLinear의 CuTe 알고리즘과
portable Triton 측정값은 서로 다른 경로이며 그림에도 이를 명시했다.

## 전체 개요와 데이터

- `index.html`: 그림 탭, 연산/측정 구간 필터, 전후 시간과 출처를 확인하는 로컬 뷰어.
- `00-graphical-abstract.*`: 융합 경계 → 내부 처리 → 측정 효과를 연결한 메인 그림.
- `01-fusion-map.*`: 연산별 수학적 단계와 실제 구현 묶음.
- `02-trimul-algorithm.*`: outgoing/incoming contraction, HBM 경계, TMA/WGMMA 처리와 학습 저장값.
- `03-performance.*`: 학습 FWD/BWD/F+B 및 TriMul 폭 확장 비교.
- `miniworld-graphical-abstract.pdf`: 네 그림을 묶은 4페이지 PDF.
- `measurements.csv`, `evidence.json`: 계산에 사용한 시간, 정확한 배속, 원본 경로와 SHA-256.

## 그림을 읽는 기준

Fusion 지도의 왼쪽은 수학적 연산 순서다. baseline의 모든 작은 상자가 별도 GPU launch라는
주장은 하지 않는다. 오른쪽 실선은 확인한 main kernel 또는 library-call 그룹이며, 점선은
하나 이상의 launch가 가능한 stage다. 준비 작업, mask 생성, weight packing과 reduction은
모든 줄에서 전부 그린 것이 아니다. 특히 PWA의 pair-weight와 MSA-value는 독립 입력 branch다.

TriMul의 K1–contraction–K3 사이에는 A/B와 T의 HBM 경계가 남아 있다. 내부 시간축은
전송/계산 overlap을 설명하는 개념도이며 NCU trace나 GPU 처리량 측정이 아니다. K1/K3의
consumer 수는 tile config에 따라 달라지므로 고정된 2-consumer 구조로 주장하지 않는다.
Training lifetime 그림은 packaged D128의 saved tuple을 따른다. 그림의 입력/출력은 주요
데이터 경로만 보여주며 모든 가중치, gradient, 보조 reduction을 펼치지 않는다.

## 수치의 범위

- **T**: `qualified_speedups.json`의 기존 Triton 대 선택 checkpoint. D256+ backward의
  공용 dispatch 통합은 아직이며, source record의 `goal_complete=false`를 그대로 인정한다.
- **X**: wired Transition D128 module의 Triton residual 경로 대비 직접 측정 F+B.
- **O/P**: S1024/L384 OPM/PWA의 own-engine 대비 기록. 원본 log는 0.001 ms로 반올림됐다.
  OPM의 대표 점은 `save_o=1`이며 CSV에는 `save_o=0`도 포함한다. 저장/재계산 tradeoff를 숨기지 않는다.
- **N**: 2026-09-26 portable Triton Norm 변경 전후. CUDA fusion의 성능 증거가 아니다.
  대표 shape는 RMSNorm D128, LayerNorm D384, LNLinear 128→16으로 고정했다. 나머지
  측정 shape와 inference도 CSV/뷰어에 남긴다. 개선이 없는 값이나 regression도 제거하지 않는다.
- TriangleAttention은 이번에 확정한 동일 범위 비교가 없어 숫자를 넣지 않았다.
- 성능 그림의 FWD는 **학습** forward. inference는 별도 phase로 저장한다. BWD를 F+B에서
  FWD를 빼서 추정하지 않는다. F+B는 각 원본의 직접 측정값을 사용한다.
- 세 성능 패널은 같은 로그 축을 사용한다. 1× 근처는 소수 셋째 자리까지 표시한다.
- 서로 다른 시점·baseline의 연산별 기록이다. 하나의 v2.1 release sweep이나 전체 모델
  배속으로 해석하거나 평균내지 않는다. GPU 종류/드라이버까지 통일한 새 sweep은 후속 작업이다.

## 재생성

CPU compute node에서, matplotlib이 있는 환경으로 실행한다. 새 GPU 실험은 실행하지 않는다.

```sh
python docs/graphical-abstract/render.py
```

그림은 `evidence.json`만 있으면 생성된다. 원본 기록에서 bundle을 새로 수집할 때는
`python docs/graphical-abstract/collect_evidence.py --engine /path/to/miniworld-engine`을 먼저 실행한다.
원본 run 디렉터리는 이 저장소의 ignored artifact일 수 있다. HTML은 JSON을 내장하여
file://로 열어도 별도 서버나 network fetch 없이 동작한다.

SVG는 벡터와 실제 텍스트를 보존하고 PDF는 폰트를 내장한다. 영어 figure label은 논문/발표
재사용을 위한 선택이며 한국어 해설과 출처는 뷰어에서 제공한다.
