# Anthropic 추론 커널 연결 및 H100 분석

**[로그인 전용 웹 현황판](https://miniworld-kernel-status.psk6950.chatgpt.site)** —
SSH/포트 포워딩 없이 브라우저에서 열고 사이트 소유자의 ChatGPT 계정으로 로그인한다.

**[인터랙티브 HTML 현황판](ANTHROPIC_STATUS.html)** — L384/L768 성능 비교,
커널 융합 배선, NCU 시간 비중, 연결 범위와 남은 작업을 한 페이지에서 확인한다.
단일 파일이므로 내려받아 오프라인으로 열 수 있다.

2026-09-19, node02 H100. Anthropic의 추론 개발을 계승한다는 방향과 출처를
engine README 및 개발 방향 문서에 명시했다. 아래는 추론 통합 당시 결과다. 이후 [TriMul K3 학습용 CUDA 확장](trimul/ANTHROPIC_TRIMUL_TRAINING.md)을 구현·검증했다.

**[전체 분석 보고서·연결 범위·원자료](../runs/trimul_sm90_parity_20260917/engine/docs/anthropic-h100-audit.md)**
· **[연결 API](../runs/trimul_sm90_parity_20260917/engine/docs/anthropic-integration.md)**

| 기존 Triton 대비, 추론 | L384 | L768 |
|---|---:|---:|
| TriMul outgoing 전체, C128 | 1.57× | 1.52× |
| TriangleAttention 전체, C128 | 1.28× | 1.44× |
| Transition C256 | 1.40× | 1.36× |
| Transition C128 | 거의 동률 | 거의 동률 |

Transition C384 후보는 기존보다 느리며 C512 후보는 시험 구성에서 미지원이다.
전체 일괄 교체 대신 명시적 `implementation="anthropic"` 및 row 선택으로 연결했다.

공통 추론 연산 16개 분류, 218개 후보/shape 중 178개 수치·CUDA Graph 검사 통과,
40개 명시적 지원범위 거절. NCU 72개 profile. LayerNorm은 HBM/L2 한계에 가깝지만
TriMul·attention·Transition 전체가 roofline에 도달한 것은 아니다.

모든 optimization kit 소스를 보존했으나 genomics/Pallas 등 모델 전용 경로의
실행·NCU, MiniWorld 전체 모델 정확도/성능, 학습 커널은 이번 완료 범위에 포함되지 않는다.
측정은 원본 source SHA를 검증한 로컬 engine checkout 기준이다.
