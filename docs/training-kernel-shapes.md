# 학습용 커널 길이 정책

2026-09-20: 학습용 튜닝·캐시 빌드는 아래 두 길이만 사용한다.

| 축 | 학습 | 추론 |
| --- | --- | --- |
| Token / pair / MSA token | **384, 768** | 기존 여러 버킷 유지 |
| Atom | **4096, 8192** | 기존 여러 버킷 유지 |

현재 학습 설정은 이미 phase-a에서384/4096, phase-b에서768/8192로
패딩한다. 이번 수정은 개발 중인 엔진의 학습 forward/backward 빌드 계획과
backward 전용 드라이버 목록에 이를 반영한다. 공유 forward 커널은 추론
범위를 유지한다. 실행 중인 학습 잡, 추론 패딩, 채널 차원은 변경하지 않았다.

- 엔진 상세: [training-shape-policy.md](../runs/trimul_sm90_parity_20260917/engine/docs/training-shape-policy.md)
- 수정 전/후 후보 목록: `runs/training_shape_policy_20260920/{before,after}.json`
- 검증 로그: `runs/training_shape_policy_20260920/tests.log` —559개 통과.
- 학습 모듈 후보6756→2188, 추론2350개와 공유 forward 드라이버2843개는 동일.
- 기존 커널 튜닝 캐시는 보존한다. 다음 정상 빌드에서 파생된 실행 계획만
  새 정책으로 생성한다. 이번에는 GPU 캐시 빌드 잡을 새로 걸지 않았다.

커널의 작은 입력 정확도 검사는 이 목록과 별개다. B1–B4의 L64 smoke
검사는 학습 버킷 추가를 의미하지 않는다.
