# TriMul 학습 개발 현황

**이전 융합·저장 정책 유지 + Anthropic 파생 front/F567 CUDA + 기존 Triton/cuBLAS backward.**
`implementation="anthropic", anthropic_row="training_saved"`로 선택한다.

양방향 L384/768에서 이전 Triton 대비 forward 시간 7.3%/12.5%, 전체 학습 시간 2.8%/3.8% 감소.
14개 GPU 테스트 및 memory/sync/race 검사 통과. 초기 튜닝 결과이며 전체 모델 검증은 남아 있다.

[배선·성능·제한·원본 출처](../../runs/trimul_sm90_parity_20260917/engine/docs/anthropic-trimul-training.md)

[기존 웹 현황판](https://miniworld-kernel-status.psk6950.chatgpt.site/trimul.html#training-progress)
