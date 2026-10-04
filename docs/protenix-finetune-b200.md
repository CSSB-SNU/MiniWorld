# Protenix 파인튜닝 스텝, B200 엔진 연결 결과

2026-10-04, B200 한 장(sm_100a, 1000 W 상한), torch 2.13.0+cu129, miniworld-engine `50916071`. 하네스와 재현 방법은
`benchmarks/protenix_finetune/README.md`에 있다.

## 조건

- Protenix v1 base(`protenix_base_default_v1.0.0`), 파인튜닝 설정(`finetune_demo.sh`): 7pzb를 앞 384 토큰으로 자른 샘플(원자 3000),
  MSA 깊이 1024 고정, 확산 학습 48샘플, mini-rollout 20스텝, confidence 헤드, distogram.
- 학습 의미는 그대로다: Protenix 손실 전체, grad clip 10, Adam(0.9, 0.95), EMA 0.999, fp32 마스터 가중치. recycle 수는 학습에서
  1~4 중 뽑히므로 1과 4를 잰다.
- 시간은 스텝 전체(순전파, 손실, 역전파, 옵티마이저, EMA)다. 엔진 경로는 CUDA 그래프 하나의 재생 시간, 원래 트레이너는 eager
  스텝 시간이다.

## 결과

| 구성 | recycle 1 | recycle 4 | 평균 | 최대 메모리 |
|---|---|---|---|---|
| Protenix 원래 트레이너(엔진 없음, cuEquivariance, Protenix 기본 활성 체크포인팅) | 4.03 s | 4.62 s | 4.32 s | 15.2 GiB |
| 엔진 B200 경로, 스텝 전체 CUDA 그래프, 체크포인팅 끔(`settings.sh`) | 314 ms | 453 ms | 384 ms | 135.1 GiB |

평균 약 11.3배다. 원래 트레이너에서 체크포인팅을 끄면 B200의 178 GiB로도 메모리가 모자라 돌지 않는다(recycle 1, 4 모두 OOM).
그래서 원래 트레이너는 실행 가능한 유일한 설정인 Protenix 기본값으로 잰 것이다. 엔진 경로는 체크포인팅 없이 메모리를 더 쓰고
그만큼 재계산을 하지 않는다.

정확성(`check_whole_graph.py`, 7r6r, 같은 상태와 같은 시드, 난수 고정): 그래프 재생과 eager의 차이는 손실 3.58225 대 3.58237,
기울기 2.2e-3, 마스터 가중치 갱신 2.2e-3, EMA 갱신 2.9e-4로, eager끼리 두 번 돌린 차이(2.3e-3, 2.1e-3, 2.8e-4)와 같은 수준이다.

## 무엇이 빠르게 했나

| 부분 | 내용 |
|---|---|
| 엔진 커널 | pair 스택(TriMul, TriAttn, Transition, APB), 템플릿 블록(은닉 폭 2배 TriMul 포함), MSA(OPM, PWA), token DiT(학습 TF32, rollout은 블록 24개를 한 러너로), atom 트랜스포머(창 32 x 128) |
| 그래프 | 스텝 전체를 CUDA 그래프 하나로. 호스트 동기화를 모두 걷어냈다(MSA 깊이, 회전, 노이즈 스케줄, 손실의 불리언 인덱싱과 SVD) |
| 손실 | 그래프 안전 Protenix 손실. smooth LDDT는 순전파와 기울기를 한 커널로(17.5 ms에서 0.58 ms), 손실 전체 41 ms에서 2.3 ms |
| 옵티마이저 | fp32 마스터, clip, Adam, EMA를 세 번의 발사로(9.75 ms에서 3.6 ms) |
| 겹치기 | mini-rollout을 두 번째 스트림에서 확산 학습과 겹쳐 돌린다(recycle 1에서 약 20 ms) |
| 설정 | 확산 샘플 48개 한 묶음(Protenix 기본은 4개씩), sparse smooth LDDT, 체크포인팅 끔 |

## 이 작업에서 찾아 엔진에서 고친 것

| 엔진 커밋 | 내용 |
|---|---|
| `8f337a24` | 은닉 폭이 pair 폭의 2배인 한 방향 TriMul(템플릿 블록, 64에서 128)을 B200 커널로. cuEquivariance 대비 추론 2.2~3.1배, 학습 2.1~4.1배 |
| `c9104b05` | 가중치 pack 캐시를 그래프 캡처 단위로. 이전에는 캡처 뒤 가중치가 바뀌어도 재생이 캡처 시점 가중치로 계산했다(token DiT, bias-only DiT, 넓은 폭 TriAttn, APB) |
| `0cbda66c` | token DiT 어텐션 커널의 P 배리어를 버퍼별로. softmax가 MMA 워프보다 두 단계 앞서면 배리어 parity가 원위치로 돌아와 영원히 기다렸다. 다른 스트림이 동시에 돌 때만 드러났다 |
| `50916071` | Triton 어텐션 memory-efficient 경로의 작은 헤드(8) 패딩 |

## 정밀도

- pair 트랙은 엔진 B200 경로에서 bf16이다(Protenix도 autocast bf16).
- 확산 모듈은 Protenix처럼 fp32다. token DiT는 TF32 텐서코어로 돈다.
- atom 트랜스포머만 bf16으로 바뀐다(엔진 창 커널이 bf16).
- 정밀도를 더 바꾸는 선택지(`--enable_tf32 true`, token DiT bf16)는 기본에 넣지 않았다.

## 남은 여지

시간의 대부분은 역전파의 엔진 커널이다. TriAttn 역전파는 하한의 약 2배이고, token DiT 학습은 SoL 52~57%다. 하네스 쪽에서 더 줄일 큰
몫은 남아 있지 않다.
