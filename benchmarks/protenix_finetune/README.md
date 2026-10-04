# protenix_finetune

Protenix v1 파인튜닝 한 스텝을 miniworld-engine의 B200 커널로 돌리고, 스텝 전체(순전파, Protenix 손실, 역전파, clip·Adam·EMA)를
CUDA 그래프 하나로 재는 하네스다. MiniWorld 모델이 아니라 외부 모델로 엔진 연결과 그래프 안전성을 검증하려고 만들었다.
측정 결과와 해석은 `docs/protenix-finetune-b200.md`에 있다.

## 준비

1. Protenix를 받아 이 디렉터리의 패치를 적용한다(기준 커밋 `2475421`).
   ```bash
   git clone https://github.com/bytedance/Protenix.git && cd Protenix && git checkout 2475421
   git apply /path/to/MiniWorld/benchmarks/protenix_finetune/protenix.patch
   ```
   패치 내용: 빠른 LayerNorm 확장이 현재 스트림에서 발사하게 한다(기본 스트림에서 발사하면 그래프 캡처가 그 커널을 놓친다).
   confidence 헤드가 미리 계산한 대표 원자 인덱스를 받게 한다(불리언 마스크 인덱싱은 호스트 동기화라 캡처할 수 없다).
   LayerNorm 확장은 빌드 디렉터리를 지우고 다시 빌드한다.
2. 체크포인트 `protenix_base_default_v1.0.0.pt`를 받는다.
3. 특징 파일을 만든다. Protenix 자체 환경에서 `featurize.py`를 돌린 뒤, 학습 크롭처럼 앞 384 토큰으로 자른다.
   ```bash
   PROTENIX_SRC=/path/to/Protenix python featurize.py          # PFX_FEATS/<이름>.pt
   python crop_feats.py 7pzb 384                               # PFX_FEATS/7pzb_c384.pt
   ```
4. 학습 환경(MiniWorld pixi 환경, miniworld-engine과 team-gm 포함)에 Protenix가 쓰는 패키지 중 빠진 것(absl, ml_collections)이 있으면
   한 디렉터리에 설치하고 `PROTENIX_EXTRA_SITE`로 넘긴다.

## 실행

```bash
export PROTENIX_SRC=/path/to/Protenix PROTENIX_CKPT=/path/to/protenix_base_default_v1.0.0.pt
export PFX_FEATS=/path/to/feats                # 기본값: 이 디렉터리의 feats/
export PROTENIX_EXTRA_SITE=/path/to/site       # 선택
source benchmarks/protenix_finetune/settings.sh
D=benchmarks/protenix_finetune
NCYC=1 PYTHONPATH=$D python $D/whole_graph.py 7pzb_c384           # recycle 1 (NCYC=4: recycle 4)
PFX_DET=1 NCYC=1 PYTHONPATH=$D python $D/check_whole_graph.py 7r6r  # 그래프 재생 대 eager 정확성
```

`whole_graph.py`는 eager 스텝을 재고 스텝 전체를 캡처한 뒤 재생 시간, 손실, 비유한 기울기, 최대 메모리를 찍는다.
`PFX_G4_STRESS=N`이면 N번 더 재생한다(동시 실행 정지 확인용). `check_whole_graph.py`는 같은 상태(파라미터, Adam 상태, EMA)와
같은 시드에서 eager 스텝과 그래프 재생의 손실, 기울기, 마스터 가중치 갱신, EMA 갱신을 비교한다. `PFX_DET=1`이면 모든 난수를
고정 패턴으로 바꾼다.

## 파일

| 파일 | 역할 |
|---|---|
| `settings.sh` | 가장 빠른 설정(스위치만, 경로 없음) |
| `pfx_common.py` | 설정과 모델 생성, 체크포인트 로드 |
| `step.py` | 학습 스텝: 트렁크(recycle, 마지막 사이클만 기울기), mini-rollout, confidence 헤드, 확산 학습, 손실, 역전파, 옵티마이저 |
| `patch_engine.py` | Protenix 블록을 엔진 모듈로 교체(pair 스택, 템플릿, MSA, token DiT, atom 트랜스포머) |
| `pfx_graphsafe.py` | 호스트 동기화 제거: 고정 MSA 깊이, GPU 회전, 토큰 수 명시 |
| `pfx_loss_gs.py` | 그래프 안전 Protenix 손실(샘플 고정 데이터 사전 계산, 강체 정렬은 Horn 쿼터니언) |
| `pfx_slddt_cuda.py` | smooth LDDT 순전파와 기울기를 한 번에 계산하는 CUDA 커널 |
| `pfx_pair_t.py` | TriAttn ending 방향의 pair 전치 커널 |
| `pfx_fused_opt.py` | clip, Adam, fp32 마스터에서 bf16 가중치로 복사, EMA를 세 번의 발사로 |
| `whole_graph.py`, `check_whole_graph.py` | 측정과 정확성 검사(위) |
| `featurize.py`, `crop_feats.py` | 특징 파일 만들기 |
| `protenix.patch` | Protenix 변경(위) |

## 주요 스위치

| 변수 | 뜻 |
|---|---|
| `NCYC` | recycle 수(Protenix 학습은 1~4에서 뽑는다) |
| `NSAMP` | 확산 학습 샘플 수(기본 48) |
| `PATCH` | 엔진으로 바꿀 부분(`pair,tokendit,msa,atom`); 비우면 Protenix 원래 모듈 |
| `PFX_TRAINER` | `fast`(그래프 가능 트레이너) 또는 `protenix`(Protenix 자체 루프: EMAWrapper, empty_cache, 호스트 NaN 검사) |
| `PFX_SIDE_CONF`, `PFX_SIDE_NOJOIN`, `PFX_CONF_LATE` | mini-rollout을 두 번째 스트림에서 돌리고 confidence 헤드는 주 역전파 뒤로 |
| `PFX_DIT_BF16` | token DiT를 bf16으로(정밀도 변경, 기본 끔) |
| `PFX_CKPT=1` | 활성 체크포인팅 켬(기본 끔) |

## 한계

- 배치 크기 1, MSA 깊이 고정(Protenix 학습은 사이클마다 깊이를 뽑는다), 정답 좌표는 합성(노이즈, 증강, 손실 입력용)이다.
  데이터 로더, 대칭 순열, 평가는 포함하지 않는다.
- B200(sm_100a)에서만 쟀다. 엔진의 B200 경로는 bf16 pair 트랙을 쓴다(Protenix는 autocast bf16). 확산 모듈은 Protenix처럼 fp32
  (token DiT는 TF32 텐서코어)이고, atom 트랜스포머만 bf16이다.
