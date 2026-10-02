# benchmarks

MiniWorld 모델 단위의 속도·정확도 측정이다. 엔진 커널의 공식 측정은 `miniworld-engine`의 `benchmarks/`가 맡고, 여기는 커널이
모델 안에서 어떻게 이어지는지(임베더, 템플릿, phase 2 학습 스텝)를 잰다. 예외가 `engine_ops/`다(아래). 어느 GPU에서든 같은 명령으로 돌고, 어디가 느린지
GPU별로 비교하는 용도다.

## 규칙

- **학습이 실제로 도는 경로를 잰다.** `torch.compile(dynamic=False)`로 컴파일한 모듈, 학습이 쓰는 dtype(bf16 모듈)과 matmul 정밀도
  (`medium`), 정상 상태, CUDA 그래프 한 번의 replay. 컴파일하지 않은 PyTorch는 발사 병목이라 기준으로 쓰지 않는다.
- **비교는 한 프로세스 안, 또는 머리줄이 같은 실행끼리만.** 같은 코드도 프로세스마다 ±3% 흔들린다(전력 상한, 메모리 배치).
  각 스크립트는 첫 줄에 `HEADER [...]`로 카드, torch/triton 버전, 엔진 설정을 찍는다.
- **엔진이 모르는 설정은 오류다.** `--engine-setting KEY=VALUE`는 `miniworld_engine.settings`에 없는 필드면 `TypeError`로 멈춘다.
  엔진이 그 스위치를 가지기 전의 결과를 새 스위치의 결과로 오해하지 않기 위해서다.
- **측정 결과는 여기 두지 않는다.** 표와 해석은 `docs/`에 적는다(예: `docs/phase2-b200-training-speed.md`). 스크립트의 docstring에는
  기준값 한 줄만 남긴다.

## 구성

| 폴더 | 재는 것 | 실행 |
|---|---|---|
| `phase2_step/` | phase 2 마이크로 스텝(recycle별, 랜덤 recycle, 그래프, 커널 배선) | `python -m benchmarks.phase2_step.bench_step --config configs/miniworld/phase2a_diffusion_v200.yaml --recycles 1,2,3,4` |
| `embedder/` | 입력 임베더 forward+backward 그래프, 커널 계열별 시간 | `python -m benchmarks.embedder.ladder` |
| `template_embedder/` | 템플릿 배치·투영 융합의 속도와 정확도, 컴파일 점검 | `python -m benchmarks.template_embedder.variants` |
| `engine_ops/` | 엔진 연산 단위: `token_pair_init`(융합 대 dense 기준), `trimul_batch`(B 샘플 한 번 대 B번) | `python -m benchmarks.engine_ops.token_pair_init` |
| `common.py` | 그래프 시간 재기, 커널 계열 분류, 머리줄, 엔진 설정 적용 | |

저장소 루트에서 `python -m`으로 실행한다. 모든 스크립트는 `--help`로 인자를 보여 준다.

## 다른 GPU나 엔진 버전과 비교하기

1. 같은 명령을 두 환경에서 돌리고 `HEADER` 줄이 의도한 차이(카드, 엔진 설정)만 다른지 확인한다.
2. 엔진 설정 하나를 바꿔 보려면 `--engine-setting b200_engine_triton=True`처럼 준다. 다른 엔진 버전은 `PYTHONPATH`로 그 엔진의
   `src`를 앞에 둔다.
3. 어느 커널이 병목인지는 `phase2_step/wiring_audit.py`(모듈 호출 수와 엔진 연산 수, 계열별 GPU 시간)와
   `embedder/ladder.py`(가장 큰 커널 목록)가 알려 준다.

## engine_ops가 여기 있는 이유

엔진의 `benchmarks/`는 타깃마다 `configs/bench.yaml`과 `bench_kernel_<이름>` 함수를 공용 러너(`runners/bench.py`)에 등록하는
방식만 허용하고, 새 커널 타깃은 레지스트리 행과 빌드 케이스까지 요구한다(`tests/layout/test_bench_target_vocabulary.py`).
`token_pair_init`과 배치 TriMul은 아직 그 타깃이 없어서, 엔진 규칙을 어기지 않으려고 호출 인터페이스만 쓰는 이 두 스크립트를
임시로 여기에 둔다. 엔진에 타깃이 생기면 이 폴더는 지운다.

## 한계

- 합성 배치 기준이다(`_build_precompile_batch`). 실제 데이터의 길이 분포는 반영하지 않는다.
- 기준값은 B200 한 곳에서만 쟀다. 다른 GPU의 기준값은 그 GPU에서 처음 재야 한다.
