<div align="center">

# Team-GM

### 간단하고 직관적인 구조 생성 모델 프레임워크

`Team-GM`은 **flow-matching**, **diffusion** 등 다양한 generative modeling을 쉽게 다루고, 공통된 프레임워크로 통합하기 위한 튜토리얼 프로젝트입니다. 누구나 자유롭게 Issue나 Pull Request를 통해 프로젝트에 기여할 수 있습니다.

</div>

## 📦 Installation
pip을 이용해 간편하게 설치할 수 있습니다:

```bash
pip install git+https://github.com/CSSB-SNU/team-gm.git
```

> [!Warning]
> Python >= 3.10 및 CUDA >= 12.0 환경만 지원


## 🛠️ Development
`Team-GM`은 [Pixi](https://pixi.sh/latest/)를 활용하여 효율적인 패키지 관리 및 개발 환경 구성을 지원합니다.

### Pixi 설치

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

### 프로젝트 환경 구성

```bash
pixi install --frozen
```

### 가상환경 실행

```bash
pixi shell
```

### 패키지 추가

```bash
pixi add torch numpy ...
```

### Pixi 환경 없이 실행 (예: Slurm)
일반적으로는 `pixi shell`로 가상환경을 실행할 수 있지만, Slurm 등과 같이 가상환경 없이 실행하려면 다음 명령어를 사용하세요:

```bash
pixi run python script.py
```

> [!Tip]
> - [pixi 튜토리얼](https://pixi.sh/latest/tutorials/python/#lets-get-started)
> - [conda와의 비교](https://pixi.sh/latest/switching_from/conda/)


## 🚀 Usage

### Training

Pre-training을 진행하려면 다음 명령어를 실행하세요:

```bash
python scripts/run_structure_flow.py train --config configs/baseline.yaml
```

특정 checkpoint에서 이어서 학습하려면:

```bash
python scripts/run_structure_flow.py train --resume_from_ckpt CKPT_PATH
```


**옵션:**

| 옵션 | 설명 |
| ---- | --- |
| `--config` | 사용할 config 파일 경로 |
| `--resume_from_ckpt` | 이어서 학습할 checkpoint 경로 |
| `-w` | W&B 로깅 활성화 (`wandb init`으로 entity, project 지정 가능) |
| `--ckpt_dir` | Checkpoint 저장 디텍토리 |


### Inference

학습된 모델로 inference를 진행하려면 다음 명령어를 사용하세요:

```bash
python scripts/run_structure_flow.py inference --ckpt CKPT_PATH --seq SEQUENCE
```

**옵션:**

| 옵션 | 설명 |
| ---- | --- |
| `--ckpt` | 사용할 checkpoint 경로 |
| `-s, --seq` | Query 단백질 서열 | 
| `--num_sample` | 생성할 샘플 수 |
| `--timesteps` | 샘플링 timestep 수 |
| `--out_dir` | 샘플 저장 디렉토리 |


### Automatic SLURM submission

자동으로 SLURM으로 job을 제출하려면 `--slurm` 플래그를 추가하세요:

```bash
python scripts/run_structure_flow.py train \
    --config configs/baseline.yaml \
    --slurm \
    --partition gpu \
    --gpus-per-node A100:4
```

**SLURM 옵션:**

| 옵션 | 설명 | 기본값 |
| ---- | --- | --- |
| `--slurm` | SLURM job으로 실행 | - |
| `--job-name` | Job 이름 | config 파일명 |
| `--partition` | 사용할 파티션 | - |
| `--mem` | 메모리 크기 | 32G |
| `--cpus-per-task` | Task당 CPU 수 | 8 |
| `--gpus-per-node` | GPU 설정 (예: A100:4) | - |
| `--ntasks-per-node` | 노드당 task 수 | 1 |
| `--time` | 시간 제한 (예: 24:00:00) | - |
| `--nodelist` | 특정 노드 지정 | - |

### Runtime Type Checking (Optional)

`Team-GM`은 `jaxtyping`과 `beartype`를 사용하여 런타임에 행렬의 shape와 dtype을 검사합니다. 이 기능을 활성화하면 개발 중 발생할 수 있는 데이터 관련 버그를 미리 잡는 데 도움이 됩니다.

타입 검사를 활성화하려면, 코드를 실행할 때 `SHOULD_TYPECHECK` 환경 변수를 `true`로 설정하세요:

```bash
SHOULD_TYPECHECK=true python script.py
``` 
