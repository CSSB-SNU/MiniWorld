# B200 환경과 실행 절차

두 노드(노드 1 `b200`, 노드 2 `b200_2`, 노드 2는 GPU 8장)는 같은 Lustre와 같은 `$W`(`/NHNHOME/WORKSPACE/26mohw002_A/psk6950`)를
공유한다. `$W/MiniWorld`, `$W/miniworld-engine`은 공유 체크아웃이므로 브랜치를 전환하지 않는다(빨리감기만).

## 환경

- `source $W/pixi_env.sh`가 캐시 경로(`XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TMPDIR`)를 정한다.
  `/tmp`는 noexec라 inductor·Triton 공유 객체를 올리지 못하므로 캐시는 `$W` 아래에 둔다.
- 실행 환경은 `$W/mwenv`(Python 3.10, torch 2.13.0+cu129)다. GitHub로 나가지 못하므로 엔진 핀을 받지 않고 로컬 체크아웃을 쓴다.
  `miniworld`은 `$W/MiniWorld`, `miniworld-engine`은 `$W/miniworld-engine/src`를 가리키는 editable 설치다
  (`site-packages/__editable__.miniworld_engine-2.2.0.pth`).
- `pixi run --frozen --manifest-path $W/mwenv/pixi.toml python ...`로 실행한다. `--frozen`이 아니면 pixi가 락을 다시 풀고,
  매니페스트에서 요구하지 않는 패키지를 환경에서 **지운다**.
- FlashAttention-4는 락에 넣지 않고 수동으로 설치한다(프리릴리스 CuTeDSL 휠):
  `uv pip install --python $W/mwenv/.pixi/envs/default/bin/python --prerelease=allow "flash-attn-4[cu12]"`.
- nvcc는 conda gcc 14 헤더를 해석하지 못하므로 `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"`가 필요하다.
  [scripts/b200/run_tests.sh](../../scripts/b200/run_tests.sh)가 이 설정을 포함한다.

- 엔진 체크아웃을 갱신(빨리감기)하거나 환경을 다시 만든 뒤에는 B200(sm_100) 커널을 미리 빌드한다. 노드 2에서, 예:
  `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++" CUDA_VISIBLE_DEVICES=4,5 pixi run --frozen --manifest-path $W/mwenv/pixi.toml python scripts/prebuild_engine.py --jobs 8`
  (`source $W/pixi_env.sh` 후, `$W/MiniWorld`에서). 결과 스탬프는 `engine-setup`이 확인한다.
  자세한 내용: [engine-prebuild.md](engine-prebuild.md).

## GPU

- 노드 1: GPU는 `gpuq`로 잡는다(`$W/gpuq/gpuq run -g <gpu> -n <이름> -- <명령>`).
- 노드 2: `gpuq`를 쓰지 않는다(상태 폴더를 두 노드가 공유해 락이 섞인다). `CUDA_VISIBLE_DEVICES`로 직접 지정한다.
  8장에 사용률용 더미가 돌고 있으므로 속도 수치는 시분할 영향을 받는다. 더미 프로세스는 건드리지 않는다.
- 다른 사람의 잡이 GPU를 쓰고 있으면 그 GPU를 쓰지 않는다(`nvidia-smi --query-compute-apps`로 확인).

## 자주 쓰는 것

| 하는 일 | 명령 |
|---|---|
| 린트·포맷·타입 검사 | `bash scripts/check.sh` |
| 테스트(B200) | `CUDA_VISIBLE_DEVICES=<gpu> bash scripts/b200/run_tests.sh` |
| phase 1 학습 스모크 | `NODE=2 GPUS=4,5,6,7 SMOKE=1 bash scripts/b200_phase1_v200.sh 1a` |
| 학습 스텝 배선 점검 | `scratch/phase1_wiring_audit.py`, `benchmarks/phase2_step/wiring_audit.py` |

## `pixi.lock` 다시 만들기

엔진 핀이 `ssh://git@github.com/...`이라 B200에서는 풀 수 없다. cssb에서 Slurm CPU 잡으로 한다:

```bash
git archive <MiniWorld 커밋> | tar -x -C <작업 디렉터리>      # 저장소를 건드리지 않는다
sbatch -p cpu -c 4 --mem=12G -t 60 --wrap "cd <작업 디렉터리> && pixi lock"
```

`CONDA_OVERRIDE_CUDA=12.9`를 주면 `__cuda` 가상 패키지 없이도 푼다. 결과는 `pixi lock --check`로 확인한다.
`pyproject.toml`의 `[project].dependencies`를 바꾸면 락을 다시 만들고, `$W/mwenv`는 `pixi install --manifest-path`로 맞춘다.

## 환경변수

커널·통합 스위치는 환경변수가 아니라 `miniworld_engine.settings`의 필드이고, 학습 스크립트의 옵션은 `train.*`이다
([docs/README.md](../README.md#configuration-instead-of-environment-variables)).
