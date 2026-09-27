# MiniWorld 저장소와 개발 상태

2026-09-27: MiniWorld, miniworld-engine, team-gm의 로컬·원격 작업 브랜치를 **main으로 통합**했다.
main에 포함된 로컬 브랜치 63개와 원격 브랜치 7개를 삭제했다.
정리 대상 7개 저장소는 `main`을 사용하며, engine에는 `mpnn`과
`backup/mpnn-pre-rebase-20260913`도 로컬·원격에 유지했다.
삭제한 브랜치의 커밋은 main 이력에 남아 있고, 이름과 SHA는
[브랜치 정리 기록](docs/repository-audit-20260927/branch-cleanup.json)에 보존했다.

## 현재 작업 경로

| 저장소 | 개발 경로 | 브랜치 |
|---|---|---|
| MiniWorld | `/home/psk6950/MiniWorld` | `main` |
| miniworld-engine | `/home/psk6950/miniworld-engine` | `main` |
| team-gm | `MiniWorld/libs/team-gm` | `main` |

`MiniWorld/.engine-release-2.0.0`은 이전 engine `3026c6bc`를 보존한 호환용 detached worktree다.
현재 v2.1 main과는 다르며, 과거 성능 기준을 보존한다.
기존 harness의 경로를 유지하기 위해 남겼다. 새 engine 수정은 canonical main에서 한다.
삭제 대상 브랜치를 사용하던 과거 worktree 10개는 HEAD와 index를 유지한 채
detached HEAD로 전환했다. 파일은 그대로이며, 최신 실행 코드는 위 경로를 기준으로 한다.

MiniWorld와 team-gm의 engine pin은 모두 `afd54410a0bf59a204b6ca00af38909a684cf50f`이다.
`pixi.lock`과 `uv.lock`도 engine 2.1.0 및 같은 SHA로 맞췄다. engine의 의존성과 extras는
이전 pin과 동일함을 비교했고 다른 패키지 버전은 바꾸지 않았다.
`.gitmodules`의 team-gm 추적 브랜치는 main이며, 실제 재현 기준은 커밋된 gitlink다.
현재 설치된 Pixi 환경은 재설치하지 않았다. 소스/pin 정리와 설치 환경 갱신은 별개다.

## v2.1.0 튜닝 정책

engine의 유지보수 소스는 canonical main 하나에서 개발한다. 기본 cache build는 실제
module dispatch를 따르며 CUDA 경로가 선택되면 쓰이지 않는 Triton 대안을 강제 탐색하지 않는다.
112개 Triton 커널은 커널당 최대 32개, 총 2,981개의 기본 후보를 사용한다. 전역 공간,
smem 예측 및 기존 측정값의 유효성 검사는 유지한다. 학습 token 길이는 384/768,
추론의 기존 shape 공간은 유지한다. atom과 MPNN edge 길이는 별개다.

CUDA도 config space가 있다. 기존 native 튜너에 더해 packaged TriMul 추론의 K1/K3와
fused Transition D128의 CTA/DW replica 탐색을 연결했다. 모든 연구 CUDA 커널의 추론
공간이 통합 튜너로 덮였다는 뜻은 아니다. 고정/manual schedule은 별도로 명시했다.

```sh
miniworld-engine build all
miniworld-engine build all --mode train
miniworld-engine build all --backend native --mode eval
miniworld-engine build all grid --include-alternatives --gpus all
```

`build trunk/diffusion/mpnn`은 기존 per-op 진단 경로다. 기본 production 정책은 `build all`
또는 module 이름을 사용한다. config 기본 공간과 backend 대안 포함 여부는 독립된 선택이다.
대규모 반복 학습/추론에는 특히 backward 중간 텐서의 메모리 왕복을 줄이는 GPU별 fused
CUDA 구현을 권장한다. GPU마다 유리한 fusion이 달라 전체 F+B 측정으로 결정한다.

CPU 전체 job 19676: 3,967 passed / 16 failed / 279 skipped, GPU 1,002 deselected.
후속 검사로 새 실패 2개를 해결했다. v2.1 관련 164 passed (job 19680), registry 69 passed와
부모 설치 계약 4 passed (job 19681). 전체 재실행 결과로 합산하지 않는다.
남은 실패는 기존 13개와 v2.1 GPU release-manifest gate다. 새 GPU 성능·graph·sanitizer 검증,
모든 CUDA 계열의 추론 tuning coverage는 아직이다. v2.1 release tag는 만들지 않았다.

상세 정책과 검증: engine `docs/releases/2.1.0.md`, `2.1.0-cpu-validation.json`.
현재 설치 환경은 바꾸지 않았으며 source pin과 실행 환경 버전을 혼동하지 않는다.

## main에 들어간 내용

- MiniWorld: v1.3 distogram diffusion, v2 데이터/loss 일관성, SWA 연결, graph 검증 기록,
  ignored runs에서 선별했던 연구 소스·기록 6,664개, oldcode 재현 설정.
- engine: 최신 upstream main, D64 barrier, Triton Norm 최적화, opt-in native Norm,
  wide TriMul/Transition 연구, TokenDiT training core와 overlap 실험,
  TriangleAttention checkpoint18246 runtime, 원격 MPNN 구현·측정 기록.
- team-gm: 최신 main의 per-block checkpoint API와 기존 segmented API 호환,
  prepared pair bias API, valid-atom diffusion loss, SWA 연구 구현,
  FoldForge adapter와 inference bucket 확장.

현재 구현과 충돌하는 오래된 코드는 main 안의 `experiments/legacy_branches/`에 파일·provenance로
보존하고 branch 이력도 연결했다. BioMol/AF3 구형 API를 현재 API에 덮어쓰거나,
불완전한 publish worktree의 164개 삭제를 현재 코드에 적용하지 않았다.
engine의 이전 연구 runtime은 `experiments/legacy_h100_runtime/`, 오래된 tuning은
`experiments/transition_fused/records/cache_snapshot_20260927/`에서 확인한다.
DiT 실험 runner는 최신 `runner.py`와 이전 `runner_multistream.py`를 함께 보존했다.

[통합 refs 및 소스 검증](docs/repository-audit-20260927/main-verification.json) ·
[초기 보존 작업의 전체 작업본 표](docs/repository-audit-20260927/BRANCHES.md) ·
[초기 커밋별 파일](docs/repository-audit-20260927/commits.json)

## v2.1 변경 전 통합 검증 기록

GPU를 요청하거나 사용하지 않았다. CPU 검사는 node02에 `--gres=none`으로 할당했다.

| 검사 | 결과 | Slurm job |
|---|---|---|
| MiniWorld v1.3/v2·설치 검사 | 64 passed, 1 GPU skip | 19661 |
| team-gm 통합·공용 블록 | 34 passed, 282 skip | 19663 |
| engine CPU 전체 | 3,929 passed, 17 failed, 279 skip, GPU 1,002 deselected | 19662 |
| engine 검사기 호환성 후속 검사 | 10 passed; 위 실패 중 4개 해결 | 19664 |

team-gm skip은 외부 reference snapshot이나 CUDA가 필요한 검사다.
engine에는 **미해결 13개 실패**가 남아 있다. stale 파생 registry, A5000/A6000 LNLinear
cache identity, MPNN dropout cache coverage, 연구 커널의 compile/빌드/명명 규칙,
CPU LayerNorm dispatch 검사 등이 포함된다. 일부는 GPU 없이 재생성할 수 없고,
일부는 연구 코드의 별도 정비가 필요하다. 이번 main은 개발 코드 통합본이며 완전 통과한
릴리스나 새 GPU 검증 완료본으로 취급하지 않는다.
[정확한 실패 목록](docs/repository-audit-20260927/engine-cpu.json) ·
[실패 상세](docs/repository-audit-20260927/engine-failures.txt)

TriangleAttention checkpoint18246의 기록된 파일 48개는 모두 기존 SHA와 일치한다.
Norm manifest 12개 중 11개는 일치한다. 나머지 `layernorm/compile_native.py`는
MPNN용 선택적 row_bucket 인수가 추가됐으며 기본 경로는 유지했다. 기존 manifest를
새 GPU 검증 결과처럼 다시 찍지 않았다. 전체 조합의 GPU 검증은 남아 있다.

## TriMul 최적화 재개 지점

연구 디렉터리: `runs/trimul_d256_bwd_sol90_stage2_20260923`.
선택된 후보는 D256 `d256_pool_checkpoint.py`, D384 `wide_checkpoint23.py`,
D512 `wide_checkpoint24.py`다. 공용 engine의 wide backward dispatch 통합은 아직이다.

| D | L | 기존 대비 BWD 배속 | 기존 대비 F+B 배속 | 둘 다 1.5배 |
|---:|---:|---:|---:|---|
| 128 | 384 | 1.523× | 1.606× | 충족 |
| 128 | 768 | 1.511× | 1.605× | 충족 |
| 256 | 384 | 1.268× | 1.353× | 미충족 |
| 256 | 768 | 1.699× | 1.662× | 충족 |
| 384 | 384 | 1.391× | 1.389× | 미충족 |
| 384 | 768 | 1.532× | 1.509× | 충족 |
| 512 | 384 | 1.319× | 1.349× | 미충족 |
| 512 | 768 | 1.504× | 1.513× | 충족 |

기준은 기존 Triton과 우리 커널의 직접 측정값이다. 원본 외부 커널 대비라는 의미가 아니다.
모든 wide shape 1.5배와 SOL90은 미달성이다. 숫자·job·원본 JSON:
[qualified_speedups.json](runs/trimul_d256_bwd_sol90_stage2_20260923/qualified_speedups.json).

## 나머지 저장소와 서버 이동

| 저장소 | 정리 상태 |
|---|---|
| DataCooker | 최신 main과 MiniWorld branch 통합; 구형 docs_practices는 main 내 보관. 현재 runtime/API 문서 유지 |
| KmerFastAlign | 로컬 빌드 변경 3개를 main에 보존; 다른 머신에서는 native rebuild 필요 |
| FoldBench | 로컬 main으로 통합; 외부 BEAM-Labs upstream에는 push하지 않음 |
| StructCooker | 기존 gitlink의 main 사용; 소스 변경 없음 |
| 연구 engine 복제본 | 다섯 작업본의 HEAD가 canonical engine main 이력에 포함됨; 비교용 파일 유지 |
| 외부 참고 커널·CUTLASS | upstream revision 유지 |

FoldBench의 MiniWorld adapter와 target 선택은
[재현 patch](patches/dependencies/foldbench-miniworld.patch)와
[기준 revision](patches/dependencies/foldbench.json)으로 MiniWorld main에 포함했다.
DataCooker/FoldBench/KmerFastAlign은 부모에서 무시하는 로컬 저장소이므로
`--recurse-submodules`만으로 설치되지 않는다.
DataCooker의 과거 `gh-pages`는 생성된 사이트 배포 이력이므로 소스 main 통합에서 제외했다.

새 서버에서는 MiniWorld main을 clone하고 `git submodule update --init --recursive`로
정확한 team-gm/StructCooker gitlink를 복원한다. `pixi install -e cu128` 후
`pixi run -e cu128 engine-setup`으로 pin과 필수 파일을 검사한다.
engine 1.x용 `apply_engine_audit_patches.py`는 역사 자료이며 main에는 자동 적용하지 않는다.

커널 개발에는 engine main을 별도로 clone해 사용한다. 연구 harness의 기존 절대경로와
PYTHONPATH는 새 서버 경로로 맞춰야 한다. source tree에 보존한 attention `.so`는 기록된
Python/PyTorch/SM90 ABI에 종속되며 일반 wheel에는 포함하지 않는다. 대상 환경에서 필요한
native rebuild 후 실제 import/dispatch, F+B, changed-input CUDA Graph, sanitizer와 같은
GPU의 baseline/candidate 시간을 다시 확인한다.

원래 백업 patch·index·파일 SHA는 `runs/repository_cleanup_20260927`에 남아 있다.
대형 profile, dataset, checkpoint, 환경 전체는 Git 통합 범위가 아니다.
