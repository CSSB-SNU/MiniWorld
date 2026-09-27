# MiniWorld 저장소와 개발 상태

2026-09-27: MiniWorld, miniworld-engine, team-gm의 로컬·원격 작업 브랜치를 **main으로 통합**했다.
예전 `integrate/*`, `research/*`, `archive/*` refs는 복구용으로 남겨두었으며,
현재 코드를 조립하기 위해 별도 브랜치를 checkout할 필요는 없다.

## 현재 작업 경로

| 저장소 | 개발 경로 | 브랜치 |
|---|---|---|
| MiniWorld | `/home/psk6950/MiniWorld` | `main` |
| miniworld-engine | `/home/psk6950/miniworld-engine` | `main` |
| team-gm | `MiniWorld/libs/team-gm` | `main` |

`MiniWorld/.engine-release-2.0.0`은 engine main과 같은 커밋의 호환용 detached worktree다.
기존 harness의 경로를 유지하기 위해 남겼다. 새 engine 수정은 canonical main에서 한다.
그 외 과거 연구 worktree는 동결된 비교·복구 자료이며, 최신 실행 코드는 위 경로를 기준으로 한다.

MiniWorld와 team-gm의 engine pin은 모두 `3026c6bcd55bf535b73a0645fd4258b41c3cb547`이다.
`pixi.lock`과 `uv.lock`도 engine 2.0.0 및 같은 SHA로 맞췄다. engine의 의존성과 extras는
이전 pin과 동일함을 비교했고 다른 패키지 버전은 바꾸지 않았다.
`.gitmodules`의 team-gm 추적 브랜치는 main이며, 실제 재현 기준은 커밋된 gitlink다.
현재 설치된 Pixi 환경은 재설치하지 않았다. 소스/pin 정리와 설치 환경 갱신은 별개다.

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

## 검증 결과와 남은 문제

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
DataCooker의 `gh-pages`는 생성된 사이트 배포 이력이므로 소스 main에 합치지 않는다.

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
