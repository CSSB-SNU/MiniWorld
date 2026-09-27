# 전체 작업본과 로컬 브랜치

이 문서는 **main 통합 전** 최초 보존 작업의 스냅샷이다. 현재 작업 경로와 main 상태는
[REPOSITORIES.md](../../REPOSITORIES.md), 통합한 refs는
[main-verification.json](main-verification.json)을 기준으로 한다.
이후 병합된 브랜치 63개(로컬)와 7개(원격)를 삭제했으며, 삭제한 refs와 보존한
worktree의 정확한 SHA는 [branch-cleanup.json](branch-cleanup.json)에 기록했다.

2026-09-27 문서 커밋 직전의 코드 스냅샷. 원격 tracking 정보는 각 저장소의 마지막 fetch 기준이다.
MiniWorld의 마지막 보고서 커밋은 아래 코드 SHA 이후에 추가된다. `archive/*`는 복구·기록용이다.

## 작업본 28개 / Git 저장소 11개

| 작업 경로 | 브랜치 | 코드 SHA | 역할/상태 |
|---|---|---|---|
| `~/MiniWorld` | `integrate/miniworld-dev-20260927` | `1f065f84da` | 문서 커밋 대기 |
| `~/MiniWorld-engine-migration` | `archive/engine-migration-20260927` | `ece87165c4` | clean |
| `~/MiniWorld-old` | `archive/oldcode-repro-20260927` | `d3f79e2416` | clean |
| `~/miniworld-engine` | `archive/engine-old-main-20260927` | `ce47f42bcb` | clean |
| `~/MiniWorld/.engine-release-2.0.0` | `integrate/h100-dev-20260927` | `017afd5b0c` | clean |
| `~/MiniWorld/.engine-v2-proof` | `archive/v2-proof-cache-20260927` | `ecd9256405` | clean |
| `~/ext/engine-main` | `(detached)` | `091962652c` | clean |
| `~/miniworld-engine-dit` | `research/token-dit-fused` | `e512d8c2ee` | clean |
| `~/miniworld-engine-dit2` | `research/token-dit-overlap-evidence-20260927` | `6ad576908b` | clean |
| `~/miniworld-engine-k1k3` | `archive/trimul-k1k3-local-20260927` | `af7a984771` | clean |
| `~/miniworld-engine-msa` | `research/msa-inference` | `b47de812e8` | clean |
| `~/miniworld-engine-tbwd` | `archive/transition-workspace-20260927` | `2254765abe` | clean |
| `~/miniworld-engine-tdt` | `feat/token-dit-train-core` | `fac86f90ed` | clean |
| `~/mwk-prefusion` | `(detached)` | `403d382c4d` | clean |
| `/tmp/claude-1019/-home-psk6950-MiniWorld/4b4d8989-664e-4f61-a593-8fd3e64e40f8/scratchpad/engine-main` | `(detached)` | `091962652c` | clean |
| `~/MiniWorld/libs/team-gm` | `integrate/miniworld-dev-20260927` | `066ae81895` | clean |
| `~/MiniWorld/libs/structcooker` | `(detached)` | `2b1543e500` | clean |
| `~/MiniWorld/libs/datacooker` | `MiniWorld` | `3c215c81a6` | clean |
| `~/MiniWorld/libs/foldbench` | `integrate/miniworld-eval-20260927` | `f5010fdd6f` | clean |
| `~/MiniWorld/libs/kmer_fast_align` | `archive/local-build-20260927` | `af22949618` | clean |
| `/tmp/team-gm-exp-miniworld-publish` | `archive/incomplete-publish-worktree-20260927` | `1ffa8b8120` | clean |
| `~/MiniWorld/runs/trimul_unidirectional_20260917/engine` | `perf/unidirectional-trimul-fusion` | `b06857c09e` | clean |
| `~/MiniWorld/runs/transition_triton_audit_20260917/engine` | `perf/transition-triton-audit` | `fbe2d6add1` | clean |
| `~/MiniWorld/runs/trimul_sm90_parity_20260917/engine` | `integrate/h100-research-runtime-20260927` | `ecaac5b44d` | clean |
| `~/MiniWorld/runs/trimul_sm90_15pct_20260917/baseline-engine` | `(detached)` | `600c8c4c1f` | clean |
| `~/MiniWorld/runs/engine_main_publish_20260917` | `integrate/local-engine-patches` | `4d28918d16` | clean |
| `~/MiniWorld/runs/anthropic_adoption_20260919/upstream` | `(detached)` | `f4f62fa659` | tracked clean; partial clone |
| `~/miniworld-engine/third_party/ct_cutlass_workbench/cutlass` | `(detached)` | `bf9da7b76c` | clean |

## ~/MiniWorld

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `archive/engine-migration-20260927` | `ece87165c4` | — | — |
| `archive/oldcode-repro-20260927` | `d3f79e2416` | — | — |
| `archive/pre-cleanup-20260927/miniworld` | `27b0abe6db` | — | — |
| `archive/pre-cleanup-20260927/mw-engine-migration` | `bfe80f8fb3` | — | — |
| `archive/pre-cleanup-20260927/mw-old-repro` | `85f13de872` | — | — |
| `docs/h100-graph-validation-20260924` | `5b4357a181` | — | — |
| `feat/distogram-diffusion-v1.3` | `efca47f5c9` | — | — |
| `fix/v2-data-loss-consistency` | `15ccef06ff` | — | — |
| `integrate/miniworld-dev-20260927` | `1f065f84da` | — | — |
| `main` | `27b0abe6db` | origin/main | [ahead 2] |
| `research/h100-experiments-20260927` | `df0d94b51d` | — | — |
| `research/swa-atom-fusion` | `20ea6ca6c7` | — | — |

## ~/miniworld-engine

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `archive/engine-h100-cache-20260927` | `f001428268` | — | — |
| `archive/engine-old-main-20260927` | `ce47f42bcb` | — | — |
| `archive/engine-wheel-fix-20260927` | `61294df387` | — | — |
| `archive/h100-legacy-runtime-20260927` | `9b980af88a` | — | — |
| `archive/pre-cleanup-20260927/engine-dit2` | `516b62c21e` | — | — |
| `archive/pre-cleanup-20260927/engine-k1k3` | `37d61f8166` | — | — |
| `archive/pre-cleanup-20260927/engine-old-main` | `b4c3f6c34e` | — | — |
| `archive/pre-cleanup-20260927/engine-release` | `55304541ff` | — | — |
| `archive/pre-cleanup-20260927/engine-tbwd` | `971992df69` | — | — |
| `archive/pre-cleanup-20260927/engine-v2-proof` | `3e81f15cba` | — | — |
| `archive/transition-tuning-20260927` | `bb6045527d` | — | — |
| `archive/transition-workspace-20260927` | `2254765abe` | — | — |
| `archive/trimul-k1k3-local-20260927` | `af7a984771` | — | — |
| `archive/v2-proof-cache-20260927` | `ecd9256405` | — | — |
| `feat/autotune-cache-builder` | `25743ad817` | — | — |
| `feat/cute-autotune-sweep` | `eebedbbc06` | — | — |
| `feat/token-dit-train-core` | `fac86f90ed` | origin/main | [ahead 2] |
| `fix/transition-d64-input-barrier` | `0043b98037` | — | — |
| `integrate/h100-dev-20260927` | `017afd5b0c` | — | — |
| `integrate/h100-research-runtime-20260927` | `ecaac5b44d` | — | — |
| `main` | `b4c3f6c34e` | origin/main | [behind 145] |
| `perf/triton-norm-h100` | `c26be24503` | — | — |
| `release/2.0.0` | `55304541ff` | origin/main | [ahead 1, behind 2] |
| `research/msa-inference` | `b47de812e8` | — | — |
| `research/native-norm-h100` | `a96ea5f7bc` | — | — |
| `research/token-dit-fused` | `e512d8c2ee` | — | — |
| `research/token-dit-overlap` | `516b62c21e` | — | — |
| `research/token-dit-overlap-evidence-20260927` | `6ad576908b` | — | — |
| `research/transition-backward-fused` | `971992df69` | origin/main | [behind 24] |
| `research/transition-local-20260927` | `9d49491b5a` | — | — |
| `research/triattn-installed-20260927` | `aa054b12d4` | — | — |
| `research/trimul-k1k3-inference` | `37d61f8166` | origin/main | [behind 39] |
| `research/trimul-local-20260927` | `cf697a1719` | — | — |
| `tmp/autotune-bucket-cap-512` | `73458b5425` | origin/tmp/autotune-bucket-cap-512 | [gone] |

## ~/MiniWorld/libs/team-gm

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `BioMol` | `0d464536ae` | origin/BioMol | [gone] |
| `archive/incomplete-publish-worktree-20260927` | `1ffa8b8120` | — | — |
| `archive/pre-cleanup-20260927/team-gm` | `ba3c07f490` | — | — |
| `archive/pre-cleanup-20260927/team-gm-incomplete-publish` | `aaf9356ccb` | — | — |
| `exp/af3-refactor` | `a8891f56ec` | origin/exp/af3-refactor | [gone] |
| `exp/miniworld` | `ba3c07f490` | origin/exp/miniworld | [ahead 6] |
| `feat/miniworld-engine-wiring-20260915` | `5d3f4a5b44` | — | — |
| `fix/diffusion-valid-atom-loss` | `c9f5a51dca` | — | — |
| `integrate/miniworld-dev-20260927` | `066ae81895` | — | — |
| `integrate/miniworld-engine-wiring` | `aaf9356ccb` | origin/exp/miniworld | [behind 8] |
| `main` | `735d476ab2` | origin/main | [behind 62] |
| `psk_add_BioMol` | `dbe63e3a02` | origin/psk_add_BioMol | [gone] |
| `research/swa-fused-h100` | `14f2c7396c` | — | — |

## ~/MiniWorld/libs/structcooker

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `main` | `2b1543e500` | origin/main | — |

## ~/MiniWorld/libs/datacooker

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `MiniWorld` | `3c215c81a6` | — | — |
| `main` | `1186cd6fa0` | origin/main | [behind 30] |

## ~/MiniWorld/libs/foldbench

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `archive/local-target-selection-20260927` | `ce463f50af` | — | — |
| `archive/pre-cleanup-20260927/foldbench` | `90a6033fed` | — | — |
| `feat/miniworld-evaluation` | `af25c944b9` | — | — |
| `integrate/miniworld-eval-20260927` | `f5010fdd6f` | — | — |
| `main` | `90a6033fed` | origin/main | — |

## ~/MiniWorld/libs/kmer_fast_align

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `archive/local-build-20260927` | `af22949618` | — | — |
| `archive/pre-cleanup-20260927/kmer-local-build` | `72a5cd922a` | — | — |
| `main` | `72a5cd922a` | origin/main | — |

## ~/MiniWorld/runs/trimul_unidirectional_20260917/engine

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `integrate/local-engine-patches` | `4d28918d16` | — | — |
| `main` | `b4c3f6c34e` | origin/main | [behind 88] |
| `perf/unidirectional-trimul-fusion` | `b06857c09e` | — | — |

## ~/MiniWorld/runs/transition_triton_audit_20260917/engine

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `archive/h100-legacy-runtime-20260927` | `9b980af88a` | — | — |
| `archive/pre-cleanup-20260927/engine-triattn-installed` | `59bdb33506` | — | — |
| `integrate/h100-research-runtime-20260927` | `ecaac5b44d` | — | — |
| `perf/transition-triton-audit` | `fbe2d6add1` | — | — |
| `perf/trimul-sm90-parity` | `59bdb33506` | — | — |
| `perf/unidirectional-trimul-fusion` | `b06857c09e` | origin/perf/unidirectional-trimul-fusion | — |
| `research/triattn-installed-20260927` | `aa054b12d4` | — | — |

## ~/MiniWorld/runs/anthropic_adoption_20260919/upstream

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
| `main` | `f4f62fa659` | origin/main | — |

## ~/miniworld-engine/third_party/ct_cutlass_workbench/cutlass

| 로컬 브랜치 | SHA | upstream | tracking 상태 |
|---|---|---|---|
