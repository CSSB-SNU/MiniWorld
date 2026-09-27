# MiniWorld 저장소와 개발 상태

기준: **2026-09-27**, CSSB 로컬 작업 공간. 브랜치 분리와 로컬 커밋을 완료했다.
원격 배포나 GPU 재검증 결과를 뜻하지 않는다. 이후 작업은 아래 경로를 기준으로 한다.

[전체 11개 저장소·28개 작업본·로컬 브랜치 표](docs/repository-audit-20260927/BRANCHES.md) ·
[커밋별 파일 목록](docs/repository-audit-20260927/commits.json) ·
[보존 및 정적 검증](docs/repository-audit-20260927/verification.json)

| 용도 | 작업 경로 | 현재 개발 브랜치 |
|---|---|---|
| 모델·데이터·학습 | `/home/psk6950/MiniWorld` | `integrate/miniworld-dev-20260927` |
| 공용 커널·Norm·Transition | `MiniWorld/.engine-release-2.0.0` | `integrate/h100-dev-20260927` |
| 공용 모델 블록·diffuser | `MiniWorld/libs/team-gm` | `integrate/miniworld-dev-20260927` |
| 설치된 TriangleAttention 연구 런타임 | `MiniWorld/runs/trimul_sm90_parity_20260917/engine` | `integrate/h100-research-runtime-20260927` |

`/home/psk6950/miniworld-engine`은 engine Git 저장소의 본체지만, 그 폴더의 작업 파일은
9월 6일의 오래된 코드다. 이 작업본은 `archive/engine-old-main-20260927`로 구분했다.
최신 공용 engine 코드는 위 `.engine-release-2.0.0` worktree에서 수정한다.
모든 경로와 파일 내용은 보존했으므로 기존 절대경로 참조는 계속 같은 파일을 가리킨다.

## 왜 서로 다른 최신 코드가 있었나

| 항목 | 확인된 기준 | 의미 |
|---|---|---|
| MiniWorld 원격 main | `e7869b3de` | 로컬 기존 main `27b0abe6d`보다 2커밋 뒤 |
| engine 원격 main | `c3101759` | `release/2.0.0`의 `55304541`과 분기: release에만 1개, main에만 2개 |
| engine v2.0.0 태그 | `c53d881c` | 이후의 로컬 수정이나 최신 연구 작업 전체를 포함하지 않음 |
| MiniWorld Pixi pin / lock | `1bc0803e` | 9월 15일 engine 계열. 현재 release 작업본과 다름 |
| cu128 설치 metadata | engine `1.0.0`, commit `1bc0803e` | 패치 적용 가능성이 있으므로 pristine 소스라는 뜻은 아님 |
| 최근 TriMul wide 실험 | `.engine-release-2.0.0/src`를 명시적으로 우선 import | Pixi pin만 보고 실제 실행 코드를 판단하면 틀림 |
| 최근 TriangleAttention 설치 | 별도 연구 engine의 `59bdb335` 기반 수정본 | 공용 release 작업본에 설치됐다는 뜻이 아님 |
| v1.3 BioAI launcher | `runs/v1.3.0/implementation_20260924/runtime/engine-src` | 서버별 동결 runtime. 이 CSSB 작업 공간에는 해당 경로 없음 |

원격 main 두 개는 정리 시작 시 `git ls-remote`로 확인했다. 나머지 오래된 tracking ref의
ahead/behind는 마지막 fetch 기준이며 별도 최신 원격 확인으로 간주하지 않는다.
MiniWorld의 `2.0.0.dev0`는 모델/파이프라인 버전이며 engine `2.0.0`과 독립적이다.

## 기능별 브랜치

통합 브랜치는 **정리 직전 파일 내용을 그대로 모은 작업용 상태**다. 기능별 브랜치는 같은
정리 전 HEAD에서 나누었으며, 통합 커밋의 부모로 연결했다. 서로 다른 기능을 한 번에
출시하거나 GPU 검증을 통과했다는 의미는 아니다.

### MiniWorld

| 브랜치 | 커밋 | 내용 / 검증 범위 |
|---|---|---|
| `feat/distogram-diffusion-v1.3` | `efca47f5c` | 96-bin distogram diffusion, graph trainer 시작/재개, BioAI 설정·런처·기존 검증 기록 |
| `fix/v2-data-loss-consistency` | `15ccef06f` | CCD fallback, mask·template·confidence·best-of-N·제한된 대칭 처리, RDKit 선언, v2 설정 |
| `research/swa-atom-fusion` | `20ea6ca6c` | SWA 공유 conditioning 전달; 대응 team-gm 연구 커밋 pin |
| `docs/h100-graph-validation-20260924` | `5b4357a18` | D64 buffer barrier 및 실제 graph 학습 기록 |
| `research/h100-experiments-20260927` | `df0d94b51` | ignored `runs/`에서 선별한 연구 소스와 검증 기록 6,664개, 약 51 MiB |

v1.3에는 기존 CPU/GPU/graph/lifecycle 검증 기록이 있다. 이는 당시 코드·runtime·shape에
대한 근거다. v2 기록은 CPU 33개 통과이며, 일반 ligand/modified-residue 대칭, 실제 복합체,
전체 GPU 실행과 수렴 검증은 남아 있다. SWA는 기본 비활성 opt-in 연구 경로다.
상세: [v1.3](docs/v1.3-distogram-diffusion.md), [v2](docs/v2-correctness.md).

### miniworld-engine

아래 브랜치는 모두 `/home/psk6950/miniworld-engine` Git 저장소에서 조회할 수 있다.

| 브랜치 | 커밋 | 내용 / 상태 |
|---|---|---|
| `fix/transition-d64-input-barrier` | `0043b980` | D64 입력 버퍼 재사용 barrier, 확장 캐시 revision, 회귀 테스트 |
| `perf/triton-norm-h100` | `c26be245` | 검증된 RMS tuning, dtype별 LN dispatch, LNLinear saved stats |
| `research/native-norm-h100` | `a96ea5f7` | 명시적 native CUDA Norm API; 전역 dispatch 교체 아님 |
| `integrate/h100-dev-20260927` | `017afd5b` | 위 세 작업을 기존 `55304541` 위에 보존한 작업본 |
| `feat/token-dit-train-core` | `fac86f90` | 기존 opt-in BF16 SM90 training attention core; 별도 작업본 유지 |
| `research/token-dit-overlap-evidence-20260927` | `6ad57690` | 최신 inference overlap 연구와 미커밋 실험 기록 보존 |
| `research/triattn-installed-20260927` | `aa054b12` | 별도 연구 engine에 설치된 attention 소스·manifest·네이티브 파일 보존 |
| `integrate/h100-research-runtime-20260927` | `ecaac5b4` | attention을 포함한 연구 engine 전체 수정 상태. 원래 로컬 저장소에서도 유지 |
| `archive/transition-workspace-20260927` | `2254765a` | 과거 Transition 작업본의 kernel, wide/TriMul 실험, tuning을 별도 커밋으로 보존 |

release와 원격 main은 자동 병합하지 않았다. main의 2개 변경은 module training benchmark와
Triton cache probe 수정이다. TokenDiT core, 최신 inference overlap, 연구 attention도 서로
자동 병합하지 않았다. 그 조합의 실행 검증은 아직 없으며, 원래 작업 파일을 보존했다.

Norm의 기존 설치 manifest 12개는 정리 후에도 모두 일치한다. 관련 60개 테스트, graph,
sanitizer, 설치 경로 검증은 [기존 보고서](.engine-release-2.0.0/docs/development/triton-norm-fixes-20260927.md)에
기록돼 있다. native Norm과 portable LNLinear가 모든 기존 경로보다 빠르다는 주장은 하지 않는다.
별도 attention runtime도 training checkpoint18246의 소스·manifest·바이너리 48개가 모두 일치한다.

기존 `research/msa-inference`, `research/token-dit-fused`, `feat/autotune-cache-builder`,
`feat/cute-autotune-sweep`의 커밋은 확인한 engine 원격 main에 포함돼 있다. 원래 refs와
작업본은 역사 확인용으로 남겼다. `research/token-dit-overlap`은 main 대비 독자 커밋 28개가
있어 단순한 정리 대상이 아니다. 두 DiT 계열의 역할을 구분해야 한다.

### team-gm와 나머지 의존성

| 저장소 / 브랜치 | 내용 |
|---|---|
| team-gm `fix/diffusion-valid-atom-loss` (`c9f5a51`) | padding을 제외한 EDM loss 평균, empty mask 처리 |
| team-gm `research/swa-fused-h100` (`14f2c73`) | opt-in SWA Triton/CUDA 코드 |
| team-gm `integrate/miniworld-dev-20260927` (`066ae81`) | 위 두 변경의 통합 상태. MiniWorld gitlink도 이 커밋으로 갱신 |
| FoldBench `integrate/miniworld-eval-20260927` | 평가 코드와 기존 target 선택/삭제 상태를 각 브랜치로 보존 |
| KmerFastAlign `archive/local-build-20260927` | 기존 tracked 로컬 `.so` 3개 보존; 배포용 소스 변경으로 취급하지 않음 |
| DataCooker | 소스 변경 없음. 생성된 bytecode만 로컬 Git exclude에 추가 |
| StructCooker | 기존 `2b1543e` gitlink 유지, 변경 없음 |

현재 `.gitmodules`의 실제 gitlink 대상은 team-gm와 StructCooker다. DataCooker, FoldBench,
KmerFastAlign은 부모에서 무시하는 로컬 저장소이므로 `git clone --recurse-submodules`만으로
함께 이동되지 않는다. 상세 SHA와 remote는 [전체 작업본 목록](docs/repository-audit-20260927/worktrees.json)에 있다.

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

## 보존과 다른 서버로의 이동

- 모든 새 커밋은 **로컬**이다. 원격 clone만으로는 이번 작업이 오지 않는다.
- 기존 main/release/research 브랜치는 남겼다. 정리 전 HEAD에는
  `archive/pre-cleanup-20260927/*` 복구 브랜치도 있다.
- 변경 전 binary patch, index, 파일 SHA는 `runs/repository_cleanup_20260927`에 보관했다.
  이 백업 디렉터리는 로컬용이며 [검증 요약](docs/repository-audit-20260927/verification.json)은 문서에 포함했다.
- 연구 파일은 현재 통합 브랜치에서는 계속 ignored다. 별도 연구 브랜치의 Git tree에 저장했다.
  [파일 manifest](docs/repository-audit-20260927/research-files.json)로 범위를 확인한다.
  빌드/컴파일 캐시, 대형 raw profile, 데이터셋, checkpoint, 환경, 전체 과거 `runs/`를
  이 Git 보존 작업에 포함한 것은 아니다.
- `/tmp/team-gm-exp-miniworld-publish`의 기존 164개 삭제 상태는
  `archive/incomplete-publish-worktree-20260927`에 격리했다. 이 커밋을 기능 브랜치로 병합하지 않는다.
- 외부 참고 소스와 CUTLASS는 기존 upstream revision과 별도 checkout으로 식별했다.
  연구 engine의 `third_party/anthropic/UPSTREAM.json`이 원본 revision과 파일 해시를 기록한다.

대여 H100에서 재개할 때는 다음 순서를 따른다.

1. MiniWorld, canonical engine, team-gm의 위 개발/연구 refs를 로컬 bundle이나 명시적 push로 전달한다.
   전달하기 전 `docs/repository-audit-20260927/commits.json`으로 대상 refs를 확인한다.
2. MiniWorld 통합 브랜치를 열고 정확한 team-gm gitlink를 복원한다. 필요한 연구 파일은
   `research/h100-experiments-20260927`를 별도 worktree로 열어 확보한다.
3. 공용 engine과 연구 runtime을 구분해서 배치한다. harness의 기존 절대경로,
   `PYTHONPATH`, 외부 참고 소스 경로를 새 환경에 맞춘다. Pixi의 오래된 pin을 최신 engine으로
   오인하거나 설치 패치 스택을 다른 계열에 무조건 재적용하지 않는다.
4. CUDA/PyTorch 환경에 맞게 네이티브 파일을 준비하고, 실제 import/dispatch 경로와
   correctness·changed-input graph·sanitizer·같은 GPU의 baseline/candidate 성능을 다시 확인한다.

이번 정리는 기존 파일 1,102개의 해시/삭제 상태 보존, Python 48개 AST 구문 확인,
Norm 12개 및 attention 48개 기존 manifest 일치 확인까지 수행했다.
전체 diff 공백 검사에서는 보존한 patch의 context 두 줄과 생성된 `.inc`의 마지막 빈 줄만
보고됐다. 해당 자료를 제외한 앱/team-gm 코드와 engine release diff는 통과했다.
GPU 작업, benchmark, 환경 설치, 원격 push는 실행하지 않았다.
