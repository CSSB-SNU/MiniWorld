# H100 학습용 TM2: CuTe F567

2026-09-16. 기존 CuTe `tm2`를 참고해, 양방향 TriMul 학습의 **F5 출력 projection + F6 gate projection + F7 sigmoid·dropout scale·residual**을 한 H100 커널로 구현했다. 커널 이름은 `F567Sm90`, 등록 op는 `trimul_output_f567_sm90_cute`다.

**구현·정확도 검증과 성능 우위는 별개다. 현재 버전은 Triton A보다 느려 설치본 자동 활성화를 예약하지 않았다.** 검증용 package에서 H100 모델 배선까지 연결해 비교하며, 기존 캐시 writer 및 예약된 이전 패치는 유지한다.

## 배선도

- [Forward 전체 SVG](../../tmp_kernel/trimul/TRIMUL_FORWARD.svg): Triton A / 기존 H100 / 신규 H100 개발 경로.
- [L768 전체 SVG](../../tmp_kernel/trimul/TRIMUL_FORWARD_L768.svg).
- [확대·경로별 비교](../../tmp_kernel/trimul/TRIMUL_STATUS.html).

Triton A와 신규 H100 열에서는 F5·F6·F7이 한 실선 경계 안에 있다. `glogit` HBM 버퍼가 사라지고, projection은 backward용으로 저장하되 forward에서는 다시 읽지 않는다. F4 LayerNorm과 dropout 난수·scale 생성은 별도다. Backward의 기존 저장값·계산 연결은 유지했다.

H100의 projection-aware backward 경로에서는 F4가 `xhat`을 저장하고, 작은 weight/bias 준비에서 `Wfold=Wp*gamma`, `bfold=sum(Wp*beta)`를 계산한다. 새 F567은 `xhat @ Wfold.T + bfold`와 gate GEMM을 함께 실행한다. 따라서 이 열의 F4 affine 처리 위치는 Triton 열과 다르며, 앞서 개발한 H100 backward 알고리즘을 보존한 것이다.

## 기존 TM2와 달라진 점

| 항목 | 기존 CuTe TM2 | 신규 학습용 F567 |
|---|---|---|
| 두 GEMM의 K | 같은 K 요구 | KP/KG 독립; 운영 shape는 256/128 |
| N 타일 | 전체 N 한 타일 | 32/64/128 타일로 분할 |
| 출력 | gated value | y, projection, gate 3개 |
| 학습 epilogue | dropout/residual 없음 | dropout scale 곱 + residual 덧셈 |
| Bias | 없음 | H100 folded-affine projection bias 지원 |
| 경계 처리 | padding/crop | TMA의 경계 처리와 epilogue predicate |
| CUDA stream | 기존 launcher 계약 | 호출 시 현재 PyTorch stream을 명시적으로 전달 |

Activation/residual/drop scale은 BF16 contiguous, 너비는 8개 BF16 원소 단위 정렬을 요구한다. Weight gate는 작은 `(N,KG)` 행렬로 변환한다. 큰 activation의 padding/copy는 하지 않는다. 공개 API는 dtype·device·shape·정렬·stride·자원 범위를 검사한다.

## H100 전용 메모리 처리

1. **TMA**로 두 GEMM의 activation/weight와 residual을 shared memory에 읽는다. TMA는 H100의 다차원 bulk copy 기능이다.
2. **WGMMA**로 서로 다른 K 길이의 두 GEMM을 계산한다. WGMMA는 warpgroup 단위 Tensor Core 행렬곱이다.
3. Projection/logit의 BF16 반올림을 유지하고 FP32 sigmoid·dropout·residual 계산을 수행한다.
4. 레지스터 값을 STSM으로 shared memory에 저장한 뒤 TMA로 y/projection/gate를 기록한다. Residual을 담았던 staging buffer를 출력 저장에 재사용하고, TMA source 읽기 완료와 CTA barrier 뒤에 덮어쓴다.

두 GEMM의 전체 K를 각각 staging한다. `tile_k`는 전송/MMA chunk 크기다. K 반복 횟수와 같지 않으며, 실제로 쓰지 않는 pipeline-depth 설정을 만들지 않았다. K를 순차적으로 나눠 읽어 shared memory를 줄이는 대안도 비교했지만 현재 실측에서는 느려 채택하지 않았다.

## Config와 캐시

설정은 엔진의 `autotune/cute_config.py`에서 관리한다. Native selector, exact tensor metadata key, registry driver, CPU precompile 계약, source identity와 shard publication에 연결했다.

| 축 | 후보 |
|---|---|
| `tile_m` | 64, 128, 192, 256 |
| `tile_n` | 32, 64, 128 |
| `tile_k` | 32, 64, 128 |
| `group_m` | 1, 2, 4, 8 |

총 **144개 조합**이다. K padding, 네 입력 staging 배열, 재사용하는 residual/output buffer 및 정렬을 계산해 **현재 H100·KP256/KG128에서 96개**를 탐색한다. Warpgroup당 128 threads와 WGMMA M atom 64는 하드웨어 계약이며, 성능 선택을 위한 임의 상수가 아니다.

캐시는 source identity, 환경, dtype, 각 operand의 shape/stride, 실제 L, 장치 shared-memory 한도를 구분한다. 이전 direct-store 버전의 측정을 새 TMA 버전의 identity로 바꾸지 않았다. 새 소스에 대해 다시 컴파일하고 측정한다. CPU precompile은 CUDA를 숨긴 별도 프로세스에서 수행한다.

## 검증·측정 증거

최종 수치는 아래 JSON에 보존한다. 시간은 **forward만**이며, optimizer나 backward 시간은 포함하지 않는다. `runtime` 보고서는 같은 GPU에서 split/Triton/CuTe를 번갈아 CUDA graph로 재생한 결과다.

- 단위 검사: M/N/K 경계, GROUP_M 나머지, bias 유무, weight stride, 지원하지 않는 정렬, static compile, graph replay 후 dropout scale 변경.
- 각 운영 shape의 모든 유효 config: y/projection/gate를 FP32 계산의 BF16 반올림 기준과 비교한 뒤 측정·저장.
- 모델 검사: holed mask, dropout, 입력과 모든 파라미터 gradient, 기존 H100 backward 연결, 실제 F567 cache hit와 CUDA profiler 호출 확인.
- 전체 forward 비교의 기존 H100 front/backward native cache는 새 source identity와 달라 기본 config를 사용한다. 두 H100 비교 경로에 동일하게 적용된다. 이전 세션의 전체 시간과 직접 비교하지 않는다.

개발 실행 디렉터리: `runs/trimul_f567_cute_20260916`.

재현 예시(GPU 할당 안에서):

```bash
export PYTHONPATH="$PWD/runs/trimul_f567_cute_20260916/package:$PWD/scripts:$PWD/libs/team-gm/src"
.pixi/envs/cu128/bin/python -m pytest tests/test_engine_trimul_f567_cute.py -q
.pixi/envs/cu128/bin/python scripts/tune_trimul_f567_cute.py --length 384 --output /tmp/f567-cute-L384.json
```

설치본과 외부 miniworld-engine checkout을 직접 변경하지 않고, 검증된 변경은 MiniWorld와 team-gm에 패치로 보존한다. 현재 native source identity는 다른 CuTe 커널 소스도 함께 구분하므로, 이 패치를 적용할 때 기존 CuTe 캐시의 별도 재검증이 필요하다.

## 최종 결과

H100 80GB, BF16, B1, pair/방향별 hidden=128, fixed-shape compile. 아래는 F4–F7 전체 출력 구간의 **동일 수식** 비교이며, 작은 weight packing·출력 할당에서 발생하는 GPU 작업도 graph에 포함된다.

| L | 기존 split | Triton A | 신규 CuTe F567 | CuTe 시간 / Triton |
|---|---:|---:|---:|---:|
| 384 | 0.2039 ms | **0.1709 ms** | 0.2646 ms | 1.55× |
| 768 | 0.7303 ms | **0.6253 ms** | 0.9505 ms | 1.52× |

결론: 이 버전의 CuTe 구현은 기능·배선 검증을 통과했지만 **Triton보다 약 52–55% 느리다.** H100 기본 경로 전환 대상으로 채택하지 않았다. H100의 기존 folded-affine 경로와 비교한 전체 TriMul forward도 L384 0.5365→0.5954 ms, L768 2.0130→2.2320 ms로 회귀했다. 이는 backward를 포함한 학습 전체 시간이 아니다.

- **96/96 configs × 두 shape 모두 CPU precompile·수치 검사·측정 통과.** Source identity를 확인한 캐시에 저장했다.
- Build winner: L384 `M128/N64/K128/group1`, L768 `M64/N64/K128/group1`. 측정된 커널 단독 시간은 각각 0.1976 / 0.7456 ms. 위 표의 F4 포함 시간과 구분한다.
- **24 tests passed; memcheck 0 errors.** 별도 [racecheck](f567-cute/racecheck.log) 6건도 통과했고, shared-memory hazard는 0건이었다. CUDA graph 첫 replay 후 결과와 scale 변경을 확인했다.
- 기존 H100 대비 출력·모든 gradient의 최대 상대 L2: L384 `4.15e-6`, L768 `3.77e-6` 이하. 두 길이 모두 실제 profiler에서 `F567Sm90` 실행과 native cache hit를 확인했다.
- 10개 엔진 파일을 포함한 패치의 전체 stack dry run, SHA-256, 정확한 이전 버전 적용, idempotent 재적용을 검증했다.

### 배포물과 원시 기록

- [엔진 패치](../../patches/miniworld-engine-trimul-f567-cute.patch), [manifest](../../patches/trimul-f567-cute-manifest.json), [커널 소스 스냅샷](f567-cute/output_f567.py)
- [L384 전체 config](f567-cute/final_tune_L384.json), [L768 전체 config](f567-cute/final_tune_L768.json)
- [L384 runtime·profiler](f567-cute/final_runtime_L384.json), [L768 runtime·profiler](f567-cute/final_runtime_L768.json)
- [테스트 XML](f567-cute/final_tests.xml), [memcheck](f567-cute/memcheck.log), [build 계약](f567-cute/final_plan.json), [패치 검증](f567-cute/patch-validation.json)
- [테스트 코드](../../tests/test_engine_trimul_f567_cute.py), [튜너](../../scripts/tune_trimul_f567_cute.py), [모델 검증·벤치마크](../../scripts/check_trimul_f567_cute_runtime.py), [도식 생성기](../../scripts/render_trimul_fusion.py)

원래 direct-store 구현 및 K streaming 대안은 실행 디렉터리에 증거로만 남긴다. 배포 패치는 최종 TMA 구현 하나만 포함한다. 설치본에는 이 패치를 아직 적용하지 않았으며 자동 활성화 잡도 등록하지 않았다.
