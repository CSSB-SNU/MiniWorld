# 설치된 Triton 융합 커널의 타일링 확인

2026-09-16. 설치본의 소스 SHA-256을 배포 manifest와 대조하고, grid/gmprobe CSV가 같은지 확인했다. 이번 확인은 소스·캐시·그룹 인덱싱 검사이며 GPU 검증은 동일 소스의 기존 기록을 참조한다.

## 현재 배선

- [Forward](../../tmp_kernel/trimul/TRIMUL_FORWARD.svg): F4는 별도, F5+F6+F7은 `_output_f567_kernel` 하나.
- [Backward](../../tmp_kernel/trimul/TRIMUL_BACKWARD.svg): B9+B10은 `_input_dual_bwd_kernel`, B11+B12는 `_ln_bwd_residual_kernel`.
- [확대 뷰어](../../tmp_kernel/trimul/TRIMUL_STATUS.html): 기본 화면은 현재 Triton. 이전 H100과 미적용 CuTe F567은 별도로 표시한다.

각 커널의 실선 경계 안에 실제 CSV의 타일·warp·stage 후보를 표시했다. 화살표는 데이터 의존성을 나타내며 병렬 실행을 뜻하지 않는다.

| 항목 | F567 | B9+B10 | B11+B12 |
|---|---|---|---|
| 타일링 | M/N/K | M/N/K | M행/K열, 전체 열 또는 2-pass reduction |
| GROUP_M | 1/2/4/8 | 1/2/4/8 | 해당 없음: LN row reduction |
| warps | 1/2/4/8 | 4/8 | 1/2/4/8/16/32 |
| stages | 2/3/4 | 2/3/4 | 1–6; covering tile은 1만 유지 |
| 선언 config 수 | 3,072 | 1,152 | 1,440 |
| 전용 튜닝 캐시 | L384·768 | L128·384·768 | L128·384·768 |

캐시 범위는 **H100·BF16·B1·D128**이다. F567의 L128은 동작하지만 전용 튜닝 캐시가 없어 fallback 후보에서 선택한다. 다른 장치·width·length 전체의 최적 성능을 보장하지 않는다.

## GROUP_M과 경계

두 GEMM 커널 모두 마지막 불완전한 M 그룹에서 실제 그룹 크기를 `min(남은 M 타일 수, GROUP_M)`로 계산한다. M 타일 수 1–130, N 타일 수 1–18, GROUP_M 1/2/4/8의 **9,360개 그리드**에서 모든 M/N 타일을 정확히 한 번 방문하는지 수학적으로 확인했다. GPU 경계 검사는 아래 기존 테스트 기록에 있다.

GROUP_M은 타일 방문 순서다. Reduction 축을 나누거나 결과를 atomic으로 합치는 설정이 아니다. N 타일이 하나면 GROUP_M별 방문 순서가 같아 중복 후보를 제거한다. M/N/K 나머지는 load/store mask로 처리한다.

## 지원 범위와 제약

1. F567과 B9+B10의 **실제 KP/KG 길이와 반복 횟수는 독립**이다. 물리 K 타일은 같은 `BLOCK_K`를 사용한다. 독립적인 두 K 타일 크기를 탐색하는 구현은 아니다.
2. F567의 `num_stages=1` 및 독립 K 타일 구현은 이전 검증에서 메모리 접근 오류가 발생해 지원 공간에서 제외됐다. 현재 배포된 공통 K 타일·stages 2–4 구현의 검사 결과와 구분한다.
3. 자원이 부족한 config는 실행 후보에서 제외한다. B9+B10의 `M128/N128/K128/stages4`는 shared memory 한도를 넘어 warps 4/8 모두 제외됐다. 선언된 모든 조합이 실행 가능하다는 뜻은 아니다.
4. B11+B12는 residual을 LN 미분 **후** 더하고, LN weight/bias gradient에 넣지 않는다. 기존 BF16 반올림 위치를 유지한다. GEMM 누적 순서 차이에 따른 gradient 차이는 backward 보고서에 수치로 기록했다.

## 근거

- [설치본 해시·CSV·캐시 키·그룹 검사 결과](tiling-audit.json)
- [F567 설정·제약·정확도](f567-config.md): **41 tests**, memcheck 오류 0건. M/N/K 경계, 불완전한 M 그룹, weight stride, 저장 projection/gate, dropout/residual을 검사했다.
- [Backward 설정·정확도·성능](triton-backward.md): **28 tests**, memcheck 오류 0건. 추가로 전체 모델·고정 dropout·zero scale·정적 compile·CUDA graph·설치본 cache hit를 검증했다.

검증한 구현과 shape 범위에서 정상 동작한다는 근거다. 모든 shape와 config 조합을 전수 검증했거나 가능한 모든 알고리즘보다 빠르다는 의미는 아니다.
