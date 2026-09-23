# B9+B10 H100 추가 최적화 4차

**유의미하고 일관된 추가 가속을 얻지 못해 기존 구현을 유지했다.**
기준 커밋은 `8c7d8b39`다. H1002장에서 병렬 실험했으며 Triton과 같은
융합·GEMM 누적 순서·BF16 경계·중간 버퍼·컨피그 축을 유지했다.
L384/L768을 대상으로 했고 L128 성능은 제외했다.

## 새로 시험한 구현

| 변경 | 실제 실행 확인 | 결과 |
|---|---|---|
| K 루프 전체 전개 / stage 주기로 부분 전개 | 주소·barrier 제어를 상수화한 cubin | 기존 최상 컨피그에서0.1–0.4% 수준 |
| 첫 front WGMMA의 명시적 초기화 / gate 값 조기 packing | 초기화 명령 제거와 레지스터 표현 확인 | 추가 개선은1% 미만; 다른 타일도 기존 최상 경로를 못 이김 |
| Front WGMMA 한 그룹을 실행 중인 채 다음 그룹 발행 | ordered wait1와 이전 슬롯 재사용 | 시험한5개 컨피그에서 느림 |
| 가중치 V를 rank-3 TMA descriptor로 표현 | 실제 `UTMALDG.3D`, 단계당 TMA3→2개 | 최상 컨피그에서는 대조군의 변동 수준 |
| M128 입력 F도 rank-3 전송 + register packing | 원래 WGMMA swizzle·buffer 유지, output 일치 | 기존 M64 최상 경로보다 느림 |

3D TMA는 새 전역 transpose나 copy가 아니다. 기존 storage의 좌표를
`(64, K, outer)`로 표현해 CuTe의 descriptor 분할을 피했다. 실제 stride를
사용하며, 적용 조건 밖에서는 원래 경로를 남겼다. 전송 명령 수 감소가
실제로 확인됐지만, 전송해야 하는 HBM 데이터는 그대로다.

마지막으로 최상 M64 컨피그에서 full-unroll·early packing·folded V를 함께
시험했다. 입력 할당2회 × 교대5라운드에서 L384 이득은1% 미만이었다.
L768 한 할당은 변동이 컸고, 다른 할당은510.70µs 대510.86µs로 거의 같았다.
이 결과를 안정적인 큰 개선이나15% 목표 달성으로 해석하지 않는다.

## 이번에는 dense BF16 roofline을 직접 수집

이전 기본 FP32/FP64 chart 대신 NCU Tensor Core roofline과
`sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off`를 수집했다.
일반 BF16 peak에는 sparse 성능이 포함되므로 dense 전용 peak를 사용했다.

유지한 커널의 L768, M=589824/KG128/KP1024/N128:

| 항목 | 값 |
|---|---:|
| 실제 dense BF16 HGMMA 연산량 |173.946GFLOP — `2MN(KG+KP)`와 정확히 일치 |
| NCU 계측 시간 |507.776µs |
| 달성 연산 처리량 |342.565TFLOP/s |
| 계측 clock의 dense BF16 compute peak |760.523TFLOP/s |
| 실제 DRAM read+write |1.50617GB |
| 실제 연산 집약도 |115.489FLOP/DRAM byte |
| 계측 clock의 HBM peak |3.352TB/s |
| 해당 집약도에서의 HBM roof |387.126TFLOP/s |
| HBM roof 대비 달성률 |**88.49%** |
| L2 throughput 사용률 |85.53% |

![B9+B10 L768 dense BF16 roofline](../../runs/trimul_sm90_bwd_round4_20260917/B9_B10_ROOFLINE_L768.svg)

이는 메모리 쪽 한계가 compute peak보다 낮다는 근거다. 측정한 DRAM 전송량을
고정하면 낙관적인 순수 메모리 처리 시간은 약449.33µs이지만, 동기화·지연을
무시한 계산이므로 실제 달성 시간을 보장하지 않는다.15% 개선이 물리적으로
불가능하다는 결론도 아니다. 계측 시간은 별도 CUDA graph 성능 측정과 구분한다.

## 연산 비용을 분리한 진단

**진단 목적으로만** gate GEMM·입력 TMA·barrier·output store를 유지하고
front HGMMA를 제거했다. 원래 수식을 계산하지 않으므로 최적화 후보나
production 코드로 취급하지 않는다.

| L | 정상 커널 µs | Front GEMM 제거 진단 µs | 시간 감소 |
|---|---:|---:|---:|
|384|132.725|128.395|3.26% |
|768|510.213|502.676|1.48% |

현재 전송 스케줄에서는 front 행렬곱 비용 대부분이 메모리 전송과 겹친다는
근거다. 모든 가능한 전송 스케줄의 하한을 증명하는 실험은 아니다.
NCU에서 L768 진단의 실제 BF16 연산량은19.327GFLOP으로 gate GEMM만 남긴
계산과 정확히 일치했다. DRAM read+write는 정상 커널과 약0.020%만 달랐다.
따라서 입력 전송까지 제거돼 빨라진 실험으로 해석하지 않는다.

## 적용 상태와 근거

기록된 정상 수식 후보들은 비교한 production-size 출력이 기존과 일치했다.
느리거나 이득이 작은 후보를 반영하지 않았으므로 새 production 안전성 검증,
전체 모듈 재측정, cache 재빌드 완료를 주장하지 않는다. 기존 엔진 구현과
배선은 유지한다. **B9+B10의 Triton 대비1.15× 목표는 아직 미달이다.**
실험용 Slurm13240의 GPU2장을 반납했으며 기존 학습·캐시 작업은 유지했다.

- [최종 source·실험 적용 상태](../../runs/trimul_sm90_bwd_round4_20260917/final-evidence.json)
- [실험 기록](../../runs/trimul_sm90_bwd_round4_20260917/README.md)
- [제어·누적·NCU 실험](../../runs/trimul_sm90_bwd_round4_20260917/control/REPORT.md)
- [파이프라인·rank-3 TMA 실험](../../runs/trimul_sm90_bwd_round4_20260917/pipeline/REPORT.md)
- [Roofline 수치와 가정](../../runs/trimul_sm90_bwd_round4_20260917/roofline-L768.json)
- [독립 할당 최종 조합](../../runs/trimul_sm90_bwd_round4_20260917/pipeline/final_combination.json)
- [이전 backward 실험](BACKWARD_ROUND3.md)
