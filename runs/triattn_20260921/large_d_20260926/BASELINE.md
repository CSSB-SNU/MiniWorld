# Original Triton wide baseline

Job19395, H100 node02, BF16, B1/H4, FP32 LN affine. Full training block, nonzero weights, masked inputs. CUDA Graph replay with CUDA-event timings. Width is pair and total QKV width, not per-head width.

| Width | Head dim | L | Direction | FWD ms | BWD ms | F+B ms |
|---:|---:|---:|:---|---:|---:|---:|
| 256 | 64 | 384 | starting | 0.843 | 2.459 | 3.294 |
| 256 | 64 | 384 | ending | 1.038 | 2.593 | 3.630 |
| 256 | 64 | 768 | starting | 4.107 | 12.919 | 17.033 |
| 256 | 64 | 768 | ending | 4.875 | 13.470 | 18.318 |
| 512 | 128 | 384 | starting | 1.639 | 4.396 | 6.035 |
| 512 | 128 | 384 | ending | 2.011 | 4.662 | 6.673 |
| 512 | 128 | 768 | starting | 8.054 | 22.530 | 30.758 |
| 512 | 128 | 768 | ending | 9.468 | 23.597 | 33.289 |

Raw JSONs retain the exact original-engine source hashes, independent original/current Triton autotune choices, every gradient comparison and profiler kernel timings. Original/current equivalence passed with independent namespace tuning and FP32 atomic LN-affine reduction differences recorded.

The first baseline harness attempts incorrectly demanded bitwise equality or reused autograd leaves from a default-stream eager backward. Final harness uses a fresh model/input for each timing regime and creates the backward forward-graph on the capture stream.
