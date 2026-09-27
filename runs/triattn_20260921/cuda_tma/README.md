# CUDA/TMA TriangleAttention surrounds

The qualified CUDA implementation is installed as the default in
`../oc/opt_core/kernels/triattn_surround_tma/`. See [CUDA_REPORT.md](../CUDA_REPORT.md)
for final performance, correctness, bandwidth definitions, and limitations.

The final build reads those installed C++ sources; `native/` and the other `.cu`
files here are development snapshots, not the serving source of truth.

From the MiniWorld root:

```sh
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/build_native.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/install_native.py
sbatch runs/triattn_20260921/cuda_tma/serving.sbatch
sbatch runs/triattn_20260921/cuda_tma/roof_serving.sbatch
```

`serving-e{0,1}-L{384,768,1024}.json` contains final same-job, alternating,
three-round results and correctness checks. `serving-roof-summary.json` is the
unit-aware summary of the final NCU CSV files. Jobs: 13867 and 13870.

Use `FPF_TRIATT_BACKEND=triton` to select the previous tuned surrounds. Historical
experiment scripts predate the default-route switch; their recorded JSON and
logs are retained as evidence, but reproducing their old baselines requires
explicitly selecting the previous backend.

The native tensor-map kernel arguments use `CUTE_GRID_CONSTANT`. Omitting it
caused descriptor copies to local memory and an illegal TMA access in the first
prototype. Register-heavy warp-specialized and scalar-copy versions were slower;
see the report for the negative results as well as the successful versions.
