#!/usr/bin/env bash
# The H100 kernel-experiment records (triattn, trimul_*, anthropic_*, norm_cuda, ...; 6,664 files, 53 MB) were removed from main
# and kept under the tag archive/h100-kernel-experiments-20260927. Docs under docs/kernels and docs/graphical-abstract link to them as raw data.
#   bash scripts/restore_kernel_experiments.sh          # extracts runs/ from the tag into this checkout (untracked, ignored)
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse -q --verify "refs/tags/archive/h100-kernel-experiments-20260927" >/dev/null || git fetch origin "refs/tags/archive/h100-kernel-experiments-20260927:refs/tags/archive/h100-kernel-experiments-20260927"
git archive "archive/h100-kernel-experiments-20260927" runs | tar -x
echo "restored $(git ls-tree -r --name-only archive/h100-kernel-experiments-20260927 runs | wc -l) files under runs/"
