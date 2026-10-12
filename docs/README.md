# MiniWorld documentation

Current guides live at the top level of `docs/`; dated validation runs, audits and one-off experiments are kept under
`archive/` unchanged (paths inside them were rewritten, content was not).

| Where | What |
|---|---|
| [repositories.md](repositories.md) | repository map, engine pin history, validation notes |
| [pipeline.md](pipeline.md) | training phases and configs |
| [known-issues.md](known-issues.md) | validation limits and open problems |
| [kernels/](kernels/) | kernel development guides and status reports |
| [operations/](operations/) | cluster and launcher procedures |
| [paper/](paper/) | technical report draft |
| [graphical-abstract/](graphical-abstract/) | figures |
| [archive/](archive/) | dated checks and records: `graph-training-20260923`, `trimul-fusion`, `repository-audit-20260927`, `history`, ... |

Related checks and commands: `bash scripts/check.sh` (lint, format, type check) and
`bash scripts/b200/run_tests.sh` (the test suite on a B200 node).
