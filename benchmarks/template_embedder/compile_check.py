"""AF3TemplateEmbedder as the training model runs it: bf16, ``torch.compile(dynamic=False)``, an inference call and one training step.

Prints whether the compiled output matches the eager one and whether every parameter gets a finite gradient. Run it on a new engine
before relying on the batched / fused template paths:

    python -m benchmarks.template_embedder.compile_check
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.common import apply_engine_settings, header, use_repo_paths

use_repo_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-setting", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder
    from tests.test_af3_template import D_PAIR, make_inputs

    apply_engine_settings(args.engine_setting)
    torch.set_float32_matmul_precision("medium")
    embedder = AF3TemplateEmbedder(d_pair=D_PAIR).cuda().to(torch.bfloat16)
    pair, template, asym, token_mask = make_inputs()
    pair = pair.to(torch.bfloat16)
    print(header("template-compile"), f"batch_templates={embedder._batch_templates}, batch_limit={embedder._batch_limit}, "  # noqa: SLF001
          f"fuse_projections={embedder._fuse_projections}", flush=True)  # noqa: SLF001

    embedder.eval()
    with torch.no_grad():
        eager = embedder(pair, template, asym, token_mask)
    compiled = torch.compile(embedder, dynamic=False)
    with torch.no_grad():
        out = compiled(pair, template, asym, token_mask)
    relative = float((out.float() - eager.float()).norm() / eager.float().norm())
    print(f"COMPILE inference: compiled against eager relative {relative:.2e}, finite {bool(torch.isfinite(out).all())}")

    embedder.train()
    out = compiled(pair, template, asym, token_mask)
    out.float().square().mean().backward()
    missing = [n for n, p in embedder.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    print(f"COMPILE training step: finite output {bool(torch.isfinite(out).all())}, parameters without a finite gradient: {missing[:5]}")


if __name__ == "__main__":
    main()
