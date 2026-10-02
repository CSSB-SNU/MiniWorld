"""Shared helpers of the benchmarks: repository paths, CUDA-graph timing, the kernel-family breakdown, and the run header.

Every benchmark here times what training runs (compiled, steady state, one CUDA graph) and prints, first, the facts that decide
whether two runs are comparable: the card, the library versions, the engine settings in force. Compare numbers only inside one
process or between runs whose headers match.
"""

from __future__ import annotations

import ast
import collections
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def use_repo_paths() -> None:
    """Put the repository root (``tests``, ``benchmarks``) and ``scripts`` (``run_miniworld_*_train``) on ``sys.path``."""
    for path in (REPO_ROOT / "scripts", REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def apply_engine_settings(pairs: list[str]) -> None:
    """Set ``miniworld_engine.settings`` fields from ``key=value`` strings (value read as a Python literal).

    Raises ``TypeError`` for a field the installed engine does not have, so a switch that this engine predates is never silently
    ignored.
    """
    if not pairs:
        return
    from miniworld_engine import settings

    settings.configure(**{key: ast.literal_eval(value) for key, value in (pair.split("=", 1) for pair in pairs)})


def header(tag: str) -> str:
    """One line naming the card, the library versions and the engine settings that change a measurement."""
    import triton

    props = torch.cuda.get_device_properties(0)
    fields = {
        "device": props.name,
        "capability": "%d.%d" % (props.major, props.minor),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        from miniworld_engine import settings

        current = settings.current()
        for name in ("b200_engine_triton", "swa_dit_fused", "swa_flash_saves_lse", "compile_wrap", "engine_backend"):
            if hasattr(current, name):
                fields[name] = getattr(current, name)
    except ImportError:
        fields["engine"] = "not installed"
    return f"HEADER [{tag}] " + ", ".join(f"{key}={value}" for key, value in fields.items())


def graph_ms(step, *, iters: int = 30, warmup: int = 3) -> float:
    """Median wall time (ms) of one replay of ``step`` captured as a CUDA graph.

    ``step`` runs the work to time (forward, or forward and backward) on tensors that stay alive between calls. It is warmed up
    on a side stream first, as ``torch.cuda.graph`` requires, then captured once and replayed.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def kernel_class(name: str) -> str:
    """Family of a CUDA kernel by its name: inductor, FlashAttention, GEMM, memcpy, or everything else (aten and engine)."""
    lowered = name.lower()
    if lowered.startswith("triton_"):
        return "inductor"
    if "flash" in lowered:
        return "FA4"
    if any(key in lowered for key in ("nvjet", "cutlass", "gemm", "cublas", "splitk")):
        return "GEMM"
    if "memcpy" in lowered or "memset" in lowered:
        return "memcpy"
    return "aten/other"


def kernel_breakdown(replay) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Profile one call of ``replay`` and return ``(per kernel name, per family)`` as ``{key: [launches, device microseconds]}``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        replay()
        torch.cuda.synchronize()
    per_kernel: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            entry = per_kernel[event.name]
            entry[0] += 1
            entry[1] += event.device_time
    per_family: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    for name, (launches, micros) in per_kernel.items():
        family = per_family[kernel_class(name)]
        family[0] += launches
        family[1] += micros
    return per_kernel, per_family

