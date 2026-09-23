"""Run the engine's unmodified module benchmark; retain repetitions and a GPU trace.

Copy benchmarks/runners/bench.py and its module YAML into --run/harness first.
The snapshot avoids the external runner prepending an older engine checkout.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--implementation', choices=['pytorch', 'triton', 'miniworld'], required=True)
    parser.add_argument('--graph', choices=['disabled', 'manual'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--ln-partial', type=Path, help='Experimental A/B config report')
    args = parser.parse_args()
    import torch
    import miniworld_engine
    from miniworld_engine.autotune import cache
    from omegaconf import OmegaConf
    if args.ln_partial:
        from trimul_ln_partial_experiment import install_from_report
        install_from_report(args.ln_partial)

    spec = importlib.util.spec_from_file_location('engine_module_bench', args.run / 'harness/benchmarks/runners/bench.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    config = OmegaConf.to_container(OmegaConf.load(args.run / 'harness/bench.yaml'))
    config.update(mode='training', min_seq_len=384, max_seq_len=384,
                  implementations=[args.implementation], cudagraph=args.graph,
                  dropout=0.25)
    conf = bench.BenchConfig(**config)
    torch.backends.cuda.matmul.allow_tf32 = conf.allow_tf32
    torch.backends.cudnn.allow_tf32 = conf.allow_tf32

    class NoFabric:
        @staticmethod
        def setup_module(module):
            return module

        @staticmethod
        def backward(tensor, gradient):
            tensor.backward(gradient)

    report = dict(engine_file=miniworld_engine.__file__, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, config=conf.model_dump(), repetitions=[],
                  scope='Module forward plus input and all parameter gradients; no optimizer. '
                        'Runner model.compile() uses default dynamic=None at one fixed L=384. '
                        'Disabled graph clears grads to None per iteration; manual graph accumulates static grads.',
                  accuracy_note='Runner compares independently drawn dropout masks; its accuracy fields '
                                'are not a matched-dropout correctness test.')
    report['cache_misses'] = []
    report['native_cache_warnings'] = []
    report['ln_partial_experiment'] = str(args.ln_partial) if args.ln_partial else None
    report['cache_lookups'] = []
    original_miss = cache._miss
    original_subset = cache._cached_subset
    original_warning = cache._warn_once
    warnings_seen = set()

    def observed_warning(op, gk, tag, reason, *pos, **kw):
        key=(op,reason)
        if not op.endswith('_triton') and key not in warnings_seen:
            warnings_seen.add(key)
            report['native_cache_warnings'].append(dict(op=op,reason=reason))
        return original_warning(op,gk,tag,reason,*pos,**kw)

    cache._warn_once = observed_warning

    def observed_miss(op, *pos, **kw):
        report['cache_misses'].append(dict(op=op, reason=str(pos[:3])))
        return original_miss(op, *pos, **kw)

    cache._miss = observed_miss

    def observed_subset(tuner, configs, nargs, meta):
        subset = original_subset(tuner, configs, nargs, meta)
        if tuner.fn.__name__ in ('_bidir_front_kernel', '_ln_mat_kernel', '_ln_bwd_kernel'):
            report['cache_lookups'].append(dict(
                kernel=tuner.fn.__name__, bucket=cache.bucket_of_autotuner(tuner,nargs,meta),
                candidates=[cache.as_cfg_dict(c) for c in subset or []]))
        return subset

    cache._cached_subset = observed_subset
    original = bench.bench_time

    def measured(func, warmup=10, rep=100, grad_to_none=None):
        for _ in range(5):
            result = original(func, warmup, rep, grad_to_none)
            report['repetitions'].append(result)
            print('TIMING', result, flush=True)
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                for _ in range(3):
                    for p in grad_to_none or []:
                        p.grad = None
                    func()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
            report['profile'] = [dict(name=e.key, count=e.count,
                                      self_cuda_us=e.self_device_time_total,
                                      total_cuda_us=e.device_time_total,
                                      shapes=str(e.input_shapes))
                                 for e in prof.key_averages(group_by_input_shape=True)
                                 if e.device_time_total > 0]
            report['profile'].sort(key=lambda e: -e['self_cuda_us'])
        return dict(median_ms=statistics.median(r['median_ms'] for r in report['repetitions']))

    bench.bench_time = measured
    with bench.forward_stream(conf):
        result = bench.bench_module_triangle_multiplication_bidirectional(conf, 384, args.implementation, NoFabric())
    report['result'] = result._asdict()
    args.output.write_text(json.dumps(report, indent=2))
    print('RESULT', json.dumps(report['result']), flush=True)


if __name__ == '__main__':
    main()
