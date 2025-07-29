import torch
import triton
import click

from pathlib import Path

from team_gm.data.features import SequenceFeatures
from team_gm.modules.primitives import Transition, LayerNorm
from team_gm.modules.attentions import (
    AttentionPairBias,
    TriangleMultiplication,
    TriangleAttention,
)
from team_gm.modules import Pairformer
from team_gm.models.structure_flow import StructureFlowModel
from team_gm.utils.rigid_utils import Rigid


DEVICE = torch.device("cuda")


def bench_layer_norm(M, N, dtype, provider, mode, eps=1e-5, device=DEVICE):
    x_shape = (M * M, N)
    w_shape = (x_shape[-1],)
    triton_model = LayerNorm(w_shape, eps, True, implementation="triton").to(
        device, dtype
    )
    torch_model = LayerNorm(w_shape, eps, True, implementation="pytorch").to(
        device, dtype
    )

    x = torch.randn(x_shape, dtype=dtype, device=device).requires_grad_(True)
    dy = 0.1 * torch.randn_like(x)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model(x)
    elif provider == "torch":
        fn = lambda: torch_model(x)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y = fn()
        fn = lambda: y.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[x])


def bench_transition(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    torch_model = Transition(N, implementation="pytorch").to(device, dtype)
    triton_model = Transition(N, implementation="triton").to(device, dtype)

    x = torch.randn(1, M, M, N, dtype=dtype, device=device).requires_grad_(True)
    dy = 0.1 * torch.randn_like(x)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model(x)
    elif provider == "torch":
        fn = lambda: torch_model(x)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y = fn()
        fn = lambda: y.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[x])


def bench_attention_pair_bias(
    M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE
):
    if dtype != torch.bfloat16:
        return
    S = 384
    H = S // 32

    torch_model = AttentionPairBias(S, N, H, implementation="pytorch").to(device, dtype)
    triton_model = AttentionPairBias(S, N, H, implementation="triton").to(device, dtype)

    single = torch.randn(1, M, S, dtype=dtype).cuda().requires_grad_()
    pair = torch.randn(1, M, M, N, dtype=dtype).cuda().requires_grad_()
    mask = torch.rand(1, M).cuda() >= 0.5
    dy = 0.1 * torch.randn_like(single)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model(single, pair, mask)
    elif provider == "torch":
        fn = lambda: torch_model(single, pair, mask)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y = fn()
        fn = lambda: y.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[single, pair])


def bench_tri_multi(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    torch_model = TriangleMultiplication(implementation="pytorch").to(device, dtype)
    triton_model = TriangleMultiplication(implementation="triton").to(device, dtype)

    pair = torch.randn(1, M, M, N).to(DEVICE, dtype).requires_grad_()
    mask = torch.rand(1, M).to(DEVICE) >= 0.5
    dy = 0.1 * torch.randn_like(pair)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model(pair, mask)
    elif provider == "torch":
        fn = lambda: torch_model(pair, mask)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y = fn()
        fn = lambda: y.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[pair])


def bench_tri_attention(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    torch_model = TriangleAttention(d_pair=N, implementation="pytorch").to(device, dtype)
    triton_model = TriangleAttention(d_pair=N, implementation="triton").to(device, dtype)

    pair = torch.randn(1, M, M, N, dtype=dtype).to(DEVICE, dtype).requires_grad_()
    mask = torch.rand(1, M).to(DEVICE, dtype) >= 0.5
    dy = 0.1 * torch.randn_like(pair)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model(pair, mask)
    elif provider == "torch":
        fn = lambda: torch_model(pair, mask)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y = fn()
        fn = lambda: y.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[pair])


def bench_pairformer(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    S = 384
    torch_model = Pairformer(
        Pairformer.Config(n_block=1, d_pair=N, d_single=S, implementation="pytorch")
    ).to(device, dtype)
    triton_model = Pairformer(
        Pairformer.Config(n_block=1, d_pair=N, d_single=S, implementation="triton")
    ).to(device, dtype)

    single = torch.randn(1, M, S, dtype=dtype).cuda().requires_grad_()
    pair = torch.randn(1, M, M, N, dtype=dtype).cuda().requires_grad_()
    mask = torch.rand(1, M).cuda() >= 0.5
    dy = 0.1 * torch.randn_like(pair)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model.forward(pair, single, mask)
    elif provider == "torch":
        fn = lambda: torch_model.forward(pair, single, mask)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        y_pair, _ = fn()
        fn = lambda: y_pair.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles, grad_to_none=[pair, single])


def bench_model(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    raise NotImplementedError()
    sequence = SequenceFeatures(
        res_type=torch.zeros(M, dtype=torch.long),
        seq_idx=torch.arange(M, dtype=torch.long),
    )
    batch = NoisyBatch(
        atom_pos=torch.randn(M, 37, 3),
        atom_mask=torch.ones(M, 37, dtype=torch.bool),
        res_type=torch.zeros(M, dtype=torch.long),
        seq_idx=torch.arange(M, dtype=torch.long),
        res_mask=torch.ones(M, dtype=torch.bool),
        rigid=Rigid.identity((M,), requires_grad=False),
        name="test",
        t=torch.rand(1),
        rigid_t=Rigid.identity((M,), requires_grad=False),
        rigid_sc=Rigid.identity((M,), requires_grad=False),
    ).to(device=DEVICE, dtype=dtype)

    def change_config(config: dict, key, value):
        for k, v in config.items():
            if isinstance(v, dict):
                change_config(v, key, value)
            elif k == key:
                config[k] = value
        return config

    torch_config = StructureFlowModel.Config().as_dict()
    torch_config = change_config(torch_config, "implementation", "pytorch")
    torch_config = StructureFlowModel.Config(**torch_config)
    torch_model = StructureFlowModel(torch_config).to(device, dtype=dtype)

    triton_config = StructureFlowModel.Config().as_dict()
    triton_config = change_config(triton_config, "implementation", "triton")
    triton_config = StructureFlowModel.Config(**triton_config)
    triton_model = StructureFlowModel(triton_config).to(device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_model.forward(batch)
    elif provider == "torch":
        fn = lambda: torch_model.forward(batch)
    else:
        ValueError(f"Unsupported provider: {provider}")

    if mode == "backward":
        rigid_pred, _ = fn()
        trans = rigid_pred.get_trans()
        dy = 0.1 * torch.randn_like(trans)
        fn = lambda: trans.backward(dy, retain_graph=True)

    return triton.testing.do_bench(fn, quantiles=quantiles)


@click.command()
@click.option("-n", default=128, type=int)
@click.option(
    "--kernel",
    type=click.Choice(
        [
            "layernorm",
            "transition",
            "attention_pair_bias",
            "tri_multi",
            "tri_attention",
            "pairformer",
            "model",
        ]
    ),
)
def main(n, kernel):
    result_dir_path = Path(__file__).parent / "results"
    result_dir_path.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)

    default_args = {
        "x_names": ["M"],
        "x_vals": list(range(32, 400, 32)),
        "line_arg": "provider",
        "line_vals": ["triton", "torch"],
        "line_names": ["Triton", "Torch"],
        "styles": [("blue", "-"), ("green", "-")],
        "ylabel": "Time (ms)",
    }
    bench_list = []
    for dtype in [torch.float32]:
        for mode in ["forward", "backward"]:
            dtype_name = str(dtype).split(".")[-1]
            bench = triton.testing.Benchmark(
                **default_args,
                plot_name=f"{kernel}_{mode}_{dtype_name}_{n}_{gpu_name}",
                args={"mode": mode, "dtype": dtype, "N": n},
            )
            bench_list.append(bench)

    if kernel == "layernorm":
        kernel = bench_layer_norm
    elif kernel == "transition":
        kernel = bench_transition
    elif kernel == "attention_pair_bias":
        kernel = bench_attention_pair_bias
    elif kernel == "tri_multi":
        kernel = bench_tri_multi
    elif kernel == "tri_attention":
        kernel = bench_tri_attention
    elif kernel == "pairformer":
        kernel = bench_pairformer
    elif kernel == "model":
        kernel = bench_model
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    triton.testing.perf_report(bench_list)(kernel).run(
        print_data=True, save_path=result_dir_path
    )


if __name__ == "__main__":
    main()
