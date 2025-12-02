import torch
import torch.nn as nn
from dataclasses import field
from torch.amp import autocast
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


def convert_dtype(dtype_str):
    if dtype_str == "torch.float32":
        return torch.float32
    elif dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.bfloat16":
        return torch.bfloat16
    return dtype_str


class DtypeConfig(BaseModel):
    # allow torch.dtype (and any other non-PEP types) without schema errors
    model_config = ConfigDict(arbitrary_types_allowed=True)

    op_dtype: str = Field(
        default="torch.float32",
        metadata={"help": "Data type for operations"},
    )
    out_dtype: str = Field(
        default="torch.float32",
        metadata={"help": "Data type for output tensors"},
    )

    # @field_validator("op_dtype", "out_dtype", mode="before")
    # def _parse_dtype(cls, v):
    #     # if user passed a string, convert it; otherwise leave dtype as-is
    #     if isinstance(v, str):
    #         return convert_dtype(v)
    #     return v


# -------------------------------------------------------------------
# 2) PrecisionConfig
# -------------------------------------------------------------------


class PrecisionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # you can pass a string here, e.g. "torch.bfloat16"
    input: str
    default: DtypeConfig
    precision_map: dict[str, DtypeConfig] = Field(default_factory=dict)
    use_scaler: bool = Field(
        default=True,
        metadata={"help": "Whether to use a GradScaler for mixed precision training."},
    )

    @field_validator("precision_map", mode="before")
    def _build_precision_map(cls, v):
        # 1) support OmegaConf DictConfig
        if isinstance(v, DictConfig):
            v = OmegaConf.to_container(v, resolve=True)
        # 2) wrap everything into DtypeConfig
        new_map: dict[str, DtypeConfig] = {}
        for name, entry in v.items():
            if isinstance(entry, dict):
                new_map[name] = DtypeConfig(**entry)
            elif isinstance(entry, DtypeConfig):
                new_map[name] = entry
            else:
                raise TypeError(
                    f"precision_map['{name}'] must be dict or DtypeConfig, got {type(entry)}"
                )
        return new_map

def cast_data(data, dtype):
    if isinstance(data, torch.Tensor):
        if (
            data.dtype != torch.float32
            and data.dtype != torch.bfloat16
            and data.dtype != torch.float16
        ):
            return data  # keep bool or integer tensors as-is
        return data.to(dtype)
    elif isinstance(data, list | tuple):
        return type(data)(cast_data(d, dtype) for d in data)
    elif isinstance(data, dict):
        return {k: cast_data(v, dtype) for k, v in data.items()}
    else:
        return data

def _wrap_with_casts(module_cls, op_dtype, out_dtype):
    class Wrapped(module_cls):  # type: ignore
        def forward(self, *args, **kwargs):
            # 1) autocast ops
            device_type = args[0].device.type
            op_dtype_torch = convert_dtype(op_dtype)
            out_dtype_torch = convert_dtype(out_dtype)

            args = cast_data(args, op_dtype_torch)
            kwargs = cast_data(kwargs, op_dtype_torch)

            if op_dtype_torch != torch.float32:
                with autocast(device_type=device_type, enabled=True, dtype=op_dtype_torch):
                    out = super().forward(*args, **kwargs)
            else:
                out = super().forward(*args, **kwargs)

            out = cast_data(out, out_dtype_torch)  # ensure output is in out_dtype
            return out

    Wrapped.__name__ = f"{module_cls.__name__}_op{op_dtype}_out{out_dtype}"
    return Wrapped


@contextmanager
def precision_manager(model: nn.Module, precision_config: PrecisionConfig):
    """
    On enter: wraps every submodule named in precision_map.
    On exit: restores their original __class__.
    """
    # record originals
    precision_map = precision_config.precision_map
    default_dtype = precision_config.default
    # if default_dtype.op_dtype == default_dtype.out_dtype== torch.float32 :
    #     default_dtype = None
    original_classes = {}
    for name, module in model.named_modules():
        for modified_name in precision_map:
            if name.split(".")[-1] == modified_name:
                original_classes[module] = module.__class__
                op_dtype = precision_map[modified_name].op_dtype
                out_dtype = precision_map[modified_name].out_dtype
                module.__class__ = _wrap_with_casts(
                    module.__class__, op_dtype, out_dtype
                )
    try:
        yield
    finally:
        # restore
        for module, orig_cls in original_classes.items():
            module.__class__ = orig_cls


if __name__ == "__main__":

    class InnerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.act = nn.ReLU()

        def forward(self, x):
            print(f" before linear: {x.dtype}")
            x1 = self.linear(x)
            print(f" before act   : {x1.dtype}")
            x2 = self.act(x1)
            print(f" after act    : {x2.dtype}")
            return x2

    class TopModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = InnerBlock()
            self.head = nn.Linear(4, 1)

        def forward(self, x):
            x = self.block(x)
            print(f" after act    : {x.dtype}")
            y = self.head(x)
            print(f" head output  : {y.dtype}")
            return y

    precision_map = {
        "block.linear": (torch.bfloat16, torch.bfloat16),
        "block.act": (torch.bfloat16, torch.float32),
        # "block": (torch.bfloat16, torch.float32),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TopModel().to(device)
    inp = torch.randn(2, 4, device=device)

    # Only inside this block are block.linear & block.act wrapped
    with precision_manager(model, precision_map):
        # You can also nest an outer autocast if you like:
        out = model(inp)

    # Outside the block, model is back to normal if you need it