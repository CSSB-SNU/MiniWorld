from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn
from torch.amp import autocast

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def convert_dtype(dtype_str: str) -> torch.dtype | str:
    """Convert string representation of dtype to torch.dtype."""
    if dtype_str == "torch.float32":
        return torch.float32
    if dtype_str == "torch.float16":
        return torch.float16
    if dtype_str == "torch.bfloat16":
        return torch.bfloat16
    return dtype_str


class DtypeConfig(BaseModel):
    """Configuration for data types used in operations and outputs."""

    # allow torch.dtype (and any other non-PEP types) without schema errors
    model_config = ConfigDict(arbitrary_types_allowed=True)

    op_dtype: str = Field(
        default="torch.float32",
        json_schema_extra={"help": "Data type for operations"},
    )
    out_dtype: str = Field(
        default="torch.float32",
        json_schema_extra={"help": "Data type for output tensors"},
    )


# -------------------------------------------------------------------
# 2) PrecisionConfig
# -------------------------------------------------------------------


class PrecisionConfig(BaseModel):
    """Configuration for precision management."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # you can pass a string here, e.g. "torch.bfloat16"
    input: str
    default: DtypeConfig
    precision_map: dict[str, DtypeConfig] = Field(default_factory=dict)
    use_scaler: bool = Field(
        default=True,
        json_schema_extra={"help": "Whether to use a GradScaler for mixed precision training."},
    )

    @field_validator("precision_map", mode="before")
    @classmethod
    def _build_precision_map(
        cls, v: dict[str, dict] | dict[str, DtypeConfig] | DictConfig
    ) -> dict[str, DtypeConfig]:
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
                msg = f"precision_map['{name}'] must be dict or DtypeConfig, got {type(entry)}"
                raise TypeError(msg)
        return new_map


def cast_data(data: Any, dtype: torch.dtype) -> Any:
    """Recursively cast data to the given dtype."""
    if isinstance(data, torch.Tensor):
        if data.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return data  # keep bool or integer tensors as-is
        return data.to(dtype)
    if isinstance(data, list | tuple):
        return type(data)(cast_data(d, dtype) for d in data)
    if isinstance(data, dict):
        return {k: cast_data(v, dtype) for k, v in data.items()}
    return data


def _wrap_with_casts(
    module_cls: type[ModuleT], op_dtype: str, out_dtype: str
) -> type[ModuleT]:
    """Return a subclass of module_cls that casts inputs/outputs to desired dtypes."""

    class Wrapped(module_cls):
        def forward(
            self, *args: Any, **kwargs: Any
        ) -> torch.Tensor | list | tuple | dict:
            """Forward with casts to op_dtype and out_dtype."""
            # 1) autocast ops
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            op_dtype_torch = convert_dtype(op_dtype)
            out_dtype_torch = convert_dtype(out_dtype)

            args = cast_data(args, op_dtype_torch)
            kwargs = cast_data(kwargs, op_dtype_torch)

            if op_dtype_torch != torch.float32:
                with autocast(
                    device_type=device_type, enabled=True, dtype=op_dtype_torch
                ):
                    out = super().forward(*args, **kwargs)
            else:
                out = super().forward(*args, **kwargs)

            return cast_data(out, out_dtype_torch)  # ensure output is in out_dtype

    Wrapped.__name__ = f"{module_cls.__name__}_op{op_dtype}_out{out_dtype}"
    return Wrapped


@contextmanager
def precision_manager(
    model: nn.Module, precision_config: PrecisionConfig
) -> Generator[None, None, None]:
    """Context manager to temporarily modify module classes for precision management."""
    # record originals
    precision_map = precision_config.precision_map
    original_classes = {}
    for name, module in model.named_modules():
        for modified_name in precision_map:
            if name.split(".")[-1] == modified_name:
                original_classes[module] = module.__class__
                op_dtype = precision_map[modified_name].op_dtype
                out_dtype = precision_map[modified_name].out_dtype
                module.__class__ = _wrap_with_casts(
                    module.__class__,
                    op_dtype,
                    out_dtype,
                )
    try:
        yield
    finally:
        # restore
        for module, orig_cls in original_classes.items():
            module.__class__ = orig_cls
