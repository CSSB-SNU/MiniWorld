from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, TypeVar, overload

import torch
import torch.nn.functional as F
from typing_extensions import Self


def auto_tensor_collate(
    data_list: list[torch.Tensor],
    batch_dim: int | None = None,
) -> torch.Tensor:
    """Automatic collate tensors with padding.

    Tensors will be padded to match the maximum size along their dimensions.
    Please note that all tensors in `data_list` must have the same number of dimensions.

    Parameters
    ----------
    data_list: list[torch.Tensor]
        A list of Tensor data.
    batch_dim: int or None, default = None
        If specified, the tensors will be concatenated along this dimension.
        If None, the tensors will be stacked into a new dimension.

    Returns
    -------
    padded_data: torch.Tensor
        Padded tensor with dimensions extended to each dimension's maximum size.

    """
    if not data_list:
        msg = "data_list cannot be empty."
        raise ValueError(msg)

    if not all(data.ndim == data_list[0].ndim for data in data_list):
        msg = (
            "All tensors in data_list must have the same number of dimensions. "
            f"Got {[data.shape for data in data_list]}."
        )
        raise ValueError(msg)

    shapes = torch.tensor([data.shape for data in data_list])
    max_shape = shapes.max(dim=0).values

    padding_needed = max_shape - shapes
    if batch_dim is not None:
        padding_needed[:, batch_dim] = 0

    if not torch.all(padding_needed == 0):
        padded_list = []
        for i, data in enumerate(data_list):
            current_padding = padding_needed[i]
            padding_spec = [
                p for pad in reversed(current_padding.tolist()) for p in (0, pad)
            ]
            padded_list.append(F.pad(data, padding_spec, value=0))
    else:
        padded_list = data_list

    if batch_dim is None:
        return torch.stack(padded_list, dim=0)
    return torch.cat(padded_list, dim=batch_dim)


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@overload
def move_to_device(data: torch.Tensor, device: str | torch.device) -> torch.Tensor: ...


@overload
def move_to_device(data: dict[K, V], device: str | torch.device) -> dict[K, V]: ...


@overload
def move_to_device(data: list[T], device: str | torch.device) -> list[T]: ...


@overload
def move_to_device(
    data: tuple[T, ...],
    device: str | torch.device,
) -> tuple[T, ...]: ...


@overload
def move_to_device(data: BaseBatch, device: str | torch.device) -> BaseBatch: ...


def move_to_device(data: Any, device: str | torch.device) -> Any:
    """Move data to a specified device.

    Parameters
    ----------
    data: Any
        Data to be moved. Can be a torch.Tensor, dict, list, tuple, batch subclass,
        or other types.
    device: str or torch.device
        Device to move the data to.

    Notes
    -----
    If data type is: dict, list, or tuple, the function will be applied recursively to
    move all elements to the specified device.
    If data is a subclass of BaseBatch, its `to` method will be used to move it to the
    device.
    If data is none of the supported types, it will be returned as is.

    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    if isinstance(data, BaseBatch):
        return data.to(device=device)
    return data


@dataclass
class BaseBatch:
    """Base class for batch data structures.

    To use, inherit from this class and attach the `@dataclass` decorator to your
    class.


    Examples
    --------
    .. code-block:: python

        @dataclass
        class MyBatch(BaseBatch):
            components: torch.Tensor
            names: list[str]

    """

    def to(self, device: str | torch.device) -> Self:
        """Move all tensor of the batch to the specified device."""
        return self.__class__(
            **{
                f.name: move_to_device(getattr(self, f.name), device)
                for f in fields(self)
            },
        )

    @classmethod
    def collate_fn(cls, batch_list: list[Self]) -> Self:
        """Collate a list of batch instances into a single batch instance.

        Parameters
        ----------
        batch_list : list[Self]
            List of batch instances to collate.

        Returns
        -------
        Self
            Collated batch instance.

        Examples
        --------
        .. code-block:: python

            collated_batch = MyBatch.collate_fn([batch1, batch2, batch3])

        """
        if not batch_list:
            msg = "batch_list cannot be empty."
            raise ValueError(msg)
        if not all(isinstance(b, cls) for b in batch_list):
            msg = (
                f"Expected all items in batch_list to be of type {cls.__name__}, "
                f"but got {[type(b) for b in batch_list]}."
            )
            raise TypeError(msg)

        collated_data = {}
        for f in fields(cls):
            data_list = [getattr(b, f.name) for b in batch_list]
            data_type = type(data_list[0])
            if not all(isinstance(d, data_type) for d in data_list):
                msg = (
                    f"Expected all items in data_list for field '{f.name}' to be of "
                    f"type {data_type.__name__}, but got "
                    f"{[type(d) for d in data_list]}."
                )
                raise TypeError(msg)

            if data_type is torch.Tensor:
                collated_data[f.name] = auto_tensor_collate(data_list, 0)
            elif issubclass(data_type, BaseBatch):
                collated_data[f.name] = data_type.collate_fn(data_list)
            elif data_type is type(None):
                collated_data[f.name] = None
            elif data_type is list:
                collated_data[f.name] = [
                    item for sublist in data_list for item in sublist
                ]
            else:
                msg = f"Unsupported data type for field '{f.name}': {data_type}."
                raise TypeError(msg)

        return cls(**collated_data)

    def duplicate(self, num: int) -> Self:
        """Duplicate the current batch instance multiple times."""
        return self.collate_fn([self] * num)

    @classmethod
    def from_sample(cls, **kwargs: Any) -> Self:
        """Create a batch instance from individual sample data.

        Parameters
        ----------
        **kwargs
            Individual sample data for each field.

        Returns
        -------
        Self
            Batch instance created from the sample data.

        Notes
        -----
        Each data field in `kwargs` will be expanded to create a batch of size 1.

        Examples
        --------
        .. code-block:: python

            batch = MyBatch.from_sample(components=torch.ones(3, 4), names="sample")
            batch.batch_size  # Should be 1
            batch.components.shape  # Should be (1, 3, 4)
            batch.names  # Should be ['sample']

        """
        batched_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                batched_kwargs[k] = v.unsqueeze(0)
            elif isinstance(v, BaseBatch):
                batched_kwargs[k] = v
            elif v is None:
                batched_kwargs[k] = None
            else:
                batched_kwargs[k] = [v]

        return cls(**batched_kwargs)

    @cached_property
    def batch_size(self) -> int:
        """Return the batch size of the current instance."""
        batch_size_dict = {}
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, torch.Tensor):
                batch_size_dict[f.name] = data.shape[0]
            elif isinstance(data, BaseBatch):
                batch_size_dict[f.name] = data.batch_size
            elif data is None:
                pass
            elif isinstance(data, list):
                batch_size_dict[f.name] = len(data)
            else:
                msg = f"Unsupported data type for field '{f.name}': {type(data)}."
                raise TypeError(msg)

        if not batch_size_dict:
            return 0

        batch_size = next(iter(batch_size_dict.values()))
        if not all(size == batch_size for size in batch_size_dict.values()):
            msg = (
                "Batch size is not consistent across all fields. "
                f"Sizes found: {batch_size_dict}."
            )
            raise ValueError(msg)
        return batch_size

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, index: int | slice) -> Self:
        """Slice the batch instance along the batch dimension.

        Parameters
        ----------
        index : int | slice
            Index or slice object to select data along the batch dimension.

        Returns
        -------
        Self
            Sliced batch instance with batch dimension preserved.

        Examples
        --------
        .. code-block:: python

            sliced_batch = my_batch[0:5]  # Returns batch of size 5
            single_batch = my_batch[2]    # Returns batch of size 1
            last_item = my_batch[-1]      # Returns last item as batch of size 1

        """
        if isinstance(index, int):
            batch_size = self.batch_size
            norm_index = index + batch_size if index < 0 else index
            if norm_index < 0 or norm_index >= batch_size:
                msg = f"Index {index} is out of bounds for batch of size {batch_size}"
                raise IndexError(msg)

            index = slice(norm_index, norm_index + 1)

        sliced_data = {}
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, (torch.Tensor, BaseBatch, list)):
                sliced_data[f.name] = data[index]
            elif data is None:
                sliced_data[f.name] = None
            else:
                msg = f"Unsupported data type for field '{f.name}': {type(data)}."
                raise TypeError(msg)

        return self.__class__(**sliced_data)
