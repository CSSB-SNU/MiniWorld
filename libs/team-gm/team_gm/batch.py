import torch

from typing_extensions import Self
from dataclasses import fields
from functools import cached_property

from team_gm.utils.data_utils import auto_tensor_collate, move_data_to_device


class BaseBatch:
    def to(self, device: str | torch.device):
        if not isinstance(device, torch.device):
            device = torch.device(device)

        return self.__class__(
            **{
                f.name: move_data_to_device(getattr(self, f.name), device)
                for f in fields(self)
            }
        )

    @classmethod
    def collate_fn(cls, batch_list: list[Self]) -> Self:
        if not batch_list:
            raise ValueError("batch_list cannot be empty.")
        if not all(isinstance(b, cls) for b in batch_list):
            raise TypeError(
                f"Expected all items in batch_list to be of type {cls.__name__}, "
                f"but got {[type(b) for b in batch_list]}."
            )

        collated_data = {}
        for f in fields(cls):
            data_list = [getattr(b, f.name) for b in batch_list]
            data_type = type(data_list[0])
            if not all(isinstance(d, data_type) for d in data_list):
                raise TypeError(
                    f"Expected all items in data_list for field '{f.name}' to be of "
                    f"type {data_type.__name__}, but got {[type(d) for d in data_list]}."
                )
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
                raise TypeError(
                    f"Unsupported data type for field '{f.name}': {data_type}."
                )

        return cls(**collated_data)

    def duplicate(self, num: int) -> Self:
        return self.collate_fn([self] * num)

    @classmethod
    def from_sample(cls, **kwargs) -> Self:
        batched_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                batched_kwargs[k] = v.unsqueeze(0)
            elif isinstance(v, BaseBatch):
                batched_kwargs[k] = v
            elif isinstance(v, type(None)):
                batched_kwargs[k] = None
            else:
                batched_kwargs[k] = [v]

        return cls(**batched_kwargs)

    @cached_property
    def batch_size(self) -> int:
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
                raise TypeError(
                    f"Unsupported data type for field '{f.name}': {type(data)}."
                )

        if not batch_size_dict:
            return 0

        batch_size = next(iter(batch_size_dict.values()))
        if not all(size == batch_size for size in batch_size_dict.values()):
            raise ValueError(
                "Batch size is not consistent across all fields. "
                f"Sizes found: {batch_size_dict}."
            )
        return batch_size
