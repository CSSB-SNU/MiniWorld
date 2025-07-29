import torch
import numpy as np
from collections.abc import Mapping, Sequence
import dataclasses

import types
from typing import Any, Literal, TypeAlias  # for Python 3.11+

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
NumpyIndex: TypeAlias = Any


@dataclasses.dataclass(frozen=True)
class PaddingShapes:
    num_tokens: int
    msa_size: int
    num_chains: int
    num_templates: int
    num_atoms: int


@dataclasses.dataclass(frozen=True)
class AtomLayout:
    """Atom layout in a fixed shape (usually 1-dim or 2-dim).

    Examples for atom layouts are atom37, atom14, and similar.
    All members are np.ndarrays with the same shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]
    All string arrays should have dtype=object to avoid pitfalls with Numpy's
    fixed-size strings

    Attributes:
        atom_name: np.ndarray of str: atom names (e.g. 'CA', 'NE2'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.label_atom_id.
        res_id: np.ndarray of int: residue index (usually starting from 1) padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_seq_id.
        chain_id: np.ndarray of str: chain names (e.g. 'A', 'B') padding elements
        can have an arbitrary value. mmCIF field: _atom_site.label_seq_id.
        atom_element: np.ndarray of str: atom elements (e.g. 'C', 'N', 'O'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.type_symbol.
        res_name: np.ndarray of str: residue names (e.g. 'ARG', 'TRP') padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_comp_id.
        chain_type: np.ndarray of str: chain types (e.g. 'polypeptide(L)'). padding
        elements can have an arbitrary value. mmCIF field: _entity_poly.type OR
        _entity.type (for non-polymers).
        shape: shape of the layout (just returns atom_name.shape)
    """

    atom_name: np.ndarray
    res_id: np.ndarray
    chain_id: np.ndarray
    atom_element: np.ndarray | None = None
    res_name: np.ndarray | None = None
    chain_type: np.ndarray | None = None

    def __post_init__(self):
        """Assert all arrays have the same shape."""
        attribute_names = (
            "atom_name",
            "atom_element",
            "res_name",
            "res_id",
            "chain_id",
            "chain_type",
        )
        _assert_all_arrays_have_same_shape(
            obj=self,
            expected_shape=self.atom_name.shape,
            attribute_names=attribute_names,
        )
        # atom_name must have dtype object, such that we can convert it to bool to
        # obtain the mask
        if self.atom_name.dtype != object:
            raise ValueError(
                "atom_name must have dtype object, such that it can "
                "be converted converted to bool to obtain the mask"
            )

    def __getitem__(self, key: NumpyIndex) -> "AtomLayout":
        return AtomLayout(
            atom_name=self.atom_name[key],
            res_id=self.res_id[key],
            chain_id=self.chain_id[key],
            atom_element=(
                self.atom_element[key] if self.atom_element is not None else None
            ),
            res_name=(self.res_name[key] if self.res_name is not None else None),
            chain_type=(self.chain_type[key] if self.chain_type is not None else None),
        )

    def __eq__(self, other: "AtomLayout") -> bool:
        if not np.array_equal(self.atom_name, other.atom_name):
            return False

        mask = self.atom_name.astype(bool)
        # Check essential fields.
        for field in ("res_id", "chain_id"):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if not np.array_equal(my_arr[mask], other_arr[mask]):
                return False

        # Check optional fields.
        for field in ("atom_element", "res_name", "chain_type"):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if (
                my_arr is not None
                and other_arr is not None
                and not np.array_equal(my_arr[mask], other_arr[mask])
            ):
                return False

        return True

    def copy_and_pad_to(self, shape: tuple[int, ...]) -> "AtomLayout":
        """Copies and pads the layout to the requested shape.

        Args:
        shape: new shape for the atom layout

        Returns:
        a copy of the atom layout padded to the requested shape

        Raises:
        ValueError: incompatible shapes.
        """
        if len(shape) != len(self.atom_name.shape):
            raise ValueError(
                f"Incompatible shape {shape}. Current layout has shape {self.shape}."
            )

        if any(new < old for old, new in zip(self.atom_name.shape, shape)):
            raise ValueError(
                "Can't pad to a smaller shape. Current layout has shape "
                f"{self.shape} and you requested shape {shape}."
            )
        pad_width = [(0, new - old) for old, new in zip(self.atom_name.shape, shape)]
        pad_val = np.array("", dtype=object)
        return AtomLayout(
            atom_name=np.pad(self.atom_name, pad_width, constant_values=pad_val),
            res_id=np.pad(self.res_id, pad_width, constant_values=0),
            chain_id=np.pad(self.chain_id, pad_width, constant_values=pad_val),
            atom_element=(
                np.pad(self.atom_element, pad_width, constant_values=pad_val)
                if self.atom_element is not None
                else None
            ),
            res_name=(
                np.pad(self.res_name, pad_width, constant_values=pad_val)
                if self.res_name is not None
                else None
            ),
            chain_type=(
                np.pad(self.chain_type, pad_width, constant_values=pad_val)
                if self.chain_type is not None
                else None
            ),
        )

    def to_array(self) -> np.ndarray:
        """Stacks the fields to a numpy array with shape (6, <layout_shape>).

        Creates a pure numpy array of type `object` by stacking the 6 fields of the
        AtomLayout, i.e. (atom_name, atom_element, res_name, res_id, chain_id,
        chain_type). This method together with from_array() provides an easy way to
        apply pure numpy methods like np.concatenate() to `AtomLayout`s.

        Returns:
        np.ndarray of object with shape (6, <layout_shape>), e.g.
        array([['N', 'CA', 'C', ..., 'CB', 'CG', 'CD'],
        ['N', 'C', 'C', ..., 'C', 'C', 'C'],
        ['LEU', 'LEU', 'LEU', ..., 'PRO', 'PRO', 'PRO'],
        [1, 1, 1, ..., 403, 403, 403],
        ['A', 'A', 'A', ..., 'D', 'D', 'D'],
        ['polypeptide(L)', 'polypeptide(L)', ..., 'polypeptide(L)']],
        dtype=object)
        """
        if self.atom_element is None or self.res_name is None or self.chain_type is None:
            raise ValueError("All optional fields need to be present.")

        return np.stack(dataclasses.astuple(self), axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "AtomLayout":
        """Creates an AtomLayout object from a numpy array with shape (6, ...).

        see also to_array()
        Args:
        arr: np.ndarray of object with shape (6, <layout_shape>)

        Returns:
        AtomLayout object with shape (<layout_shape>)
        """
        if arr.shape[0] != 6:
            raise ValueError(
                "Given array must have shape (6, ...) to match the 6 fields of "
                "AtomLayout (atom_name, atom_element, res_name, res_id, chain_id, "
                f"chain_type). Your array has {arr.shape=}"
            )
        return cls(*arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.atom_name.shape


@dataclasses.dataclass(frozen=True)
class GatherInfo:
    """Gather indices to translate from one atom layout to another.

    All members are np or jnp ndarray (usually 1-dim or 2-dim) with the same
    shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]

    Attributes:
      gather_idxs: np or jnp ndarray of int: gather indices into a flattened array
      gather_mask: np or jnp ndarray of bool: mask for resulting array
      input_shape: np or jnp ndarray of int: the shape of the unflattened input
        array
      shape: output shape. Just returns gather_idxs.shape
    """

    gather_idxs: ArrayLike
    gather_mask: ArrayLike
    input_shape: np.ndarray

    def __post_init__(self):
        if self.gather_mask.shape != self.gather_idxs.shape:
            raise ValueError(
                "All arrays must have the same shape. Got\n"
                f"gather_idxs.shape = {self.gather_idxs.shape}\n"
                f"gather_mask.shape = {self.gather_mask.shape}\n"
            )

    def __getitem__(self, key: NumpyIndex) -> "GatherInfo":
        return GatherInfo(
            gather_idxs=self.gather_idxs[key],
            gather_mask=self.gather_mask[key],
            input_shape=self.input_shape,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.gather_idxs.shape

    def as_torch_from_np(self, xnp: types.ModuleType) -> "GatherInfo":
        return GatherInfo(
            gather_idxs=xnp.from_numpy(self.gather_idxs),
            gather_mask=xnp.from_numpy(self.gather_mask),
            input_shape=np.array(self.input_shape),
        )

    def as_np_from_torch(self, xnp: types.ModuleType) -> "GatherInfo":
        return GatherInfo(
            gather_idxs=xnp.numpy(self.gather_idxs),
            gather_mask=xnp.numpy(self.gather_mask),
            input_shape=self.input_shape,
        )

    def as_dict(
        self,
        key_prefix: str | None = None,
    ) -> dict[str, ArrayLike]:
        prefix = f"{key_prefix}:" if key_prefix else ""
        return {
            prefix + "gather_idxs": self.gather_idxs,
            prefix + "gather_mask": self.gather_mask,
            prefix + "input_shape": self.input_shape,
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, ArrayLike],
        key_prefix: str | None = None,
    ) -> "GatherInfo":
        """Creates GatherInfo from a given dictionary."""
        assert isinstance(d[key_prefix + "input_shape"], np.ndarray), (
            f"input shape is not np.ndarray"
        )
        prefix = f"{key_prefix}:" if key_prefix else ""
        return cls(
            gather_idxs=d[prefix + "gather_idxs"],
            gather_mask=d[prefix + "gather_mask"],
            input_shape=d[prefix + "input_shape"],
        )

    def to(
        self,
        *,
        device: str | torch.device | None = None,
    ):
        def _check_is_same(data: torch.Tensor) -> bool:
            if device is None or data.device == device:
                return True
            return False

        def _to(data):
            if isinstance(data, torch.Tensor):
                if _check_is_same(data):
                    return data
                return data.to(device=device)
            elif isinstance(data, np.ndarray):
                return data
            else:
                return data

        kwargs = {
            field.name: _to(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

        return self.__class__(**kwargs)


@dataclasses.dataclass(frozen=True)
class AtomCrossAtt:
    """Operate on flat atoms."""

    token_atoms_to_queries: GatherInfo
    tokens_to_queries: GatherInfo
    tokens_to_keys: GatherInfo
    queries_to_keys: GatherInfo
    queries_to_token_atoms: GatherInfo

    @classmethod
    def compute_features(
        cls,
        all_token_atoms_layout: AtomLayout,  # (num_tokens, num_dense)
        queries_subset_size: int,
        keys_subset_size: int,
        padding_shapes: PaddingShapes,
        gatherinfo_type: Literal["numpy", "pytorch"] = "numpy",
    ):
        """Computes gather indices and meta data to work with a flat atom list."""

        token_atoms_layout = all_token_atoms_layout.copy_and_pad_to(
            (padding_shapes.num_tokens, all_token_atoms_layout.shape[1])
        )
        token_atoms_mask = token_atoms_layout.atom_name.astype(bool)
        flat_layout = token_atoms_layout[token_atoms_mask]
        num_atoms = flat_layout.shape[0]

        padded_flat_layout = flat_layout.copy_and_pad_to((padding_shapes.num_atoms,))

        # Create the layout for queries
        assert padding_shapes.num_atoms % queries_subset_size == 0
        num_subsets = padding_shapes.num_atoms // queries_subset_size
        lay_arr = padded_flat_layout.to_array()
        queries_layout = AtomLayout.from_array(
            lay_arr.reshape((6, num_subsets, queries_subset_size))
        )

        # Create the layout for the keys (the key subsets are centered around the
        # query subsets)
        # Create initial gather indices (contain out-of-bound indices)
        subset_centers = np.arange(
            queries_subset_size / 2, padding_shapes.num_atoms, queries_subset_size
        )
        flat_to_key_gathers = (
            subset_centers[:, None]
            + np.arange(-keys_subset_size / 2, keys_subset_size / 2)[None, :]
        )
        flat_to_key_gathers = flat_to_key_gathers.astype(int)

        # Shift subsets with out-of-bound indices, such that they are fully within
        # the bounds.
        for row in range(flat_to_key_gathers.shape[0]):
            if flat_to_key_gathers[row, 0] < 0:
                flat_to_key_gathers[row, :] -= flat_to_key_gathers[row, 0]
            elif flat_to_key_gathers[row, -1] > num_atoms - 1:
                overflow = flat_to_key_gathers[row, -1] - (num_atoms - 1)
                flat_to_key_gathers[row, :] -= overflow

        # Create the keys layout.
        keys_layout = padded_flat_layout[flat_to_key_gathers]

        # Create gather indices for conversion between token atoms layout,
        # queries layout and keys layout.
        token_atoms_to_queries = compute_gather_idxs(
            source_layout=token_atoms_layout, target_layout=queries_layout
        )

        token_atoms_to_keys = compute_gather_idxs(
            source_layout=token_atoms_layout, target_layout=keys_layout
        )

        queries_to_keys = compute_gather_idxs(
            source_layout=queries_layout, target_layout=keys_layout
        )

        queries_to_token_atoms = compute_gather_idxs(
            source_layout=queries_layout, target_layout=token_atoms_layout
        )

        # Create gather indices for conversion of tokens layout to
        # queries and keys layout
        token_idxs = np.arange(padding_shapes.num_tokens).astype(np.int64)
        token_idxs = np.broadcast_to(token_idxs[:, None], token_atoms_layout.shape)
        tokens_to_queries = GatherInfo(
            gather_idxs=convert(token_atoms_to_queries, token_idxs, layout_axes=(0, 1)),
            gather_mask=convert(
                token_atoms_to_queries, token_atoms_mask, layout_axes=(0, 1)
            ),
            input_shape=np.array((padding_shapes.num_tokens,)),
        )

        tokens_to_keys = GatherInfo(
            gather_idxs=convert(token_atoms_to_keys, token_idxs, layout_axes=(0, 1)),
            gather_mask=convert(
                token_atoms_to_keys, token_atoms_mask, layout_axes=(0, 1)
            ),
            input_shape=np.array((padding_shapes.num_tokens,)),
        )

        if gatherinfo_type == "pytorch":
            token_atoms_to_queries = token_atoms_to_queries.as_torch_from_np(torch)
            tokens_to_queries = tokens_to_queries.as_torch_from_np(torch)
            tokens_to_keys = tokens_to_keys.as_torch_from_np(torch)
            queries_to_keys = queries_to_keys.as_torch_from_np(torch)
            queries_to_token_atoms = queries_to_token_atoms.as_torch_from_np(torch)

        return cls(
            token_atoms_to_queries=token_atoms_to_queries,
            tokens_to_queries=tokens_to_queries,
            tokens_to_keys=tokens_to_keys,
            queries_to_keys=queries_to_keys,
            queries_to_token_atoms=queries_to_token_atoms,
        )

    def to(
        self,
        *,
        device: str | torch.device | None = None,
    ):
        def _check_is_same(data: torch.Tensor) -> bool:
            if device is None or data.device == device:
                return True
            return False

        def _to(data):
            if isinstance(data, torch.Tensor):
                if _check_is_same(data):
                    return data
                return data.to(device=device)
            elif isinstance(data, GatherInfo):
                return data.to(device=device)
            else:
                return data

        kwargs = {
            field.name: _to(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

        return self.__class__(**kwargs)


def compute_gather_idxs(
    *,
    source_layout: AtomLayout,
    target_layout: AtomLayout,
    fill_value: int = 0,
) -> GatherInfo:
    """Produce gather indices and mask to convert from source layout to target."""
    source_uid_to_idx = {
        uid: idx
        for idx, uid in enumerate(
            zip(
                source_layout.chain_id.ravel(),
                source_layout.res_id.ravel(),
                source_layout.atom_name.ravel(),
                strict=True,
            )
        )
    }
    gather_idxs = []
    gather_mask = []
    for uid in zip(
        target_layout.chain_id.ravel(),
        target_layout.res_id.ravel(),
        target_layout.atom_name.ravel(),
        strict=True,
    ):
        if uid in source_uid_to_idx:
            gather_idxs.append(source_uid_to_idx[uid])
            gather_mask.append(True)
        else:
            gather_idxs.append(fill_value)
            gather_mask.append(False)
    target_shape = target_layout.atom_name.shape
    return GatherInfo(
        gather_idxs=np.array(gather_idxs, dtype=int).reshape(target_shape),
        gather_mask=np.array(gather_mask, dtype=bool).reshape(target_shape),
        input_shape=np.array(source_layout.atom_name.shape),
    )


def convert(
    gather_info: GatherInfo,
    arr: ArrayLike,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> ArrayLike:
    """Convert an array from one atom layout to another."""
    # Translate negative indices to the corresponding positives.
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)

    # Ensure that layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1

    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f"layout_axes must be continuous. Got {layout_axes}.")
    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]

    # Ensure that the layout shape is compatible
    # with the gather_info. I.e. the first axis size must be equal or greater
    # than the gather_info.input_shape, and all subsequent axes sizes must match.
    if (len(layout_shape) != gather_info.input_shape.size) or (
        isinstance(gather_info.input_shape, np.ndarray)
        and (
            (layout_shape[0] < gather_info.input_shape[0])
            or (np.any(layout_shape[1:] != gather_info.input_shape[1:]))
        )
    ):
        raise ValueError(
            "Input array layout axes are incompatible. You specified layout "
            f"axes {layout_axes} with an input array of shape {arr.shape}, but "
            f"the gather info expects shape {gather_info.input_shape}. "
            "Your first axis size must be equal or greater than the "
            "gather_info.input_shape, and all subsequent axes sizes must "
            "match."
        )

    # Compute the shape of the input array with flattened layout.
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    arr_flattened_shape = batch_shape + (np.prod(layout_shape),) + features_shape

    # Flatten input array and perform the gather.
    arr_flattened = arr.reshape(arr_flattened_shape)
    if layout_axes_begin == 0:
        out_arr = arr_flattened[gather_info.gather_idxs, ...]
    elif layout_axes_begin == 1:
        out_arr = arr_flattened[:, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 2:
        out_arr = arr_flattened[:, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 3:
        out_arr = arr_flattened[:, :, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 4:
        out_arr = arr_flattened[:, :, :, :, gather_info.gather_idxs, ...]
    else:
        raise ValueError(
            "Only 4 batch axes supported. If you need more, the code is easy to extend."
        )
    # Broadcast the mask and apply it.
    broadcasted_mask_shape = (
        (1,) * len(batch_shape)
        + gather_info.gather_mask.shape
        + (1,) * len(features_shape)
    )
    out_arr *= gather_info.gather_mask.reshape(broadcasted_mask_shape)
    return out_arr


def _assert_all_arrays_have_same_shape(
    *,
    obj: AtomLayout | GatherInfo,
    expected_shape: tuple[int, ...],
    attribute_names: Sequence[str],
) -> None:
    """Checks that given attributes of the object have the expected shape."""
    attribute_shapes_description = []
    all_shapes_are_valid = True

    for attribute_name in attribute_names:
        attribute = getattr(obj, attribute_name)

        if attribute is None:
            attribute_shape = None
        else:
            attribute_shape = attribute.shape

        if attribute_shape is not None and expected_shape != attribute_shape:
            all_shapes_are_valid = False

        attribute_shape_name = attribute_name + ".shape"
        attribute_shapes_description.append(
            f"{attribute_shape_name:25} = {attribute_shape}"
        )

    if not all_shapes_are_valid:
        raise ValueError(
            f"All arrays must have the same shape ({expected_shape=}). Got\n"
            + "\n".join(attribute_shapes_description)
        )
