from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miniworld.data.constants import ResidueMapping

if TYPE_CHECKING:
    from miniworld.data.mols import TemplateMol


class ProteinTemplate:
    """Dense AF3-style template features for proteins.

    This class stores the model-facing template representation, not raw
    ``TemplateMol`` objects. Use ``from_mols`` to convert raw templates and
    ``empty`` to create the neutral/default value when no template exists.
    """

    _TEMPLATE_FILL_VALUE = -1
    _RES_TYPE_FILL_VALUE = ResidueMapping().AA_UNKNOWN
    _ATOMS_PER_RESIDUE = 4
    _BACKBONE_ATOM_NUM = 3

    def __init__(
        self,
        n_residues: int | None = None,
        template_list: list[TemplateMol] | None = None,
        ids: list[int] | None = None,
    ) -> None:
        if n_residues is not None and template_list is not None:
            msg = "Only one of n_residues or template_list should be provided."
            raise ValueError(msg)

        if template_list is not None:
            template = self.from_mols(template_list, ids=ids)
        elif n_residues is not None:
            template = self.empty(n_residues)
        else:
            msg = "Either n_residues or template_list must be provided."
            raise ValueError(msg)

        self._assign_from(template)

    def _assign_from(self, other: ProteinTemplate) -> None:
        """Copy another template object into ``self``."""
        self.view = other.view
        self.mask = other.mask.copy()
        self.ids = other.ids.copy()
        self.res_type = other.res_type.copy()
        self.cb_xyz = other.cb_xyz.copy()
        self.cb_mask = other.cb_mask.copy()
        self.bb_xyz = other.bb_xyz.copy()
        self.bb_mask = other.bb_mask.copy()

    @classmethod
    def _from_arrays(
        cls,
        *,
        mask: np.ndarray,
        ids: np.ndarray,
        res_type: np.ndarray,
        cb_xyz: np.ndarray,
        cb_mask: np.ndarray,
        bb_xyz: np.ndarray,
        bb_mask: np.ndarray,
    ) -> ProteinTemplate:
        """Construct a template object directly from dense arrays."""
        obj = cls.__new__(cls)
        obj.view = ResidueMapping().protein
        obj.mask = np.asarray(mask, dtype=bool)
        obj.ids = np.asarray(ids, dtype=object)
        obj.res_type = np.asarray(res_type, dtype=np.int32)
        obj.cb_xyz = np.asarray(cb_xyz, dtype=float)
        obj.cb_mask = np.asarray(cb_mask, dtype=bool)
        obj.bb_xyz = np.asarray(bb_xyz, dtype=float)
        obj.bb_mask = np.asarray(bb_mask, dtype=bool)
        cls._validate_arrays(
            mask=obj.mask,
            ids=obj.ids,
            res_type=obj.res_type,
            cb_xyz=obj.cb_xyz,
            cb_mask=obj.cb_mask,
            bb_xyz=obj.bb_xyz,
            bb_mask=obj.bb_mask,
        )
        return obj

    @classmethod
    def _validate_arrays(
        cls,
        *,
        mask: np.ndarray,
        ids: np.ndarray,
        res_type: np.ndarray,
        cb_xyz: np.ndarray,
        cb_mask: np.ndarray,
        bb_xyz: np.ndarray,
        bb_mask: np.ndarray,
    ) -> None:
        """Validate the internal dense representation."""
        if ids.ndim != 2:
            msg = "ids must have shape (n_templates, n_residues)."
            raise ValueError(msg)

        slot_num, res_num = ids.shape
        if mask.shape != (slot_num,):
            msg = "mask must have shape (n_templates,)."
            raise ValueError(msg)
        if res_type.shape != (slot_num, res_num):
            msg = "res_type must have shape (n_templates, n_residues)."
            raise ValueError(msg)
        if cb_xyz.shape != (slot_num, res_num, 3):
            msg = "cb_xyz must have shape (n_templates, n_residues, 3)."
            raise ValueError(msg)
        if cb_mask.shape != (slot_num, res_num):
            msg = "cb_mask must have shape (n_templates, n_residues)."
            raise ValueError(msg)
        if bb_xyz.shape != (slot_num, res_num, cls._BACKBONE_ATOM_NUM, 3):
            msg = "bb_xyz must have shape (n_templates, n_residues, 3, 3)."
            raise ValueError(msg)
        if bb_mask.shape != (slot_num, res_num):
            msg = "bb_mask must have shape (n_templates, n_residues)."
            raise ValueError(msg)

    @classmethod
    def empty(cls, n_residues: int) -> ProteinTemplate:
        """Create the neutral/default template value for a sequence length."""
        n_residues = int(n_residues)
        if n_residues < 0:
            msg = "n_residues must be non-negative."
            raise ValueError(msg)

        return cls._from_arrays(
            mask=np.zeros((0,), dtype=bool),
            ids=np.full(
                (0, n_residues),
                fill_value=cls._TEMPLATE_FILL_VALUE,
                dtype=object,
            ),
            res_type=np.full(
                (0, n_residues),
                fill_value=cls._RES_TYPE_FILL_VALUE,
                dtype=np.int32,
            ),
            cb_xyz=np.full((0, n_residues, 3), fill_value=np.nan, dtype=float),
            cb_mask=np.zeros((0, n_residues), dtype=bool),
            bb_xyz=np.full(
                (0, n_residues, cls._BACKBONE_ATOM_NUM, 3),
                fill_value=np.nan,
                dtype=float,
            ),
            bb_mask=np.zeros((0, n_residues), dtype=bool),
        )

    @classmethod
    def from_mols(
        cls,
        template_list: list[TemplateMol],
        ids: list[int] | None = None,
    ) -> ProteinTemplate:
        """Convert raw ``TemplateMol`` objects into dense template features."""
        if len(template_list) == 0:
            msg = "template_list must be non-empty."
            raise ValueError(msg)

        n_residues = [len(template.residues) for template in template_list]
        if not all(n == n_residues[0] for n in n_residues):
            msg = "All templates must have the same number of residues."
            raise ValueError(msg)

        if ids is None:
            ids = [cls._TEMPLATE_FILL_VALUE] * len(template_list)
        if len(ids) != len(template_list):
            msg = "ids and template_list must have the same length."
            raise ValueError(msg)

        n_residue = n_residues[0]
        atom_xyz_list = []
        for template in template_list:
            atom_xyz = np.asarray(template.atoms.xyz.value, dtype=float)
            expected_shape = (n_residue * cls._ATOMS_PER_RESIDUE, 3)
            if atom_xyz.shape != expected_shape:
                msg = (
                    "Each TemplateMol must contain 4 atoms per residue in "
                    "N/CA/C/CB order."
                )
                raise ValueError(msg)
            atom_xyz_list.append(atom_xyz)

        ids_array = np.repeat(
            np.asarray(ids, dtype=object)[:, None],
            n_residue,
            axis=1,
        )
        res_type = np.stack(
            [
                ResidueMapping().protein.map(
                    template.residues.one_letter_code_can.value,
                )
                for template in template_list
            ],
            axis=0,
        )

        n_xyz = np.stack([xyz[0::4] for xyz in atom_xyz_list], axis=0)
        ca_xyz = np.stack([xyz[1::4] for xyz in atom_xyz_list], axis=0)
        c_xyz = np.stack([xyz[2::4] for xyz in atom_xyz_list], axis=0)
        cb_xyz = np.stack([xyz[3::4] for xyz in atom_xyz_list], axis=0)
        bb_xyz = np.stack([n_xyz, ca_xyz, c_xyz], axis=2)
        cb_mask = np.asarray(np.isfinite(cb_xyz).all(axis=-1), dtype=bool)
        bb_mask = np.asarray(np.isfinite(bb_xyz).all(axis=(-1, -2)), dtype=bool)

        return cls._from_arrays(
            mask=np.ones((len(template_list),), dtype=bool),
            ids=ids_array,
            res_type=res_type,
            cb_xyz=cb_xyz,
            cb_mask=cb_mask,
            bb_xyz=bb_xyz,
            bb_mask=bb_mask,
        )

    def copy(self) -> ProteinTemplate:
        """Return a deep copy of the template features."""
        return self._from_arrays(
            mask=self.mask.copy(),
            ids=self.ids.copy(),
            res_type=self.res_type.copy(),
            cb_xyz=self.cb_xyz.copy(),
            cb_mask=self.cb_mask.copy(),
            bb_xyz=self.bb_xyz.copy(),
            bb_mask=self.bb_mask.copy(),
        )

    @property
    def has_template(self) -> bool:
        """Whether there is at least one real template row."""
        return self.template_num > 0 and self.res_num > 0

    @property
    def template_num(self) -> int:
        """Number of real template rows, excluding padding rows."""
        return int(self.mask.sum())

    @property
    def slot_num(self) -> int:
        """Number of allocated template slots, including padding rows."""
        return self.res_type.shape[0]

    @property
    def res_num(self) -> int:
        """Number of residues."""
        return self.res_type.shape[1]

    def padded(
        self,
        *,
        n_templates: int | None = None,
        n_residues: int | None = None,
    ) -> ProteinTemplate:
        """Return a padded copy of the template features."""
        target_template_num = self.slot_num if n_templates is None else int(n_templates)
        target_res_num = self.res_num if n_residues is None else int(n_residues)

        if target_template_num < self.slot_num:
            msg = "n_templates must be greater than or equal to the current slot_num."
            raise ValueError(msg)
        if target_res_num < self.res_num:
            msg = "n_residues must be greater than or equal to the current res_num."
            raise ValueError(msg)

        pad_width = (
            (0, target_template_num - self.slot_num),
            (0, target_res_num - self.res_num),
        )

        return self._from_arrays(
            mask=np.pad(
                self.mask,
                pad_width=(0, target_template_num - self.slot_num),
                mode="constant",
                constant_values=False,
            ),
            ids=np.pad(
                self.ids,
                pad_width=pad_width,
                mode="constant",
                constant_values=self._TEMPLATE_FILL_VALUE,
            ),
            res_type=np.pad(
                self.res_type,
                pad_width=pad_width,
                mode="constant",
                constant_values=self._RES_TYPE_FILL_VALUE,
            ),
            cb_xyz=np.pad(
                self.cb_xyz,
                pad_width=(*pad_width, (0, 0)),
                mode="constant",
                constant_values=np.nan,
            ),
            cb_mask=np.pad(
                self.cb_mask,
                pad_width=pad_width,
                mode="constant",
                constant_values=False,
            ),
            bb_xyz=np.pad(
                self.bb_xyz,
                pad_width=(*pad_width, (0, 0), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            ),
            bb_mask=np.pad(
                self.bb_mask,
                pad_width=pad_width,
                mode="constant",
                constant_values=False,
            ),
        )

    def padding(self, n: int, axis: int = 0) -> ProteinTemplate:
        """Return a padded copy along the requested axis.

        This keeps the previous public API but no longer mutates ``self``.
        """
        if axis == 0:
            return self.padded(n_templates=n)
        if axis == 1:
            return self.padded(n_residues=n)

        msg = "Axis must be either 0 or 1."
        raise ValueError(msg)

    @classmethod
    def concat(
        cls,
        templates: list[ProteinTemplate],
    ) -> ProteinTemplate:
        """Concatenate templates along the residue axis."""
        if not templates:
            msg = "No templates to concatenate."
            raise ValueError(msg)

        max_slot_num = max(template.slot_num for template in templates)
        padded_templates = [
            template.padded(n_templates=max_slot_num) for template in templates
        ]
        mask = np.asarray(
            np.stack(
                [template.mask for template in padded_templates],
                axis=0,
            ).any(axis=0),
            dtype=bool,
        )

        return cls._from_arrays(
            mask=mask,
            ids=np.concatenate(
                [template.ids for template in padded_templates],
                axis=1,
            ),
            res_type=np.concatenate(
                [template.res_type for template in padded_templates],
                axis=1,
            ),
            cb_xyz=np.concatenate(
                [template.cb_xyz for template in padded_templates],
                axis=1,
            ),
            cb_mask=np.concatenate(
                [template.cb_mask for template in padded_templates],
                axis=1,
            ),
            bb_xyz=np.concatenate(
                [template.bb_xyz for template in padded_templates],
                axis=1,
            ),
            bb_mask=np.concatenate(
                [template.bb_mask for template in padded_templates],
                axis=1,
            ),
        )
