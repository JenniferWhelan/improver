# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Calculation of each ensemble member's rank fraction within the ensemble."""

import numpy as np
from iris.cube import Cube
from scipy.stats import rankdata

from improver import BasePlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)


class EnsembleRankFraction(BasePlugin):
    """Compute, for each ensemble member at each point, the fraction of the
    ensemble whose value is strictly greater than that member's value.

    For a realization dimension of length N, member i with value x_i receives

        f_i = |{ j : x_j > x_i }| / N

    so the largest member is always 0 and tied members share a value. The
    fraction rather than an integer rank is used so that the result does not
    depend on ensemble size, which lets tree models trained on one ensemble be
    applied to another and tolerates missing members.

    Masked or non-finite (NaN or infinite) members are excluded from N at that
    point and remain masked in the output. A cube with no realization
    coordinate, or a single member, yields zeros with its coordinates
    unchanged, so the plugin can be applied uniformly to ensemble and
    deterministic forecasts.

    The output is intended as a per-member feature for Rainforests calibration.
    """

    @staticmethod
    def _check_input(cube: Cube) -> None:
        """Reject probability cubes.

        Args:
            cube:
                Input cube.

        Raises:
            ValueError:
                If the cube carries a threshold coordinate, identified by the
                IMPROVER convention ``var_name == "threshold"``.
        """
        threshold_coords = cube.coords(var_name="threshold")
        if threshold_coords:
            raise ValueError(
                "EnsembleRankFraction requires a cube of forecast values, but "
                f"{cube.name()} has a threshold coordinate "
                f"({threshold_coords[0].name()}) and so is a probability cube."
            )

    @staticmethod
    def _rank_fraction(data: np.ndarray, axis: int) -> np.ma.MaskedArray:
        """Compute the rank fraction of data along one axis.

        Args:
            data:
                Array (possibly masked) with the ensemble along ``axis``.
            axis:
                Axis along which members are ranked.

        Returns:
            Masked float32 array of the same shape as ``data``. The mask marks
            entries that were masked or non-finite in the input.
        """
        masked = np.ma.masked_invalid(np.ma.asarray(data))
        valid = ~np.ma.getmaskarray(masked)
        # Invalid members are filled with -inf so they sort below every valid
        # member. Each valid member's "max" rank is then inflated by exactly
        # the number of invalid members at that point, which is subtracted.
        filled = np.where(valid, np.ma.getdata(masked), -np.inf)
        ranks = rankdata(filled, method="max", axis=axis)
        n_invalid = np.sum(~valid, axis=axis, keepdims=True)
        n_valid = np.sum(valid, axis=axis, keepdims=True)
        valid_rank = ranks - n_invalid
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (n_valid - valid_rank) / n_valid
        fraction = np.where(valid, fraction, np.nan).astype(np.float32)
        return np.ma.masked_array(fraction, mask=~valid)

    def process(self, cube: Cube) -> Cube:
        """Compute the ensemble rank fraction of each member of the input cube.

        Args:
            cube:
                Cube of forecast values, normally with a realization dimension.

        Returns:
            Cube named ``ensemble_rank_fraction_of_<input name>`` with units of 1,
            float32 data and the same coordinates as the input.
        """
        self._check_input(cube)
        realization_dims = (
            cube.coord_dims("realization") if cube.coords("realization") else ()
        )
        if realization_dims:
            fraction = self._rank_fraction(cube.data, realization_dims[0])
        else:
            # No realization dimension: rank a one-member ensemble along a
            # temporary leading axis, which yields zeros wherever data are valid.
            fraction = self._rank_fraction(cube.data[np.newaxis, ...], 0)[0]

        if not np.ma.is_masked(fraction):
            fraction = fraction.data

        return create_new_diagnostic_cube(
            f"ensemble_rank_fraction_of_{cube.name()}",
            "1",
            cube,
            generate_mandatory_attributes([cube]),
            data=fraction,
        )
