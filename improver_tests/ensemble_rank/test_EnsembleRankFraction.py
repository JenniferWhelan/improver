# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the EnsembleRankFraction plugin."""

import numpy as np
import pytest
from iris.coords import CellMethod
from numpy.testing import assert_allclose

from improver.ensemble_rank import EnsembleRankFraction
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

ATTRIBUTES = {
    "source": "Unit test",
    "institution": "Test institution",
    "title": "Test forecast on a grid",
}


@pytest.fixture
def distinct_cube():
    """Three members, 2x2 grid. At every point member 1 is largest,
    member 0 is in the middle and member 2 is smallest."""
    data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    return set_up_variable_cube(data, realizations=[0, 1, 2], attributes=ATTRIBUTES)


@pytest.fixture
def tied_cube():
    """Five members, 2x2 grid. Members 0, 1 and 2 are all zero at every
    point; member 3 is 1 and member 4 is 2."""
    data = np.zeros((5, 2, 2), dtype=np.float32)
    data[3] = 1.0
    data[4] = 2.0
    return set_up_variable_cube(
        data, realizations=[0, 1, 2, 3, 4], attributes=ATTRIBUTES
    )


def test_distinct_values(distinct_cube):
    """Fractions follow value order, not member order."""
    result = EnsembleRankFraction()(distinct_cube)
    expected = np.empty((3, 2, 2), dtype=np.float32)
    expected[0] = 1.0 / 3.0
    expected[1] = 0.0
    expected[2] = 2.0 / 3.0
    assert_allclose(result.data, expected, rtol=1e-6)


def test_ties_share_a_value(tied_cube):
    """Tied members all receive the fraction strictly above them."""
    result = EnsembleRankFraction()(tied_cube)
    expected = np.empty((5, 2, 2), dtype=np.float32)
    expected[0:3] = 2.0 / 5.0
    expected[3] = 1.0 / 5.0
    expected[4] = 0.0
    assert_allclose(result.data, expected, rtol=1e-6)


def test_all_members_equal():
    """A dry point (every member zero) gives zero for every member."""
    data = np.zeros((3, 2, 2), dtype=np.float32)
    cube = set_up_variable_cube(data, realizations=[0, 1, 2], attributes=ATTRIBUTES)
    result = EnsembleRankFraction()(cube)
    assert_allclose(result.data, np.zeros((3, 2, 2), dtype=np.float32))
    assert not np.ma.is_masked(result.data)


def test_metadata(distinct_cube):
    """Name, units, dtype, coordinates and attributes are as specified;
    cell methods are not carried over."""
    distinct_cube.add_cell_method(CellMethod("sum", coords="time"))
    result = EnsembleRankFraction()(distinct_cube)
    assert result.name() == "ensemble_rank_fraction_of_air_temperature"
    assert result.units == "1"
    assert result.dtype == np.float32
    assert result.cell_methods == ()
    assert result.attributes == ATTRIBUTES
    assert result.coords() == distinct_cube.coords()
    assert result.shape == distinct_cube.shape
    for coord in distinct_cube.coords():
        assert result.coord_dims(coord.name()) == distinct_cube.coord_dims(coord.name())
