# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the EnsembleRankFraction plugin."""

import warnings

import numpy as np
import pytest
from iris.coords import CellMethod
from numpy.testing import assert_allclose

from improver.ensemble_rank import EnsembleRankFraction
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)

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


def test_realization_not_leading_axis(distinct_cube):
    """Ranking is along realization wherever that dimension sits, and the
    input dimension order is preserved."""
    transposed = distinct_cube.copy()
    transposed.transpose([1, 2, 0])
    result = EnsembleRankFraction()(transposed)
    expected = np.empty((2, 2, 3), dtype=np.float32)
    expected[..., 0] = 1.0 / 3.0
    expected[..., 1] = 0.0
    expected[..., 2] = 2.0 / 3.0
    assert result.coord_dims("realization") == (2,)
    assert_allclose(result.data, expected, rtol=1e-6)


def _expected_with_member_one_missing_at_origin():
    """Expected data for distinct_cube when member 1 is missing at [0, 0]:
    only members 0 (value 1) and 2 (value 0) remain there, so member 0
    gets 0 and member 2 gets 1/2. Every other point is unchanged."""
    expected = np.ma.empty((3, 2, 2), dtype=np.float32)
    expected[0] = 1.0 / 3.0
    expected[1] = 0.0
    expected[2] = 2.0 / 3.0
    expected[0, 0, 0] = 0.0
    expected[2, 0, 0] = 0.5
    expected[1, 0, 0] = np.ma.masked
    return expected


def test_masked_member(distinct_cube):
    """A masked member is excluded from the count at that point and stays masked."""
    data = np.ma.masked_array(distinct_cube.data, mask=np.zeros((3, 2, 2), bool))
    data[1, 0, 0] = np.ma.masked
    distinct_cube.data = data
    result = EnsembleRankFraction()(distinct_cube)
    expected = _expected_with_member_one_missing_at_origin()
    assert np.array_equal(np.ma.getmaskarray(result.data), np.ma.getmaskarray(expected))
    assert_allclose(result.data.filled(-1.0), expected.filled(-1.0), rtol=1e-6)
    assert np.isnan(np.ma.getdata(result.data)[1, 0, 0])


def test_nan_member(distinct_cube):
    """A NaN member behaves exactly like a masked one."""
    data = distinct_cube.data.copy()
    data[1, 0, 0] = np.nan
    distinct_cube.data = data
    result = EnsembleRankFraction()(distinct_cube)
    expected = _expected_with_member_one_missing_at_origin()
    assert np.ma.is_masked(result.data)
    assert np.array_equal(np.ma.getmaskarray(result.data), np.ma.getmaskarray(expected))
    assert_allclose(result.data.filled(-1.0), expected.filled(-1.0), rtol=1e-6)
    assert np.isnan(np.ma.getdata(result.data)[1, 0, 0])


def test_no_realization_coordinate():
    """A deterministic cube gives zeros with its coordinates unchanged."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cube = set_up_variable_cube(data, attributes=ATTRIBUTES)
    result = EnsembleRankFraction()(cube)
    assert not cube.coords("realization")
    assert not result.coords("realization")
    assert_allclose(result.data, np.zeros((2, 2), dtype=np.float32))
    assert result.coords() == cube.coords()


def test_single_member_dimension():
    """A length-one realization dimension gives zeros and keeps the dimension."""
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    cube = set_up_variable_cube(data, realizations=[0], attributes=ATTRIBUTES)
    result = EnsembleRankFraction()(cube)
    assert result.coord_dims("realization") == (0,)
    assert_allclose(result.data, np.zeros((1, 2, 2), dtype=np.float32))
    assert result.coords() == cube.coords()


def test_single_member_scalar_coordinate(distinct_cube):
    """A scalar realization coordinate gives zeros and keeps the coordinate."""
    cube = distinct_cube[0]
    result = EnsembleRankFraction()(cube)
    assert result.coord_dims("realization") == ()
    assert result.coord("realization") == cube.coord("realization")
    assert_allclose(result.data, np.zeros((2, 2), dtype=np.float32))


def test_probability_cube_raises():
    """Probability cubes are rejected, naming the threshold coordinate."""
    data = np.full((2, 2, 2), 0.5, dtype=np.float32)
    cube = set_up_probability_cube(data, thresholds=[273.0, 275.0])
    with pytest.raises(ValueError, match="threshold"):
        EnsembleRankFraction()(cube)


def test_integer_input(distinct_cube):
    """Integer data are ranked like floats and still produce float32 output."""
    distinct_cube.data = distinct_cube.data.astype(np.int32)
    result = EnsembleRankFraction()(distinct_cube)
    expected = np.empty((3, 2, 2), dtype=np.float32)
    expected[0] = 1.0 / 3.0
    expected[1] = 0.0
    expected[2] = 2.0 / 3.0
    assert result.dtype == np.float32
    assert_allclose(result.data, expected, rtol=1e-6)


def test_all_members_invalid_at_a_point(distinct_cube):
    """A point where every member is masked stays masked, holds NaN beneath
    the mask, and raises no warnings."""
    data = np.ma.masked_array(distinct_cube.data, mask=np.zeros((3, 2, 2), bool))
    data[:, 1, 1] = np.ma.masked
    distinct_cube.data = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = EnsembleRankFraction()(distinct_cube)
    assert np.all(np.ma.getmaskarray(result.data)[:, 1, 1])
    assert np.all(np.isnan(np.ma.getdata(result.data)[:, 1, 1]))
    assert not np.any(np.ma.getmaskarray(result.data)[:, 0, 0])
    assert_allclose(
        np.ma.getdata(result.data)[:, 0, 0], [1.0 / 3.0, 0.0, 2.0 / 3.0], rtol=1e-6
    )
