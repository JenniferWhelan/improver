# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate each ensemble member's rank fraction within the ensemble."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """Calculate the ensemble rank fraction of each ensemble member.

    For each member at each point, the output is the fraction of the ensemble
    whose value is strictly greater than that member's value, in the range 0 to
    1. Tied members share a value and the largest member is always 0. Masked or
    non-finite (NaN or infinite) members are excluded from the count at that
    point and remain masked.

    A cube with no realization coordinate, or with a single member, produces
    zeros with its coordinates unchanged, so this CLI can be applied uniformly
    to ensemble and deterministic forecasts. The output is intended as a
    per-member feature for Rainforests calibration.

    Args:
        cube (iris.cube.Cube):
            Cube of forecast values, normally with a realization dimension.
            Probability cubes (with a threshold coordinate) are rejected.

    Returns:
        iris.cube.Cube:
            Cube named ensemble_rank_fraction_of_<input name> with units of 1
            and the same coordinates as the input.
    """
    from improver.ensemble_rank import EnsembleRankFraction

    return EnsembleRankFraction()(cube)
