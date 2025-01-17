#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""CLI to enforce consistency between two forecasts."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    ref_name: str = None,
    additive_amount: float = None,
    multiplicative_amount: float = None,
    comparison_operator: str = ">=",
    diff_for_warning: float = None,
):
    """
    Module to enforce that the values in the forecast cube are not less than or not
    greater than a linear function of the corresponding values in the
    reference forecast.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                forecast_cube (iris.cube.Cube):
                    Cube of forecasts to be updated by using the reference forecast to
                    create a bound on the value of the forecasts.
                ref_forecast (iris.cube.Cube)
                    Cube of forecasts used to create a bound for the values in
                    forecast_cube. It must be the same shape as forecast_cube but have
                    a different name.
        ref_name (str): Name of ref_forecast cube
        additive_amount (float): The amount to be added to the reference forecast (in
            the units of the reference forecast) prior to enforcing consistency between
            the forecast and reference forecast. If both an additive_amount and
            multiplicative_amount are specified then addition occurs after
            multiplication. This option cannot be used for probability forecasts, if it
            is then an error will be raised.
        multiplicative_amount (float): The amount to multiply the reference forecast by
            prior to enforcing consistency between the forecast and reference
            forecast. If both an additive_amount and multiplicative_amount are
            specified then addition occurs after multiplication. This option cannot be
            used for probability forecasts, if it is then an error will be raised.
        comparison_operator (str): Determines whether the forecast is enforced to be not
            less than or not greater than the reference forecast. Valid choices are
            ">=", for not less than, and "<=" for not greater than.
        diff_for_warning (float): If assigned, the plugin will raise a warning if any
            absolute change in forecast value is greater than this value.

    Returns:
        iris.cube.Cube:
            A forecast cube with identical metadata to forecast but the forecasts are
            enforced to be not less than or not greater than a linear function of
            reference_forecast.
    """
    from iris.cube import CubeList

    from improver.utilities.enforce_consistency import EnforceConsistentForecasts
    from improver.utilities.flatten import flatten

    cubes = flatten(cubes)

    if len(cubes) != 2:
        raise ValueError(
            f"Exactly two cubes should be provided but received {len(cubes)}"
        )

    ref_forecast = CubeList(cubes).extract_cube(ref_name)
    cubes.remove(ref_forecast)

    forecast_cube = cubes[0]

    plugin = EnforceConsistentForecasts(
        additive_amount=additive_amount,
        multiplicative_amount=multiplicative_amount,
        comparison_operator=comparison_operator,
        diff_for_warning=diff_for_warning,
    )

    return plugin(forecast_cube, ref_forecast)
