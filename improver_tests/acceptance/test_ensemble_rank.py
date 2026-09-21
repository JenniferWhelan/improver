# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the ensemble-rank CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_realizations(tmp_path):
    """Test ranking of a realization forecast against a known good output.

    Skips until the KGO data for this CLI have been generated and their
    checksums added, so the test cannot fail on a KGO tree that predates it.
    """
    kgo_dir = acc.kgo_root() / "ensemble-rank"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    if not (kgo_path.exists() and input_path.exists()):
        pytest.skip("ensemble-rank KGO data not yet generated")
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
