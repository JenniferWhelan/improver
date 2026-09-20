# Ensemble Rank Fraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `EnsembleRankFraction` plugin and `ensemble-rank` CLI that output, for every ensemble member at every grid point, the fraction of members whose value is strictly greater than that member's value.

**Architecture:** One plugin module (`improver/ensemble_rank.py`) does the calculation with `scipy.stats.rankdata(method="max")` along the realization axis and builds the output cube with the existing `create_new_diagnostic_cube` helper. A thin CLI (`improver/cli/ensemble_rank.py`) wraps it, following the `expected_value` pattern. Nothing in `apply-rainforests-calibration` changes: the output is an ordinary feature cube that already carries a realization dimension.

**Tech Stack:** Python 3.12, iris 3.12, numpy 2.2, scipy 1.15, pytest. Tests run inside a micromamba environment in WSL Ubuntu because the Windows system Python has no iris.

**Spec:** `docs/superpowers/specs/2026-09-20-ensemble-rank-design.md`

## Global Constraints

- Feature definition (spec §2): `f_i = |{ j : x_j > x_i }| / N`, computed with `scipy.stats.rankdata(..., method="max", axis=realization_axis)` then `f = (N - rank_max) / N`. Ties share a value. Largest member is 0.
- Masked or NaN members are excluded from N at that point and stay masked in the output (spec §2, Missing data).
- No realization coordinate, scalar realization coordinate, or length-one realization dimension: all zeros, coordinates unchanged (spec §2, Degenerate inputs).
- A cube carrying a coordinate with `var_name == "threshold"` raises `ValueError` (spec §3.1 step 1).
- Output name `ensemble_rank_fraction_of_<input cube name>`, units `1`, dtype float32, coordinates identical to input in the same dimension order, no cell methods, mandatory attributes from input (spec §3.1 table).
- CLI is `improver ensemble-rank INPUT --output OUTPUT` with no other options (spec §3.2).
- Every new Python file starts with the exact four-line Crown Copyright header (enforced by `./copyright_check`):

  ```
  # (C) Crown Copyright, Met Office. All rights reserved.
  #
  # This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
  # See LICENSE in the root of the repository for full licensing details.
  ```

- Lint and format with ruff (config in `pyproject.toml`: isort, pycodestyle, pyflakes, bandit; `assert` is allowed under `improver_tests/`).
- Do not commit to `master`. All work is on branch `feature/ensemble-rank` (already exists, cut from `master`, holds the spec commit).
- Every commit message ends with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never fabricate acceptance-test checksums or KGO files (spec §4.2).

## Running commands

The repo is at `C:\Users\jaber\git_repos\improver` on Windows, visible in WSL Ubuntu as `/mnt/c/Users/jaber/git_repos/improver`. Git commands run from the Windows shell. Python and pytest run inside WSL. Invoke WSL from the **PowerShell tool**, not Git Bash, because Git Bash rewrites `/mnt/...` arguments into Windows paths:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/jaber/git_repos/improver && ~/bin/micromamba run -n improver python -m pytest improver_tests/ensemble_rank -v"
```

Throughout this plan, `RUN_IN_ENV` means that `wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/jaber/git_repos/improver && ~/bin/micromamba run -n improver <command>"` wrapper.

---

### Task 1: Test environment in WSL

**Files:**
- None in the repo. Creates `~/bin/micromamba` and the `improver` micromamba environment inside WSL Ubuntu.

**Interfaces:**
- Produces: a working `RUN_IN_ENV` wrapper (see "Running commands") that every later task uses to run pytest and ruff.

- [ ] **Step 1: Confirm no usable environment exists**

Run (PowerShell):
```powershell
wsl.exe -d Ubuntu -- bash -lc "python3 -c 'import iris' ; ls ~/bin/micromamba"
```
Expected: `ModuleNotFoundError: No module named 'iris'` and `No such file or directory` for micromamba. If micromamba already exists, skip Step 2.

- [ ] **Step 2: Install micromamba**

Run (PowerShell):
```powershell
wsl.exe -d Ubuntu -- bash -lc "cd ~ && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && ~/bin/micromamba --version"
```
Expected: a version string such as `2.x.y`.

- [ ] **Step 3: Create the environment**

This is `envs/environment_a.yml` with the heavy optional packages (tensorflow, pysteps, geopandas, kmedoids, sphinx tooling) left out. Optional-dependency tests in `improver_tests/calibration` use `pytest.importorskip`, so they skip cleanly.

Run (PowerShell, allow up to 10 minutes):
```powershell
wsl.exe -d Ubuntu -- bash -lc "~/bin/micromamba create -y -n improver -c conda-forge python=3.12 iris=3.12 numpy=2.2 scipy=1.15 'pandas<3' proj=9.7.1 cartopy clize joblib pytz pytest pytest-xdist filelock mock numba statsmodels scikit-learn fastparquet lightgbm ruff"
```
Expected: ends with a line containing `Transaction finished`.

- [ ] **Step 4: Install improver into the environment without dependencies**

Run (PowerShell):
```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/jaber/git_repos/improver && ~/bin/micromamba run -n improver python -m pip install --no-deps -e ."
```
Expected: `Successfully installed improver-...`.

- [ ] **Step 5: Verify the environment runs an existing test module**

Run (PowerShell):
```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/jaber/git_repos/improver && ~/bin/micromamba run -n improver python -m pytest improver_tests/expected_value -q"
```
Expected: all tests pass, output ends with `passed`. If any fail because of a missing module, install that module with `~/bin/micromamba install -y -n improver -c conda-forge <module>` and re-run.

No commit for this task; nothing in the repo changed.

---

### Task 2: Plugin happy path — ranking, ties, output cube metadata

**Files:**
- Create: `improver/ensemble_rank.py`
- Create: `improver_tests/ensemble_rank/__init__.py` (empty file, no header needed: existing test package `__init__.py` files are empty)
- Create: `improver_tests/ensemble_rank/test_EnsembleRankFraction.py`

**Interfaces:**
- Consumes: `improver.BasePlugin`; `improver.metadata.utilities.create_new_diagnostic_cube(name, units, template_cube, mandatory_attributes, data=...)`; `improver.metadata.utilities.generate_mandatory_attributes([cube])`; `improver.synthetic_data.set_up_test_cubes.set_up_variable_cube(data, realizations=..., attributes=...)`.
- Produces: `class EnsembleRankFraction(BasePlugin)` with `process(self, cube: Cube) -> Cube` and a static helper `_rank_fraction(data: np.ndarray, axis: int) -> np.ma.MaskedArray`. Tasks 3 and 4 extend `_rank_fraction` behaviour and `process`; Task 5 calls `EnsembleRankFraction()(cube)`.

- [ ] **Step 1: Create the empty test package file**

Create `improver_tests/ensemble_rank/__init__.py` with no content.

- [ ] **Step 2: Write the failing tests**

Create `improver_tests/ensemble_rank/test_EnsembleRankFraction.py`:

```python
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
        assert result.coord_dims(coord.name()) == distinct_cube.coord_dims(
            coord.name()
        )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank -v`
Expected: collection error `ModuleNotFoundError: No module named 'improver.ensemble_rank'`.

- [ ] **Step 4: Write the implementation**

Create `improver/ensemble_rank.py`:

```python
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

    Masked or NaN members are excluded from N at that point and remain masked
    in the output. A cube with no realization coordinate, or a single member,
    yields zeros with its coordinates unchanged, so the plugin can be applied
    uniformly to ensemble and deterministic forecasts.

    The output is intended as a per-member feature for Rainforests calibration.
    """

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
        filled = masked.filled(-np.inf)
        ranks = rankdata(filled, method="max", axis=axis)
        n_invalid = np.sum(~valid, axis=axis, keepdims=True)
        n_valid = np.sum(valid, axis=axis, keepdims=True)
        valid_rank = ranks - n_invalid
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (n_valid - valid_rank) / n_valid
        return np.ma.masked_array(fraction.astype(np.float32), mask=~valid)

    def process(self, cube: Cube) -> Cube:
        """Compute the ensemble rank fraction of each member of the input cube.

        Args:
            cube:
                Cube of forecast values, normally with a realization dimension.

        Returns:
            Cube named ``ensemble_rank_fraction_of_<input name>`` with units of 1,
            float32 data and the same coordinates as the input.
        """
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank -v`
Expected: 4 passed.

- [ ] **Step 6: Lint and format**

Run: `RUN_IN_ENV ruff check improver/ensemble_rank.py improver_tests/ensemble_rank` then `RUN_IN_ENV ruff format improver/ensemble_rank.py improver_tests/ensemble_rank` then `RUN_IN_ENV ./copyright_check improver/ensemble_rank.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py`
Expected: ruff reports no errors; format reports files unchanged or reformatted (re-run the tests if it reformatted); copyright_check prints nothing or reports the headers are correct.

- [ ] **Step 7: Commit**

```bash
git add improver/ensemble_rank.py improver_tests/ensemble_rank/__init__.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py
git commit -m "feat: EnsembleRankFraction plugin ranking members along realization" -m "Fraction of ensemble members strictly greater than each member, ties shared, output as a new diagnostic cube." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Realization axis position and missing data

**Files:**
- Modify: `improver/ensemble_rank.py` (only if a test fails; the Task 2 implementation is expected to already satisfy these)
- Test: `improver_tests/ensemble_rank/test_EnsembleRankFraction.py`

**Interfaces:**
- Consumes: `EnsembleRankFraction` from Task 2 and the `distinct_cube` fixture.
- Produces: nothing new; locks in behaviour for non-leading realization axes and for masked and NaN members.

- [ ] **Step 1: Write the failing tests**

Append to `improver_tests/ensemble_rank/test_EnsembleRankFraction.py`:

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank -v`
Expected: 7 passed. The Task 2 implementation already handles axes, masks and NaN. If any of the three new tests fail, the failure is in `_rank_fraction`; fix it there (check that `masked_invalid` is applied before `filled`, and that `n_invalid` and `n_valid` use `keepdims=True`) and re-run until 7 pass.

- [ ] **Step 3: Lint and format**

Run: `RUN_IN_ENV ruff check improver/ensemble_rank.py improver_tests/ensemble_rank` then `RUN_IN_ENV ruff format improver/ensemble_rank.py improver_tests/ensemble_rank`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add improver/ensemble_rank.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py
git commit -m "test: ensemble rank fraction axis position, masked and NaN members" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Degenerate inputs and threshold rejection

**Files:**
- Modify: `improver/ensemble_rank.py`
- Test: `improver_tests/ensemble_rank/test_EnsembleRankFraction.py`

**Interfaces:**
- Consumes: `EnsembleRankFraction` from Task 2; `improver.synthetic_data.set_up_test_cubes.set_up_probability_cube(data, thresholds)`.
- Produces: `EnsembleRankFraction._check_input(cube)` static method raising `ValueError` for probability cubes; `process` returns zeros for deterministic and single-member input.

- [ ] **Step 1: Write the failing tests**

Add `set_up_probability_cube` to the existing import line at the top of the test file so it reads:

```python
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)
```

Append to the test file:

```python
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
```

- [ ] **Step 2: Run the tests to verify the new failure**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank -v`
Expected: 10 passed, 1 failed. `test_probability_cube_raises` fails with `DID NOT RAISE`. The three degenerate-input tests pass already because `process` routes cubes without a realization dimension through a temporary leading axis.

- [ ] **Step 3: Add the input check**

In `improver/ensemble_rank.py`, add this static method to `EnsembleRankFraction`, directly above `_rank_fraction`:

```python
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
```

Then make the first line of `process` a call to it:

```python
        self._check_input(cube)
        realization_dims = (
            cube.coord_dims("realization") if cube.coords("realization") else ()
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank -v`
Expected: 11 passed.

- [ ] **Step 5: Lint and format**

Run: `RUN_IN_ENV ruff check improver/ensemble_rank.py improver_tests/ensemble_rank` then `RUN_IN_ENV ruff format improver/ensemble_rank.py improver_tests/ensemble_rank`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add improver/ensemble_rank.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py
git commit -m "feat: ensemble rank fraction handles deterministic input, rejects probability cubes" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: CLI, plugin registry entry and acceptance test

**Files:**
- Create: `improver/cli/ensemble_rank.py`
- Modify: `improver/api/__init__.py:57-61` (insert one line in alphabetical order)
- Create: `improver_tests/acceptance/test_ensemble_rank.py`

**Interfaces:**
- Consumes: `EnsembleRankFraction` from Task 2; `improver.cli.clizefy`, `improver.cli.with_output`, `improver.cli.inputcube`; `improver_tests.acceptance.acceptance` helpers `run_cli`, `cli_name_with_dashes`, `kgo_root`, `compare`, `skip_if_kgo_missing`.
- Produces: the `improver ensemble-rank` command and the registry key `"EnsembleRankFraction"`.

- [ ] **Step 1: Write the failing registry test run**

The registry is checked by the existing parametrised test `improver_tests/test_api.py::test_all_values_are_callable`. There is nothing to write; the new entry is verified by re-running that test after Step 3.

- [ ] **Step 2: Write the CLI**

Create `improver/cli/ensemble_rank.py`:

```python
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
    NaN members are excluded from the count at that point and remain masked.

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
```

- [ ] **Step 3: Register the plugin**

In `improver/api/__init__.py`, the `PROCESSING_MODULES` dictionary is alphabetical. Insert this line between the `"EnforceConsistentForecasts"` entry (line 57) and the `"EnsembleReordering"` entry (line 58):

```python
    "EnsembleRankFraction": "improver.ensemble_rank",
```

- [ ] **Step 4: Write the acceptance test**

Create `improver_tests/acceptance/test_ensemble_rank.py`:

```python
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
    """Test ranking of a realization forecast against a known good output."""
    kgo_dir = acc.kgo_root() / "ensemble-rank"
    kgo_path = kgo_dir / "kgo.nc"
    input_path = kgo_dir / "input.nc"
    output_path = tmp_path / "output.nc"
    args = [input_path, "--output", output_path]
    run_cli(args)
    acc.compare(output_path, kgo_path)
```

- [ ] **Step 5: Verify the registry test, the acceptance skip, and the CLI help**

Run: `RUN_IN_ENV python -m pytest "improver_tests/test_api.py::test_all_values_are_callable[EnsembleRankFraction]" improver_tests/acceptance/test_ensemble_rank.py -v`
Expected: the registry test passes; the acceptance test is reported as `SKIPPED` because the known-good-output directory is not configured.

Run: `RUN_IN_ENV improver ensemble-rank --help`
Expected: usage text beginning `Usage: improver ensemble-rank [OPTIONS] cube` followed by the docstring above.

- [ ] **Step 6: End-to-end check of the CLI on a generated file**

Run (single command; the here-doc is inside the WSL shell):

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/jaber/git_repos/improver && ~/bin/micromamba run -n improver python -c \"
import numpy as np
from iris import save
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
data = np.array([[[1.,2.],[3.,4.]],[[2.,3.],[4.,5.]],[[0.,1.],[2.,3.]]], dtype=np.float32)
save(set_up_variable_cube(data, realizations=[0,1,2]), '/tmp/rank_input.nc')
\" && ~/bin/micromamba run -n improver improver ensemble-rank /tmp/rank_input.nc --output /tmp/rank_output.nc && ~/bin/micromamba run -n improver python -c \"
import iris
c = iris.load_cube('/tmp/rank_output.nc')
print(c.name(), c.units, c.dtype)
print(c.data[:, 0, 0])
\""
```

Expected output:
```
ensemble_rank_fraction_of_air_temperature 1 float32
[0.33333334 0.         0.6666667 ]
```

- [ ] **Step 7: Lint, format and copyright**

Run: `RUN_IN_ENV ruff check improver/cli/ensemble_rank.py improver/api/__init__.py improver_tests/acceptance/test_ensemble_rank.py` then `RUN_IN_ENV ruff format improver/cli/ensemble_rank.py improver/api/__init__.py improver_tests/acceptance/test_ensemble_rank.py` then `RUN_IN_ENV ./copyright_check improver/cli/ensemble_rank.py improver_tests/acceptance/test_ensemble_rank.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add improver/cli/ensemble_rank.py improver/api/__init__.py improver_tests/acceptance/test_ensemble_rank.py
git commit -m "feat: ensemble-rank CLI and plugin registry entry" -m "Acceptance test is skipped until KGO data exist; no checksum is added here." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Full verification and pull request

**Files:**
- None created or modified unless verification finds a defect.

**Interfaces:**
- Consumes: everything from Tasks 2 to 5.
- Produces: a pushed branch `feature/ensemble-rank` and an open pull request against `master`.

- [ ] **Step 1: Run the new tests plus the calibration suite**

Run: `RUN_IN_ENV python -m pytest improver_tests/ensemble_rank improver_tests/calibration improver_tests/test_api.py -q -n 4`
Expected: all pass or skip; no failures. Optional-dependency tests report as skipped. If a calibration test fails, confirm it also fails on `master` before treating it as caused by this branch:

```bash
git stash --include-untracked
git checkout master
```
re-run the failing test, then
```bash
git checkout feature/ensemble-rank
git stash pop
```

- [ ] **Step 2: Run ruff over the whole diff against master**

Run: `RUN_IN_ENV ruff check $(git diff --name-only master -- '*.py')` and `RUN_IN_ENV ruff format --check $(git diff --name-only master -- '*.py')`
Expected: `All checks passed!` and `N files already formatted`.

- [ ] **Step 3: Run the repo's own init and copyright hooks on the new files**

Run: `RUN_IN_ENV ./init_check improver/ensemble_rank.py improver/cli/ensemble_rank.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py improver_tests/acceptance/test_ensemble_rank.py` and `RUN_IN_ENV ./copyright_check improver/ensemble_rank.py improver/cli/ensemble_rank.py improver_tests/ensemble_rank/test_EnsembleRankFraction.py improver_tests/acceptance/test_ensemble_rank.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Confirm the working tree is clean and the log is as expected**

Run: `git status --short` and `git log --oneline master..feature/ensemble-rank`
Expected: no tracked changes (the pre-existing untracked `analysis/` directory and `modelfiles.tar.gz` are not part of this work and are left alone); five commits: spec, plugin, axis/missing-data tests, degenerate inputs, CLI.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feature/ensemble-rank
```
Expected: branch created on the remote.

- [ ] **Step 6: Open the pull request**

```bash
gh pr create --base master --head feature/ensemble-rank --title "Ensemble rank fraction feature for Rainforests" --body-file docs/superpowers/plans/pr-body-ensemble-rank.md
```

Where `docs/superpowers/plans/pr-body-ensemble-rank.md` is written first (and not committed; delete it after the PR is open) with this content:

```markdown
## Summary

Adds an `EnsembleRankFraction` plugin and `improver ensemble-rank` CLI. For each ensemble member at each point the output is the fraction of members whose value is strictly greater than that member's value (0 to 1, ties shared, largest member 0). It is a per-member feature for Rainforests calibration that expresses ensemble consensus without collapsing member conditionality.

- Ranks with `scipy.stats.rankdata(method="max")` along the realization axis.
- Masked or NaN members are excluded from the count and stay masked.
- Deterministic or single-member input returns zeros with coordinates unchanged, so one call works for every NWP model in a pipeline.
- Probability cubes are rejected.
- No change to `apply-rainforests-calibration`; the output is an ordinary feature cube with a realization dimension.

Design: `docs/superpowers/specs/2026-09-20-ensemble-rank-design.md`.

## Testing

- 11 unit tests in `improver_tests/ensemble_rank/` covering ordering, ties, axis position, masked and NaN members, deterministic and single-member input, metadata and probability rejection.
- Acceptance test `improver_tests/acceptance/test_ensemble_rank.py` is skipped until known-good-output data are added; no checksum is included in this PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Expected: `gh` prints the PR URL. Report that URL to the user.

- [ ] **Step 7: Remove the temporary PR body file**

```bash
rm docs/superpowers/plans/pr-body-ensemble-rank.md
git status --short
```
Expected: no tracked changes.
