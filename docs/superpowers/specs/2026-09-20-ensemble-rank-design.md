# Ensemble rank fraction feature — design

**Date:** 2026-09-20
**Branch:** `feature/ensemble-rank`
**Status:** approved design, awaiting implementation plan

## 1. Purpose

Add a plugin and CLI to IMPROVER that, for every ensemble member at every grid
point, computes the fraction of the ensemble whose value is strictly greater than
that member's value. The output is a per-member feature cube for Rainforests
calibration.

### Motivation

The Rainforests 3 h diagnostics (analysis branch
`analysis/rainforests-3h-diagnostics`, findings doc
`docs/superpowers/specs/2026-08-16-rainforests-3h-diagnostics-findings.md`, §5.3
and §6.4) found that:

- each tree answers P(obs ≥ t | one member's fields), and averaging 51 such
  conditionals cannot express ensemble consensus, so the vicinity-only feature
  set underforecasts the tail;
- the ensemble-mean feature restores the level but collapses member
  conditionality (every member gives the same answer);
- a **member-rank feature** ("fraction of ensemble above my value") recovers most
  of the ensemble mean's information while remaining member-conditional, and is
  the recommended remedy.

This feature is that quantity, implemented once so that training-data
extraction and apply time use identical code.

## 2. Definition

For a cube with realization dimension of length N, at each grid point (and each
other non-realization coordinate point), member *i* with value *x_i* receives

    f_i = |{ j : x_j > x_i }| / N

- Range is 0 to (N−1)/N. The largest member is always 0.
- Tied members share a value. A point where all members are 0 mm yields 0 for
  every member.
- The fraction (rather than an integer rank) is used so that the feature is
  invariant to ensemble size and to member dropouts, allowing models trained on
  one ensemble to be applied to another.

### Ties

Handled by `scipy.stats.rankdata(data, method="max", axis=realization_axis)`,
which gives each member the count of members less than or equal to it (ties
shared). Then

    f = (N - rank_max) / N

This is preferred over the two ranking idioms already in IMPROVER:

- ECC (`rank_ecc`) breaks ties by member index or randomly via `np.lexsort` and a
  double `argsort`, producing ordinal ranks. That is correct for building a
  permutation but would inject member-numbering noise into a feature.
- Quantile mapping (`_forecast_to_quantiles`) uses a stable 1D `argsort` with
  midpoint plotting positions; also ordinal, and written for 1D samples.

SciPy is already a hard dependency (EMOS, beta recalibration, ECC), so no new
dependency is introduced.

### Missing data

- Masked or NaN members are excluded from N at that point and remain masked in
  the output. A point with 48 valid members of 51 is ranked over 48.
- Output mask equals input mask.

### Degenerate inputs

- **No realization coordinate:** output is all zeros with the input's
  coordinates unchanged. `ApplyRainForestsCalibration._align_feature_variables`
  adds the length-one realization dimension itself, as for every other feature.
- **Single member (scalar realization coordinate, or a length-one realization
  dimension):** output is all zeros, the coordinate is preserved as it was.
- **Probabilistic input** (a threshold coordinate): raises `ValueError` naming
  the coordinate. Ranking probabilities across members is not the intended use.

These choices mean the CLI works unchanged for every NWP model in a pipeline,
ensemble or deterministic, which is required because all models share one
apply-rainforests-calibration call signature and feature list. For a
deterministic model the feature is a constant-zero column that LightGBM never
splits on.

## 3. Components

### 3.1 Plugin — `improver/ensemble_rank.py`

```python
class EnsembleRankFraction(BasePlugin):
    def process(self, cube: Cube) -> Cube
```

Steps:

1. Reject cubes carrying a coordinate with `var_name == "threshold"` (the
   IMPROVER convention for probability cubes) by raising `ValueError`.
2. If no realization dimension coordinate: return zeros (see §2).
3. Locate the realization axis with `cube.coord_dims("realization")`.
4. Extract data as a masked array; build a validity mask from the mask and
   `np.isnan`.
5. Fill invalid entries with `-inf` so they sort to the bottom, compute
   `rankdata(..., method="max", axis=axis)`; subtract the per-point count of
   invalid members from the rank (invalid members sit below every valid one, so
   they inflate every valid rank by exactly the invalid count); compute
   `N_valid = valid.sum(axis, keepdims=True)`; `f = (N_valid - rank) / N_valid`.
6. Re-apply the mask; cast to float32.
7. Build the output with `create_new_diagnostic_cube(name, "1", cube,
   generate_mandatory_attributes([cube]), data=f)`. Cell methods are not copied.

Output metadata:

| Field | Value |
|---|---|
| name | `ensemble_rank_fraction_of_<input cube name>` |
| units | `1` |
| dtype | float32 |
| coordinates | identical to input, same dimension order |
| cell methods | none |
| attributes | mandatory attributes from input |

### 3.2 CLI — `improver/cli/ensemble_rank.py`

```python
@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
```

Command: `improver ensemble-rank INPUT --output OUTPUT`. No options. Imports the
plugin lazily inside `process`, as `improver/cli/expected_value.py` does.

### 3.3 Files

| Path | Change |
|---|---|
| `improver/ensemble_rank.py` | new |
| `improver/cli/ensemble_rank.py` | new |
| `improver_tests/ensemble_rank/__init__.py` | new |
| `improver_tests/ensemble_rank/test_EnsembleRankFraction.py` | new |
| `improver_tests/acceptance/test_ensemble_rank.py` | new |
| `improver/api/__init__.py` | one line: register `"EnsembleRankFraction": "improver.ensemble_rank"` in `PROCESSING_MODULES` |

No other existing file is modified. `apply-rainforests-calibration` already
accepts feature cubes carrying a realization dimension.

## 4. Testing

### 4.1 Unit tests

Synthetic cubes from `improver.synthetic_data.set_up_test_cubes.set_up_variable_cube`
on a small grid with hand-computable answers:

| Case | Expectation |
|---|---|
| Distinct values, 3 members | fractions 2/3, 1/3, 0 ordered by value, not member index |
| Ties, 5 members, 3 share the minimum | tied members 2/5; others 1/5 and 0 |
| All members equal | 0 everywhere |
| Realization not leading axis | same answers; input dim order preserved |
| One masked member at one point | that entry masked; others ranked over N−1 |
| One NaN member at one point | identical to the masked case |
| No realization coordinate | zeros; coordinates unchanged |
| Single member (scalar realization) | zeros; coordinate preserved |
| Metadata | name, units `1`, float32, coords equal input, no cell methods, mandatory attributes carried |
| Threshold cube | `ValueError` naming the coordinate |

### 4.2 Acceptance test

`improver_tests/acceptance/test_ensemble_rank.py` follows the expected-value
pattern: run the CLI on a realization input and compare to a known-good output
under `kgo_root() / "ensemble-rank"`. It is skipped automatically until the KGO
data and its `SHA256SUMS` entry exist. The checksum is **not** to be fabricated;
the user generates the KGO from real data and adds the entry.

### 4.3 Verification before completion

- new unit tests pass;
- `improver_tests/calibration` still passes;
- pre-commit formatting and lint hooks pass.

## 5. Out of scope

- Configurable direction ("fraction below") or tie method — fixed by design.
- Deriving the feature inside `apply-rainforests-calibration` when the cube is
  absent — rejected so that training and apply cannot drift.
- Any change to the Rainforests training pipeline. Training data must contain
  **all members** of each ensemble at each point so that ranks populate
  correctly (findings doc §6.4); that is a data-extraction requirement, not a
  code change here.
