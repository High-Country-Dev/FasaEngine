"""Per-user nutrient overrides for the ingredient database.

WorldFish asked for a "background database which is not editable, and also a
user database which is", so a miller can correct the nutrient composition of a
raw material they actually buy. This module is that second database: a set of
per-request replacements applied on top of the reference values.

Two invariants matter more than the feature:

Toxin ceilings are never overridable. `constraint_builder` calls them
non-negotiable, and a miller lowering an aflatoxin figure would manufacture a
false safety pass on the one check they are least able to make themselves.

Nothing here mutates shared state. `load_ficd_wide()` is `lru_cache`d, so one
process holds a single frame; writing to it would silently make one miller's
figures everyone's until the process restarted.
"""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from . import crosswalk

PROXIMATE_PARAMS = (
    "moisture_percent",
    "crude_protein_percent",
    "crude_lipids_percent",
    "crude_fibre_percent",
    "ash_percent",
)

MAX_OVERRIDDEN_CODES = 300
MAX_PARAMS_PER_CODE = 40


def locked_params() -> frozenset[str]:
    """FICD parameters backing a TX* toxicity ceiling."""
    locked: set[str] = set()
    for spec_code in crosswalk._crosswalk_raw():
        if not spec_code.startswith("TX"):
            continue
        param, _factor = crosswalk.resolve(spec_code)
        if param and param != "__ratio__":
            locked.add(param)
    return frozenset(locked)


def _check_bounds(param: str, value: float) -> None:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{param} must be a real number.")
    if value < 0:
        raise ValueError(f"{param} cannot be negative.")
    if param.endswith("_percent") and value > 100:
        raise ValueError(f"{param} cannot exceed 100%.")


def apply_overrides(
    ficd_pool: pd.DataFrame,
    overrides: Optional[Mapping[str, Mapping[str, float]]],
) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with the caller's values substituted, plus warnings.

    `ficd_pool` is expected to be the per-pool copy from `attach_ficd_rows`;
    it is copied again here so callers cannot be surprised.

    Raises
    ------
    ValueError
        On an unknown ingredient, a locked parameter, an unknown parameter, an
        impossible value, or proximate components summing above 100%.
    """
    if not overrides:
        return ficd_pool, []

    if len(overrides) > MAX_OVERRIDDEN_CODES:
        raise ValueError(
            f"At most {MAX_OVERRIDDEN_CODES} ingredients can be overridden at once."
        )

    frame = ficd_pool.copy()
    known_codes = set(frame["code"])
    locked = locked_params()
    warnings: list[str] = []

    for code, params in overrides.items():
        if code not in known_codes:
            raise ValueError(
                f"{code} is not in the active ingredient pool, so its "
                f"composition cannot be changed."
            )
        if len(params) > MAX_PARAMS_PER_CODE:
            raise ValueError(
                f"At most {MAX_PARAMS_PER_CODE} values can be changed per ingredient."
            )

        row_mask = frame["code"] == code
        for param, raw in params.items():
            if param in locked:
                raise ValueError(
                    f"{param} backs a toxicity limit and cannot be changed. "
                    f"Safety limits always use the reference database."
                )
            if param not in frame.columns:
                raise ValueError(f"{param} is not a known composition parameter.")

            value = float(raw)
            _check_bounds(param, value)

            before = frame.loc[row_mask, param]
            reference = None if before.empty else before.iloc[0]
            if reference is not None and reference == reference and reference > 0:
                if value > reference * 3 or value < reference / 3:
                    warnings.append(
                        f"{code} {param} was changed from {reference:g} to "
                        f"{value:g}, which is a large difference from the "
                        f"reference value."
                    )
            frame.loc[row_mask, param] = value

        present = [p for p in PROXIMATE_PARAMS if p in frame.columns]
        total = float(frame.loc[row_mask, present].fillna(0.0).sum(axis=1).iloc[0])
        if total > 100.0:
            raise ValueError(
                f"{code}: moisture, protein, fat, fibre and ash add up to "
                f"{total:.1f}%, which is more than the whole ingredient."
            )

    return frame, warnings
