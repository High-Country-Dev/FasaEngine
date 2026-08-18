"""Evaluate a user-supplied recipe against the ASNS specification.

This answers "is the feed I already make any good?" rather than "what should I
make?". It runs no optimization: the caller supplies the mix, and we recompute
the nutrient composition and compare it with the same constraint set the LP
would have been given.

Toxin (TX*) specs are reported separately because `constraint_builder` never
masks them — they are the one part of the specification that is non-negotiable.

Read-only and additive: nothing here mutates state or influences the LP.
"""

from __future__ import annotations

from typing import Optional

from .config.defaults import (
    DEFAULT_PREMIX_RATE,
    DEFAULT_PROCESSING_METHOD,
    normalize_country,
)
from .constraint_builder import LinearConstraint, build_constraints
from .ingredient_pool import load_pool
from .models import EvaluateRecipeResponse, NutrientLine

FRACTION_SUM_TOLERANCE = 0.005
TOXIN_PREFIX = "TX"

# A self-reported recipe is entered as rounded percentages, so its constraint
# values carry rounding noise. We size the allowance against the magnitude of the
# terms that actually make up this constraint for this recipe, not against the
# pool: bounding it by the largest coefficient anywhere would let a feed with no
# omega-3 at all pass an omega-3 minimum. Scaling with the recipe means a feed
# that genuinely lacks a nutrient still fails, because its terms are near zero.
#
# Toxicity ceilings get no allowance. The constraint builder calls them
# non-negotiable, and telling someone they are marginally over a safe limit is
# the right outcome.
ROUNDING_NOISE = 0.002
EXACT = 1e-6


def _tolerance(con: LinearConstraint, fractions: dict[str, float]) -> float:
    if con.spec_code.startswith(TOXIN_PREFIX):
        return EXACT
    scale = sum(
        abs(con.coeffs.get(code, 0.0) * share) for code, share in fractions.items()
    )
    return max(EXACT, ROUNDING_NOISE * scale)


def _line_for(con: LinearConstraint, fractions: dict[str, float]) -> NutrientLine:
    achieved = sum(
        con.coeffs.get(code, 0.0) * share for code, share in fractions.items()
    ) + con.constant
    tol = _tolerance(con, fractions)

    if con.restriction_type == "Minimum":
        in_spec = achieved >= con.rhs - tol
    elif con.restriction_type == "Maximum":
        in_spec = achieved <= con.rhs + tol
    else:
        in_spec = abs(achieved - con.rhs) <= tol

    return NutrientLine(
        code=con.spec_code,
        spec_label=con.spec_label,
        restriction_type=con.restriction_type,
        target=con.rhs,
        achieved=round(achieved, 6),
        unit=con.unit,
        in_spec=bool(in_spec),
    )


def unknown_codes(fractions: dict[str, float]) -> list[str]:
    """Codes the caller supplied that the configured pool does not contain."""
    known = {r.code for r in load_pool(only_codes=set(fractions.keys()))}
    return sorted(set(fractions.keys()) - known)


def _guidance_for(line: NutrientLine) -> Optional[str]:
    if line.in_spec or line.target is None or line.achieved is None:
        return None

    unit = f" {line.unit}".rstrip()
    yours = f"{line.achieved:g}{unit}"
    limit = f"{line.target:g}{unit}"

    if line.code.startswith(TOXIN_PREFIX):
        return (
            f"{line.spec_label} is above the safe limit. "
            f"Yours is {yours}, the limit is {limit}."
        )
    if line.restriction_type == "Minimum":
        return (
            f"{line.spec_label} is below the minimum. "
            f"Yours is {yours}, the minimum is {limit}."
        )
    if line.restriction_type == "Maximum":
        return (
            f"{line.spec_label} is above the maximum. "
            f"Yours is {yours}, the maximum is {limit}."
        )
    return (
        f"{line.spec_label} is off target. "
        f"Yours is {yours}, the target is {limit}."
    )


def evaluate_recipe(
    species: str,
    stage: str,
    production_system: str,
    fractions: dict[str, float],
    *,
    processing_method: str = DEFAULT_PROCESSING_METHOD,
    premix_enabled: bool = True,
    premix_rate: float = DEFAULT_PREMIX_RATE,
    custom_premix_mask_codes: Optional[list[str]] = None,
    country: Optional[str] = None,
    ficd_overrides: Optional[dict[str, dict[str, float]]] = None,
) -> EvaluateRecipeResponse:
    """Score an explicit recipe against the active specification.

    Raises
    ------
    ValueError
        If the fractions do not form a usable recipe. Evaluating a partial or
        over-100% mix would silently misreport every nutrient, so we refuse.
    """
    country = normalize_country(country)

    total = sum(fractions.values())
    if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"Ingredient shares must add up to 100%; these add up to {total * 100:.1f}%."
        )

    missing = unknown_codes(fractions)
    if missing:
        raise ValueError(
            "These ingredients are not in the database, so the feed cannot be "
            f"checked: {', '.join(missing)}."
        )

    pool = load_pool(only_codes=set(fractions.keys()))

    constraints, warnings = build_constraints(
        species=species,
        stage=stage,
        production_system=production_system,
        pool=pool,
        processing_method=processing_method,
        premix_enabled=premix_enabled,
        premix_rate=premix_rate,
        premix_mask_override=custom_premix_mask_codes,
        ficd_overrides=ficd_overrides,
    )

    report = [_line_for(con, fractions) for con in constraints]

    toxicity = [line for line in report if line.code.startswith(TOXIN_PREFIX)]
    nutrients = [line for line in report if not line.code.startswith(TOXIN_PREFIX)]

    guidance = [g for g in (_guidance_for(line) for line in toxicity) if g]
    guidance += [g for g in (_guidance_for(line) for line in nutrients) if g]

    return EvaluateRecipeResponse(
        status="ok",
        species=species,
        stage=stage,
        production_system=production_system,
        country=country,
        total_fraction=round(total, 6),
        in_spec=all(line.in_spec for line in report),
        safe=all(line.in_spec for line in toxicity),
        composition=nutrients,
        toxicity=toxicity,
        guidance=guidance,
        warnings=warnings,
    )
