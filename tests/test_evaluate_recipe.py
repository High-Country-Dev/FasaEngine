import os

from fastapi.testclient import TestClient

from fasa_api.main import app
from fasa_core.ingredient_pool import load_pool

TOKEN = "abc123"


def _client() -> TestClient:
    os.environ["FASA_REQUIRE_AUTH"] = "true"
    os.environ["FASA_API_TOKEN"] = TOKEN
    return TestClient(app)


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _a_stage(client: TestClient, species: str, system: str) -> str:
    r = client.get("/supported", headers=_auth())
    stages = r.json()["stages_by_species_and_system"][species][system]
    return stages[0]


def _priced_pool() -> dict[str, float]:
    """The engine caps /formulate at 300 priced entries; stay under it."""
    return {r.code: 0.5 for r in load_pool()[:280]}


def test_requires_auth():
    client = _client()
    r = client.post("/evaluate-recipe", json={})
    assert r.status_code == 401


def test_rejects_shares_that_do_not_sum_to_one():
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = list(_priced_pool().keys())
    r = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": {pool[0]: 0.5},
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_recipe"
    assert "100%" in r.json()["detail"]["message"]


def test_rejects_an_unknown_ingredient_rather_than_ignoring_it():
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = list(_priced_pool().keys())
    r = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": {pool[0]: 0.9, "not-a-real-code": 0.1},
        },
    )
    assert r.status_code == 400
    assert "not-a-real-code" in r.json()["detail"]["message"]


def test_rejects_a_fraction_outside_zero_to_one():
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = list(_priced_pool().keys())
    r = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": {pool[0]: 1.4, pool[1]: -0.4},
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_fraction"


def test_a_recipe_the_optimiser_produced_evaluates_as_in_spec():
    """The round trip: what /formulate emits must pass /evaluate-recipe."""
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")

    formulated = client.post(
        "/formulate",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "prices": _priced_pool(),
            "premix_enabled": True,
        },
    )
    assert formulated.status_code == 200
    body = formulated.json()
    assert body["status"] == "optimal", body.get("warnings")

    fractions = {
        line["code"]: line["inclusion_percent"] / 100.0 for line in body["recipe"]
    }
    total = sum(fractions.values())
    fractions = {k: v / total for k, v in fractions.items()}

    evaluated = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": fractions,
            "premix_enabled": True,
        },
    )
    assert evaluated.status_code == 200
    verdict = evaluated.json()
    assert verdict["safe"] is True
    assert verdict["in_spec"] is True, verdict["guidance"]
    assert verdict["guidance"] == []


def test_separates_toxicity_from_the_rest_and_reports_plainly():
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = list(_priced_pool().keys())

    r = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": {pool[0]: 1.0},
        },
    )
    assert r.status_code == 200
    body = r.json()

    assert all(line["code"].startswith("TX") for line in body["toxicity"])
    assert not any(line["code"].startswith("TX") for line in body["composition"])
    assert body["in_spec"] is False
    assert body["guidance"], "a single-ingredient feed should fail something"
    assert all("TX" not in g for g in body["guidance"]), "guidance must not leak spec codes"


def test_toxicity_breaches_are_listed_before_nutrient_shortfalls():
    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = list(_priced_pool().keys())

    r = client.post(
        "/evaluate-recipe",
        headers=_auth(),
        json={
            "species": "Nile Tilapia",
            "stage": stage,
            "production_system": "General-LowCost",
            "fractions": {pool[0]: 1.0},
        },
    )
    body = r.json()
    breached_toxins = [l for l in body["toxicity"] if not l["in_spec"]]
    if breached_toxins:
        assert "safe limit" in body["guidance"][0]
        assert body["safe"] is False


def _constraints_for(stage: str, pool):
    from fasa_core.constraint_builder import build_constraints

    cons, _ = build_constraints(
        species="Nile Tilapia",
        stage=stage,
        production_system="General-LowCost",
        pool=pool,
    )
    return cons


def test_a_nutrient_five_percent_below_its_minimum_still_fails():
    """The allowance must absorb typed rounding, never a real shortfall."""
    from fasa_core.evaluator import _line_for

    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = load_pool()[:280]
    fractions = {code: 1.0 / 280 for code in (r.code for r in pool)}

    checked = 0
    for con in _constraints_for(stage, pool):
        if con.restriction_type != "Minimum" or not con.rhs or not con.coeffs:
            continue
        line = _line_for(con, fractions)
        if line.achieved is None or line.achieved <= 0:
            continue
        scaled = {k: v * (con.rhs * 0.95 / line.achieved) for k, v in fractions.items()}
        assert _line_for(con, scaled).in_spec is False, con.spec_code
        checked += 1
        if checked >= 10:
            break
    assert checked, "expected some minimum constraints to test"


def test_toxicity_ceilings_get_no_rounding_allowance():
    from fasa_core.evaluator import EXACT, _tolerance

    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = load_pool()[:280]
    fractions = {code: 1.0 / 280 for code in (r.code for r in pool)}

    toxins = [c for c in _constraints_for(stage, pool) if c.spec_code.startswith("TX")]
    assert toxins, "expected the spec to carry toxicity ceilings"
    for con in toxins:
        assert _tolerance(con, fractions) == EXACT


def test_a_nutrient_at_half_its_minimum_still_fails():
    from fasa_core.evaluator import _line_for

    client = _client()
    stage = _a_stage(client, "Nile Tilapia", "General-LowCost")
    pool = load_pool()[:280]
    fractions = {code: 1.0 / 280 for code in (r.code for r in pool)}

    checked = 0
    for con in _constraints_for(stage, pool):
        if con.restriction_type != "Minimum" or not con.rhs or not con.coeffs:
            continue
        line = _line_for(con, fractions)
        if line.achieved is None or line.achieved <= 0:
            continue
        halved = {k: v * (con.rhs * 0.5 / line.achieved) for k, v in fractions.items()}
        assert _line_for(con, halved).in_spec is False, con.spec_code
        checked += 1
        if checked >= 5:
            break
    assert checked, "expected some minimum constraints to test"
