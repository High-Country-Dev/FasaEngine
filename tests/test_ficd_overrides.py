import os

from fastapi.testclient import TestClient

from fasa_api.main import app
from fasa_core.data_loader import load_ficd_wide
from fasa_core.ingredient_pool import load_pool
from fasa_core.overrides import apply_overrides, locked_params

TOKEN = "abc123"
RICE_BRAN = "30936"


def _client() -> TestClient:
    os.environ["FASA_REQUIRE_AUTH"] = "true"
    os.environ["FASA_API_TOKEN"] = TOKEN
    return TestClient(app)


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _stage(client: TestClient) -> str:
    r = client.get("/supported", headers=_auth())
    return r.json()["stages_by_species_and_system"]["Nile Tilapia"]["General-LowCost"][0]


def _prices() -> dict[str, float]:
    return {r.code: 0.5 for r in load_pool()[:280]}


def _formulate(client, stage, overrides=None):
    body = {
        "species": "Nile Tilapia",
        "stage": stage,
        "production_system": "General-LowCost",
        "prices": _prices(),
    }
    if overrides is not None:
        body["ficd_overrides"] = overrides
    return client.post("/formulate", headers=_auth(), json=body)


def test_locked_params_cover_the_toxicity_ceilings():
    locked = locked_params()
    assert "aflatoxin_b_ppb" in locked
    assert "zeralenone_zon_ppb" in locked
    assert "gossypol_mg_kg" in locked
    assert "crude_protein_percent" not in locked


def test_an_empty_override_map_changes_nothing():
    client = _client()
    stage = _stage(client)

    plain = _formulate(client, stage)
    with_empty = _formulate(client, stage, {})

    assert plain.status_code == 200
    assert with_empty.status_code == 200
    assert plain.json()["recipe"] == with_empty.json()["recipe"]
    assert plain.json()["composition"] == with_empty.json()["composition"]


def test_an_override_reaches_the_optimiser():
    """Gut the protein of whatever the LP leaned on most and it must react."""
    client = _client()
    stage = _stage(client)

    before = _formulate(client, stage).json()
    leaned_on = max(before["recipe"], key=lambda line: line["inclusion_percent"])

    after = _formulate(
        client, stage, {leaned_on["code"]: {"crude_protein_percent": 0.5}}
    )
    assert after.status_code == 200, after.json()

    protein_before = next(
        c for c in before["composition"] if c["code"] == "PA03"
    )["achieved"]
    protein_after = next(
        c for c in after.json()["composition"] if c["code"] == "PA03"
    )["achieved"]

    changed_recipe = before["recipe"] != after.json()["recipe"]
    changed_protein = protein_before != protein_after
    assert changed_recipe or changed_protein, (
        f"overriding {leaned_on['description']} protein changed nothing"
    )


def test_a_toxin_parameter_is_refused():
    client = _client()
    stage = _stage(client)

    r = _formulate(client, stage, {RICE_BRAN: {"aflatoxin_b_ppb": 0.0}})
    assert r.status_code == 400
    assert "toxicity limit" in str(r.json()["detail"])


def test_an_unknown_ingredient_is_refused():
    client = _client()
    stage = _stage(client)

    r = _formulate(client, stage, {"not-a-code": {"crude_protein_percent": 20.0}})
    assert r.status_code == 400
    assert "not-a-code" in str(r.json()["detail"])


def test_an_unknown_parameter_is_refused():
    client = _client()
    stage = _stage(client)

    r = _formulate(client, stage, {RICE_BRAN: {"vibes_percent": 20.0}})
    assert r.status_code == 400
    assert "vibes_percent" in str(r.json()["detail"])


def test_impossible_values_are_refused():
    client = _client()
    stage = _stage(client)

    assert _formulate(client, stage, {RICE_BRAN: {"crude_protein_percent": -1.0}}).status_code == 400
    assert _formulate(client, stage, {RICE_BRAN: {"crude_protein_percent": 140.0}}).status_code == 400


def test_proximate_components_cannot_exceed_the_whole_ingredient():
    client = _client()
    stage = _stage(client)

    r = _formulate(
        client,
        stage,
        {RICE_BRAN: {"crude_protein_percent": 60.0, "crude_lipids_percent": 60.0}},
    )
    assert r.status_code == 400
    assert "more than the whole ingredient" in str(r.json()["detail"])


def test_one_request_cannot_change_another():
    """The FICD frame is lru_cached and shared; overrides must not reach it."""
    client = _client()
    stage = _stage(client)

    baseline = _formulate(client, stage).json()["recipe"]
    overridden = _formulate(client, stage, {RICE_BRAN: {"crude_protein_percent": 4.0}})
    assert overridden.status_code == 200

    after = _formulate(client, stage).json()["recipe"]
    assert after == baseline


def test_the_reference_database_is_never_written_to():
    reference = load_ficd_wide()
    before = reference.loc[reference["code"] == RICE_BRAN, "crude_protein_percent"].iloc[0]

    pool = load_pool()[:280]
    from fasa_core.ingredient_pool import attach_ficd_rows

    apply_overrides(attach_ficd_rows(pool), {RICE_BRAN: {"crude_protein_percent": 4.0}})

    again = load_ficd_wide()
    after = again.loc[again["code"] == RICE_BRAN, "crude_protein_percent"].iloc[0]
    assert after == before


def test_a_wildly_different_value_is_allowed_but_warned_about():
    pool = load_pool()[:280]
    from fasa_core.ingredient_pool import attach_ficd_rows

    frame, warnings = apply_overrides(
        attach_ficd_rows(pool), {RICE_BRAN: {"crude_protein_percent": 1.0}}
    )
    assert frame.loc[frame["code"] == RICE_BRAN, "crude_protein_percent"].iloc[0] == 1.0
    assert any("large difference" in w for w in warnings)


def test_a_plausible_correction_is_not_warned_about():
    pool = load_pool()[:280]
    from fasa_core.ingredient_pool import attach_ficd_rows

    _frame, warnings = apply_overrides(
        attach_ficd_rows(pool), {RICE_BRAN: {"crude_protein_percent": 8.0}}
    )
    assert warnings == []


def test_reference_nutrients_covers_every_pool_ingredient():
    from fasa_core.ingredient_pool import load_pool, reference_nutrients
    from fasa_core.overrides import REPORTED_PARAMS

    pool = load_pool()
    reported = reference_nutrients([r.code for r in pool])

    assert set(reported) == {r.code for r in pool}
    for values in reported.values():
        assert set(values) == set(REPORTED_PARAMS)


def test_reference_nutrients_never_reports_a_locked_parameter():
    from fasa_core.ingredient_pool import load_pool, reference_nutrients
    from fasa_core.overrides import locked_params

    pool = load_pool()
    reported = reference_nutrients([r.code for r in pool[:5]])
    locked = locked_params()

    for values in reported.values():
        assert not (set(values) & locked)


def test_reference_nutrients_does_not_mutate_the_cached_frame():
    from fasa_core.data_loader import load_ficd_wide
    from fasa_core.ingredient_pool import reference_nutrients

    before = load_ficd_wide().copy(deep=True)
    reference_nutrients(["30937"])
    after = load_ficd_wide()

    assert before.equals(after)
