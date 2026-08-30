"""The items config is the only place a store's economics enter the system.

It is also the file model/backtest.py loads instead of importing sim/params.py, so the
nine records must reproduce results/results.json's critical fractiles exactly -- that is
what makes the severance from the simulator provable rather than asserted. The rest of
these tests are about a store filling the file in wrong, which is the normal case: a cost
above the price, a batch of zero, a weighed item declared as pieces.
"""
import json

import pytest

from ht import config
from model import newsvendor

# results/results.json, q_star block. If these move, the proposal's dollar figures moved.
PUBLISHED_Q_STAR = {
    "bread": 0.724, "cake": 0.654, "doughnut": 0.790, "hotbar-lb": 0.650,
    "pizza-slice": 0.742, "pizza-whole": 0.675, "rotisserie": 0.599, "sub": 0.699,
    "sushi": 0.577,
}


def test_shipped_config_loads(items):
    assert set(items) == set(PUBLISHED_Q_STAR)
    for key, it in items.items():
        assert it["price"] > it["cost"] >= 0
        assert it["batch"] > 0
        assert it["name"] and it["dept"]


def test_critical_fractiles_reproduce_the_published_q_star(items):
    got = config.critical_fractiles(items)
    assert {k: round(v, 3) for k, v in got.items()} == PUBLISHED_Q_STAR


def test_load_items_returns_sorted_keys(items):
    # load-bearing: backtest dumps this dict's order straight into results.json
    assert list(items) == sorted(items)


def test_defaults_merge_in_the_documented_order(write_json):
    path = write_json("items.json", {
        "schema": "ht-items/1",
        "defaults": {"salvage": 0.10, "shelf_life_days": 2},
        "items": {
            "a": {"name": "A", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1},
            "b": {"name": "B", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1,
                  "salvage": 0.90},
        },
    })
    loaded = config.load_items(path)
    assert loaded["a"]["salvage"] == 0.10           # the file's defaults block
    assert loaded["b"]["salvage"] == 0.90           # the item's own field wins
    assert loaded["a"]["shelf_life_days"] == 2
    assert loaded["a"]["unit"] == "each"            # DEFAULT_ITEM, untouched by either


def _one_item(**overrides):
    item = {"name": "A", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1}
    item.update(overrides)
    return {"schema": "ht-items/1", "items": {"a": item}}


@pytest.mark.parametrize("overrides,needle", [
    ({"cost": 9.0}, "cost"),                        # cost >= price: q* non-positive
    ({"cost": 4.0}, "cost"),
    ({"batch": 0}, "batch"),
    ({"price": 0}, "price"),
    ({"salvage": 2.0}, "salvage"),                  # salvage >= cost makes waste free
    ({"shelf_life_days": 0}, "shelf_life_days"),
    ({"unit": "kg"}, "unit"),
    ({"continuous": True}, "continuous"),           # continuous with unit "each"
    ({"sigma": 0.3}, "sigma"),                      # a simulator latent, not a store's answer
])
def test_bad_item_fields_raise_config_error(overrides, needle, write_json):
    path = write_json("items.json", _one_item(**overrides))
    with pytest.raises(config.ConfigError) as exc:
        config.load_items(path)
    assert needle in str(exc.value)


def test_missing_required_field_raises(write_json):
    path = write_json("items.json",
                      {"schema": "ht-items/1", "items": {"a": {"name": "A", "dept": "D"}}})
    with pytest.raises(config.ConfigError) as exc:
        config.load_items(path)
    assert "price" in str(exc.value)


def test_unknown_schema_tag_raises(write_json):
    path = write_json("items.json", dict(_one_item(), schema="ht-items/2"))
    with pytest.raises(config.ConfigError):
        config.load_items(path)


def test_every_problem_is_reported_at_once(write_json):
    path = write_json("items.json", {"schema": "ht-items/1", "items": {
        "a": {"name": "A", "dept": "D", "price": 1.0, "cost": 2.0, "batch": 1},
        "b": {"name": "B", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 0},
        "c": {"name": "C", "dept": "D", "price": 4.0},
    }})
    with pytest.raises(config.ConfigError) as exc:
        config.load_items(path)
    message = str(exc.value)
    assert "a" in message and "b" in message and "c" in message


def test_inactive_items_are_dropped_unless_asked_for(write_json):
    doc = _one_item()
    doc["items"]["b"] = {"name": "B", "dept": "D", "price": 4.0, "cost": 1.0,
                         "batch": 1, "active": False}
    path = write_json("items.json", doc)
    assert list(config.load_items(path)) == ["a"]
    assert list(config.load_items(path, include_inactive=True)) == ["a", "b"]


def test_resolve_tolerance_is_by_unit():
    # produced pounds and sold pounds never match exactly; pieces do
    assert config.resolve_tolerance({"unit": "each", "sellout_tolerance": None}) == 0.0
    assert config.resolve_tolerance({"unit": "lb", "sellout_tolerance": None}) == 0.5
    assert config.resolve_tolerance({"unit": "lb", "sellout_tolerance": 0.0}) == 0.0


def test_validate_items_warns_about_the_things_a_report_must_disclose(write_json):
    path = write_json("items.json", {"schema": "ht-items/1", "items": {
        "multi": {"name": "M", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1,
                  "shelf_life_days": 3},
        "guessed": {"name": "G", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1,
                    "cost_imputed": True},
        "edge": {"name": "E", "dept": "D", "price": 100.0, "cost": 0.5, "batch": 1},
    }})
    warns = " | ".join(config.validate_items(config.load_items(path)))
    assert "shelf_life_days" in warns
    assert "cost_imputed" in warns or "gross margin" in warns
    assert "tau grid" in warns or "fractile" in warns     # q* 0.995 is off the grid


def test_validate_items_is_quiet_on_the_shipped_config(items):
    assert config.validate_items(items) == []


def test_critical_fractile_is_the_newsvendor_definition():
    assert newsvendor.critical_fractile(4.0, 1.0) == pytest.approx(3.0 / 4.0)
    assert newsvendor.critical_fractile(4.0, 1.0, 0.5) == pytest.approx(3.0 / 3.5)


def test_config_hash_is_stable_and_content_sensitive(items_path, tmp_path):
    first = config.config_hash(items_path)
    assert first == config.config_hash(items_path)
    copy = tmp_path / "items.json"
    copy.write_text(open(items_path, encoding="utf-8").read() + "\n", encoding="utf-8")
    assert config.config_hash(str(copy)) != first


def test_shipped_mapping_loads_against_the_shipped_items(items):
    from tests.conftest import MAPPING_JSON
    mapping = config.load_mapping(MAPPING_JSON, items)
    assert mapping["sellout"]["rule"] in config.SELLOUT_RULES
    assert mapping["date"]["format"] != "auto"
    assert [f["role"] for f in mapping["files"]].count("sales") == 1


def _mapping(**overrides):
    doc = {
        "schema": "ht-source-mapping/1", "store": "0123",
        "files": [{"role": "sales", "path": "S.CSV"}],
        "columns": {"sales": {"date": "D", "item_code": "I", "units": "U"}},
        "date": {"format": "%m/%d/%y"},
        "sellout": {"rule": "none"},
    }
    doc.update(overrides)
    return doc


@pytest.mark.parametrize("overrides,needle", [
    ({"date": {"format": "auto"}}, "auto"),
    ({"numbers": {"units_are_dollars": True}}, "units_are_dollars"),
    ({"sellout": {}}, "sellout"),
    ({"sellout": {"rule": "waste_zero"}}, "waste_zero"),
    ({"sellout": {"rule": "last_sale_gap"}}, "last_sale_gap"),
    ({"files": []}, "sales"),
    ({"columns": {"sales": {"date": "D"}}}, "item_code"),
])
def test_bad_mapping_raises_with_the_field_named(overrides, needle, write_json):
    path = write_json("mapping.json", _mapping(**overrides))
    with pytest.raises(config.MappingError) as exc:
        config.load_mapping(path)
    assert needle in str(exc.value)


def test_mapping_items_map_must_target_a_real_item_key(write_json, items):
    doc = _mapping()
    doc["items"] = {"map": {"771002": "not-an-item"}}
    path = write_json("mapping.json", doc)
    with pytest.raises(config.MappingError):
        config.load_mapping(path, items)


def test_mapping_defaults_fill(write_json):
    path = write_json("mapping.json", _mapping())
    mapping = config.load_mapping(path)
    entry = mapping["files"][0]
    assert entry["encoding"] == "utf-8"
    assert entry["header_row"] == 1
    assert mapping["dedupe"]["policy"] in ("sum", "last")
    assert mapping["calendar"]["payday_days"] == [1, 2, 3, 15, 16, 17]


def test_validate_mapping_warns_about_a_missing_production_file(write_json, items):
    path = write_json("mapping.json", _mapping())
    warns = " | ".join(config.validate_mapping(config.load_mapping(path), items))
    assert "production" in warns


def test_the_cost_basis_and_overrun_policy_are_named_mapping_fields(write_json):
    """Nothing in a cost column says whether it is a cost each or already cost x units, and
    guessing multiplies every settlement dollar by the day's line count."""
    assert config.MAPPING_DEFAULTS["price_cost"]["cost_basis"] == "per_unit"
    assert config.MAPPING_DEFAULTS["production"]["overrun_policy"] == "warn"

    with pytest.raises(config.MappingError) as exc:
        config.load_mapping(write_json("m1.json", _mapping(price_cost={"cost_basis": "x"})))
    assert "cost_basis" in str(exc.value)


def test_a_dedupe_key_without_store_is_refused(write_json):
    """It is a natural edit for a single-store pilot, and it used to be a KeyError from
    inside a pandas groupby naming a column they deliberately removed."""
    with pytest.raises(config.MappingError) as exc:
        config.load_mapping(write_json("m2.json",
                                       _mapping(dedupe={"key": ["item", "date"]})))
    assert "dedupe.key" in str(exc.value)


def test_a_validation_labels_file_nobody_reads_is_refused(write_json):
    """Telling a store their sold-out log is being scored against when nothing opens it is
    worse than not offering the field."""
    with pytest.raises(config.MappingError) as exc:
        config.load_mapping(write_json("m3.json", _mapping(
            sellout={"rule": "none", "validation_labels": "OOS.CSV"})))
    assert "validation_labels" in str(exc.value)
