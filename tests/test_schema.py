"""The canonical schema is the whole design's load-bearing claim.

The claim: the simulator's own CSV IS a canonical panel, so the dress rehearsal runs the
real pipeline rather than a translation of it, and model/features.py needed parameterizing
rather than rewriting. If conform() ever stops accepting data/store_synth.csv untouched,
that claim is gone and results/results.json becomes unreachable -- which is why the first
test here reads the frozen file and the rest are about the ways a real export is wrong.
"""
import numpy as np
import pandas as pd
import pytest

from ht import schema


def test_canonical_shape():
    assert len(schema.CANONICAL) == 21
    assert schema.SIM_ONLY == ("true_demand", "true_mean", "lost_sales")
    assert schema.WEATHER_KINDS == ("sunny", "cloudy", "rain", "snow", "unknown")
    assert schema.ROW_STATUS == ("ok", "closed", "partial", "not_carried", "missing", "suspect")
    assert schema.SELLOUT_SOURCES == ("produced_vs_sold", "flag", "none", "unknown")
    # the four provenance columns must all default, or the simulator's file cannot conform
    for name in ("store", "row_status", "stockout_known", "sellout_source"):
        assert schema.BY_NAME[name].default is not None


def test_synthetic_csv_conforms_untouched(synth_raw):
    panel = schema.conform(synth_raw)
    assert list(panel.columns) == list(schema.NAMES)
    assert len(panel) == len(synth_raw)
    assert sorted(panel.attrs["dropped"]) == sorted(schema.SIM_ONLY)
    assert panel.dtypes.astype(str).to_dict() == schema.dtypes()


def test_conform_preserves_every_observable_value(synth_raw):
    panel = schema.conform(synth_raw)
    src = synth_raw.sort_values(["item", "date"], kind="stable").reset_index(drop=True)
    got = panel.sort_values(["item", "date"], kind="stable").reset_index(drop=True)
    for col in ("sold", "produced", "wasted", "tmax_f", "unit_price", "unit_cost"):
        assert np.nanmax(np.abs(got[col].astype(float) - src[col].astype(float))) < 1e-4
    for col in ("dow", "stockout", "is_closed", "payday", "snow_tomorrow"):
        assert (got[col].astype(int) == src[col].astype(int)).all()


def test_conform_never_mutates_its_argument(synth_raw):
    before = list(synth_raw.columns)
    schema.conform(synth_raw)
    assert list(synth_raw.columns) == before


def test_defaults_fill_and_holiday_is_never_null():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "item": ["bread", "bread"], "item_name": ["Bread", "Bread"],
        "dept": ["Bakery", "Bakery"], "sold": [10.0, 12.0],
    })
    panel = schema.conform(frame)
    assert list(panel["store"]) == ["default", "default"]
    assert list(panel["row_status"]) == ["ok", "ok"]
    assert list(panel["stockout_known"]) == [1, 1]
    assert list(panel["sellout_source"]) == ["unknown", "unknown"]
    assert list(panel["holiday"]) == ["", ""]
    assert panel["holiday"].isna().sum() == 0
    assert list(panel["dow"]) == [3, 4]          # 2026-01-01 is a Thursday
    assert panel["produced"].isna().all()        # optional, no default, stays NaN


def test_missing_required_columns_are_all_reported_at_once():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "junk": [1]})
    with pytest.raises(schema.SchemaError) as exc:
        schema.conform(frame)
    message = str(exc.value)
    for name in ("item", "item_name", "dept", "sold"):
        assert name in message


@pytest.mark.parametrize("column,value", [
    ("sold", "twelve"),
    ("date", "not-a-date"),
    ("stockout_known", 0.5),          # a fractional flag means something was derived wrong
])
def test_uncoercible_values_raise_with_the_column_named(column, value, make_panel):
    frame = make_panel(days=3, conform=False)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(schema.SchemaError) as exc:
        schema.conform(frame)
    assert column in str(exc.value)


@pytest.mark.parametrize("column,value", [
    ("weather", "FOG"),               # the provider normalizes; features must never meet this
    ("row_status", "wat"),
    ("sellout_source", "waste_zero"),
])
def test_out_of_vocabulary_values_raise(column, value, make_panel):
    frame = make_panel(days=3, conform=False)
    frame.loc[0, column] = value
    with pytest.raises(schema.SchemaError) as exc:
        schema.conform(frame)
    assert column in str(exc.value)


def test_assert_no_truth_names_each_simulator_column(synth_raw):
    for column in schema.SIM_ONLY:
        frame = pd.DataFrame({column: [1.0]})
        with pytest.raises(schema.SchemaError) as exc:
            schema.assert_no_truth(frame)
        assert column in str(exc.value)
    schema.assert_no_truth(schema.conform(synth_raw))    # a conformed panel passes


def test_keep_extra_still_drops_simulator_truth(synth_raw):
    frame = synth_raw.copy()
    frame["promo_flag"] = 0
    panel = schema.conform(frame, keep_extra=True)
    assert "promo_flag" in panel.columns
    assert not [c for c in schema.SIM_ONLY if c in panel.columns]


def test_panel_hash_survives_a_round_trip_to_disk(synth_raw, tmp_path):
    panel = schema.conform(synth_raw)
    path = str(tmp_path / "panel.csv")
    schema.write_panel(panel, path)
    back = schema.read_panel(path)
    assert schema.panel_hash(back) == schema.panel_hash(panel)
    assert schema.panel_hash(synth_raw) == schema.panel_hash(panel)


def test_read_panel_keeps_a_leading_zero_item_key(tmp_path):
    # "00451" is a real hot-bar PLU; a bare read_csv turns it into 451 and the join dies
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]), "item": ["00451"], "item_name": ["Hot Bar"],
        "dept": ["Hot Foods"], "sold": [52.0],
    })
    path = str(tmp_path / "panel.csv")
    schema.write_panel(frame, path)
    assert list(schema.read_panel(path)["item"]) == ["00451"]


def test_empty_panel_has_the_declared_dtypes():
    empty = schema.empty_panel()
    assert list(empty.columns) == list(schema.NAMES)
    assert len(empty) == 0
    assert empty.dtypes.astype(str).to_dict() == schema.dtypes()


def test_a_horizon_row_may_carry_a_null_sold(make_panel):
    frame = make_panel(days=3, conform=False)
    frame.loc[2, "sold"] = np.nan
    panel = schema.conform(frame)                 # tomorrow has not sold anything yet
    assert np.isnan(panel["sold"].iloc[-1])
    frame.loc[2, "item"] = None
    with pytest.raises(schema.SchemaError):       # but a null key is still an error
        schema.conform(frame)
