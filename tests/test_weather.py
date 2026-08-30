"""Weather is normalized into a closed five-value vocabulary at the ingest boundary.

That boundary is the whole point. model/features.py looks its weather value up in a fixed
list, so an unmapped "FOG" from a real feed is a ValueError on the morning a store first
points the tool at live data. Every messy vocabulary therefore lives in a provider, and
anything unrecognized becomes "unknown" and is COUNTED -- never guessed into a real
category, because silently wrong is worse than loudly missing.
"""
import numpy as np
import pandas as pd
import pytest

from ht import schema, weather


@pytest.mark.parametrize("raw,kind", [
    ("CLEAR", "sunny"), ("Sunny", "sunny"), ("MOSTLY SUNNY", "sunny"),
    ("PARTLY CLOUDY", "cloudy"), ("OVERCAST", "cloudy"), ("FOG", "cloudy"),
    ("RAIN", "rain"), ("SHOWERS", "rain"), ("T-STORM", "rain"), ("DRIZZLE", "rain"),
    ("SNOW", "snow"), ("SLEET", "snow"), ("WINTRY MIX", "snow"), ("FLURRIES", "snow"),
])
def test_normalize_kind_resolves_a_real_feed_vocabulary(raw, kind):
    assert weather.normalize_kind(raw) == kind


def test_every_normalized_kind_is_in_the_closed_vocabulary():
    for raw in list(weather.KIND_ALIASES) + ["BLOWING DUST", "", "???"]:
        assert weather.normalize_kind(raw) in schema.WEATHER_KINDS


def test_an_unrecognized_condition_is_counted_not_guessed():
    import collections
    counter = collections.Counter()
    assert weather.normalize_kind("BLOWING DUST", counter=counter) == "unknown"
    assert counter["BLOWING DUST"] == 1


def test_a_store_kind_map_beats_the_alias_table():
    assert weather.normalize_kind("CLEAR", {"CLEAR": "cloudy"}) == "cloudy"


def test_a_kind_map_pointing_outside_the_vocabulary_raises():
    with pytest.raises(weather.WeatherError):
        weather.normalize_kind("FOG", {"FOG": "foggy"})


def _write_weather(tmp_path, rows, header="DATE,TMAX,CONDITION"):
    path = tmp_path / "WEATHER.CSV"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def _csv_provider(path, **kw):
    return weather.CsvWeather(path, {"date": "DATE", "tmax_f": "TMAX", "kind": "CONDITION"},
                              **kw)


def test_csv_provider_returns_the_canonical_frame(tmp_path):
    path = _write_weather(tmp_path, [
        "2026-01-01,41,PARTLY CLOUDY",
        "2026-01-02,38,SNOW",
        "2026-01-03,44,BLOWING DUST",
    ])
    frame = _csv_provider(path).frame(pd.date_range("2026-01-01", periods=3))
    assert list(frame.columns) == list(weather.FRAME_COLUMNS)
    assert list(frame["weather"]) == ["cloudy", "snow", "unknown"]
    assert frame["tmax_f"].dtype == np.float32
    assert frame["snow_tomorrow"].dtype == np.int8
    assert set(frame["weather"]) <= set(schema.WEATHER_KINDS)


def test_derive_snow_tomorrow_is_a_lookup_by_date_with_a_zero_at_the_end():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3),
                          "weather": ["sunny", "snow", "snow"]})
    assert list(weather.derive_snow_tomorrow(frame)) == [1, 1, 0]


def test_a_real_tomorrow_in_the_file_is_used_for_the_last_requested_day(tmp_path):
    path = _write_weather(tmp_path, [
        "2026-01-01,41,CLEAR", "2026-01-02,30,SNOW", "2026-01-03,44,SNOW",
        "2026-01-04,50,CLEAR",
    ])
    provider = _csv_provider(path)
    frame = provider.frame(pd.date_range("2026-01-01", periods=3))
    assert list(frame["snow_tomorrow"]) == [1, 1, 0]
    assert provider.report()["snow_tomorrow_is_hindcast"] is True


def test_the_hindcast_stops_at_the_last_observation(tmp_path):
    """A day past the end of the feed must not inherit today's weather.

    The spec is explicit that the hindcast is "next day's kind == snow, with the last date
    always 0", and derive_snow_tomorrow honours it. But the provider's alignment step
    forward-fills the kind one day past the last observation before hindcasting, so a feed
    that ends on a snowy day reports snow_tomorrow=1 for that day. That is the exact shape
    of the morning-sheet failure: a panel ends yesterday, yesterday was snowy, and the sheet
    prints a snow forecast for today that nobody forecast.
    """
    path = _write_weather(tmp_path, [
        "2026-01-01,41,CLEAR", "2026-01-02,30,SNOW", "2026-01-03,44,SNOW",
    ])
    frame = _csv_provider(path).frame(pd.date_range("2026-01-01", periods=3))
    assert list(frame["snow_tomorrow"]) == [1, 1, 0]


def test_a_real_forecast_column_beats_the_hindcast(tmp_path):
    path = tmp_path / "W.CSV"
    path.write_text("DATE,TMAX,CONDITION,SNOWTMW\n2026-01-01,41,CLEAR,Y\n"
                    "2026-01-02,30,CLEAR,no\n", encoding="utf-8")
    provider = weather.CsvWeather(str(path), {"date": "DATE", "tmax_f": "TMAX",
                                              "kind": "CONDITION",
                                              "snow_tomorrow": "SNOWTMW"})
    frame = provider.frame(pd.date_range("2026-01-01", periods=2))
    assert list(frame["snow_tomorrow"]) == [1, 0]
    assert provider.report()["snow_tomorrow_is_hindcast"] is False


def test_a_requested_date_is_never_dropped(tmp_path):
    path = _write_weather(tmp_path, ["2026-01-01,41,CLEAR"])
    frame = _csv_provider(path).frame(pd.date_range("2026-01-01", periods=6))
    assert len(frame) == 6
    assert np.isnan(frame["tmax_f"].iloc[-1])            # past the fill limit
    assert frame["weather"].iloc[-1] == "unknown"


def test_a_short_gap_is_filled_and_a_long_one_is_left_unknown(tmp_path):
    path = _write_weather(tmp_path, ["2026-01-01,41,CLEAR", "2026-01-10,50,RAIN"])
    provider = _csv_provider(path)
    frame = provider.frame(pd.date_range("2026-01-01", periods=10))
    assert list(frame["weather"])[:4] == ["sunny"] * 4    # 3 days carried, then unknown
    assert frame["weather"].iloc[5] == "unknown"
    report = provider.report()
    assert report["filled_days"] >= 1 and report["missing_days"] >= 1


def test_no_weather_degrades_without_changing_any_shape():
    provider = weather.NoWeather()
    frame = provider.frame(pd.date_range("2026-01-01", periods=5))
    assert list(frame.columns) == list(weather.FRAME_COLUMNS)
    assert frame["tmax_f"].isna().all()
    assert (frame["weather"] == "unknown").all()
    assert (frame["snow_tomorrow"] == 0).all()


def test_live_weather_raises_and_opens_no_socket():
    provider = weather.LiveWeather("KBOS", "")
    with pytest.raises(NotImplementedError) as exc:
        provider.frame(pd.date_range("2026-01-01", periods=1))
    assert "CsvWeather" in str(exc.value) or "CSV" in str(exc.value)


def test_make_provider_dispatches_on_the_mapping(tmp_path):
    path = _write_weather(tmp_path, ["2026-01-01,41,CLEAR"])
    mapping = {
        "files": [{"role": "weather", "path": "WEATHER.CSV"}],
        "columns": {"weather": {"date": "DATE", "tmax_f": "TMAX", "kind": "CONDITION"}},
        "weather": {"provider": "csv"},
    }
    assert isinstance(weather.make_provider(mapping, root=str(tmp_path)), weather.CsvWeather)
    assert isinstance(weather.make_provider({"weather": {"provider": "none"}}),
                      weather.NoWeather)
    with pytest.raises(weather.WeatherError):
        weather.make_provider({"weather": {"provider": "wat"}})
    with pytest.raises(weather.WeatherError):     # csv provider, no file with role weather
        weather.make_provider({"weather": {"provider": "csv"}, "files": []})


def test_csv_provider_reproduces_the_synthetic_weather_columns(synth_panel, tmp_path):
    """The strongest check available: rebuild the feed from the frozen panel's own columns
    in a real-world vocabulary and prove the provider recovers all three model inputs."""
    vocab = {"sunny": "CLEAR", "cloudy": "PARTLY CLOUDY", "rain": "T-STORM", "snow": "SNOW"}
    days = synth_panel.drop_duplicates("date").sort_values("date")
    rows = [f"{d:%m/%d/%Y},{t:.1f},{vocab[w]}"
            for d, t, w in zip(days.date, days.tmax_f, days.weather)]
    path = _write_weather(tmp_path, rows)
    frame = _csv_provider(path, date_format="%m/%d/%Y").frame(
        pd.DatetimeIndex(days.date.values))
    assert (frame["weather"].values == days.weather.values).all()
    assert np.allclose(frame["tmax_f"].astype(float), days.tmax_f.astype(float), atol=0.05)
    # the simulator's own snow_tomorrow is a noiseless lookahead; the hindcast matches it
    assert (frame["snow_tomorrow"].values == days.snow_tomorrow.values).all()


def test_gauges_classify_a_day_with_no_condition_text():
    assert weather.kind_from_measurements(0.4, 0.0, 55.0) == "rain"
    assert weather.kind_from_measurements(0.4, 2.0, 28.0) == "snow"
    assert weather.kind_from_measurements(0.4, None, 28.0) == "snow"
    # a rain gauge cannot tell sunny from cloudy, so a dry day is not guessed
    assert weather.kind_from_measurements(0.0, 0.0, 70.0) == "unknown"
