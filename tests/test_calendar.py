"""Holidays derived from real dates, for any year -- including 2026, the pilot year.

sim/calendar_events.py reads a three-entry Easter table covering 2023-2025 and raises a
bare KeyError for anything else, so the simulator's calendar cannot be pointed at the year
the pitch is for. These tests check the replacement two ways: that it reproduces the frozen
dataset's own holiday, dow and payday columns exactly (so the switch is lossless), and that
it keeps working outside the training range.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ht import calendar
from ht.schema import ConfigError
from model import features
from sim import params as sim_params


def test_holiday_names_match_the_frozen_covariate_layout():
    # cov[11:25] is a 14-slot one-hot and demandnet.pt is shape-locked to it
    assert list(calendar.HOLIDAY_NAMES) == features.HOLIDAY_NAMES
    assert len(calendar.HOLIDAY_NAMES) == 14
    assert tuple(calendar.PRECEDENCE) == tuple(calendar.HOLIDAY_NAMES)


def test_computus_reproduces_the_simulator_table():
    for year, (month, day) in sim_params.EASTER.items():
        assert calendar.easter(year) == dt.date(year, month, day)


@pytest.mark.parametrize("year,expected", [
    (2019, "2019-04-21"), (2021, "2021-04-04"), (2022, "2022-04-17"),
    (2026, "2026-04-05"), (2027, "2027-03-28"), (2028, "2028-04-16"),
])
def test_computus_continues_past_the_simulator_table(year, expected):
    assert calendar.easter(year).isoformat() == expected


def test_easter_is_always_a_sunday_in_march_or_april():
    for year in range(1900, 2200):
        e = calendar.easter(year)
        assert e.weekday() == 6
        assert e.month in (3, 4)


@pytest.mark.parametrize("year,expected", [
    (2019, "2019-02-03"),    # LIII, first Sunday -- the 16-game era
    (2021, "2021-02-07"),    # LV, still first Sunday
    (2022, "2022-02-13"),    # LVI, the first 17-game season: second Sunday from here on
    (2026, "2026-02-08"),
    (2027, "2027-02-14"),
])
def test_super_bowl_handles_the_era_change(year, expected):
    assert calendar.super_bowl(year).isoformat() == expected


def test_super_bowl_collides_with_valentines_in_the_years_it_really_does():
    # The frozen spec lists 2021, 2027 and 2038. 2021 comes from applying the second-Sunday
    # rule to a year the game was actually played on Feb 7, so the collision years under the
    # correct two-era rule are 2027 and 2038 -- and 2021 must NOT collide.
    collide = [y for y in range(2015, 2061)
               if calendar.super_bowl(y) == dt.date(y, 2, 14)]
    assert 2027 in collide and 2038 in collide
    assert 2021 not in collide


def test_a_collision_keeps_the_higher_precedence_name_and_reports_the_loser():
    mapping = calendar.holiday_map(dt.date(2027, 2, 1), dt.date(2027, 2, 28))
    assert mapping[dt.date(2027, 2, 14)] == "super_bowl"
    dropped = [c for c in calendar.holiday_map.collisions if c[0] == "2027-02-14"]
    assert dropped and dropped[0][2] == "valentines"


def test_every_holiday_name_fires_in_every_year():
    for year in (2023, 2026, 2030):
        names = {n for names in calendar.us_holidays(year).values() for n in names}
        assert names == set(calendar.HOLIDAY_NAMES)


def test_us_holidays_returns_a_list_per_date_so_a_collision_is_visible():
    day = calendar.us_holidays(2027)[dt.date(2027, 2, 14)]
    assert sorted(day) == ["super_bowl", "valentines"]


def test_annotate_reproduces_the_synthetic_holiday_dow_and_payday(synth_panel):
    # every one of the 9864 rows, not a sample: the covariates must be identical or the
    # rehearsal panel is not the frozen panel
    got = calendar.annotate(synth_panel.drop(columns=["holiday", "payday"]))
    assert (got["holiday"].astype(str) == synth_panel["holiday"].astype(str)).all()
    assert (got["dow"].astype(int) == synth_panel["dow"].astype(int)).all()
    assert (got["payday"].astype(int) == synth_panel["payday"].astype(int)).all()
    assert got["holiday"].isna().sum() == 0


def test_annotate_returns_a_copy_and_overwrites_a_wrong_dow(make_panel):
    panel = make_panel(days=10)
    panel = panel.copy()
    panel["dow"] = 6
    out = calendar.annotate(panel)
    assert (panel["dow"] == 6).all()                      # untouched
    assert list(out["dow"]) == list(out["date"].dt.dayofweek)


def test_payday_days_are_configurable(make_panel):
    panel = make_panel(days=31, start="2026-01-01")
    out = calendar.annotate(panel, payday_days=(5,))
    assert set(out.loc[out.payday == 1, "date"].dt.day) == {5}


def test_days_to_next_holiday_is_capped_and_zero_on_the_day():
    dates = pd.date_range("2026-12-20", periods=12).values
    holidays = np.array([np.datetime64("2026-12-25")])
    got = calendar.days_to_next_holiday(dates, holidays, horizon=21)
    assert got[5] == 0                                    # 2026-12-25 itself
    assert got[0] == 5
    assert got[-1] == 21                                  # nothing ahead inside the horizon
    assert got.max() <= 21


def test_extra_holidays_csv_overrides_and_reports(tmp_path):
    path = tmp_path / "local.csv"
    path.write_text("date,name\n2026-02-08,super_bowl_watch_party\n", encoding="utf-8")
    extra = calendar.load_extra_holidays(str(path))
    mapping = calendar.holiday_map(dt.date(2026, 2, 1), dt.date(2026, 2, 28), extra=extra)
    assert mapping[dt.date(2026, 2, 8)] == "super_bowl_watch_party"
    assert any(c[0] == "2026-02-08" for c in calendar.holiday_map.collisions)


@pytest.mark.parametrize("body", [
    "date\n2026-01-01\n",                                 # one column
    "date,name\nnot-a-date,x\n2026-01-02,y\n",
    "date,name\n2026-01-01,a\n2026-01-01,b\n",            # duplicate date
    "date,name\n2026-01-01,\n",                           # empty name
])
def test_a_broken_local_events_file_raises_rather_than_losing_an_entry(body, tmp_path):
    path = tmp_path / "local.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        calendar.load_extra_holidays(str(path))
