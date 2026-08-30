"""Holidays, days-to-next-holiday and payday, derived from real dates for any year.

sim/calendar_events.py cannot come along to a real store. Its Easter is a three-entry
lookup table covering 2023-2025, so holidays_for_year(2026) -- the pilot year -- dies with
a bare KeyError. Here Easter is computed with the anonymous Gregorian computus, which
reproduces all three of the simulator's shipped values exactly (2023-04-09, 2024-03-31,
2025-04-20) and keeps going, so the switch is lossless against the frozen dataset.

Two other things are wrong in the simulator and right here. The Super Bowl moved a week
later when the season went to seventeen games, so it is the second Sunday of February from
2022 and the first Sunday before that -- a single 2nd-Sunday rule applied backwards puts
Super Bowl LV on 2021-02-14, which is Valentine's Day and is not when it was played. And
because the simulator collapses each year into one dict literal, a floating holiday landing
on a fixed one silently overwrites it: super_bowl really does fall on valentines in 2027
and 2038, and super_bowl is the largest pizza signal in the data. Here a date carries a
LIST of names, holiday_map() collapses it by a stated precedence, and the loser is reported
rather than lost.

dow and payday live here too, because a real POS export gives a date and nothing else.
Both were checked against all 9864 rows of data/store_synth.csv and reproduce the
simulator's columns exactly, so a store need not supply either.

No network, ever: the rules are code and a store's own closures are a local CSV.
"""
import csv
import datetime as dt

import numpy as np
import pandas as pd

from .schema import ConfigError

# The simulator's fourteen names, in the simulator's order. THIS ORDER IS FROZEN: it is the
# layout of one-hot slots 11:25 of the covariate vector that model/artifacts/demandnet.pt
# was trained on. It must stay equal to model.features.HOLIDAY_NAMES.
HOLIDAY_NAMES = (
    "new_years_day", "super_bowl", "valentines", "easter", "mothers_day",
    "memorial_day", "july4", "labor_day", "halloween", "thanksgiving_eve",
    "thanksgiving", "christmas_eve", "christmas", "new_years_eve",
)

# Which name survives when two rules fire on one date. Reusing the covariate order means
# the earlier one-hot slot wins, which puts super_bowl ahead of valentines -- the collision
# that actually happens, and the one where the dropped name would cost the most signal.
PRECEDENCE = HOLIDAY_NAMES

# The 2nd-Sunday rule starts with Super Bowl LVI; the 1st-Sunday rule below it holds back
# to 2002, when the game first moved into February. Anything earlier was played in January
# on no rule this file can express, so it belongs in the table.
SUPER_BOWL_SECOND_SUNDAY_FROM = 2022
SUPER_BOWL_FIRST_SUNDAY_FROM = 2002
SUPER_BOWL_OVERRIDES = {
    2018: dt.date(2018, 2, 4), 2019: dt.date(2019, 2, 3), 2020: dt.date(2020, 2, 2),
    2021: dt.date(2021, 2, 7),
}

DEFAULT_PAYDAY_DAYS = (1, 2, 3, 15, 16, 17)
HOLIDAY_HORIZON = 21


def _as_date(value):
    """datetime.date out of a date, datetime, Timestamp, datetime64 or ISO string."""
    if isinstance(value, dt.datetime):      # pd.Timestamp is a datetime subclass
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _as_days(values):
    """A datetime64[D] array out of anything date-shaped, for whole-day arithmetic."""
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([], dtype="datetime64[D]")
    return pd.to_datetime(arr).values.astype("datetime64[D]")


def _nth_weekday(year, month, weekday, n):
    """n-th (1-based) given weekday (Mon=0) of a month."""
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        last = dt.date(year, 12, 31)
    else:
        last = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def easter(year):
    """Easter Sunday by the anonymous Gregorian computus."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def super_bowl(year):
    """Super Bowl Sunday. Second Sunday of February from 2022, first Sunday 2002-2021.

    Years listed in SUPER_BOWL_OVERRIDES win outright. Before 2002 the game was played in
    January on no rule, so the first-Sunday answer returned there is wrong; add the year to
    the overrides if a panel ever reaches back that far.
    """
    if year in SUPER_BOWL_OVERRIDES:
        return SUPER_BOWL_OVERRIDES[year]
    n = 2 if year >= SUPER_BOWL_SECOND_SUNDAY_FROM else 1
    return _nth_weekday(year, 2, 6, n)


def us_holidays(year):
    """{date: [name, ...]} for one year, each list in PRECEDENCE order.

    A list rather than a name because rules collide: super_bowl lands on valentines in
    2027 and 2038 (and 2044, 2049, 2055, ...), and whoever reads this map is entitled to
    know that both fired rather than being handed whichever one a dict literal wrote last.
    """
    thanksgiving = _nth_weekday(year, 11, 3, 4)          # 4th Thursday of November
    rules = [
        ("new_years_day", dt.date(year, 1, 1)),
        ("super_bowl", super_bowl(year)),
        ("valentines", dt.date(year, 2, 14)),
        ("easter", easter(year)),
        ("mothers_day", _nth_weekday(year, 5, 6, 2)),    # 2nd Sunday of May
        ("memorial_day", _last_weekday(year, 5, 0)),     # last Monday of May
        ("july4", dt.date(year, 7, 4)),
        ("labor_day", _nth_weekday(year, 9, 0, 1)),      # 1st Monday of September
        ("halloween", dt.date(year, 10, 31)),
        ("thanksgiving_eve", thanksgiving - dt.timedelta(days=1)),
        ("thanksgiving", thanksgiving),
        ("christmas_eve", dt.date(year, 12, 24)),
        ("christmas", dt.date(year, 12, 25)),
        ("new_years_eve", dt.date(year, 12, 31)),
    ]
    out = {}
    for name, day in rules:
        out.setdefault(day, []).append(name)
    return out


def load_extra_holidays(path):
    """Read a store's local-events CSV (two columns, date,name) into {date: name}.

    This is where a store's own closures and local events go -- the county fair, the plant
    shutdown week, the Friday the high school plays at home. Names are deliberately not
    checked against HOLIDAY_NAMES: the vocabulary is open by design and ht.validate is the
    layer that warns the checkpoint's one-hot will have to grow.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise ConfigError(f"{path} line {lineno}: expected two columns, date,name, "
                                  f"got {row!r}")
            raw, name = row[0].strip(), row[1].strip()
            try:
                day = dt.date.fromisoformat(raw)
            except ValueError:
                if lineno == 1:
                    continue                    # a header row; anything later is an error
                raise ConfigError(f"{path} line {lineno}: {raw!r} is not an ISO date "
                                  "(YYYY-MM-DD)")
            if not name:
                raise ConfigError(f"{path} line {lineno}: {raw} has no name")
            if day in out:
                raise ConfigError(f"{path} line {lineno}: {raw} appears twice ({out[day]!r} "
                                  f"and {name!r}); one name per date")
            out[day] = name
    return out


def holiday_map(start, end, extra=None):
    """{date: name} over [start, end], one primary name per date.

    Rule collisions are resolved by PRECEDENCE; `extra` (a store's local events) overrides
    a rule-derived name for the same date. Every name that lost is recorded on
    holiday_map.collisions as (iso_date, kept, dropped) -- a module-level list refreshed on
    every call, which ht.ingest copies into its report and ht.validate turns into a warning.
    """
    start, end = _as_date(start), _as_date(end)
    collisions, out = [], {}
    for year in range(start.year, end.year + 1):
        for day, names in us_holidays(year).items():
            if not start <= day <= end:
                continue
            kept = min(names, key=PRECEDENCE.index)
            collisions.extend((day.isoformat(), kept, n) for n in names if n != kept)
            out[day] = kept
    for day, name in (extra or {}).items():
        day = _as_date(day)
        if not start <= day <= end:
            continue
        if day in out and out[day] != name:
            collisions.append((day.isoformat(), name, out[day]))
        out[day] = name
    holiday_map.collisions = sorted(collisions)
    return out


holiday_map.collisions = []


def days_to_next_holiday(dates, holiday_dates, horizon=HOLIDAY_HORIZON):
    """Whole days from each date to the next holiday on or after it, capped at horizon.

    A holiday itself scores 0. Dates with no holiday ahead of them score the horizon, which
    is also what the covariate saturates at, so the tail of a panel is not a special case.
    """
    days = _as_days(dates)
    holidays = np.unique(_as_days(holiday_dates))
    gap = np.full(len(days), horizon, dtype=np.int64)
    if len(holidays):
        idx = np.searchsorted(holidays, days, side="left")
        found = idx < len(holidays)
        gap[found] = (holidays[idx[found]] - days[found]).astype(np.int64)
    return np.minimum(gap, horizon)


def annotate(df, extra=None, payday_days=DEFAULT_PAYDAY_DAYS):
    """Return a copy of df with dow, holiday and payday (re)derived from its date column.

    Any dow the export supplied is discarded rather than trusted: a POS "day" column may
    count from Sunday, and day-of-week is the model's strongest signal. holiday is "" where
    there is none and never null, which is what the canonical panel promises.
    """
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    out["dow"] = dates.dt.dayofweek.astype("int8")
    out["payday"] = dates.dt.day.isin(list(payday_days)).astype("int8")
    if dates.notna().any():
        holidays = holiday_map(dates.min(), dates.max(), extra)
        out["holiday"] = dates.dt.date.map(holidays).fillna("").astype(str)
    else:
        holiday_map.collisions = []
        out["holiday"] = ""
    return out
