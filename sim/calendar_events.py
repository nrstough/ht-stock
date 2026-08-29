"""Holiday calendar for the simulation years."""
import datetime as dt

from . import params


def nth_weekday(year, month, weekday, n):
    """n-th (1-based) given weekday (Mon=0) of a month."""
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def last_weekday(year, month, weekday):
    if month == 12:
        d = dt.date(year, 12, 31)
    else:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def holidays_for_year(year):
    """Map date -> holiday key (keys match params.HOLIDAYS)."""
    e = params.EASTER[year]
    thanksgiving = nth_weekday(year, 11, 3, 4)  # 4th Thursday
    days = {
        dt.date(year, 1, 1): "new_years_day",
        nth_weekday(year, 2, 6, 2): "super_bowl",       # 2nd Sunday of Feb
        dt.date(year, 2, 14): "valentines",
        dt.date(year, e[0], e[1]): "easter",
        nth_weekday(year, 5, 6, 2): "mothers_day",      # 2nd Sunday of May
        last_weekday(year, 5, 0): "memorial_day",       # last Monday of May
        dt.date(year, 7, 4): "july4",
        nth_weekday(year, 9, 0, 1): "labor_day",        # 1st Monday of Sep
        dt.date(year, 10, 31): "halloween",
        thanksgiving - dt.timedelta(days=1): "thanksgiving_eve",
        thanksgiving: "thanksgiving",
        dt.date(year, 12, 24): "christmas_eve",
        dt.date(year, 12, 25): "christmas",
        dt.date(year, 12, 31): "new_years_eve",
    }
    return days


def holiday_map(start, end):
    out = {}
    for year in range(start.year, end.year + 1):
        for d, name in holidays_for_year(year).items():
            if start <= d <= end:
                out[d] = name
    return out
