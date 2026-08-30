"""Weather for the model: three fields, one closed vocabulary, whatever the store can get.

The model consumes exactly three weather things -- tmax_f, a day kind, and whether snow is
expected tomorrow. No humidity, wind, tmin or precipitation amount is read anywhere in
model/, so none of it is asked for. That keeps the ask to a store (or to a one-time
download) small enough to actually happen.

Everything messy lives here rather than in model/features.py, which does
WEATHER_KINDS.index(row.weather) and raises a bare ValueError on the first "FOG" it meets.
A stack trace on the morning a store first points the tool at its own weather file is the
worst possible moment for this project's credibility, so the vocabulary is closed at this
boundary: a raw condition maps through the store's kind_map, then KIND_ALIASES, and
anything still unrecognised becomes "unknown" and is COUNTED by raw text. It is never
guessed into a real category -- silently calling an unmapped string "sunny" is worse than
loudly reporting it, and the count is what tells the store which strings to add.

Two providers cover the real cases (a CSV the store exports, or none at all) and the third
fixes the interface for a live feed without pretending to implement it. NoWeather is a
first-class mode: the panel keeps its columns, the shapes never change, and the model still
trains -- see that class for exactly which covariates go dead.

No network, ever. Not at import, not at test time, not in CI. There is no HTTP client in
this file and LiveWeather.frame() raises rather than opening one.
"""
import collections
import glob
import os

import numpy as np
import pandas as pd

from .schema import WEATHER_KINDS, HtError

# Raw condition text -> the four real kinds, matched ignoring case, spacing and punctuation.
# The judgement calls, stated once: fog, mist, haze and smoke are dry obscurations and go to
# "cloudy", the dry-but-not-bright bucket; sleet, freezing rain and a wintry mix close a
# parking lot the way snow does, so they are one operational category with it; hail arrives
# with a thunderstorm and goes to "rain".
KIND_ALIASES = {
    "CLEAR": "sunny", "SUNNY": "sunny", "FAIR": "sunny", "MOSTLY SUNNY": "sunny",
    "MOSTLY CLEAR": "sunny", "PARTLY SUNNY": "sunny", "SKC": "sunny", "CLR": "sunny",

    "PARTLY CLOUDY": "cloudy", "MOSTLY CLOUDY": "cloudy", "CLOUDY": "cloudy",
    "OVERCAST": "cloudy", "SCATTERED CLOUDS": "cloudy", "BROKEN CLOUDS": "cloudy",
    "FEW CLOUDS": "cloudy", "FOG": "cloudy", "MIST": "cloudy", "HAZE": "cloudy",
    "SMOKE": "cloudy", "PC": "cloudy", "MC": "cloudy", "OVC": "cloudy", "BKN": "cloudy",
    "SCT": "cloudy", "FG": "cloudy", "BR": "cloudy", "HZ": "cloudy",

    "RAIN": "rain", "LIGHT RAIN": "rain", "HEAVY RAIN": "rain", "SHOWERS": "rain",
    "RAIN SHOWERS": "rain", "DRIZZLE": "rain", "THUNDERSTORM": "rain",
    "THUNDERSTORMS": "rain", "T-STORM": "rain", "TSTM": "rain", "HAIL": "rain",
    "RA": "rain", "DZ": "rain", "SHRA": "rain", "TS": "rain", "GR": "rain",

    "SNOW": "snow", "LIGHT SNOW": "snow", "HEAVY SNOW": "snow", "SNOW SHOWERS": "snow",
    "FLURRIES": "snow", "SLEET": "snow", "FREEZING RAIN": "snow", "WINTRY MIX": "snow",
    "BLIZZARD": "snow", "ICE": "snow", "SN": "snow", "FZRA": "snow", "PL": "snow",
    "IP": "snow",
}

# Gauge thresholds, for a feed that reports numbers and no words at all (the NOAA case).
SNOW_MIN_IN = 0.1      # a trace dusting is not a snow day to anyone shopping
PRCP_MIN_IN = 0.01     # the NWS "measurable precipitation" threshold
SNOW_BELOW_F = 34.0    # measurable precip this cold is snow, when there is no snow gauge

MAX_FILL_DAYS = 3      # a weather gap longer than this is left unknown rather than invented
FRAME_COLUMNS = ("date", "tmax_f", "weather", "snow_tomorrow")


class WeatherError(HtError):
    pass


def _squash(text):
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


_ALIASES = {_squash(k): v for k, v in KIND_ALIASES.items()}


def _prepare_kind_map(kind_map):
    """Squash a store's kind_map for matching, refusing values outside the vocabulary."""
    out = {}
    for raw, kind in (kind_map or {}).items():
        if kind not in WEATHER_KINDS:
            raise WeatherError(
                f"weather.kind_map maps {raw!r} to {kind!r}, which is not a weather kind | "
                f"the vocabulary is closed: use one of {list(WEATHER_KINDS)}")
        out[_squash(raw)] = kind
    return out


def _as_dates(values):
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def _empty_frame():
    return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"),
                         "tmax_f": pd.Series([], dtype="float32"),
                         "weather": pd.Series([], dtype="str"),
                         "snow_tomorrow": pd.Series([], dtype="int8")})


def normalize_kind(text, kind_map=None, counter=None):
    """One raw condition string -> a member of schema.WEATHER_KINDS.

    The store's kind_map wins over KIND_ALIASES, so a private vocabulary is a mapping entry
    rather than a code change. Matching ignores case, spacing and punctuation, so "T-STORM",
    "t storm" and "TStorm" are one key. Blank is a missing observation and returns "unknown"
    quietly; a non-blank string nobody recognises also returns "unknown" but is counted in
    `counter` by its raw text, which is what ht.validate prints back to the store.
    """
    raw = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).strip()
    if not raw:
        return "unknown"
    key = _squash(raw)
    kind = _prepare_kind_map(kind_map).get(key) or _ALIASES.get(key)
    if kind is None:
        if counter is not None:
            counter[raw] += 1
        return "unknown"
    return kind


def kind_from_measurements(prcp_in, snow_in=None, tmax_f=None, dry="unknown"):
    """Gauge readings -> the model's vocabulary, for a feed that carries no condition text.

    This is the downloaded-daily-summary case: TMAX, PRCP and SNOW and not one word. Snow
    wins over rain, because a day with both is operationally a snow day. With no snowfall
    column, measurable precipitation below SNOW_BELOW_F is called snow.

    A DRY day is deliberately not classified. Nothing in a rain gauge distinguishes sunny
    from cloudy, so `dry` defaults to "unknown" rather than inventing a sky; pass
    dry="sunny" only if the feed genuinely means clear whenever it is dry. Losing the
    sunny/cloudy split costs far less than mislabelling three quarters of the year.

    Accepts scalars or arrays and returns the same shape.
    """
    prcp = np.asarray(prcp_in, dtype=float)
    snow = np.asarray(np.nan if snow_in is None else snow_in, dtype=float)
    tmax = np.asarray(np.nan if tmax_f is None else tmax_f, dtype=float)
    scalar = prcp.ndim == 0 and snow.ndim == 0 and tmax.ndim == 0
    prcp, snow, tmax = np.broadcast_arrays(*np.atleast_1d(prcp, snow, tmax))

    out = np.full(prcp.shape, "unknown", dtype=object)
    wet = prcp >= PRCP_MIN_IN
    out[(prcp < PRCP_MIN_IN) & (np.isnan(snow) | (snow < SNOW_MIN_IN))] = dry
    out[wet] = "rain"
    out[wet & np.isnan(snow) & (tmax < SNOW_BELOW_F)] = "snow"
    out[snow >= SNOW_MIN_IN] = "snow"
    return out[0] if scalar else out


def derive_snow_tomorrow(frame):
    """HINDCAST "snow is expected tomorrow": 1 where the next calendar day's kind is snow.

    Honest about what it is. On the evening of day t a store has a FORECAST -- probabilistic,
    sometimes wrong, and worth less than this. Looking the answer up afterwards trains the
    model to lean on a signal that will be noisier in production, a train/serve skew that
    ht.ingest records as weather.snow_tomorrow_is_hindcast and the weekly report repeats.
    The synthetic column has exactly the same defect by construction, so a store supplying a
    real archived forecast column is strictly better than this and should say so in the
    mapping.

    The lookup is by calendar date, not row position, so a missing day yields 0 instead of
    reaching across the gap. The last date is always 0: tomorrow is outside the frame, and
    "no snow expected" is the safe answer. That also means a morning sheet built from a
    panel ending yesterday sees snow_tomorrow=0 for today unless the store supplies a real
    forecast -- the one day of the year this matters most is the one day a hindcast cannot
    speak to.
    """
    dates = _as_dates(frame["date"])
    kinds = np.asarray(frame["weather"], dtype=object)
    snow_days = set(dates[kinds == "snow"])
    return np.array([1 if d in snow_days else 0 for d in dates + pd.Timedelta(days=1)],
                    dtype=np.int8)


def _as_flag(values, where):
    """A 0/1 forecast column out of numbers or words, refusing anything it cannot read.

    Silently zeroing a "Y" would throw away the one weather signal a store took the trouble
    to record, so an unreadable value stops the ingest instead.
    """
    num = pd.to_numeric(values, errors="coerce")
    out = num.fillna(0).gt(0).astype(np.int8)
    words = {"true": 1, "t": 1, "y": 1, "yes": 1, "snow": 1,
             "false": 0, "f": 0, "n": 0, "no": 0, "none": 0}
    unreadable = []
    for i in values.index[num.isna()]:
        raw = values[i]
        key = "" if raw is None or pd.isna(raw) else str(raw).strip().lower()
        if not key:
            continue
        if key in words:
            out[i] = words[key]
        else:
            unreadable.append(raw)
    if unreadable:
        raise WeatherError(
            f"{where}: {len(unreadable)} snow_tomorrow value(s) are neither a number nor a "
            f"yes/no word ({unreadable[:5]}) | map the column to null in "
            "mapping.columns.weather.snow_tomorrow to hindcast it instead, or fix the export")
    return out


class WeatherProvider:
    """Daily tmax_f, weather kind and snow_tomorrow for a range of dates. One method.

    Subclasses produce observations keyed by date and hand them to _align(), which does the
    reindexing, the short-gap fill, the snow_tomorrow hindcast and the dtypes -- so every
    provider returns the identical table and ht.ingest never branches on which one it got.
    """
    name = "base"

    def __init__(self):
        self.unknown_conditions = collections.Counter()
        self.missing_days = 0
        self.filled_days = 0
        self.snow_tomorrow_is_hindcast = False

    def frame(self, dates):
        """-> DataFrame(date, tmax_f, weather, snow_tomorrow), one row per requested date."""
        raise NotImplementedError

    def report(self):
        """What ht.ingest puts under `weather` in its report, and ht.validate warns from."""
        return dict(provider=self.name, missing_days=self.missing_days,
                    filled_days=self.filled_days,
                    snow_tomorrow_is_hindcast=self.snow_tomorrow_is_hindcast,
                    unknown_conditions=dict(self.unknown_conditions))

    def _align(self, obs, dates):
        """Put observations on the requested dates; fill gaps of at most MAX_FILL_DAYS days.

        A requested date with no observation is never dropped -- it comes back tmax_f NaN and
        weather "unknown", which is exactly what the model reads as "no information" -- and it
        is counted. The fill runs over a complete daily index so its limit counts DAYS rather
        than rows, and it reaches one day past the end so the last requested date can still
        see tomorrow's kind.
        """
        idx = _as_dates(dates)
        if len(idx) == 0:
            return _empty_frame()

        obs = obs[~obs.index.duplicated(keep="last")].sort_index()
        for col in ("tmax_f", "weather", "snow_tomorrow"):
            if col not in obs.columns:
                obs[col] = np.nan
        daily = obs.reindex(pd.date_range(idx.min(), idx.max() + pd.Timedelta(days=1), freq="D"))

        observed = daily[["tmax_f", "weather"]].notna().any(axis=1)
        daily[["tmax_f", "weather"]] = daily[["tmax_f", "weather"]].ffill(limit=MAX_FILL_DAYS)
        # snow_tomorrow is deliberately NOT carried forward: it is a claim about one specific
        # tomorrow, and a stale claim about snow is worse than no claim at all.
        if daily["snow_tomorrow"].isna().all():
            self.snow_tomorrow_is_hindcast = bool(observed.any())
            # Hindcast from OBSERVED kinds only. A ffilled day is today's weather wearing
            # tomorrow's date, and reading it back as a forecast is the carry-forward this
            # column refuses to do -- worst on the last day of a feed, where it would print
            # a snow forecast on a morning sheet that nobody forecast.
            seen = daily["weather"].where(observed).fillna("unknown")
            daily["snow_tomorrow"] = derive_snow_tomorrow(
                pd.DataFrame({"date": daily.index, "weather": seen.to_numpy()}))

        have = daily[["tmax_f", "weather"]].notna().any(axis=1)
        uniq = idx.unique()
        self.filled_days += int((have.loc[uniq] & ~observed.loc[uniq]).sum())
        self.missing_days += int((~have.loc[uniq]).sum())

        rows = daily.reindex(idx)
        kind = np.asarray(rows["weather"].astype(object).where(rows["weather"].notna(),
                                                              "unknown"), dtype=object)
        off = set(kind) - set(WEATHER_KINDS)
        if off:
            raise WeatherError(f"provider {self.name!r} produced weather kind(s) {sorted(off)} "
                               f"outside {list(WEATHER_KINDS)}")
        out = pd.DataFrame({
            "date": np.asarray(idx),
            "tmax_f": pd.to_numeric(rows["tmax_f"], errors="coerce").to_numpy(dtype="float32"),
            "weather": kind,
            "snow_tomorrow": pd.to_numeric(rows["snow_tomorrow"], errors="coerce")
                               .fillna(0).gt(0).to_numpy().astype("int8"),
        })
        out["weather"] = out["weather"].astype("str")
        return out


class CsvWeather(WeatherProvider):
    """A weather CSV: one the store exports, or one downloaded once and kept beside the panel.

    `columns` maps canonical names onto that file's own headers. Only `date` is required:
        date            the observation date
        tmax_f          daily high (temp_units="C" converts)
        kind            condition text, read through kind_map then KIND_ALIASES
        snow_tomorrow   a REAL forecast flag if the store has one; leave it null to hindcast
        prcp_in         precipitation depth, used where there is no condition text
        snow_in         snowfall depth, same
    A file with prcp/snow and no words is the common downloaded case; a file with both uses
    the words first and falls back to the gauges only where the words were unrecognised,
    which recovers a day whose condition text is unmappable but whose rain gauge is not.

    Dates parse as ISO-8601 unless date_format names an explicit strftime string -- the same
    rule ht.ingest applies to the sales export, for the same reason: 3/4/25 is two dates.
    """
    name = "csv"

    def __init__(self, path, columns, kind_map=None, date_format=None, encoding="utf-8",
                 delimiter=",", header_row=1, skip_footer_rows=0, na_values=None,
                 temp_units="F", precip_units="in", dry_kind="unknown"):
        super().__init__()
        self.path = path
        self.columns = dict(columns or {})
        self.kind_map = _prepare_kind_map(kind_map)
        self.date_format = date_format
        self.encoding = encoding
        self.delimiter = delimiter
        self.header_row = header_row
        self.skip_footer_rows = skip_footer_rows
        self.na_values = list(na_values) if na_values else None
        self.temp_units = str(temp_units).upper()
        self.precip_units = str(precip_units).lower()
        self.dry_kind = dry_kind
        if self.temp_units not in ("F", "C"):
            raise WeatherError(f"weather.temp_units is {temp_units!r} | expected 'F' or 'C'")
        if self.precip_units not in ("in", "mm"):
            raise WeatherError(f"weather.precip_units is {precip_units!r} | expected 'in' or 'mm'")
        if dry_kind not in WEATHER_KINDS:
            raise WeatherError(f"weather.dry_kind is {dry_kind!r} | expected one of "
                               f"{list(WEATHER_KINDS)}")
        if not self.columns.get("date"):
            raise WeatherError("no date column mapped for the weather file | set "
                               "mapping.columns.weather.date to the raw header that carries it")

    def _read(self):
        paths = sorted(glob.glob(self.path)) or ([self.path] if os.path.exists(self.path) else [])
        if not paths:
            raise WeatherError(f"weather file {self.path!r} matches nothing | fix the path in "
                               "mapping.files, or set mapping.weather.provider to 'none'")
        parts = []
        for p in paths:
            try:
                raw = pd.read_csv(p, encoding=self.encoding, sep=self.delimiter,
                                  skiprows=max(self.header_row - 1, 0),
                                  na_values=self.na_values, dtype=str, keep_default_na=True)
            except UnicodeDecodeError as exc:
                raise WeatherError(f"{p}: not readable as {self.encoding} ({exc}) | set the "
                                   "encoding on this file's mapping.files entry, 'cp1252' for "
                                   "anything with AS/400 lineage") from None
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                raise WeatherError(
                    f"{p}: will not parse as a {self.delimiter!r}-delimited CSV with its header "
                    f"on row {self.header_row} ({exc}) | fix header_row, skip_footer_rows or "
                    "delimiter on this file's mapping.files entry") from None
            if self.skip_footer_rows:
                raw = raw.iloc[:-self.skip_footer_rows]
            missing = [f"{name}={header!r}" for name, header in self.columns.items()
                       if header and header not in raw.columns]
            if missing:
                raise WeatherError(
                    f"{p}: mapped column(s) {', '.join(missing)} are not in the file | its "
                    f"headers are {list(raw.columns)}; fix mapping.columns.weather, or "
                    f"mapping.files header_row if the header is not on row {self.header_row}")
            parts.append(raw)
        return pd.concat(parts, ignore_index=True)

    def _number(self, raw, name):
        col = self.columns.get(name)
        if not col:
            return None
        return pd.to_numeric(raw[col].astype(str).str.replace(",", "", regex=False),
                             errors="coerce")

    def frame(self, dates):
        raw = self._read()
        parsed = _parse_dates(raw[self.columns["date"]], self.date_format, self.path)
        raw = raw[parsed.notna()]
        obs = pd.DataFrame(index=pd.DatetimeIndex(parsed[parsed.notna()]))

        tmax = self._number(raw, "tmax_f")
        if tmax is not None:
            if self.temp_units == "C":
                tmax = tmax * 9.0 / 5.0 + 32.0
            # catches a missing-value sentinel (NOAA writes -9999) and a gross unit error. It
            # cannot catch a plausible-looking Celsius file declared as Fahrenheit: -5..38 is a
            # real range in both, so that one is a question for the store, not for the code.
            sane = tmax.dropna()
            if len(sane) and (sane.min() < -100 or sane.max() > 150):
                raise WeatherError(
                    f"{self.path}: tmax runs {sane.min():.1f}..{sane.max():.1f} read as degrees "
                    f"{self.temp_units} | no store ever saw that; add the sentinel to this "
                    "file's na_values, or set mapping.weather.temp_units")
            obs["tmax_f"] = tmax.to_numpy()

        prcp, snow = self._number(raw, "prcp_in"), self._number(raw, "snow_in")
        if self.precip_units == "mm":
            prcp = None if prcp is None else prcp / 25.4
            snow = None if snow is None else snow / 25.4
        kind = None
        if self.columns.get("kind"):
            texts = raw[self.columns["kind"]]
            # map the distinct raw strings, not every row: a three-year file has a dozen.
            # The counter is filled from the row counts afterwards, because "two conditions
            # we could not read" and "two conditions covering a quarter of the year" are the
            # difference between ignoring the report and adding two kind_map entries.
            lookup = {t: normalize_kind(t, self.kind_map) for t in texts.dropna().unique()}
            kind = texts.map(lookup).fillna("unknown").to_numpy(dtype=object)
            unread = texts[[lookup.get(t) == "unknown" for t in texts]]
            self.unknown_conditions.update(str(t).strip().upper() for t in unread.dropna())
        if prcp is not None or snow is not None:
            gauged = kind_from_measurements(
                np.nan if prcp is None else prcp.to_numpy(),
                None if snow is None else snow.to_numpy(),
                None if tmax is None else tmax.to_numpy(), dry=self.dry_kind)
            gauged = np.asarray(gauged, dtype=object)
            kind = gauged if kind is None else np.where(kind == "unknown", gauged, kind)
        if kind is not None:
            obs["weather"] = kind

        if self.columns.get("snow_tomorrow"):
            obs["snow_tomorrow"] = _as_flag(raw[self.columns["snow_tomorrow"]],
                                            self.path).to_numpy()
        return self._align(obs, dates)


class NoWeather(WeatherProvider):
    """No weather feed at all: the default, and a supported mode rather than a broken one.

    Every date comes back tmax_f NaN, weather "unknown", snow_tomorrow 0. What that costs in
    model/features.py, exactly: tmax_z is 0.0 wherever tmax_f is NaN, so covariate 27 and
    context channel 3 are constant; "unknown" leaves the weather one-hot (covariates 28:32)
    all zeros under spec["unknown_vocab"] == "zero" and makes context channels 4 and 5 (rain,
    snow) constant; covariate 32 (snow_tomorrow) is constant. Six covariate dimensions and
    three context channels carry no information and the GRU learns to ignore them. Nothing
    changes SHAPE -- ctx_dim stays 6, cov_dim stays 35 -- because there is deliberately no
    "drop the weather block" code path here to get wrong against a saved checkpoint.

    The model still trains and the forecast is still a forecast. What is lost is the weather
    response: cold-day and rain-day levels, and the pantry-loading spike ahead of snow. A
    store with no feed should know that is the gap, and that a downloaded daily-summary CSV
    for the nearest station closes it for the cost of one download.
    """
    name = "none"

    def frame(self, dates):
        return self._align(pd.DataFrame(index=pd.DatetimeIndex([])), dates)


class LiveWeather(WeatherProvider):
    """Not wired, on purpose: this codebase makes no network call at import, test or CI time.

    The interface is fixed here so that nobody bolts an HTTP client into ht/ingest.py later.
    What a real implementation looks like, written out so the next person need not guess:

        GET https://api.weather.gov/stations/<station>/observations
            ?start=<iso>&end=<iso>          headers: {"User-Agent": "<a contact email>"}
        or an NOAA CDO daily-summary request for GHCND:<station> with datatypes
        TMAX,PRCP,SNOW -- one request covering the whole range, not one per day.
        Take the daily high, the day's dominant textualDescription (through
        normalize_kind) or the gauges (through kind_from_measurements), and tomorrow's snow
        probability from the forecast product. Write date,tmax_f,kind,snow_tomorrow to
        cache_path and read it back with CsvWeather -- this class should never be the thing
        the pipeline calls at 5:30am.

    Two things that implementation must get right, neither of them plumbing:
      - Fetch ARCHIVED FORECASTS for training, not archived observations. Training on what
        happened while serving on what was predicted is a train/serve skew that no offline
        metric in this repo can see.
      - Cache to a local CSV OUTSIDE the pipeline and keep it, so rebuilding the panel is
        reproducible and does not depend on a service being up on a Tuesday morning.
    """
    name = "live"

    def __init__(self, station, cache_path):
        super().__init__()
        self.station = station
        self.cache_path = cache_path

    def frame(self, dates):
        raise NotImplementedError(
            "live weather is not wired: this codebase makes no network calls at import, test "
            "or CI time. Export a CSV and use CsvWeather. A real implementation should fetch "
            "ARCHIVED FORECASTS for training rather than archived observations, and cache to "
            "a local CSV outside the pipeline.")


def _parse_dates(values, fmt, where):
    """Explicit format or ISO-8601, never a guess. Blanks drop out; garbage raises."""
    parsed = pd.to_datetime(values, format=fmt or "ISO8601", errors="coerce").dt.normalize()
    blank = values.isna() | (values.astype(str).str.strip() == "")
    bad = parsed.isna() & ~blank
    if bad.any():
        samples = list(values[bad].head(3))
        raise WeatherError(
            f"{where}: {int(bad.sum())} date(s) will not parse as "
            f"{fmt or 'ISO-8601 (YYYY-MM-DD)'} -- {samples} | set mapping.weather.date_format "
            "to an explicit strftime string such as '%m/%d/%Y'")
    return parsed


def make_provider(mapping, root="."):
    """Build the provider mapping.weather.provider names: "csv", "none" or "live".

    The csv provider takes its file from the mapping.files entry with role "weather" (so the
    encoding and header row a store's report needs are declared in one place), and its
    column names from mapping.columns.weather, falling back to mapping.weather.columns.
    """
    cfg = dict((mapping.get("weather") or {}))
    kind = str(cfg.get("provider") or "none").lower()
    if kind == "none":
        return NoWeather()
    if kind == "live":
        cache = cfg.get("cache_path") or ""
        return LiveWeather(cfg.get("station", ""),
                           os.path.join(root, cache) if cache else cache)
    if kind != "csv":
        raise WeatherError(f"weather.provider is {kind!r} | expected 'csv', 'none' or 'live'")

    files = [f for f in (mapping.get("files") or []) if f.get("role") == "weather"]
    if not files:
        raise WeatherError("weather.provider is 'csv' but no file has role 'weather' | add one "
                           "to mapping.files, or set weather.provider to 'none' and train "
                           "without weather")
    if len(files) > 1:
        raise WeatherError(f"{len(files)} files have role 'weather' | one station per panel; "
                           "use a glob in a single entry if the export is split by year")
    entry = files[0]
    columns = (mapping.get("columns") or {}).get("weather") or cfg.get("columns") or {}
    return CsvWeather(
        os.path.join(root, entry["path"]), columns, kind_map=cfg.get("kind_map"),
        date_format=cfg.get("date_format") or entry.get("date_format"),
        encoding=entry.get("encoding", "utf-8"), delimiter=entry.get("delimiter", ","),
        header_row=entry.get("header_row", 1),
        skip_footer_rows=entry.get("skip_footer_rows", 0), na_values=entry.get("na_values"),
        temp_units=cfg.get("temp_units", "F"), precip_units=cfg.get("precip_units", "in"),
        dry_kind=cfg.get("dry_kind", "unknown"))
