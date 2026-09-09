"""The daily loop, served to a back-room browser. It adds no arithmetic.

Phase 3 works and is reachable only from a terminal: six commands with four path flags
each, one of them at 5:30am, every morning for twenty-eight consecutive days. The model is
not what will kill the pilot; that is. This module puts the same six commands behind a
local HTTP API so a store employee can run them from a browser, and it is deliberately a
wrapper: every number it returns is a number `python -m model.shadow` would have printed.

Four properties of the record make the pilot worth anything (see docs/features/SHADOW_MODE.md),
and putting HTTP in front of them threatens the first one in a way a terminal does not:

  `log_predictions` appends and `read_predictions` keeps the LAST row per (for_date, item).
  Under a person typing one command that is safe. Under a browser -- a refresh, a prefetch,
  a double-click, a framework that mounts twice -- a second run for a date replaces the
  morning's forecast with a later one, and nothing in the record says so. So no GET here
  writes, and a second POST for a date is refused rather than appended.

The second hazard is the opposite of a refusal: `score_day` freezes a day's verdict by file
existence, so Score pressed before the store's export lands writes an all-missing verdict
that `catch-up` cannot repair and that counts against G1 completeness for the rest of the
pilot. A CLI makes that hard to do by accident. A button does not, so the precondition is
enforced here, on the server.

Nothing in this module is a new dependency: it is the standard library, plus the pandas
that `ht/` and `model/` already require.

Handlers are pure functions of (method, path, query, body) and the http.server glue is
thin. That is a testing constraint made structural: tests/conftest.py patches socket.socket
to raise for every test, so an API test cannot bind a port, and the whole route surface is
exercised by calling Api.handle directly.
"""
import argparse
import contextlib
import datetime as dt
import http.server
import json
import math
import mimetypes
import os
import posixpath
import re
import socketserver
import sys
import threading
import time

try:
    import fcntl
except ImportError:      # pragma: no cover - POSIX only
    fcntl = None
import urllib.parse
import zoneinfo

import numpy as np
import pandas as pd

from ht import config as ht_config, schema
from model import shadow

# The private helpers of model.shadow are used on purpose. _panel, _load_meta, _yesterday,
# _write_state, _score_rows and _read_csv are the exact steps model/shadow.py's own
# subcommands take; reimplementing any of them here would be a second definition of the
# daily loop that could drift from the one the CLI runs, which is the one thing this module
# must never become.
DEFAULT_PORT = 8765
LOOPBACK = "127.0.0.1"
MAX_BODY_BYTES = 1 << 20
LOCK_NAME = ".serve.lock"

# How far back to look for an item's last sale when deciding whether a missing row is a gap
# the export can still fill or an item that has stopped. Seven days is exactly one cycle, so a
# weekly item survives with no margin at all and one skipped week -- a holiday, a stockout, a
# closure -- would reclassify it. Fourteen gives it two.
RECENT_DAYS = 14
WRITE_TIMEOUT_S = 20.0

# The one status that says "the store's export has not landed for this item-day yet", and the
# only one waiting can fix. A closed or partial day is settled rather than pending --
# weekly_report counts it as covered -- so it stays scoreable. missing_sheet is permanent by
# construction: no sheet was logged for that item, and no amount of waiting produces one, so a
# day whose only gap is missing_sheet rows is as complete as it will ever be.
UNSETTLED = {"missing_data"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# weekly_report names its file after res["week"], an ISO year-week. Anything else reaching
# os.path.join would read a file the caller chose -- and an absolute path would discard the
# directory prefix entirely.
WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")


class ServeError(schema.HtError):
    """One sentence a person at a back-room terminal can act on. Never a traceback."""

    def __init__(self, message, status=400, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


# ---- JSON, without lying about what is missing ----

def json_safe(obj):
    """Whatever pandas and numpy hand back, as something json.dumps accepts.

    NaN becomes null and never 0. An item the checkpoint never saw carries NaN quantiles by
    design (forecast() gives it its trailing par instead), and a zero there would print
    "make none" against a real item on a real morning.
    """
    if isinstance(obj, pd.DataFrame):
        return [json_safe(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return str(pd.Timestamp(obj).date())
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (str, int)):
        return obj
    # Anything else -- np.datetime64, Timedelta, Period, Decimal, bytes -- is rendered rather
    # than passed through. Returning it unchanged would push the failure into json.dumps at
    # the socket, where it drops the connection instead of answering.
    with contextlib.suppress(Exception):
        return str(obj)
    return None


# ---- one writer at a time, and readers that never see half a row ----

class RWLock:
    """Many readers or one writer.

    log_predictions and record_actuals append with csv.DictWriter and no locking, while
    read_predictions and read_overrides go straight to pd.read_csv. Serialising writers
    against each other is only half the problem: under ThreadingHTTPServer a GET runs
    beside a POST, and a reader that catches a half-written line gets a pandas tokenizer
    error that names no file -- exactly the failure _overrides_columns exists to prevent.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextlib.contextmanager
    def read(self, timeout=WRITE_TIMEOUT_S):
        with self._cond:
            # writer preference: a reader that arrives while a writer is queued waits behind
            # it. Without this the page's own status polling can hold the 5:30am morning POST
            # off until it times out into a 503 -- readers are frequent and cheap here, and a
            # writer that never acquires is the one failure a store would actually notice.
            if not self._cond.wait_for(
                    lambda: not self._writer and not self._waiting_writers, timeout):
                raise ServeError("the server is busy writing; try again in a moment", 503)
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def write(self, timeout=WRITE_TIMEOUT_S):
        with self._cond:
            self._waiting_writers += 1
            try:
                if not self._cond.wait_for(lambda: not self._writer and not self._readers,
                                           timeout):
                    # catch-up loops score_day over every logged date; a morning sheet queued
                    # behind it would otherwise hang the browser with nothing on screen
                    raise ServeError("another operation is still running; try again in a "
                                     "moment", 503)
            finally:
                self._waiting_writers -= 1
                self._cond.notify_all()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


# ---- the lock file ----

def _write_note(path):
    """Who holds the lock, for a person reading the directory. Never load-bearing.

    Written whole via a temp file and rename, so a reader never catches it half-written and
    reports "(its pid could not be read)" about a server that is perfectly healthy.
    """
    tmp = f"{path}.note"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(),
                   "since": dt.datetime.now().astimezone().isoformat(timespec="seconds")}, fh)
    os.replace(tmp, path)


def _held_by(path):
    """The pid recorded in a lock file, for the refusal message only. 0 when unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return int(json.load(fh).get("pid") or 0)
    except (OSError, ValueError, TypeError):
        return 0


@contextlib.contextmanager
def hold_lock(out_dir):
    """One server per shadow directory, enforced by the kernel rather than by inspection.

    Three earlier attempts at this got it wrong in three different ways, and all three shared
    a root cause: they tried to work out whether the *previous* holder was still alive, from
    a pid written in a file. That question cannot be answered without a race. Checking a pid
    and then removing the file is a time-of-check/time-of-use gap wide enough that two
    starters racing a stale lock both removed it and both claimed -- measured at roughly one
    start in five, in exactly the situation the reclaim existed for: a power cut leaves a
    lock behind, the box reboots, and cron and the operator both start the server.

    flock does not ask the question. The lock lives on an open file descriptor, the kernel
    releases it when the holder exits for ANY reason -- clean shutdown, kill, power loss --
    and a second attempt either gets it or does not. There is nothing to reclaim, no
    staleness to detect, no pid to trust, and no window in which the file says nothing. The
    pid inside the file is only there so a person can see who holds it.

    Yields True when it had to CREATE the directory. Taking a lock means creating it, which
    quietly undoes the CLI's rule that only `morning` does -- and that rule is what stops a
    typo in --out from reading an empty log and reporting a pilot that lost its record. The
    server cannot refuse instead (it needs the lock before it serves anything), so it says so
    at startup and on every status response rather than letting an empty record pass for an
    empty week.
    """
    if fcntl is None:                      # pragma: no cover - POSIX only in practice
        raise ServeError(
            "this platform has no flock, so ht.serve cannot guarantee that only one server "
            "writes to a shadow directory. Run it on Linux or macOS.")
    root = os.path.realpath(out_dir)
    fresh = not os.path.isdir(root)
    try:
        os.makedirs(root, exist_ok=True)
        # The lock is taken on the DIRECTORY, not on a file inside it. flock lives on an
        # inode while a path is only a name for one, so locking .serve.lock would leave the
        # obvious hole: delete the name, and the next starter's O_CREAT mints a fresh inode
        # carrying no lock at all. That is not a hypothetical -- the file is deliberately
        # left behind after a clean shutdown, so a leftover .serve.lock is the normal state
        # of a pilot record and "tidy up that stale lock" is exactly what somebody reaches
        # for. A directory cannot be replaced the same way without taking the record with it.
        dir_fd = os.open(root, os.O_RDONLY)
    except OSError as exc:
        raise ServeError(
            f"--out: cannot use {out_dir} as a shadow directory "
            f"({exc.strerror}).") from None
    path = os.path.join(root, LOCK_NAME)
    try:
        try:
            fcntl.flock(dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            held = _held_by(path)
            who = f"(pid {held})" if held else "(its pid could not be read)"
            raise ServeError(
                f"another ht.serve {who} is already serving {root}. Stop that process "
                f"first -- deleting {path} will not help, because the lock is held on the "
                f"directory itself and that file is only a note saying who holds it.") from None
        with contextlib.suppress(OSError):
            _write_note(path)
        yield fresh
    finally:
        with contextlib.suppress(OSError):
            os.close(dir_fd)


# ---- settings ----

class Settings:
    """The flags, once. Shared-flag defaults mirror the `morning` subparser.

    model.shadow has six subparsers whose defaults diverge -- --artifacts is required for
    morning and defaults to None for weekly -- so "the CLI's defaults" is not one set. The
    morning subparser is the one this server's own morning route reproduces, so it is the
    one the defaults are pinned to.
    """

    def __init__(self, panel, items, artifacts, out="shadow", store="",
                 max_staleness=shadow.MAX_STALENESS_DAYS, timezone=None, entered_by="",
                 web_dist=None):
        self.panel_path = panel
        self.items_path = items
        self.artifacts_dir = artifacts
        self.out_dir = out
        self.store = store
        self.max_staleness = int(max_staleness)
        self.timezone = timezone
        self.entered_by = entered_by
        self.web_dist = web_dist

    def today(self):
        """The store's today, in the store's zone.

        Nothing in model/shadow.py computes this: --date is required on every subcommand
        that takes one. A server has to default it, and a server whose clock runs ahead of
        the store rolls over to tomorrow while the store is still on today -- which no guard
        refuses, because a FUTURE date is never backfilled. So the zone is a required flag
        rather than a guess, and the UI shows the date it resolved.
        """
        return str(dt.datetime.now(zoneinfo.ZoneInfo(self.timezone)).date())


# ---- the API ----

class Api:
    """Every route is a thin wrapper over model.shadow. No route computes a number."""

    def __init__(self, settings, fresh_out=False):
        self.s = settings
        self.fresh_out = fresh_out
        self.lock = RWLock()
        self._panel_cache = (None, None)
        self._items_cache = (None, None)
        # the caches are written under the READ lock too (several readers hold it at once),
        # so they get their own mutex rather than relying on which bytecodes happen to be
        # atomic in one interpreter
        self._cache_lock = threading.Lock()

    # -- cached inputs, refreshed when the file underneath them moves --

    @staticmethod
    def _stamp(path):
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)

    def _cached(self, attr, path, build, flag):
        try:
            stamp = self._stamp(path)
        except OSError as exc:
            # The person holding the browser did not type this path and cannot act on it, and
            # this module's own rule is that an absolute path in a browser helps nobody. The
            # flag and the file's name are enough to find it; the server's console has the
            # rest.
            raise ServeError(
                f"--{flag}: {os.path.basename(path)} is no longer readable "
                f"({exc.strerror}). The service was started with a path that has since moved "
                f"or been removed; restart it with the right --{flag}.", 503) from None
        with self._cache_lock:
            held_stamp, value = getattr(self, attr)
            if held_stamp == stamp:
                return value
        value = build(path)                       # built outside the mutex: it reads a file
        with self._cache_lock:
            setattr(self, attr, (stamp, value))
        return value

    def _panel(self):
        """The canonical panel, re-read when a fresh ingest replaces it.

        schema.read_panel conforms what it reads, and conform drops the simulator-only
        columns whatever it is asked, so a panel file carrying them cannot deliver them
        here. shadow._panel's assert_no_truth is a belt-and-braces check on an
        already-clean frame rather than the refusal it looks like.
        """
        return self._cached("_panel_cache", self.s.panel_path, shadow._panel, "panel")

    def _items(self):
        return self._cached("_items_cache", self.s.items_path, ht_config.load_items,
                            "items")

    def _items_hash(self):
        # recomputed per logging request: a config edited while the server runs would
        # otherwise stamp every later prediction with the hash it booted on
        return ht_config.config_hash(self.s.items_path)

    def _require_out(self):
        """Only `morning` creates the shadow directory; everything else refuses its absence.

        shadow.status() does not refuse -- it returns a dict of Nones -- so a mistyped --out
        would report a pilot that lost its record rather than a path that does not exist.
        The guard has to live here, in the same words model/shadow.py uses.
        """
        if not os.path.isdir(self.s.out_dir):
            raise ServeError(f"--out: no shadow directory at {self.s.out_dir}; the morning "
                             f"sheet creates it and every other command works inside it")

    # -- helpers --

    @staticmethod
    def _date(value, field="date"):
        if value in (None, ""):
            raise ServeError(f"{field} is required, as YYYY-MM-DD")
        text = str(value)
        if not DATE_RE.match(text):
            raise ServeError(f"{field}: {text!r} is not a date; write it as YYYY-MM-DD")
        try:
            return pd.Timestamp(text).normalize()
        except ValueError as exc:
            raise ServeError(f"{field}: {text!r} is not a date ({exc}); write it as "
                             f"YYYY-MM-DD") from None

    def _window(self, query):
        """from/to as dates or not at all -- pandas' parse errors are not sentences."""
        return tuple(self._date(query[k], k) if query.get(k) else None
                     for k in ("from", "to"))

    def _unsettled(self, panel, items, for_date):
        """Items whose sales data has not landed for a day the export otherwise covers.

        score_day freezes the whole file at once, so any row still waiting would be recorded
        as missing for the rest of the pilot. Two cases are deliberately NOT "waiting":

        a day the panel does not cover at all -- that is "the export has not arrived", which
        the caller reports differently; and an item with no rows anywhere near the date, which
        is a discontinued item or one added to the items file ahead of the export. Those never
        become scoreable, and treating them as waiting would refuse the day forever with no
        way out. Their absence is reported and frozen rather than blocking the other eight.
        """
        fresh = shadow._score_rows(panel, items, self.s.out_dir, for_date)
        if not len(fresh):
            return fresh, [], []
        waiting = set(fresh.loc[fresh["status"].astype(str).isin(UNSETTLED),
                                "item"].astype(str))
        if not waiting:
            return fresh, [], []
        df = panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        if not len(df[df["date"] == for_date]):
            # the export for this date has not arrived at all -- everything is still coming
            return fresh, sorted(waiting), []
        # It HAS arrived, so an item missing from it is either a partial ingest or an item
        # that has stopped. Recent history separates them: one with rows the day before and
        # none today is a gap the store can still fill; one with no row for a fortnight is
        # discontinued, or was added to the items file ahead of the export, and no amount of
        # waiting produces a row. Refusing on the second kind would refuse the day forever.
        # Note this measures rows in the panel, not units sold: an item present every day
        # with nothing sold is still one the export is covering.
        recent = df[(df["date"] < for_date)
                    & (df["date"] >= for_date - pd.Timedelta(days=RECENT_DAYS))]
        alive = set(recent["item"].astype(str))
        pending = sorted(k for k in waiting if k in alive)
        gone = sorted(k for k in waiting if k not in alive)
        return fresh, pending, gone

    def _runs_for(self, for_date):
        """Every logged run for a date, oldest first -- before keep="last" collapses them.

        This is the only honest evidence that a sheet was made twice: PREDICTION_COLUMNS has
        no field saying so, and the printed sheet carries no marker. read_predictions cannot
        show it because resolving duplicates is exactly its job.
        """
        path = os.path.join(self.s.out_dir, "predictions.csv")
        df = shadow._read_csv(path, ("for_date",), shadow.PREDICTION_COLUMNS)
        if not len(df):
            return []
        df = df[pd.to_datetime(df["for_date"]).eq(for_date)]
        if not len(df):
            return []
        seen, runs = set(), []
        for _, row in df.iterrows():
            rid = str(row.get("run_id", ""))
            if rid in seen:
                continue
            seen.add(rid)
            runs.append({"run_id": rid, "made_at": str(row.get("made_at", "")),
                         "backfilled": int(pd.to_numeric(row.get("backfilled", 0),
                                                         errors="coerce") or 0)})
        return runs

    def _sheet_meta_path(self, for_date):
        return os.path.join(self.s.out_dir, "sheets", f"{for_date.date()}.meta.json")

    # ---- routes ----

    def status(self):
        self._require_out()
        with self.lock.read():
            # status reads predictions.csv and scores/ through read_predictions/read_scores,
            # so it tears on a concurrent append exactly like /api/predictions would. It is
            # also the route the page polls, which makes it the likeliest one to be reading
            # while a morning sheet is being written.
            st = shadow.status(self.s.out_dir)
        st["today"] = self.s.today()
        st["store"] = self.s.store
        # an empty record and a mistyped --out look identical from in here, and the CLI's
        # protection (only `morning` creates the directory) cannot apply to a server that
        # must take a lock inside it before serving
        st["new_shadow_dir"] = bool(self.fresh_out and not st["last_sheet_date"])
        st["shadow_dir"] = os.path.realpath(self.s.out_dir)
        return 200, json_safe(st)

    def config(self):
        with self.lock.read():
            items = self._items()
            panel = self._panel()
        dates = pd.to_datetime(panel["date"])
        meta, version = {}, ""
        with contextlib.suppress(OSError, ValueError, KeyError):
            meta = shadow._load_meta(self.s.artifacts_dir)
            # the same function forecast() stamps onto every logged row; meta.json's own
            # fields are not it, and a config page quoting a different one would let an
            # operator compare a version against a log that never carried it
            version = shadow.evaluate.model_version(self.s.artifacts_dir)
        return 200, json_safe({
            "store": self.s.store,
            "today": self.s.today(),
            "timezone": self.s.timezone,
            "entered_by": self.s.entered_by,
            "items": {k: dict(v) for k, v in items.items()},
            "panel_first_date": dates.min(),
            "panel_last_date": dates.max(),
            "model_version": version,
            "taus": list(meta.get("taus", [])),
            "max_staleness": self.s.max_staleness,
        })

    def morning_get(self, query):
        """The sheet that was logged for a date. Never logs one."""
        self._require_out()
        for_date = self._date(query.get("date"))
        with self.lock.read():
            runs = self._runs_for(for_date)
            if not runs:
                raise ServeError(f"no morning sheet has been made for {for_date.date()}",
                                 404)
            preds = shadow.read_predictions(self.s.out_dir, for_date, for_date)
            meta = {}
            path = self._sheet_meta_path(for_date)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    meta = json.load(fh)
        return 200, json_safe({
            "for_date": for_date, "rows": preds, "runs": runs,
            "superseded": len(runs) > 1, "live_run_id": runs[-1]["run_id"],
            "conditions": meta.get("conditions", {}), "caveats": meta.get("caveats", []),
            "yesterday": meta.get("yesterday", []),
            "day_source": meta.get("day_source", ""),
            "staleness_days": meta.get("staleness_days"),
        })

    def morning_post(self, body):
        """Forecast, log, render both sheets. The only route that creates the shadow dir."""
        for_date = self._date(body.get("date"))
        backfill = bool(body.get("backfill"))
        reforecast = bool(body.get("reforecast"))
        with self.lock.write():
            runs = self._runs_for(for_date)
            if runs and not reforecast:
                raise ServeError(
                    f"a sheet was already logged for {for_date.date()} (run "
                    f"{runs[-1]['run_id']} at {runs[-1]['made_at']}). The prediction log is "
                    f"append-only and the newest run becomes the live one, so re-forecasting "
                    f"replaces what the morning said. Confirm deliberately to do it anyway.",
                    409, runs=runs)
            panel = self._panel()
            items = self._items()
            recs = shadow.forecast(panel, self.s.artifacts_dir, items, for_date,
                                   allow_backfill=backfill,
                                   max_staleness=self.s.max_staleness)
            n = shadow.log_predictions(recs, self.s.out_dir, store=self.s.store,
                                       items_config_hash=self._items_hash())
            meta = shadow._load_meta(self.s.artifacts_dir)
            known = pd.to_numeric(panel.get("stockout_known"), errors="coerce")
            caveats = shadow.sheet_caveats(recs, items, meta,
                                           float(known.mean()) if known is not None else 1.0)
            yest = shadow._yesterday(panel, self.s.out_dir, for_date, items)
            conditions = dict(recs.attrs["conditions"])

            sheets = os.path.join(self.s.out_dir, "sheets")
            os.makedirs(sheets, exist_ok=True)
            for fmt, ext in (("text", "txt"), ("html", "html")):
                body_text = shadow.morning_sheet(
                    recs, store=self.s.store, for_date=for_date, conditions=conditions,
                    yesterday=yest, caveats=caveats, fmt=fmt)
                with open(os.path.join(sheets, f"{for_date.date()}.{ext}"), "w",
                          encoding="utf-8") as fh:
                    fh.write(body_text)
            # recs.attrs survives neither to_dict("records") nor the CSV, and morning_sheet
            # requires conditions, so a sheet can never be re-rendered from predictions.csv.
            # What the GET route needs is written down here instead, while it is in hand.
            with open(self._sheet_meta_path(for_date), "w", encoding="utf-8") as fh:
                json.dump(json_safe({
                    "conditions": conditions, "caveats": caveats,
                    "yesterday": yest, "day_source": recs.attrs.get("day_source", ""),
                    "warnings": recs.attrs.get("warnings", []),
                    "staleness_days": recs.attrs.get("staleness_days"),
                }), fh, indent=1)

            shadow._write_state(
                self.s.out_dir, last_sheet_date=str(for_date.date()),
                current_artifacts_dir=self.s.artifacts_dir,
                last_ingested_date=str(pd.Timestamp(panel["date"].max()).date()),
                current_model_version=recs["model_version"].iloc[0],
                sellout_source=recs["sellout_source"].iloc[0])
            runs = self._runs_for(for_date)
        return 200, json_safe({
            "for_date": for_date, "logged": n, "rows": recs, "caveats": caveats,
            "conditions": conditions, "yesterday": yest, "runs": runs,
            "superseded": len(runs) > 1,
            "backfilled": int(recs["backfilled"].iloc[0]),
            "day_source": recs.attrs.get("day_source", ""),
            "staleness_days": recs.attrs.get("staleness_days"),
        })

    def sheet(self, for_date, ext):
        self._require_out()
        for_date = self._date(for_date)
        if ext not in ("txt", "html"):
            raise ServeError(f"no sheet format {ext!r}; the sheet is written as .txt and "
                             f".html", 404)
        path = os.path.join(self.s.out_dir, "sheets", f"{for_date.date()}.{ext}")
        if not os.path.exists(path):
            raise ServeError(f"no {ext} sheet for {for_date.date()}", 404)
        with self.lock.read(), open(path, encoding="utf-8") as fh:
            return 200, {"content_type": "text/html; charset=utf-8" if ext == "html"
                         else "text/plain; charset=utf-8", "body": fh.read()}

    def entry_order(self, query):
        """The items in the order the paper printed them, with what the sheet said."""
        self._require_out()
        for_date = self._date(query.get("date"))
        items = self._items()
        with self.lock.read():
            preds = shadow.read_predictions(self.s.out_dir, for_date, for_date)
            keys = shadow.entry_order(self.s.out_dir, for_date, items)
        said = dict(zip(preds["item"].astype(str), preds["rec_qty"])) if len(preds) else {}
        # entry_order falls back to alphabetical when the date has no sheet. On a mistyped
        # date that silently presents a different page from the one in the operator's hand,
        # so the fallback is named rather than just used.
        no_sheet = not len(preds)
        rows = [{"item": k, "name": items[k]["name"], "dept": items[k]["dept"],
                 "unit": items[k].get("unit", "each"),
                 "said": said.get(k), "source": None} for k in keys]
        if len(preds):
            source = dict(zip(preds["item"].astype(str), preds["source"].astype(str)))
            for row in rows:
                row["source"] = source.get(row["item"])
        out = {"for_date": for_date, "rows": rows, "no_sheet_logged": no_sheet,
               "entered_by": self.s.entered_by}
        if no_sheet:
            out["warning"] = (f"no sheet was logged for {for_date.date()}; this page is in "
                              f"alphabetical order, not the order the sheet printed. Check "
                              f"the date on the paper in your hand before entering it.")
        return 200, json_safe(out)

    def enter(self, body):
        """Key the returned paper sheet in. All of it parses, or none of it is written."""
        self._require_out()
        for_date = self._date(body.get("date"))
        lines = body.get("lines")
        if lines is None:
            raise ServeError("lines is required: 'item, made, sold out at[, note]' per entry")
        if isinstance(lines, str):
            lines = lines.splitlines()
        if not isinstance(lines, list):
            raise ServeError("lines must be a list of 'item, made, sold out at' strings")
        entered_by = str(body.get("by") or self.s.entered_by or "")
        items = self._items()
        rows, errors = shadow.parse_entries([str(x) for x in lines], items)
        if errors:
            # nothing is written: a half-keyed day that looks entered is worse than one that
            # obviously is not
            return 400, {"error": "nothing was written. Fix these and send it again:",
                         "errors": errors}
        if not rows:
            return 200, {"for_date": str(for_date.date()), "written": 0,
                         "message": "nothing to record"}
        with self.lock.write():
            warning = None
            if not self._runs_for(for_date):
                warning = (f"no sheet was logged for {for_date.date()}; recording it anyway. "
                           f"Check the date on the paper in your hand.")
            path = shadow.record_actuals(self.s.out_dir, for_date, rows,
                                         entered_by=entered_by)
        return 200, json_safe({
            "for_date": for_date, "written": len(rows), "path": os.path.basename(path),
            "entered_by": entered_by, "warning": warning,
            "with_production": sum(1 for r in rows if r["actual_produced"] != ""),
            "sold_out": sum(1 for r in rows if r["sold_out_at"]),
        })

    def score(self, body):
        """Freeze one day's verdict -- but only once there is something to freeze.

        score_day is write-once by file existence, and on a day whose export has not landed
        every row comes back missing_data. Writing that freezes an empty verdict: catch_up
        calls the same function, sees the file and only records revisions, so the day never
        enters the settled set and counts against G1 completeness for the rest of the pilot.
        A person typing a command rarely does this. A button would do it every week.
        """
        self._require_out()
        for_date = self._date(body.get("date"))
        panel = self._panel()
        items = self._items()
        with self.lock.write():
            path = os.path.join(self.s.out_dir, "scores", f"{for_date.date()}.csv")
            gone = []
            if not os.path.exists(path):
                fresh, pending, gone = self._unsettled(panel, items, for_date)
                if not len(fresh):
                    raise ServeError(
                        f"{for_date.date()} has nothing to score: no sheet was logged for it "
                        f"and the panel carries no rows for it.", 409)
                if pending:
                    raise ServeError(
                        f"{for_date.date()} is not fully in the panel yet -- "
                        f"{len(pending)} item(s) have no sales data "
                        f"({', '.join(pending[:4])}{', …' if len(pending) > 4 else ''}). A "
                        f"day's verdict is frozen once and cannot be re-scored, so scoring now "
                        f"would record those items as missing for the rest of the pilot. "
                        f"Ingest the day's export, then score it -- or use catch-up, which "
                        f"scores the days that are complete and leaves this one alone.", 409)
            rows = shadow.score_day(panel, items, self.s.out_dir, for_date)
            # Same write _cmd_score makes, kept for parity even though scoring an earlier day
            # after a later one moves the stored value backwards. Nothing reads it: status()
            # derives last_scored_date from the score files themselves. catch-up does NOT
            # write it, because the CLI's catch-up does not either.
            shadow._write_state(self.s.out_dir, last_scored_date=str(for_date.date()))
        counts = rows["status"].astype(str).value_counts().to_dict()
        out = {"for_date": for_date, "counts": counts, "rows": int(len(rows))}
        if gone:
            # named rather than buried in a missing_data count: freezing a row is permanent,
            # and the operator is the only one who can say whether the item really has
            # stopped or the export is simply behind
            out["frozen_without_data"] = gone
            one = len(gone) == 1
            out["note"] = (
                f"{', '.join(gone)} {'has' if one else 'have'} no sales data for this day and "
                f"{'has' if one else 'have'} no row in the panel at all for the last "
                f"{RECENT_DAYS} days, so {'it was' if one else 'they were'} recorded as "
                f"missing rather than holding the day back. If "
                f"{'that item is' if one else 'those items are'} still selling, the export is "
                f"behind and this day's verdict for "
                f"{'it' if one else 'them'} cannot be changed.")
        return 200, json_safe(out)

    def catch_up(self, body):
        self._require_out()
        since = body.get("since")
        since = self._date(since, "since") if since else None
        panel = self._panel()
        items = self._items()
        with self.lock.write():
            # shadow.catch_up skips a date only when the panel has NO rows for it, so on its
            # own it would freeze exactly the partly-landed day POST /api/score refuses -- and
            # the refusal names this route as the safe one. Days with a pending item are held
            # back here so that sentence is true.
            logged = shadow.read_predictions(self.s.out_dir)
            dates = sorted(pd.Timestamp(d) for d in logged["for_date"].unique()) \
                if len(logged) else []
            if since:
                dates = [d for d in dates if d >= since]
            done, held, frozen = [], {}, {}
            for day in dates:
                if os.path.exists(os.path.join(self.s.out_dir, "scores",
                                               f"{day.date()}.csv")):
                    continue
                fresh, pending, gone = self._unsettled(panel, items, day)
                if not len(fresh):
                    continue
                if pending:
                    held[str(day.date())] = pending
                    continue
                if gone:
                    frozen[str(day.date())] = gone
                shadow.score_day(panel, items, self.s.out_dir, day)
                done.append(str(day.date()))
            # deliberately no _write_state here: the CLI's catch-up writes none, and
            # last_scored_date is derived from the score files anyway, so writing the last
            # date scored in THIS pass could move the stored value backwards when an earlier
            # day is filled in after a later one
            st = shadow.status(self.s.out_dir)
        return 200, json_safe({"scored": done, "unscored": st["unscored_dates"],
                               "waiting_on_data": held, "frozen_without_data": frozen})

    def weekly_post(self, body):
        """The district-manager page. A POST because the CLI's weekly writes files."""
        self._require_out()
        week_ending = self._date(body.get("week_ending"), "week_ending")
        weeks = body.get("weeks") or 1
        try:
            weeks = int(weeks)
        except (TypeError, ValueError):
            raise ServeError(f"weeks: {weeks!r} is not a whole number of weeks") from None
        if not 1 <= weeks <= 52:
            raise ServeError(f"weeks: {weeks} is not between 1 and 52") from None
        include_backfilled = bool(body.get("include_backfilled"))
        panel = self._panel()
        items = self._items()
        with self.lock.write():
            res = shadow.weekly_report(self.s.out_dir, panel, items, week_ending,
                                       weeks=weeks, include_backfilled=include_backfilled)
            root = os.path.join(self.s.out_dir, "weekly")
            os.makedirs(root, exist_ok=True)
            stem = os.path.join(root, res["week"])
            with open(stem + ".txt", "w", encoding="utf-8") as fh:
                fh.write(shadow.format_weekly(res, "text", shadow.WEEKLY_WIDTH) + "\n")
            with open(stem + ".json", "w", encoding="utf-8") as fh:
                fh.write(shadow.format_weekly(res, "json") + "\n")
            shadow._write_state(self.s.out_dir, last_gates=res["gates"])
        return 200, json_safe(res)

    def weekly_get(self, query):
        """A report that was already written. Never writes one."""
        self._require_out()
        week = query.get("week")
        if week:
            # the shape is checked before the path is built, the same way the sheet route
            # leans on DATE_RE: a name that cannot be a week cannot name a file either
            if not WEEK_RE.match(str(week)):
                raise ServeError(f"{week!r} is not a week; write it as 2026-W01", 404)
            root = os.path.realpath(os.path.join(self.s.out_dir, "weekly"))
            path = os.path.realpath(os.path.join(root, f"{week}.json"))
            if os.path.dirname(path) != root or not os.path.exists(path):
                raise ServeError(f"no weekly report has been made for {week}", 404)
            with self.lock.read(), open(path, encoding="utf-8") as fh:
                return 200, json.load(fh)
        root = os.path.join(self.s.out_dir, "weekly")
        if not os.path.isdir(root):
            return 200, {"weeks": []}
        # only weeks the single-week branch will actually serve: listing a name it then 404s
        # sends the page after a report that is not fetchable
        weeks = sorted(f[:-5] for f in os.listdir(root)
                       if f.endswith(".json") and WEEK_RE.match(f[:-5])
                       and os.path.isfile(os.path.realpath(os.path.join(root, f)))
                       and os.path.dirname(os.path.realpath(os.path.join(root, f)))
                       == os.path.realpath(root))
        return 200, {"weeks": weeks}

    def scores(self, query):
        self._require_out()
        with self.lock.read():
            df = shadow.read_scores(self.s.out_dir, *self._window(query))
        return 200, json_safe({"rows": df})

    def predictions(self, query):
        self._require_out()
        with self.lock.read():
            df = shadow.read_predictions(self.s.out_dir, *self._window(query))
        return 200, json_safe({"rows": df})

    # ---- dispatch ----

    ROUTES = (
        ("GET", "/api/status"), ("GET", "/api/config"), ("GET", "/api/morning"),
        ("POST", "/api/morning"), ("GET", "/api/entry-order"), ("POST", "/api/enter"),
        ("POST", "/api/score"), ("POST", "/api/catch-up"), ("POST", "/api/weekly"),
        ("GET", "/api/weekly"), ("GET", "/api/scores"), ("GET", "/api/predictions"),
    )

    def handle(self, method, path, query=None, body=None):
        """(status, payload). The only entry point; the HTTP glue adds nothing."""
        query = query or {}
        body = body if body is not None else {}
        try:
            return self._route(method, path, query, body)
        except ServeError as exc:
            return exc.status, dict({"error": str(exc)}, **json_safe(exc.extra))
        except schema.HtError as exc:
            return 400, {"error": str(exc)}
        except (OSError, ValueError, KeyError) as exc:
            # forecast() raises these carrying the sentence a person needs ("panel carries
            # actuals through ... pass --backfill"); the traceback above it helps nobody in a
            # back room, and an absolute path in a browser helps nobody at all.
            return 400, {"error": str(exc)}

    def _route(self, method, path, query, body):
        path = path.rstrip("/") or "/"
        if path.startswith("/api/sheet/"):
            if method != "GET":
                raise ServeError(f"{method} is not allowed on {path}", 405)
            name = path[len("/api/sheet/"):]
            if "/" in name or ".." in name:
                raise ServeError("not found", 404)
            stem, _, ext = name.rpartition(".")
            return self.sheet(stem, ext)
        table = {
            ("GET", "/api/status"): lambda: self.status(),
            ("GET", "/api/config"): lambda: self.config(),
            ("GET", "/api/morning"): lambda: self.morning_get(query),
            ("POST", "/api/morning"): lambda: self.morning_post(body),
            ("GET", "/api/entry-order"): lambda: self.entry_order(query),
            ("POST", "/api/enter"): lambda: self.enter(body),
            ("POST", "/api/score"): lambda: self.score(body),
            ("POST", "/api/catch-up"): lambda: self.catch_up(body),
            ("POST", "/api/weekly"): lambda: self.weekly_post(body),
            ("GET", "/api/weekly"): lambda: self.weekly_get(query),
            ("GET", "/api/scores"): lambda: self.scores(query),
            ("GET", "/api/predictions"): lambda: self.predictions(query),
        }
        if (method, path) in table:
            return table[(method, path)]()
        if any(p == path for _, p in self.ROUTES):
            raise ServeError(f"{method} is not allowed on {path}", 405)
        raise ServeError(f"no route {path}", 404)


# ---- static files ----

def static_response(dist, url_path):
    """Serve the built frontend, refusing anything outside it."""
    if not dist or not os.path.isdir(dist):
        return 503, "text/plain; charset=utf-8", (
            "The web interface has not been built yet. Run:\n\n"
            "    npm --prefix web ci && npm --prefix web run build\n")
    rel = posixpath.normpath(urllib.parse.unquote(url_path)).lstrip("/")
    if rel in ("", "."):
        rel = "index.html"
    full = os.path.realpath(os.path.join(dist, rel))
    root = os.path.realpath(dist)
    if full != root and not full.startswith(root + os.sep):
        return 404, "text/plain; charset=utf-8", "not found"
    if not os.path.isfile(full):
        full = os.path.join(root, "index.html")      # the SPA's own routing
        if not os.path.isfile(full):
            return 404, "text/plain; charset=utf-8", "not found"
    ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
    with open(full, "rb") as fh:
        return 200, ctype, fh.read()


# ---- the http.server glue, which does nothing but move bytes ----

def make_handler(api):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ht.serve"

        def _send(self, status, ctype, payload):
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(payload)

        def _dispatch(self, method):
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                status, ctype, payload = static_response(api.s.web_dist, parsed.path)
                return self._send(status, ctype, payload)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            body = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self._send(413, "application/json",
                                  json.dumps({"error": "that request is too large"}))
            if length:
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return self._send(400, "application/json",
                                      json.dumps({"error": "the request body is not JSON"}))
                if not isinstance(body, dict):
                    return self._send(400, "application/json",
                                      json.dumps({"error": "the request body must be a "
                                                           "JSON object"}))
            status, payload = api.handle(method, parsed.path, query, body)
            if isinstance(payload, dict) and "content_type" in payload:
                return self._send(status, payload["content_type"], payload["body"])
            try:
                body_text = json.dumps(payload)
            except (TypeError, ValueError):
                # json_safe should have made this impossible; if a field ever slips past it,
                # answer with a sentence rather than dropping the connection mid-response
                status, body_text = 500, json.dumps(
                    {"error": "the server could not encode its own answer; check the log"})
            return self._send(status, "application/json", body_text)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def log_message(self, fmt, *args):
            pass

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---- the CLI ----

def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m ht.serve",
        description="serve the shadow-mode daily loop to a browser on this machine")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", default="shadow")
    ap.add_argument("--store", default="")
    ap.add_argument("--max-staleness", type=int, default=shadow.MAX_STALENESS_DAYS)
    ap.add_argument("--timezone", required=True,
                    help="the STORE's timezone, e.g. America/Chicago. Required rather than "
                         "guessed: a server clock in a zone ahead of the store would default "
                         "the date to tomorrow, and no guard refuses a future date")
    ap.add_argument("--by", default=os.environ.get("USER", ""),
                    help="who is keying sheets in; recorded on every row")
    ap.add_argument("--web", default=None, help="the built frontend (default web/dist)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=LOOPBACK)
    ap.add_argument("--allow-remote", action="store_true",
                    help="required to bind anything but 127.0.0.1. There is no "
                         "authentication: anyone who can reach the port can write to the "
                         "pilot's record")
    return ap


def check_args(args):
    """Refuse a mistyped path or flag in one sentence, before anything is read or served."""
    for flag, what in (("panel", "panel csv"), ("items", "items config")):
        path = getattr(args, flag)
        if not os.path.exists(path):
            raise schema.HtError(f"--{flag}: no {what} at {path}")
    if not os.path.exists(os.path.join(args.artifacts, "meta.json")):
        raise schema.HtError(f"--artifacts: {args.artifacts} has no meta.json -- point it at "
                             f"a trained model directory, e.g. model/artifacts")
    try:
        zoneinfo.ZoneInfo(args.timezone)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise schema.HtError(f"--timezone: {args.timezone!r} is not a timezone; write it as "
                             f"a region, e.g. America/Chicago") from None
    if os.path.exists(args.out) and not os.path.isdir(args.out):
        raise schema.HtError(f"--out: {args.out} is a file, not a directory; the shadow "
                             f"directory is where the pilot's record lives")
    if args.host != LOOPBACK and not args.allow_remote:
        raise schema.HtError(
            f"--host {args.host} would serve the pilot's record to the network, and there is "
            f"no authentication in front of it: anyone who can reach port {args.port} could "
            f"write to it. Pass --allow-remote if that is really what you want, or leave the "
            f"default and open the page on this machine.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        check_args(args)
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    # log_predictions and record_actuals stamp made_at with datetime.now().astimezone(), which
    # reads the PROCESS timezone. In the CLI that is the operator's shell; a long-lived server
    # inherits whatever started it, usually UTC, and a 5:30am sheet would then be filed as
    # 11:30. Ordering survives either way because the offset is recorded, but the person
    # auditing "was this made before the store opened?" should not have to convert.
    os.environ["TZ"] = args.timezone
    with contextlib.suppress(AttributeError):
        time.tzset()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings = Settings(panel=args.panel, items=args.items, artifacts=args.artifacts,
                        out=args.out, store=args.store, max_staleness=args.max_staleness,
                        timezone=args.timezone, entered_by=args.by,
                        web_dist=args.web or os.path.join(repo, "web", "dist"))
    try:
        with hold_lock(args.out) as fresh_out:
            api = Api(settings, fresh_out=fresh_out)
            httpd = Server((args.host, args.port), make_handler(api))
            print(f"ht.serve  ->  http://{args.host}:{args.port}")
            print(f"  panel {args.panel}\n  shadow dir {os.path.realpath(args.out)}")
            print(f"  the store's today is {settings.today()} ({args.timezone})")
            if fresh_out:
                print(f"  NOTE: {os.path.realpath(args.out)} did not exist and has been "
                      f"created.\n        If you expected an existing pilot record here, "
                      f"stop and check --out.")
            if not os.path.isdir(settings.web_dist):
                print("  the web interface is not built yet: "
                      "npm --prefix web ci && npm --prefix web run build")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nstopped")
            finally:
                httpd.server_close()
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
