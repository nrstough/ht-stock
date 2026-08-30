"""Shared fixtures, and the two rules the whole suite is built on.

RULE ONE: no network, ever. A comment saying so would not survive the first person who
adds a "just fetch the station's archive" line to a weather provider, so socket.socket is
monkeypatched to raise for every test. A module that opens a socket at import time is a
collection error; one that opens it at call time is a failing test.

RULE TWO: nothing is written inside the repo. Every fixture that needs a file writes into
tmp_path, and the three frozen artifacts (results.json, demandnet.pt, meta.json) plus
data/store_synth.csv are only ever read.

The panel factory builds the smallest thing that is a legal canonical panel, so a test
about the split policy can say "126 days" and mean it, rather than carrying a fixture that
also happens to exercise weather, holidays and sellouts.
"""
import json
import os
import socket
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_CSV = os.path.join(REPO, "data", "store_synth.csv")
ITEMS_JSON = os.path.join(REPO, "config", "items.example.json")
MAPPING_JSON = os.path.join(REPO, "config", "source_mapping.example.json")
ARTIFACTS = os.path.join(REPO, "model", "artifacts")
RESULTS_JSON = os.path.join(REPO, "results", "results.json")

# The repo root, not tests/, is the import root: `ht` and `model` are top-level packages
# and there is no pytest.ini or pyproject.toml in this project to say so.
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the frozen backtest end to end")


class _NoNetwork(socket.socket):
    def __init__(self, *a, **kw):
        raise RuntimeError(
            "a test opened a network socket. Nothing in this project may reach the network "
            "at import, test or CI time -- export a CSV and point a provider at it.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoNetwork)
    monkeypatch.setattr(socket, "create_connection", _NoNetwork)


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def items_path():
    return ITEMS_JSON


@pytest.fixture(scope="session")
def items(items_path):
    from ht import config
    return config.load_items(items_path)


@pytest.fixture(scope="session")
def synth_raw():
    """data/store_synth.csv exactly as it sits on disk, simulator columns included."""
    return pd.read_csv(SYNTH_CSV, parse_dates=["date"])


@pytest.fixture(scope="session")
def synth_panel(synth_raw):
    """The same file as a canonical panel -- the observable columns only."""
    from ht import schema
    return schema.conform(synth_raw)


def _panel_frame(item_keys, start, days, seed=7, **columns):
    """A legal canonical panel: one row per item per day, deterministic sales."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for i, key in enumerate(item_keys):
        base = 20.0 + 10.0 * i
        sold = np.round(base + 5.0 * np.sin(np.arange(days) / 7.0 * 2 * np.pi)
                        + rng.normal(0, 2.0, days), 1).clip(0)
        frame = pd.DataFrame({
            "date": dates,
            "store": "0123",
            "item": key,
            "item_name": key.title(),
            "dept": "Test",
            "dow": dates.dayofweek,
            "holiday": "",
            "payday": dates.day.isin([1, 2, 3, 15, 16, 17]).astype(int),
            "is_closed": 0,
            "row_status": "ok",
            "tmax_f": 60.0 + 20.0 * np.sin(np.arange(days) / 365.0 * 2 * np.pi),
            "weather": "sunny",
            "snow_tomorrow": 0,
            "sold": sold,
            "produced": sold + 5.0,
            "wasted": 5.0,
            "stockout": 0,
            "stockout_known": 1,
            "sellout_source": "produced_vs_sold",
            "unit_price": 4.0,
            "unit_cost": 1.0,
        })
        for name, value in columns.items():
            frame[name] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def make_panel():
    """make_panel(["bread"], start="2025-01-01", days=200) -> a conformed panel."""
    from ht import schema

    def build(item_keys=("bread",), start="2025-01-01", days=200, conform=True, **columns):
        frame = _panel_frame(list(item_keys), start, days, **columns)
        return schema.conform(frame) if conform else frame
    return build


@pytest.fixture
def write_json(tmp_path):
    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return str(path)
    return write


@pytest.fixture
def items_doc():
    """The shipped items file as a dict, for tests that break one field at a time."""
    with open(ITEMS_JSON, encoding="utf-8") as fh:
        return json.load(fh)
