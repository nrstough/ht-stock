"""The item economics live in two files, and results/results.json needs them to agree.

model/backtest.py used to read sim/params.py; it now reads config/items.example.json, which
is what lets it run against a store's export instead of the simulator's. The nine records in
sim/params.ITEMS were left in place because sim/generate.py still needs a price to write
unit_price with -- so the same five numbers per item now exist twice, and every dollar in
results/results.json is settled with the config's copy while the panel those dollars are
settled over was generated with the simulator's. Nothing checked that. These tests do:
first that the two files still say the same thing, then that the frozen file's dollars are
arithmetic on the config's numbers, so a mismatch is a reproduction failure and not a
matter of taste.
"""
import json

import pytest

import sim.params as sim_params
from tests.conftest import RESULTS_JSON

# name and dept are on the morning sheet and in every report; price, cost and batch are the
# newsvendor's whole input. Everything else in a sim/params.py record is a simulator latent
# that has no business in a file a store manager signs off on.
SHARED = ("name", "dept", "price", "cost", "batch")

WHY = (
    "config/items.example.json is the authority -- model/backtest.py reads it, and "
    "results/results.json is settled with its price, cost and batch. sim/params.py is what "
    "GENERATED the panel those dollars are settled over. Edit whichever copy is wrong so the "
    "two agree, then re-run `python -m model.backtest` and confirm results/results.json comes "
    "back byte-identical; if it does not, the published dollar figures have moved."
)


@pytest.fixture(scope="module")
def frozen():
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_two_item_catalogs_hold_the_same_nine_keys(items):
    only_config = sorted(set(items) - set(sim_params.ITEMS))
    only_sim = sorted(set(sim_params.ITEMS) - set(items))
    assert not (only_config or only_sim), (
        f"the item rosters have drifted: only in config/items.example.json {only_config}, "
        f"only in sim/params.py {only_sim}. {WHY}")
    assert len(items) == 9


def test_price_cost_batch_name_and_dept_match_for_every_item(items):
    """The five fields both files carry. A silent disagreement moves the frozen dollars."""
    bad = []
    for key in sorted(items):
        sim_item = sim_params.ITEMS[key]
        for field in SHARED:
            want, got = sim_item[field], items[key][field]
            if want != got:
                bad.append(f"  {key}.{field}: sim/params.py {want!r} != "
                           f"config/items.example.json {got!r}")
        if bool(sim_item.get("continuous", False)) != bool(items[key]["continuous"]):
            bad.append(f"  {key}.continuous: sim/params.py "
                       f"{bool(sim_item.get('continuous', False))} != "
                       f"config/items.example.json {items[key]['continuous']}")
    assert not bad, "the two item catalogs disagree:\n" + "\n".join(bad) + f"\n{WHY}"


def test_the_frozen_critical_fractiles_are_the_configs_economics(items, frozen):
    """q* = (price - cost) / (price - salvage), rounded to 3dp by model/backtest.py."""
    from ht import config as ht_config

    want = {k: round(v, 3) for k, v in ht_config.critical_fractiles(items).items()}
    assert want == frozen["q_star"], (
        "config/items.example.json no longer produces the critical fractiles frozen in "
        f"results/results.json:\n  from the config {want}\n  in results.json "
        f"{frozen['q_star']}\nq* is the quantile every policy orders to, so a changed price "
        f"or cost changes every production quantity in the file. {WHY}")


def test_the_frozen_dollars_are_arithmetic_on_the_configs_price_and_cost(items, frozen):
    """The concrete dependency, item by item: results.json's dollars are units x config.

    waste_retail = waste_units x price, waste_cost = waste_units x cost, and
    lost_margin = lost_units x (price - cost). Nothing in results/results.json carries the
    price it was settled with, so this is the only place the file's dollars can be tied back
    to a number a store manager can check.
    """
    bad = []
    for key, detail in sorted(frozen["per_item"].items()):
        it = items[key]
        if detail["name"] != it["name"]:
            bad.append(f"  {key}: results.json names it {detail['name']!r}, the config says "
                       f"{it['name']!r}")
        sq = detail["sq"]
        for field, expected in (("waste_retail", sq["waste_units"] * it["price"]),
                                ("waste_cost", sq["waste_units"] * it["cost"]),
                                ("lost_margin", sq["lost_units"] * (it["price"] - it["cost"]))):
            if abs(sq[field] - expected) > 0.011:
                bad.append(f"  {key}.sq.{field}: results.json {sq[field]:.2f}, the config's "
                           f"numbers give {expected:.2f}")
    assert not bad, ("results/results.json was settled with different economics than "
                     "config/items.example.json now holds:\n" + "\n".join(bad) + f"\n{WHY}")
