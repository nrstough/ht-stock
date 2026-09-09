"""Regenerate tests/fixtures/entry_vectors.json from model.shadow.parse_entries.

The returned paper sheet is parsed twice: once in the browser, so a mistake shows up next to
the box that caused it, and once on the server, whose answer is the one that decides whether
anything is written. Two parsers drift. A browser that accepts what the server refuses costs
a keystroke; one that refuses what the server accepts loses a returned sheet, and for a store
with no sellout column in its export that sheet is the only observation the pilot will ever
get of that day.

So neither side owns the expectations. This script asks the Python parser -- the authority --
what each vector means, and both test suites assert against the file it writes. Adding a case
here and re-running is how a new rule reaches both implementations at once.

    python scripts/make_entry_vectors.py

It rewrites the fixture in place; commit the result. If the file changes when you did not
mean to change parsing behaviour, that diff is the finding.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ht import config as ht_config          # noqa: E402
from model import shadow                    # noqa: E402

ITEMS = os.path.join(REPO, "config", "items.example.json")
OUT = os.path.join(REPO, "tests", "fixtures", "entry_vectors.json")

# A store whose POS descriptions are longer than the sheet's 18-character ITEM column. No
# item in config/items.example.json is, so the shipped catalogue cannot exercise truncation
# at all -- and truncation is the guard that decides whether a returned sheet is read or
# lost over a column width, and whether two similar names get filed under each other.
LONG_NAMES = {
    "rotisserie-lp": {"name": "Rotisserie Chicken Lemon Pepper"},
    "rotisserie-bbq": {"name": "Rotisserie Chicken Barbecue"},
    "cake-sheet": {"name": "Sheet Cake Half Vanilla"},
    "bread": {"name": "Bread Loaf"},
}

# Each case is (what it is testing, the lines a person or a file would supply, and optionally
# the items catalogue it is read against). Keep the labels readable: they are what a failing
# test prints.
CASES = [
    ("a plain row", ["bread,20,14:30"]),
    ("no time written", ["bread,20,"]),
    ("zero is a negative not midnight", ["bread,20,0"]),
    ("written negatives", ["bread,20,no", "cake,3,none", "sushi,1,n/a"]),
    ("bare affirmative", ["bread,20,yes", "cake,3,circled"]),
    ("12 hour clock", ["bread,20,2:30pm", "cake,3,2pm", "sushi,1,11:15am"]),
    ("military without a colon", ["bread,20,1430"]),
    ("dots and spaces are ignored", ["bread,20,2.30 pm"]),
    ("a decimal quantity for a weighed item", ["hotbar-lb,12.5,"]),
    ("blank and comment lines are skipped", ["", "# yesterday", "bread,20,"]),
    ("a pasted header row is skipped", ["item,made,sold out at", "bread,20,"]),
    ("a note with its own comma", ["bread,20,14:30,ran out, no more flour"]),
    ("the printed name instead of the key", ["Bread Loaf,20,"]),
    ("a name the sheet did not truncate is still just a name",
     ["Rotisserie Chicken,8,"]),
    ("a half-typed name is a near miss, not a truncation", ["Rotisserie Chicke,8,"]),
    ("the name as the 18-character ITEM column printed it",
     ["Rotisserie Chicken,8,"], LONG_NAMES),
    ("two names that truncate alike are refused rather than misfiled",
     ["Rotisserie Chicken,8,"], LONG_NAMES),
    ("an unambiguous truncation is accepted", ["Sheet Cake Half Va,2,"], LONG_NAMES),
    ("the full long name is accepted too",
     ["Sheet Cake Half Vanilla,2,"], LONG_NAMES),
    ("case does not matter", ["BREAD,20,"]),
    ("an unknown item", ["kombucha,3,"]),
    ("a near miss gets a hint", ["Bread L,3,"]),
    ("the same item twice", ["bread,20,", "bread,21,"]),
    ("made is not a number", ["bread,banana,"]),
    ("made is negative", ["bread,-5,"]),
    ("an unreadable time", ["bread,20,half past two"]),
    ("an impossible minute", ["bread,20,14:75"]),
    ("an impossible 12 hour value", ["bread,20,14:30pm"]),
    ("an impossible 24 hour value", ["bread,20,25:00"]),
    ("one bad line poisons the whole sheet", ["bread,20,", "cake,banana,"]),
]

COMMENT = ("Generated from model.shadow.parse_entries -- the authority. Both "
           "tests/test_serve.py and web/src/lib/parseEntries.test.ts read this file, so the "
           "browser's parser and the server's cannot drift apart without a test failing. "
           "Regenerate with scripts/make_entry_vectors.py.")


def build():
    default = {k: {"name": v["name"]} for k, v in ht_config.load_items(ITEMS).items()}
    cases = []
    for case in CASES:
        label, lines = case[0], case[1]
        items = case[2] if len(case) > 2 else default
        rows, errors = shadow.parse_entries(lines, items)
        entry = {"label": label, "lines": lines, "rows": rows, "errors": errors}
        if len(case) > 2:
            entry["items"] = items
        cases.append(entry)
    return {"_comment": COMMENT, "items": default, "cases": cases}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=1, sort_keys=False)
        fh.write("\n")
    doc = build()
    print(f"{OUT}: {len(doc['cases'])} cases, "
          f"{sum(len(c['errors']) for c in doc['cases'])} errors, "
          f"{sum(len(c['rows']) for c in doc['cases'])} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
