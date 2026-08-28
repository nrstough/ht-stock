# HT Stock — Prepared Food Waste Tracker

A phone-friendly tool for documenting prepared-food waste (bakery, pizza, hot foods, fresh foods)
at a grocery store, and turning that log into the dollar numbers that anchor a waste-reduction pitch.

This is **Phase 1** of a larger idea (predicting demand to cut over-production waste). Phase 1 is
deliberately the unglamorous part: **measure the baseline first.** Nobody funds "reduces waste 15%";
people fund "recovers $X/year at this store." This tool produces $X.

## What it does

- **Log** — tap an item, set a quantity, pick a reason, done. Built for one-handed use in the back
  room. Supports backdating and decimal quantities (hot bar by the pound).
- **History** — every entry, grouped by day, with day totals.
- **Stats** — documented waste in units and dollars, weekly run rate, annualized baseline
  (weekly × 52), 10/15/20% recovery scenarios, breakdowns by item, department, and day of week.
- **Pitch** — a copy-ready, print-ready one-page summary generated from your real data, including
  district-level extrapolation. It warns you if you have less than 14 days of coverage.
- **Setup** — manage items and retail prices, export CSV for spreadsheets, and back up / restore
  your data as JSON.

## How to use it

The whole app is a single file with no dependencies, no build step, and no server.

- **On a computer:** open `index.html` in any browser.
- **On your phone (recommended):** enable GitHub Pages for this repo (Settings → Pages → deploy
  from branch), open the URL on your phone, and add it to your home screen. It behaves like an app.
- Everything is stored in the browser's local storage **on that device only**. Nothing is uploaded
  anywhere. Back up regularly from the Setup tab — clearing the browser clears the data.

### The two-week baseline

1. In **Setup**, fix the item list and enter your store's real retail prices. The defaults are
   placeholders.
2. For at least **14 consecutive days**, log every discarded item you see across your pilot items.
3. Open **Pitch**, set your district's store count, and copy or print the summary.

## How the math works

- **Weekly run rate** = total documented waste ÷ calendar days covered × 7. Calendar span is used
  (not just days with entries), so days you logged nothing count as zero-waste days — conservative
  by design.
- **Annualized baseline** = weekly × 52. This ignores seasonality; the app says so wherever the
  number appears.
- **Recovery scenarios** = annualized × 10% / 15% / 20%, the typical first-year range for retail
  perishable waste-reduction pilots. Pitch the middle, not the top.

## Roadmap

- **Phase 1 (this):** hand-logged baseline. ✅
- **Phase 2:** with written permission and real sales/production data, a demand model that suggests
  daily production quantities per item (day-of-week + trend, later weather/holidays).
- **Phase 3:** shadow mode — the model makes suggestions, nobody follows them, and you measure how
  much waste following them *would have* avoided.
- **Phase 4:** live pilot with a measured before/after.

## A note on data and permission

This tool is a digital clipboard for what you personally observe being thrown away — the same
thing you could do on paper. It does not connect to, scrape, or store anything from company
systems. Before Phase 2 (which needs real sales data), get written permission for data access and
clarity on IP ownership first. That ordering protects both the project and you.
