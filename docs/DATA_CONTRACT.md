# What we need from the store

This is the whole data ask for the Fresh Forecast pilot. It is one export you almost
certainly already run, plus two small files. If you read only one section, read
**The short version**.

---

## The short version

| | |
|---|---|
| **What** | Item movement: how many units of each prepared-foods item sold, each day. |
| **Grain** | One row per **store x item x business day**. Not per transaction. Not per hour. |
| **Span** | **104 weeks** (2 years) of history, plus a daily file during the pilot. |
| **Format** | CSV. One file, or one file per year, or one per department — all fine. |
| **Contains** | Item numbers, dates, units, dollars. |
| **Does not contain** | Any customer, any employee, any card, any basket. See [What we do not want](#what-we-do-not-want). |
| **Costs the store** | One report run, once, plus a daily scheduled export during the pilot. |

Two years sounds like a lot for a 90-day pilot. It is not history for its own sake: the
model learns each item's weekday shape, its season, and its holidays, and there is only
one Thanksgiving per year to learn from. Most chains keep 2-3 years of item movement
online without an IT project. If you can only reach 12 months, say so — the tool runs,
it just cannot see an annual season. The hard floor is 126 days of history to train at
all, and 84 days of selling history for any individual item; below that the tool refuses
rather than printing a number it cannot stand behind.

**The pilot's 6 weeks of logging are not a training set.** They are the validation
window. Training needs the historical export. This is the single most common
misunderstanding about this project, so it is stated first.

---

## 1. The required file: item movement

The report is usually called *item movement*, *product movement*, *PLU movement*,
*scan movement*, or *sales by item by day*.

| We need | Typically called | What we do with it | If you cannot supply it |
|---|---|---|---|
| **Business date** | BUS DT, TRAN DATE, DAY | Every calendar feature: weekday, season, holiday, payday. | Fatal. There is no forecast without a date. |
| **Store number** | STORE, LOC, SITE | Labels the panel. One store is fine. | We label it `default`. No impact. |
| **Item number** | ITEM NBR, PLU, UPC, SKU | The join key for everything. | Fatal. A description alone is not stable enough to key on. |
| **Item description** | ITEM DESC | Human check that we mapped the number to the right thing, and a backstop when an item number changes. | Minor. Mapping gets harder to review. |
| **Department** | DEPT, CATEGORY | Grouping on reports only. Never a model input. | None. We take department from our own item sheet. |
| **Units sold** | UNITS, QTY, MOVEMENT | **The thing we forecast.** Pieces for packaged items, pounds for weighed items. | Fatal. |
| **Net sales dollars** | NET SALES $, EXT RETAIL | Settling waste in dollars, and detecting promotion weeks (a big gap between realized and shelf price). | Reports lose dollar figures. Forecasting is unaffected. |
| **Unit cost** | UNIT COST, COST, COGS | See [section 2](#2-the-three-things-that-change-the-outcome-most). | See section 2. |

Rules that matter:

- **Units, not dollars.** We never divide dollars by price to get units. Dollars are net
  of markdowns, so that arithmetic is wrong exactly on promotion days, which are the days
  we most need to be right about. If your export has dollars but no units, it cannot be used.
- **Zero-sales days may simply be missing rows.** That is normal and we handle it. Tell us
  if it is true so we do not read a missing row as a data gap.
- **Refunds and voids** may be present as negative rows. Leave them in; we net them.
- **One row per item per day.** If your export has multiple rows per item-day (one per
  register, one per department roll-up), leave them; tell us to sum them.

---

## 2. The three things that change the outcome most

Everything above gets a forecast. These three decide whether the pilot can *prove* it
saved money, and how good the numbers are. In order of value:

### a. A production count — how many were made or put out

**The single most valuable field a store can supply, ahead of sales.** A photographed
clipboard keyed into a spreadsheet counts. So does a scale label-printer log ("labels
printed per item per day"), which most stores already have and nobody looks at.

- With it: waste is **measured**, not modeled — for day-fresh items it is `made - sold`,
  on the assumption that every unit that left the case either scanned or was thrown out.
  Employee meals, samples, a catering tray rung to another department and damage all break
  that assumption in the same direction: they make `made - sold` an **upper bound** on what
  was really discarded, and they make the sold-out flag miss real sell-outs rather than
  invent them. Both errors are the safe direction, and both are why the pilot reports a
  lower bound on the saving rather than a point estimate.
- A label log undercounts on the days the kitchen ran a second batch nobody printed for.
  That is expected: those days are counted and reported, and only an implausible share of
  them (`production.max_overrun_share`, default 2%) blocks training.
- Without it: there is **no measured waste baseline at all**. The pilot can still show
  forecast accuracy, but it cannot say "this would have cut waste by at least $X" —
  the sentence the whole business case rests on. It also forces the sold-out signal to
  "none" (see below).

### b. Unit cost

- With it: the production recommendation is set by the real trade-off between the cost of
  one too many and the margin on one too few.
- Without it: we impute cost from a department gross-margin percentage you give us, mark
  those items `cost_imputed`, and **every report that quotes their dollars prints the
  assumption**. The forecast is unaffected; the recommended quantity is built on a guess,
  and the guess goes straight into the recommendation. This is often a more sensitive ask
  than sales — if the answer is no, say so and we will use margin percentages instead.

### c. A sold-out signal

When an item runs out at 2pm, that day's sales are not that day's demand — they are a
**floor** under it. A model trained without knowing which days those were learns to
forecast sell-outs rather than demand.

Anything that marks an item-day as sold out works: an out-of-stock scan log, a "sold out"
button on the scale, a zero-out event, or the paper morning sheets from shadow mode with
the sell-out time circled. If you supply a production count (a), we derive this for free
and need nothing more.

- Without any of it: a first-class, supported mode. We set the signal to `none` and say
  so on every artifact. Consequence, stated plainly: **recommended quantities run roughly
  1-8% low on the busiest days**, worst on the items that sell out most. We do **not**
  compensate by padding the numbers — that would trade a measured, disclosed bias for an
  unmeasured one.

---

## 3. Useful, cheap, optional

| | Why | Without it |
|---|---|---|
| **Closure and early-close dates** | A holiday closure and a demand collapse look identical in the data. | Nothing infers a closure. An undeclared closed day trains as a real zero-demand day and the model learns the collapse. The validator warns when every item sells zero on the same date, which is all it can honestly do. |
| **Store hours** | Confirms a closure rather than inferring it. | As above. |
| **Daily weather** (high temp + condition) | Hot bar loves cold days; rain suppresses traffic; the day *before* snow spikes bread. | We can source a local CSV ourselves. Nothing breaks; four inputs go quiet. |
| **The weekly ad / promo calendar** | An ad item can double for a week. | **The largest gap in the model.** We currently see the spike and learn noise from it. Ad weeks should be excluded from accuracy reporting once we can identify them. |
| **Shelf life per item** | Day-fresh math (`waste = made - sold`, one-day newsvendor) is only valid for one-day items. Real bread and cake are 2-5 days. | Multi-day items are labelled SHADOW ONLY, excluded from the waste figures, and still get a forecast. Better to know than to guess. |

---

## What we do not want

Please **do not** include, and we will delete on receipt if it arrives anyway:

- **Any customer data.** No loyalty numbers, no names, no addresses, no emails, no phone
  numbers, no household IDs, no basket or transaction detail, no card data of any kind.
- **Any employee data.** No cashier IDs, no operator numbers, no labor hours, no schedules.
- **Any financials beyond item price and item cost.** No P&L, no store totals, no margin
  reports, no shrink dollars by category, no invoices, no vendor terms.
- **Anything from a system you do not have the right to export.** If a field requires a
  permission you do not hold, leave it out and tell us; every one of them has a
  documented fallback above.

We also never ask for **true demand**, **lost sales**, or a **fill rate**. Those quantities
do not exist in any store's data — nobody records the customer who looked at an empty case
and left. Our proof-of-concept simulator has them because a simulator can; the real
pipeline is built so that no code path can even read them, and the numbers it reports are
either measured or conservative lower bounds.

Everything runs locally. The pipeline makes no network calls — not at import, not in tests,
not in continuous integration.

---

## 4. Format notes for whoever runs the export

You do not need to clean anything. The mess below is expected and handled by configuration,
not by editing your file. Just tell us which of it applies.

- **CSV.** We do not read `.xlsx` — save as CSV. (Reading Excel would mean adding a
  dependency this project does not have.)
- **Encoding and line endings:** cp1252/latin-1 and CRLF are fine. Tell us which.
- **Title blocks and totals rows:** a three-line report header above the column names and a
  `DEPT TOTAL` footer are fine. Tell us the row number the column names are on.
- **Dates:** any consistent format, including `MM/DD/YY`. Tell us which — we require an
  explicit format and refuse to guess, because `3/4/25` is ambiguous and a wrong guess is
  silent and catastrophic.
- **Numbers:** `$`, thousands separators and `(12.34)` for negatives are all handled.
- **Split files:** one per year or per department is fine.
- **Random-weight barcodes:** weighed items often scan a different 10-13 digit barcode per
  package. We extract the embedded PLU. Tell us the barcode pattern.
- **Item numbers that changed:** if an item was discontinued and re-added under a new
  number, tell us both numbers and the switch date, or the item's history splits in two and
  both halves get dropped for being too short.

### Five questions we need answered in writing

These cannot be inferred from the data, and each one silently corrupts the result if wrong.

1. **Where is the day-close cutoff?** Is the business date the store's own local day close,
   or a UTC day from a cloud BI tool? A UTC export shifts a US store 4-5 hours and pushes
   evening sales into tomorrow, which scrambles day-of-week — the model's strongest signal.
2. **Which items are sold by weight?** So `52` is never read as 52 pieces of hot bar.
3. **Are all prepared-foods departments in this report?** Sushi and some bakery programs are
   run by licensed third parties and their sales may not appear in your movement report.
4. **Are the dollars net of markdowns and coupons?** We assume yes and treat them as
   realized, not shelf, price.
5. **Do zero-sales item-days appear as rows with 0, or as no row at all?**

---

## 5. Worked example

Five rows out of a real-shaped export, with a report header, a totals footer, a refund, an
item number that changed mid-history, and two random-weight barcodes:

```text
FRESH FOODS MOVEMENT BY ITEM
STORE 0123           02/03/25 - 02/09/25
RUN 02/10/25 06:14
BUS DT,ITEM NBR,ITEM DESC,DEPT,UNITS,NET SALES $
02/07/25,330901,BREAD LOAF WHT 24OZ,BAKERY,41,"$163.59"
02/07/25,884213,ROTIS CHKN ORIG,DELI HOT,52,"$415.48"
02/07/25,884219,ROTIS CHKN LEM PEP,DELI HOT,(2),"($15.98)"
02/07/25,2004511898,HOT BAR SELF SERVE,DELI HOT,1.9,"$18.98"
02/07/25,2004511499,HOT BAR SELF SERVE,DELI HOT,1.5,"$14.99"
DEPT TOTAL,,,,94.4,"$597.06"
```

And the scale label-printer log for the same day:

```text
PRINT DT,ITEM NBR,LABELS PRINTED
02/07/25,330901,44
02/07/25,884213,56
02/07/25,00451,5.0
```

Those become exactly three rows of our canonical panel:

```text
date        store item        item_name           dept       dow  sold  produced  stockout  stockout_known  sellout_source     unit_price
2025-02-07  0123  bread       Bread Loaf          Bakery      4   41.0      44.0         0               1  produced_vs_sold         3.99
2025-02-07  0123  rotisserie  Rotisserie Chicken  Hot Foods   4   50.0      56.0         0               1  produced_vs_sold         7.99
2025-02-07  0123  hotbar-lb   Hot Bar (per lb)    Hot Foods   4    3.4       5.0         0               1  produced_vs_sold         9.99
```

What happened, line by line — all of it configuration, none of it hand-editing:

- The three title lines and the `DEPT TOTAL` footer **have no parsable date, so they are
  dropped and counted**. That is deliberate: it is how report furniture is removed without
  a rule that could also drop a real row.
- `884213` and `884219` are the same product before and after a pack change. Both map to
  `rotisserie`, so the history stays one series instead of two short ones. The `(2)` refund
  is a negative and nets: `52 - 2 = 50`.
- `2004511898` and `2004511499` are random-weight barcodes: `2` + PLU `00451` + the price
  in cents. Both collapse to `hotbar-lb` and the pounds sum: `1.9 + 1.5 = 3.4`. Grouping on
  the raw barcode instead would turn one hot-bar item into thousands of one-row items,
  which is not a data quirk but a model-shape failure.
- `dow=4` is Friday, recomputed from the date. We discard any day column in the export,
  because a POS day-of-week may not start on Monday.
- `dept` comes from our item sheet (`Hot Foods`), not from the export's `DELI HOT`.
  `item_name` likewise — the export's all-caps description is re-keyed by staff and drifts.
- `unit_price` is realized: `(415.48 - 15.98) / 50 = 7.99`. It settles dollars. It is **not**
  what sets the recommended quantity — that always uses the planning price from the item
  sheet, so a promotion cannot quietly move the recommendation.
- `produced` comes from the label log, so `stockout` is derivable: bread sold 41 of 44 made,
  so it did not sell out, and `stockout_known=1` records that we could actually tell.
  Hot bar is weighed, so it only counts as sold out within half a pound of what was put
  out; 3.4 against 5.0 is not close, so the flag is 0.
- Two-digit years and `$` are read per the format you confirm; nothing was cleaned by hand.

---

## 6. The one page we need from a person, not a system

Item cost, price, batch size and shelf life do not live in a movement report. We need one
short table from someone who knows the department — see `config/items.example.json` for the
exact shape:

| field | example | note |
|---|---|---|
| item key | `rotisserie` | our name for it; the export's item numbers map onto it |
| name | `Rotisserie Chicken` | as it should print on the morning sheet |
| department | `Hot Foods` | report grouping |
| price | `7.99` | normal shelf price, not the ad price |
| cost | `3.20` | or a department margin % if cost cannot be shared |
| batch | `4` | the production rounding unit: one bake, one tray, one spit |
| unit | `each` or `lb` | so the sheet never reads 52 pounds as 52 pieces |
| shelf life (days) | `1` | 1 = day-fresh. Bread and cake are usually 2-5 |

Nine items is a good pilot scope. Items with a shelf life over one day still get a forecast,
but they are marked SHADOW ONLY and left out of the waste math, because
`waste = made - sold` is simply false for them.

---

## 7. What we deliberately do not ask for, and the one future exception

We do not ask for basket or transaction-line data. It is the most sensitive export in the
building, it is 1-3 GB per year, and we do not need it.

There is one thing it would eventually buy, recorded here so nobody has to rediscover it:
**basket-line timestamps** would let us infer sell-outs from the silence after an item's
last sale of the day, without any production count. That is the best sell-out signal
available to a store that cannot count production. It is out of scope for this pilot — it
needs a second ingest path, a per-item hourly rate model, and register-outage detection —
and it is named here as a documented extension point, not a request.

---

## Checklist

- [ ] Item movement, item x day x store, 104 weeks, CSV, units **and** dollars
- [ ] The five questions above, answered in writing
- [ ] Production or label counts, in any form, even a photographed clipboard
- [ ] Unit cost, or department gross-margin percentages
- [ ] A sold-out signal, if one exists in any system
- [ ] Closure and early-close dates
- [ ] The item sheet from section 6
- [ ] Written permission covering the export, the retention period, and who owns what is built
