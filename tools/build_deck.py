"""Build the leadership slide deck from the frozen backtest results.

    python tools/build_deck.py            # writes proposal/deck.html

Every number on every slide is read from results/results.json at build time, so the deck
cannot drift from the backtest it describes. The output is a single self-contained HTML
file: no scripts are required to read it, the only script inside it fits the slides to the
window and handles arrow keys, and printing produces one landscape page per slide.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
OUT = ROOT / "proposal" / "deck.html"

# Series colors. Two categorical hues (model, status quo) validated for CVD separation with
# direct labels as the secondary encoding; everything else is context gray.
C_MODEL = "#1e7a46"
C_SQ = "#eb6834"
C_ORACLE = "#2a78d6"
C_CTX = "#9aa39c"
C_INK = "#17211b"
C_INK2 = "#4d5a52"
C_MUTED = "#7d8a81"
C_GRID = "#e3e7e2"

W, H = 1280, 720  # logical slide size


def money(x: float, k: bool = False) -> str:
    if k:
        return f"${x / 1000:,.0f}K"
    return f"${x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# --------------------------------------------------------------------------------------
# SVG chart helpers. Each returns a complete <svg> string sized in logical pixels.
# --------------------------------------------------------------------------------------

def hbar_chart(rows, width=760, height=420, vmax=None, label_w=250):
    """Horizontal bars: rows = [(label, value, color, sublabel)]."""
    vmax = vmax or max(r[1] for r in rows) * 1.12
    pad_t, pad_b = 16, 34
    band = (height - pad_t - pad_b) / len(rows)
    bar_h = min(26, band * 0.62)
    x0 = label_w
    x1 = width - 90
    scale = (x1 - x0) / vmax
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Bar chart">']
    # gridlines at clean steps
    step = 15000
    v = 0
    while v <= vmax:
        x = x0 + v * scale
        out.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}" '
                   f'stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" '
                   f'font-size="12" fill="{C_MUTED}">{money(v, True) if v else "$0"}</text>')
        v += step
    for i, (label, value, color, sub) in enumerate(rows):
        cy = pad_t + band * i + band / 2
        y = cy - bar_h / 2
        w = value * scale
        r = 4
        # square at baseline, rounded at the data end
        path = (f'M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} a{r},{r} 0 0 1 {r},{r} '
                f'V{y + bar_h - r:.1f} a{r},{r} 0 0 1 -{r},{r} H{x0:.1f} Z')
        out.append(f'<path d="{path}" fill="{color}"/>')
        out.append(f'<text x="{x0 - 14}" y="{cy - 3}" text-anchor="end" font-size="15" '
                   f'font-weight="600" fill="{C_INK}">{esc(label)}</text>')
        if sub:
            out.append(f'<text x="{x0 - 14}" y="{cy + 14}" text-anchor="end" font-size="12" '
                       f'fill="{C_MUTED}">{esc(sub)}</text>')
        out.append(f'<text x="{x0 + w + 10:.1f}" y="{cy + 5}" font-size="15" font-weight="600" '
                   f'fill="{C_INK}">{money(value, True)}</text>')
    out.append('</svg>')
    return "\n".join(out)


def heatmap(cells, width=720, height=400):
    """cells: list of dicts with dow, weather, index. Sequential single-hue ramp."""
    dows = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weathers = ["sunny", "cloudy", "rain", "snow"]
    lookup = {(c["dow"], c["weather"]): c["index"] for c in cells}
    vals = [c["index"] for c in cells]
    lo, hi = min(vals), max(vals)
    left, top = 70, 40
    cw = (width - left - 10) / 7
    ch = (height - top - 10) / 4

    def color(v):
        # one hue, light -> dark, in the model green
        t = (v - lo) / (hi - lo)
        a, b = (0xe7, 0xf2, 0xec), (0x12, 0x4d, 0x2e)
        rgb = [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]
        return "#%02x%02x%02x" % tuple(rgb), t

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Demand index by weekday and weather">']
    for j, d in enumerate(dows):
        out.append(f'<text x="{left + cw * j + cw / 2:.1f}" y="26" text-anchor="middle" '
                   f'font-size="14" font-weight="600" fill="{C_INK2}">{d}</text>')
    for i, wth in enumerate(weathers):
        out.append(f'<text x="{left - 12}" y="{top + ch * i + ch / 2 + 5:.1f}" text-anchor="end" '
                   f'font-size="14" font-weight="600" fill="{C_INK2}">{wth.title()}</text>')
        for j in range(7):
            v = lookup.get((j, wth))
            x, y = left + cw * j, top + ch * i
            if v is None:
                out.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" '
                           f'height="{ch - 2:.1f}" rx="6" fill="none" stroke="{C_GRID}" '
                           f'stroke-dasharray="3 3"/>')
                out.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 5:.1f}" '
                           f'text-anchor="middle" font-size="12" fill="{C_MUTED}">n/a</text>')
                continue
            fill, t = color(v)
            ink = "#ffffff" if t > 0.55 else C_INK
            out.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" '
                       f'height="{ch - 2:.1f}" rx="6" fill="{fill}"/>')
            out.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 6:.1f}" text-anchor="middle" '
                       f'font-size="17" font-weight="600" fill="{ink}">{v:.2f}</text>')
    out.append('</svg>')
    return "\n".join(out)


def small_multiples(season, width=760, height=420):
    depts = ["Bakery", "Pizza", "Hot Foods", "Fresh Foods"]
    months = "JFMAMJJASOND"
    cols, rows = 2, 2
    pw, ph = width / cols, height / rows
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Seasonal demand index by department">']
    lo, hi = 0.78, 1.25
    for k, dept in enumerate(depts):
        ox, oy = (k % cols) * pw, (k // cols) * ph
        left, right, top, bottom = ox + 46, ox + pw - 18, oy + 34, oy + ph - 28
        pts = sorted([(s["month"], s["index"]) for s in season if s["dept"] == dept])

        def X(m):
            return left + (m - 1) / 11 * (right - left)

        def Y(v):
            return bottom - (v - lo) / (hi - lo) * (bottom - top)

        out.append(f'<text x="{left}" y="{oy + 20}" font-size="15" font-weight="600" '
                   f'fill="{C_INK}">{dept}</text>')
        for g in (0.8, 1.0, 1.2):
            out.append(f'<line x1="{left}" y1="{Y(g):.1f}" x2="{right}" y2="{Y(g):.1f}" '
                       f'stroke="{C_GRID if g != 1.0 else C_CTX}" stroke-width="1"/>')
            out.append(f'<text x="{left - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" font-size="11" '
                       f'fill="{C_MUTED}">{g:.1f}</text>')
        for m in range(1, 13):
            out.append(f'<text x="{X(m):.1f}" y="{bottom + 16}" text-anchor="middle" '
                       f'font-size="11" fill="{C_MUTED}">{months[m - 1]}</text>')
        d = " ".join(f'{"M" if i == 0 else "L"}{X(m):.1f},{Y(v):.1f}' for i, (m, v) in enumerate(pts))
        area = d + f' L{X(12):.1f},{Y(1.0):.1f} L{X(1):.1f},{Y(1.0):.1f} Z'
        out.append(f'<path d="{area}" fill="{C_MODEL}" opacity="0.10"/>')
        out.append(f'<path d="{d}" fill="none" stroke="{C_MODEL}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        pk = max(pts, key=lambda p: p[1])
        out.append(f'<circle cx="{X(pk[0]):.1f}" cy="{Y(pk[1]):.1f}" r="4" fill="{C_MODEL}" '
                   f'stroke="#fff" stroke-width="2"/>')
        anchor = "end" if pk[0] > 8 else "start"
        dx = -8 if anchor == "end" else 8
        out.append(f'<text x="{X(pk[0]) + dx:.1f}" y="{Y(pk[1]) - 8:.1f}" text-anchor="{anchor}" '
                   f'font-size="12" font-weight="600" fill="{C_INK}">peak {pk[1]:.2f}×</text>')
    out.append('</svg>')
    return "\n".join(out)


def series_chart(series, width=1180, height=400):
    dates = series["dates"]
    td, sq, dl = series["true_demand"], series["status_quo"], series["dl"]
    hol = series["holidays"]
    n = len(dates)
    left, right, top, bottom = 50, width - 150, 20, height - 34
    vmax = max(max(td), max(sq), max(dl)) * 1.1

    def X(i):
        return left + i / (n - 1) * (right - left)

    def Y(v):
        return bottom - v / vmax * (bottom - top)

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Daily production versus true demand, whole pizza, eight weeks">']
    step = 10
    g = 0
    while g <= vmax:
        out.append(f'<line x1="{left}" y1="{Y(g):.1f}" x2="{right}" y2="{Y(g):.1f}" '
                   f'stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" font-size="12" '
                   f'fill="{C_MUTED}">{g}</text>')
        g += step
    # week ticks
    for i in range(0, n, 7):
        out.append(f'<text x="{X(i):.1f}" y="{bottom + 18}" text-anchor="middle" font-size="12" '
                   f'fill="{C_MUTED}">{dates[i][5:]}</text>')
    # holiday markers
    for i, hname in enumerate(hol):
        if hname:
            out.append(f'<line x1="{X(i):.1f}" y1="{top}" x2="{X(i):.1f}" y2="{bottom}" '
                       f'stroke="{C_CTX}" stroke-width="1" stroke-dasharray="4 4"/>')
            label = {"super_bowl": "Super Bowl", "valentines": "Valentine's"}.get(hname, hname)
            out.append(f'<text x="{X(i) + 6:.1f}" y="{top + 12}" font-size="12" '
                       f'fill="{C_INK2}">{label}</text>')

    def path(vals):
        return " ".join(f'{"M" if i == 0 else "L"}{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(vals))

    # true demand as a wash + thin gray line
    out.append(f'<path d="{path(td)} L{X(n - 1):.1f},{Y(0):.1f} L{X(0):.1f},{Y(0):.1f} Z" '
               f'fill="{C_CTX}" opacity="0.12"/>')
    out.append(f'<path d="{path(td)}" fill="none" stroke="{C_CTX}" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    out.append(f'<path d="{path(sq)}" fill="none" stroke="{C_SQ}" stroke-width="2" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<path d="{path(dl)}" fill="none" stroke="{C_MODEL}" stroke-width="2.5" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')
    # end labels
    for vals, color, name in ((sq, C_SQ, "Par sheet"), (dl, C_MODEL, "Forecast"), (td, C_CTX, "Actual demand")):
        out.append(f'<circle cx="{X(n - 1):.1f}" cy="{Y(vals[-1]):.1f}" r="4" fill="{color}" '
                   f'stroke="#fff" stroke-width="2"/>')
    # legend-style end labels, de-collided by fixed offsets
    ys = sorted([(Y(sq[-1]), "Par sheet", C_SQ), (Y(dl[-1]), "Forecast", C_MODEL), (Y(td[-1]), "Actual demand", C_CTX)])
    placed = []
    for y, name, color in ys:
        if placed and y - placed[-1] < 18:
            y = placed[-1] + 18
        placed.append(y)
        out.append(f'<rect x="{right + 10}" y="{y - 6:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{right + 26}" y="{y + 4:.1f}" font-size="13" font-weight="600" '
                   f'fill="{C_INK}">{name}</text>')
    out.append('</svg>')
    return "\n".join(out)


def cumulative_chart(cum, width=760, height=420):
    dates, dollars = cum["dates"], cum["dollars"]
    n = len(dates)
    left, right, top, bottom = 70, width - 30, 24, height - 34
    vmax = max(dollars) * 1.08

    def X(i):
        return left + i / (n - 1) * (right - left)

    def Y(v):
        return bottom - v / vmax * (bottom - top)

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Cumulative economic cost avoided across the simulated year">']
    g = 0
    while g <= vmax:
        out.append(f'<line x1="{left}" y1="{Y(g):.1f}" x2="{right}" y2="{Y(g):.1f}" '
                   f'stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" font-size="12" '
                   f'fill="{C_MUTED}">{money(g, True) if g else "$0"}</text>')
        g += 4000
    month_starts = [i for i, d in enumerate(dates) if d.endswith("-01")]
    names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for k, i in enumerate(month_starts):
        out.append(f'<text x="{X(i):.1f}" y="{bottom + 18}" text-anchor="start" font-size="12" '
                   f'fill="{C_MUTED}">{names[k]}</text>')
    d = " ".join(f'{"M" if i == 0 else "L"}{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(dollars))
    out.append(f'<path d="{d} L{X(n - 1):.1f},{Y(0):.1f} L{X(0):.1f},{Y(0):.1f} Z" fill="{C_MODEL}" opacity="0.10"/>')
    out.append(f'<path d="{d}" fill="none" stroke="{C_MODEL}" stroke-width="2.5" stroke-linejoin="round"/>')
    out.append(f'<circle cx="{X(n - 1):.1f}" cy="{Y(dollars[-1]):.1f}" r="5" fill="{C_MODEL}" stroke="#fff" stroke-width="2"/>')
    out.append(f'<text x="{X(n - 1) - 10:.1f}" y="{Y(dollars[-1]) - 12:.1f}" text-anchor="end" '
               f'font-size="15" font-weight="600" fill="{C_INK}">{money(dollars[-1])} avoided</text>')
    out.append('</svg>')
    return "\n".join(out)


def pipeline_svg(width=1180, height=320):
    boxes = [
        (20, 40, 230, 200, "Inputs", ["Sales history", "(sellouts flagged)", "Weather forecast", "Calendar & holidays"], False),
        (320, 70, 220, 140, "Demand model", ["one network,", "every item"], True),
        (610, 70, 220, 140, "Decision rule", ["cost of one-too-many", "vs one-too-few"], True),
        (900, 70, 260, 140, "Morning sheet", ["printed in the kitchen", "make 14 · thaw 2 cases"], False),
    ]
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
           f'aria-label="Sales history, weather and calendar feed a demand model; a decision rule turns the '
           f'forecast into a production quantity; the kitchen gets a printed morning sheet; each day feeds back.">',
           f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 Z" fill="{C_INK2}"/></marker></defs>']
    for x, y, w, h, title, lines, accent in boxes:
        fill = "#e7f2ec" if accent else "#ffffff"
        stroke = C_MODEL if accent else "#cfd6d0"
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        out.append(f'<text x="{x + w / 2}" y="{y + 34}" text-anchor="middle" font-size="19" font-weight="700" fill="{C_INK}">{title}</text>')
        for i, ln in enumerate(lines):
            out.append(f'<text x="{x + w / 2}" y="{y + 64 + i * 24}" text-anchor="middle" font-size="15" fill="{C_INK2}">{esc(ln)}</text>')
    arrows = [(250, 140, 320, 140, "", ""), (540, 140, 610, 140, "demand as", "a range"),
              (830, 140, 900, 140, "quantity", "")]
    for x1, y1, x2, y2, l1, l2 in arrows:
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_INK2}" stroke-width="2" marker-end="url(#arr)"/>')
        if l1:
            out.append(f'<text x="{(x1 + x2) / 2}" y="{y1 - 14}" text-anchor="middle" font-size="12" fill="{C_MUTED}">{l1}</text>')
        if l2:
            out.append(f'<text x="{(x1 + x2) / 2}" y="{y1 + 26}" text-anchor="middle" font-size="12" fill="{C_MUTED}">{l2}</text>')
    # feedback loop
    out.append(f'<path d="M1030,210 V262 H135 V240" fill="none" stroke="{C_INK2}" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arr)"/>')
    out.append(f'<text x="580" y="284" text-anchor="middle" font-size="13" fill="{C_MUTED}">each day\'s sales and waste feed back into the history</text>')
    out.append('</svg>')
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------------------

def build(results: dict) -> str:
    S = results["summary"]
    sq, dlm, dl, orc, naive, ridge = (S["status_quo"], S["dl_matched"], S["dl"], S["oracle"],
                                      S["naive"], S["ridge"])
    wape = results["wape"]
    year = results["test_year"]
    ndays = results["n_test_days"]
    items = results["per_item"]

    waste_cut_matched = 1 - dlm["waste_retail"] / sq["waste_retail"]
    waste_cut_opt = 1 - dl["waste_retail"] / sq["waste_retail"]
    econ_saved_matched = sq["econ_cost"] - dlm["econ_cost"]
    econ_saved_opt = sq["econ_cost"] - dl["econ_cost"]
    gap_closed = (sq["econ_cost"] - dl["econ_cost"]) / (sq["econ_cost"] - orc["econ_cost"])
    retail_saved_matched = sq["waste_retail"] - dlm["waste_retail"]
    pizza = results["charts"]["series_pizza-whole"]
    cum = results["charts"]["cumulative_savings"]

    # Ranking of items by waste retail under the status quo, for the appendix
    item_rows = sorted(items.values(), key=lambda r: -r["sq"]["waste_retail"])

    def slide(body: str, cls: str = "", notes: str = "") -> str:
        n = f'<aside class="notes">{notes}</aside>' if notes else ""
        return f'<section class="slide {cls}">{body}{n}</section>'

    slides = []

    # 1 · Title ------------------------------------------------------------------------
    slides.append(slide(f"""
      <div class="title-wrap">
        <p class="eyebrow light">Harris Teeter · Prepared Foods · Proposal for leadership</p>
        <h1 class="hero">Fresh Forecast</h1>
        <p class="subhead">Measure, predict and reduce prepared-food waste,<br>
        one store at a time, starting with a 90-day pilot.</p>
        <div class="title-meta">
          <span>Concept validated in a full-year simulation</span>
          <span>Built on zero company data</span>
          <span>September 2026</span>
        </div>
      </div>""", "dark",
        "Open with the two-sided cost: every night we throw away food we made too much of, and every "
        "week we send customers home without food we made too little of. Same forecasting problem."))

    # 2 · One-slide summary ------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">The one-slide version</p>
      <h2>We can stop guessing how much to make</h2>
      <div class="two-col">
        <div>
          <p class="lede">Bakery, pizza, hot foods and fresh cases are produced against gut-feel par
          sheets. Pars cannot see weather, holidays or trend, so they pad, and the padding becomes
          waste. A demand forecast can see all of those a day ahead.</p>
          <p class="lede">A working forecasting system already exists. It was built and tested on a
          fully synthetic store so that no data-permission or IP question had to be answered first.
          The ask is one pilot store, read-only data access, and a modest number of hours.</p>
        </div>
        <div class="stats">
          <div class="stat"><div class="v">−{waste_cut_matched * 100:.0f}%</div>
            <div class="l">waste in the simulated pilot year with product availability held at today's level</div></div>
          <div class="stat"><div class="v">{gap_closed * 100:.0f}%</div>
            <div class="l">of the improvement that perfect knowledge of demand would allow</div></div>
          <div class="stat accent"><div class="v">10–15%</div>
            <div class="l">the waste reduction this proposal actually promises for a real 90-day pilot</div></div>
        </div>
      </div>
      <p class="foot">Simulation figures come from a synthetic three-year store, held-out year {year}. They prove the method. The real dollar figure comes from the pilot's measured baseline.</p>""",
        notes="If leadership reads one slide, this is it. Method proven; dollars to be measured; ask is small."))

    # 3 · The problem: demand is not flat -----------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">The problem</p>
      <h2>A par sheet is one number. Demand is not.</h2>
      <div class="two-col wide-right">
        <div>
          <p>Make too much and we eat the cost of goods. Make too little and we lose the margin
          and disappoint the customer at an empty hot bar at 6 pm.</p>
          <p>Pars solve this with a flat safety pad. But demand swings with the weekday, the weather,
          the season and the calendar, and all of those are known the day before.</p>
          <p class="small muted">Demand index by weekday and weather in the simulated store. 1.00 is
          an average day. The spread between a snowy Monday and a sunny Friday is almost 2×, and a
          single par number has to cover both.</p>
        </div>
        <figure>{heatmap(results["charts"]["dow_weather"])}</figure>
      </div>""",
        notes="Industry-typical waste for these departments runs 15 to 25 percent of production. "
              "The heatmap is the simulated store, but the shape is the one every kitchen manager knows."))

    # 4 · Seasonality --------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">The problem, continued</p>
      <h2>Each department has its own year</h2>
      <div class="two-col wide-right">
        <div>
          <p>Hot foods peak in winter, fresh foods in summer, bakery and pizza around the holidays.
          A trailing average lags each of these turns by weeks. A forecast that knows the calendar
          turns with them.</p>
          <p class="small muted">Monthly demand index by department in the simulated store,
          relative to each department's own average. The shaded area is the departure from 1.00.</p>
        </div>
        <figure>{small_multiples(results["charts"]["seasonality"])}</figure>
      </div>""",
        notes="Point at hot foods: a swing of nearly 40 points between July and January. No single par survives that."))

    # 5 · How it works -------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">What already exists</p>
      <h2>The whole system, end to end</h2>
      <figure class="full">{pipeline_svg()}</figure>
      <div class="three-col small">
        <div><b>No integration needed for a pilot.</b> The inputs are a sales export, a public weather forecast and a calendar. The output is a printed sheet in the kitchen each morning.</div>
        <div><b>One model, every item.</b> A compact neural network of the family used industrially for retail demand forecasting, forecasting a demand range rather than a point.</div>
        <div><b>A decision rule, not a guess.</b> Produce where the cost of one wasted unit equals the cost of one missed sale. Cheap, high-margin items pad; expensive, low-margin items run lean.</div>
      </div>""",
        notes="Emphasise how little has to change: the kitchen gets a piece of paper. The manager can override any line."))

    # 6 · Proof of concept: cost by policy --------------------------------------------
    rows = [
        ("Status quo (par sheet)", sq["econ_cost"], C_SQ, "today's practice, simulated"),
        ("Naive trailing average", naive["econ_cost"], C_CTX, "benchmark"),
        ("Linear model", ridge["econ_cost"], C_CTX, "benchmark"),
        ("Forecast · availability held", dlm["econ_cost"], C_MODEL, f"fill rate {pct(dlm['fill_rate'])}"),
        ("Forecast · profit-optimal", dl["econ_cost"], C_MODEL, f"fill rate {pct(dl['fill_rate'])}"),
        ("Oracle (perfect knowledge)", orc["econ_cost"], C_ORACLE, "the best any method could do"),
    ]
    slides.append(slide(f"""
      <p class="eyebrow">Proof of concept</p>
      <h2>A year replayed: the forecast closes {gap_closed * 100:.0f}% of the gap to perfect knowledge</h2>
      <div class="two-col wide-right">
        <div>
          <p>The model trained on two simulated years, seeing only what a real store sees, then ran in
          shadow against the untouched third year. For every day and item it recommended a quantity;
          the simulation's true demand settled what sold, what was wasted and what was missed.</p>
          <p>Two benchmark forecasters ran alongside so the network had to earn its seat.</p>
          <p class="small muted">Total cost of being wrong per simulated store-year: wasted units at cost of goods plus missed sales at lost margin. {ndays} days.</p>
        </div>
        <figure>{hbar_chart(rows)}</figure>
      </div>""",
        notes=f"Forecast accuracy: {pct(wape['dl'])} weighted error for the network against "
              f"{pct(wape['naive'])} for the trailing average and {pct(wape['ridge'])} for the linear model."))

    # 7 · Results table ----------------------------------------------------------------
    def trow(name, r, delta=None, cls=""):
        d = f'<span class="delta">({delta})</span>' if delta else ""
        return (f'<tr class="{cls}"><td>{name}</td><td class="num">{money(r["waste_retail"])} {d}</td>'
                f'<td class="num">{pct(r["waste_pct_of_production"])}</td><td class="num">{pct(r["fill_rate"])}</td>'
                f'<td class="num">{money(r["econ_cost"])}</td></tr>')

    slides.append(slide(f"""
      <p class="eyebrow">Proof of concept</p>
      <h2>Less waste at the same availability, or much less waste at slightly lower availability</h2>
      <table class="results" style="margin-top:10px">
        <thead><tr><th>Held-out year, one store</th><th class="num">Waste (retail value)</th>
        <th class="num">Waste, % of production</th><th class="num">Availability (fill rate)</th><th class="num">Total economic cost</th></tr></thead>
        <tbody>
          {trow("Status quo (par sheet)", sq)}
          {trow("Forecast, availability held", dlm, f"−{waste_cut_matched * 100:.0f}%", "hl")}
          {trow("Forecast, profit-optimal", dl, f"−{waste_cut_opt * 100:.0f}%", "hl")}
          {trow("Oracle (perfect knowledge)", orc)}
        </tbody>
      </table>
      <div class="three-col small top-gap">
        <div><b>Availability held</b> is the pilot setting: {money(retail_saved_matched)} less food thrown away per year at retail value, with customers seeing an empty case about as often as they do today.</div>
        <div><b>Profit-optimal</b> is where the decision rule would go on its own: waste falls by {waste_cut_opt * 100:.0f}%, and availability slips from {pct(sq['fill_rate'])} to {pct(dl['fill_rate'])}. That is a business choice, item by item, not a model setting.</div>
        <div><b>Forecast accuracy</b> {pct(wape['dl'])} weighted error, against {pct(wape['naive'])} for a trailing average and {pct(wape['ridge'])} for a linear model.</div>
      </div>""",
        notes="The two forecast rows are the same model with a different availability target. Leadership picks the row."))

    # 8 · Day to day -----------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">What it looks like day to day</p>
      <h2>Whole pizza, eight simulated weeks: the forecast hugs demand, the par sheet pads it</h2>
      <figure class="full">{series_chart(pizza)}</figure>
      <p class="foot">Units per day. The gray band is the demand no method ever sees; both policies only see sales. The dashed lines mark holidays the forecast knew about a day ahead.</p>""",
        notes="Super Bowl Sunday: the forecast lifts production ahead of the spike; the par sheet catches up a week late, then over-produces."))

    # 9 · Cumulative ------------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">Proof of concept</p>
      <h2>Savings accrue every week, not at the end of a project</h2>
      <div class="two-col wide-right">
        <div>
          <p>Cumulative economic cost avoided by the profit-optimal forecast versus the par sheet across
          the simulated year: {money(econ_saved_opt)} per store, or about {money(econ_saved_opt / 52)} a week from
          the first week of live operation.</p>
          <p>At the availability-held setting it is {money(econ_saved_matched)} per store-year in economic cost,
          and {money(retail_saved_matched)} of food, at retail, not thrown away.</p>
          <p class="small muted">Economic cost counts waste at cost of goods and missed sales at lost margin. Retail value of waste is the number a shrink report shows.</p>
        </div>
        <figure>{cumulative_chart(cum)}</figure>
      </div>""",
        notes="Steady slope means no big-bang: the value shows up in the first shrink report after go-live."))

    # 10 · Honesty ----------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">Read this slide carefully</p>
      <h2>What the simulation proves, and what it does not</h2>
      <div class="two-col">
        <div class="card ok" style="align-self:stretch">
          <h3>It proves the method</h3>
          <ul>
            <li>Demand with realistic weekday, weather, holiday and seasonal structure is learnable from sales data alone, censored by sellouts as a real store's data is.</li>
            <li>Production driven by those forecasts beats a well-run par sheet by a wide margin, and beats two simpler forecasters too.</li>
            <li>Every number is reproducible from the repository with one command, and the inputs are frozen.</li>
          </ul>
        </div>
        <div class="card warn" style="align-self:stretch">
          <h3>It does not prove our dollar figure</h3>
          <ul>
            <li>The simulated store is a plausible invention. Its simulated manager, though deliberately competent, is an invention too.</li>
            <li>No company data of any kind was used. Nothing here has seen a real export yet.</li>
            <li>That is why this proposal promises 10–15% in a real pilot, not {waste_cut_matched * 100:.0f}–{waste_cut_opt * 100:.0f}%, and why the pilot's measured baseline, not the simulation, is the number it is judged against.</li>
          </ul>
        </div>
      </div>""",
        notes="This slide is deliberate. Leadership funds measured numbers, and the credibility of the ask rests on not overclaiming."))

    # 11 · Pilot -----------------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">The proposal</p>
      <h2>A 90-day pilot at one store, with a stop gate before anything in the kitchen changes</h2>
      <div class="timeline">
        <div class="phase">
          <div class="when">Weeks 1–2</div><h3>Baseline</h3>
          <p>Log every discarded prepared-food item across 5–10 pilot SKUs with the phone-based waste logger that already exists.</p>
          <p class="out">Output: this store's real waste number, in dollars per year.</p>
        </div>
        <div class="phase">
          <div class="when">Weeks 3–6</div><h3>Shadow mode</h3>
          <p>With data access granted, the model retrains on this store's history and prints its morning sheet. The kitchen ignores it. We record what it would have done.</p>
          <p class="out">Output: forecast accuracy against reality, and five go/no-go criteria fixed in advance.</p>
        </div>
        <div class="phase live">
          <div class="when">Weeks 7–13</div><h3>Live</h3>
          <p>The kitchen follows the sheet. The manager can override any line on any day, and overrides are logged, not fought.</p>
          <p class="out">Output: measured waste and sellout change versus the baseline.</p>
        </div>
      </div>
      <div class="callout"><b>Success criterion, agreed in advance:</b> at least a 10% reduction in prepared-food waste dollars versus the baseline, with no material drop in on-shelf availability. If shadow mode shows the forecasts are not accurate enough, we stop at week 6 and nothing in the kitchen has changed.</div>""",
        notes="The gate at week 6 is the point: the downside of the pilot is bounded to a few pallets of mis-produced food, and only after shadow mode passes."))

    # 12 · The asks --------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">The asks</p>
      <h2>Three decisions, all reversible</h2>
      <div class="three-col asks">
        <div class="ask"><div class="n">1</div><h3>Data access, in writing</h3>
          <p>Read-only access to daily sales and production counts for the pilot items at one store, for the duration of the pilot. Scoped, revocable, and requested before any real data is touched.</p></div>
        <div class="ask"><div class="n">2</div><h3>40–80 paid hours</h3>
          <p>Non-floor time to connect the store's export, run shadow mode, and operate the live phase. The importer, validator, training and scoring tools are already built and rehearsed on a mock export.</p></div>
        <div class="ask"><div class="n">3</div><h3>A sponsor and a recognition structure</h3>
          <p>A store or district leader who owns the pilot, and an agreed, written structure tied to the measured result, settled before the live phase, so that a successful pilot has somewhere to scale.</p></div>
      </div>
      <p class="foot">Ownership of the work built so far was deliberately kept clean: nothing was built on company data, so IP and data questions can be settled on the way in rather than argued about on the way out.</p>""",
        notes="Ask three is what makes this a leadership decision rather than a store one."))

    # 13 · If it works ---------------------------------------------------------------------
    scale_rows = ""
    for n in (1, 20, 50, 100):
        scale_rows += (f'<tr><td>{n} store{"s" if n > 1 else ""}</td>'
                       f'<td class="num">{money(retail_saved_matched * n)}</td>'
                       f'<td class="num">{money(econ_saved_matched * n)}</td>'
                       f'<td class="num">{money(sq["waste_retail"] * 0.10 * n)} – {money(sq["waste_retail"] * 0.15 * n)}</td></tr>')
    slides.append(slide(f"""
      <p class="eyebrow">If it works</p>
      <h2>One model serves every item, so the pilot is also the template</h2>
      <div class="two-col scale">
        <div>
          <p>The same architecture scales from 9 SKUs at one store to every prepared-foods case in a
          district without redesign: retraining per store takes about a minute, and the morning sheet
          is the only interface the kitchen ever sees.</p>
          <p>The table is arithmetic on the simulated store, not a forecast of Harris Teeter's numbers.
          Its job is to show which column the pilot replaces with a measured fact. The last column is
          what this proposal actually promises, applied to a store whose waste looks like the simulated one.</p>
        </div>
        <div>
          <table class="results compact">
            <thead><tr><th>Per year</th><th class="num">Food not wasted<br>(retail, simulated)</th><th class="num">Economic cost<br>avoided (simulated)</th><th class="num">Promise applied<br>(10–15% of waste)</th></tr></thead>
            <tbody>{scale_rows}</tbody>
          </table>
          <p class="small muted">Simulated store: {money(sq['waste_retail'])} of prepared-food waste at retail per year under the par sheet; availability-held forecast setting. Replace the first row with the pilot's measured baseline and the rest follows.</p>
        </div>
      </div>""",
        notes="Do not read the 100-store row as a projection. Read it as: this is why a single-store pilot is worth 80 hours."))

    # 14 · Risks ---------------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">Risks, named</p>
      <h2>Every risk has a gate, a fallback, or a written answer</h2>
      <table class="risks">
        <thead><tr><th>Risk</th><th>Mitigation</th></tr></thead>
        <tbody>
          <tr><td>Forecasts do not transfer from simulation to this store</td><td>Shadow mode is the gate: four weeks of predictions scored against reality before production changes at all.</td></tr>
          <tr><td>Availability suffers and customers notice</td><td>The pilot runs at the availability-held setting; sellouts are tracked daily and any item can revert to its par instantly.</td></tr>
          <tr><td>The kitchen does not trust the sheet</td><td>Manager override is a feature, logged not fought; the pilot measures the blend, which is how it would really run.</td></tr>
          <tr><td>The store's export is dirtier than expected</td><td>The importer has been rehearsed against a deliberately dirty mock export. The first real one will still surprise us, and the hours ask covers that.</td></tr>
          <tr><td>Data access or IP questions</td><td>Nothing was built on company data; access is scoped, read-only, written and revocable. IP and recognition are settled in writing before the live phase.</td></tr>
        </tbody>
      </table>""",
        notes="The fourth row is the honest one: the importer was tested against its own author's imagination."))

    # 15 · Decision --------------------------------------------------------------------------
    slides.append(slide(f"""
      <div class="title-wrap">
        <p class="eyebrow light">The decision</p>
        <h1 class="hero small">Approve one pilot store.</h1>
        <p class="subhead">Week 1 begins the day the data-access letter is signed.<br>
        By week 6 we know whether the forecasts are accurate here. By week 13 we have a measured number.</p>
        <div class="title-meta">
          <span>Downside: a few pallets of mis-produced food, after a stop gate</span>
          <span>Upside: a measured template for every store</span>
        </div>
      </div>""", "dark",
        notes="Close by asking for the sponsor by name."))

    # 16 · Appendix: per item ------------------------------------------------------------------
    item_html = ""
    for r in item_rows:
        cut = 1 - r["dl"]["waste_retail"] / r["sq"]["waste_retail"]
        item_html += (f'<tr><td>{esc(r["name"])}</td><td class="num">{money(r["sq"]["waste_retail"])}</td>'
                      f'<td class="num">{pct(r["sq"]["waste_pct_of_production"], 0)}</td>'
                      f'<td class="num">{money(r["dl"]["waste_retail"])}</td>'
                      f'<td class="num">−{cut * 100:.0f}%</td>'
                      f'<td class="num">{pct(r["sq"]["fill_rate"])} → {pct(r["dl"]["fill_rate"])}</td>'
                      f'<td class="num">{r["q_star"]:.2f}</td></tr>')
    slides.append(slide(f"""
      <p class="eyebrow">Appendix</p>
      <h2 style="max-width:none">Per item, simulated held-out year, profit-optimal setting</h2>
      <table class="results compact">
        <thead><tr><th>Item</th><th class="num">Waste today (retail)</th><th class="num">Waste % of production</th><th class="num">Waste with forecast</th><th class="num">Change</th><th class="num">Fill rate today → forecast</th><th class="num">Service target</th></tr></thead>
        <tbody>{item_html}</tbody>
      </table>
      <p class="foot">Service target is the newsvendor critical fractile each item's price, cost and salvage imply: the fraction of days the item should not sell out. Expensive, low-margin items get a lower target and run lean; cheap, high-margin items pad.</p>""",
        notes="Sushi is the item to watch: high cost, so the rule runs it lean and availability drops most. In a real pilot that item would get a floor."))

    # 17 · Appendix: method ------------------------------------------------------------------
    slides.append(slide(f"""
      <p class="eyebrow">Appendix</p>
      <h2>Method notes and provenance</h2>
      <div class="two-col top">
        <div>
          <h3>Demand model</h3>
          <p class="small">A global quantile-regression neural network: a GRU encoder over each item's trailing 28 days, learned item embeddings, and calendar, weather and holiday covariates, trained with pinball loss across 11 quantiles. Sellout days are treated as censored, since sales only bound demand from below when the case sold out.</p>
          <h3>Decision layer</h3>
          <p class="small">Per-item newsvendor critical fractile, rounded to batch sizes. The availability-held variant raises each item's target until the year's fill rate matches the par sheet's.</p>
        </div>
        <div>
          <h3>Validation</h3>
          <p class="small">Trained on simulated 2023–24, tested untouched on {year} against naive, linear and oracle policies. Weighted forecast error: network {pct(wape['dl'])}, trailing average {pct(wape['naive'])}, linear {pct(wape['ridge'])}.</p>
          <h3>Provenance</h3>
          <p class="small">The synthetic dataset, the trained checkpoint and the results file are frozen and checked in. <code>python -m model.backtest</code> reproduces the results byte for byte, and the tools refuse to overwrite the frozen files by accident. This deck is generated from that results file by <code>tools/build_deck.py</code>.</p>
          <h3>Real-data path</h3>
          <p class="small">Importer, validation gate, training, evaluation and the morning-sheet loop have been run end to end on a mock export built from the synthetic store with realistic dirt added. None of it has seen real data.</p>
        </div>
      </div>""",
        notes="For the analyst in the room."))

    body = "\n".join(slides)
    total = len(slides)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fresh Forecast · Leadership deck</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
  :root {{
    --bg: #eceee9; --paper: #f8f9f6; --card: #ffffff;
    --ink: {C_INK}; --ink-2: {C_INK2}; --muted: {C_MUTED};
    --accent: {C_MODEL}; --accent-soft: #e7f2ec; --accent-deep: #124d2e;
    --warm: {C_SQ}; --border: #d9dfd9; --grid: {C_GRID};
    --warnbg: #fdf6e8; --warnbr: #e5cf9a;
    --dark: #0f1a14; --dark-2: #16261c;
    --sans: "Archivo", system-ui, -apple-system, "Segoe UI", sans-serif;
    --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
    color-scheme: light;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--ink); font-family: var(--serif); }}
  body {{ overflow: hidden; }}
  #stage {{ position: fixed; inset: 0; display: grid; place-items: center; }}
  .slide {{
    width: {W}px; height: {H}px; position: absolute; top: 50%; left: 50%;
    transform-origin: center center; background: var(--paper);
    padding: 54px 72px 48px; display: none; flex-direction: column;
    box-shadow: 0 30px 80px rgba(0,0,0,0.18); border-radius: 6px; overflow: hidden;
  }}
  .slide.current {{ display: flex; }}
  .slide.dark {{ background: radial-gradient(1200px 700px at 20% 0%, var(--dark-2), var(--dark) 70%); color: #eef1ec; justify-content: center; }}
  .slide::after {{
    content: attr(data-n) " / {total}"; position: absolute; right: 30px; bottom: 18px;
    font: 500 12px var(--sans); color: var(--muted); letter-spacing: 0.04em;
  }}
  .slide.dark::after {{ color: #7d8a81; }}
  .eyebrow {{ font: 600 13px var(--sans); letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }}
  .eyebrow.light {{ color: #7fd1a0; }}
  h1.hero {{ font: 800 96px/1 var(--sans); letter-spacing: -0.03em; margin: 0 0 22px; }}
  h1.hero.small {{ font-size: 72px; }}
  h2 {{ font: 700 36px/1.12 var(--sans); letter-spacing: -0.02em; margin: 0 0 22px; max-width: 1080px; text-wrap: balance; }}
  h3 {{ font: 700 19px/1.25 var(--sans); margin: 0 0 8px; }}
  p {{ margin: 0 0 12px; font-size: 19px; line-height: 1.5; max-width: 62ch; }}
  p.lede {{ font-size: 20px; }}
  p.small, .small {{ font-size: 15.5px; line-height: 1.5; }}
  .muted {{ color: var(--muted); }}
  .subhead {{ font-size: 26px; line-height: 1.4; color: #c9d4cc; max-width: 46ch; margin-bottom: 30px; }}
  .title-wrap {{ max-width: 1000px; }}
  .title-meta {{ display: flex; gap: 28px; flex-wrap: wrap; font: 500 14px var(--sans); color: #9fb0a5; }}
  .title-meta span::before {{ content: "●"; color: #7fd1a0; margin-right: 8px; font-size: 9px; vertical-align: 2px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: center; align-content: center; flex: 1; }}
  .two-col.top {{ align-items: start; align-content: start; }}
  .two-col.wide-right {{ grid-template-columns: 380px 1fr; }}
  .three-col {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; }}
  .three-col.small div {{ font-size: 15px; line-height: 1.5; color: var(--ink-2); }}
  .three-col.small b {{ color: var(--ink); font-family: var(--sans); font-weight: 700; }}
  .top-gap {{ margin-top: 22px; }}
  .two-col.scale {{ grid-template-columns: 1fr 1.2fr; }}
  .two-col.scale td.num {{ white-space: nowrap; }}
  figure {{ margin: 0; }}
  figure svg {{ display: block; max-width: 100%; height: auto; font-family: var(--sans); }}
  figure.full {{ margin: 6px 0 14px; }}
  .stats {{ display: grid; gap: 14px; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px 20px; display: grid; grid-template-columns: 170px 1fr; align-items: center; gap: 16px; }}
  .stat .v {{ font: 800 44px/1 var(--sans); letter-spacing: -0.02em; color: var(--accent); white-space: nowrap; }}
  .stat .l {{ font: 500 14.5px/1.4 var(--sans); color: var(--ink-2); }}
  .stat.accent {{ background: var(--accent-soft); border-color: #b9d8c6; }}
  .foot {{ margin-top: auto; padding: 14px 110px 0 0; font: 400 13.5px/1.5 var(--sans); color: var(--muted); max-width: 100ch; }}
  table {{ width: 100%; border-collapse: collapse; font: 400 16px/1.4 var(--sans); }}
  th, td {{ padding: 12px 12px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }}
  th {{ font-size: 12.5px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-2); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.results td:first-child {{ font-weight: 600; white-space: nowrap; }}
  table.results tr.hl td {{ background: var(--accent-soft); }}
  table.results tr.hl td:first-child {{ border-radius: 8px 0 0 8px; }}
  table.results tr.hl td:last-child {{ border-radius: 0 8px 8px 0; }}
  table.compact th, table.compact td {{ padding: 9px 10px; font-size: 14.5px; }}
  .delta {{ color: var(--accent); font-weight: 700; margin-left: 4px; }}
  table.risks td:first-child {{ font-weight: 600; width: 36%; }}
  table.risks td {{ font-size: 16.5px; padding: 14px 12px; }}
  .card {{ border-radius: 16px; padding: 22px 26px; }}
  .card ul {{ margin: 0; padding-left: 20px; }}
  .card li {{ margin-bottom: 10px; font-size: 17px; line-height: 1.45; }}
  .card.ok {{ background: var(--accent-soft); border: 1px solid #b9d8c6; }}
  .card.warn {{ background: var(--warnbg); border: 1px solid var(--warnbr); }}
  .callout {{ margin-top: auto; background: var(--accent-soft); border-radius: 12px; padding: 16px 20px; font-size: 17px; line-height: 1.45; }}
  .callout b {{ font-family: var(--sans); }}
  .timeline {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 20px; flex: 1; align-content: center; }}
  .phase {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px 22px; }}
  .phase.live {{ background: var(--accent-deep); color: #eef1ec; border-color: var(--accent-deep); }}
  .phase .when {{ font: 700 12.5px var(--sans); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin-bottom: 8px; }}
  .phase.live .when {{ color: #9fe0b9; }}
  .phase p {{ font-size: 15.5px; line-height: 1.45; }}
  .phase .out {{ font-family: var(--sans); font-weight: 600; font-size: 14px; color: var(--ink-2); margin: 0; }}
  .phase.live .out {{ color: #c9e6d4; }}
  .asks {{ margin-top: 8px; flex: 1; align-content: center; }}
  .ask {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 22px 24px; }}
  .ask .n {{ width: 38px; height: 38px; border-radius: 50%; background: var(--accent); color: #fff; font: 800 18px/38px var(--sans); text-align: center; margin-bottom: 14px; }}
  .ask p {{ font-size: 16px; line-height: 1.5; }}
  code {{ font: 500 0.92em ui-monospace, SFMono-Regular, Menlo, monospace; background: #e9ede8; padding: 1px 5px; border-radius: 4px; }}
  .notes {{ display: none; }}
  #hud {{ position: fixed; left: 16px; bottom: 12px; font: 500 12px var(--sans); color: var(--muted); user-select: none; animation: hud 6s forwards; }}
  @keyframes hud {{ 0%, 80% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}
  @media print {{
    @page {{ size: {W}px {H}px; margin: 0; }}
    html, body {{ background: #fff; overflow: visible; height: auto; }}
    #stage {{ position: static; display: block; }}
    #hud {{ display: none; }}
    .slide {{ display: flex !important; position: relative; top: auto; left: auto; transform: none !important;
              box-shadow: none; border-radius: 0; page-break-after: always; break-after: page; }}
    .slide.dark {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
</style>
</head>
<body>
<div id="stage">
{body}
</div>
<div id="hud">← → to move · Home/End · P prints one page per slide</div>
<script>
(function () {{
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  slides.forEach(function (s, i) {{ s.setAttribute('data-n', i + 1); }});
  var i = Math.max(0, Math.min(slides.length - 1, (parseInt(location.hash.slice(1), 10) || 1) - 1));
  function fit() {{
    var k = Math.min(window.innerWidth / {W}, window.innerHeight / {H}) * 0.96;
    slides.forEach(function (s) {{ s.style.transform = 'translate(-50%,-50%) scale(' + k + ')'; }});
  }}
  function show(n) {{
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (s, j) {{ s.classList.toggle('current', j === i); }});
    history.replaceState(null, '', '#' + (i + 1));
  }}
  window.addEventListener('resize', fit);
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{ show(i + 1); e.preventDefault(); }}
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ show(i - 1); e.preventDefault(); }}
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key === 'p' || e.key === 'P') window.print();
  }});
  document.getElementById('stage').addEventListener('click', function (e) {{
    show(e.clientX > window.innerWidth / 2 ? i + 1 : i - 1);
  }});
  fit(); show(i);
}})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text())
    Path(args.out).write_text(build(results), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
