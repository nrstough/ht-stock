#!/usr/bin/env node
/* Build the leadership deck as a PowerPoint file from the frozen backtest results.

     npm install pptxgenjs          # once, anywhere on NODE_PATH
     node tools/build_deck_pptx.js  # writes proposal/Fresh-Forecast-Leadership-Deck.pptx

   Same slides and the same numbers as tools/build_deck.py: every figure is read from
   results/results.json at build time. Charts are native PowerPoint charts, so they stay
   editable, and every slide carries speaker notes. */
'use strict';
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const RESULTS = path.join(ROOT, 'results', 'results.json');
const OUT = process.argv[2] || path.join(ROOT, 'proposal', 'Fresh-Forecast-Leadership-Deck.pptx');

const R = JSON.parse(fs.readFileSync(RESULTS, 'utf8'));
const S = R.summary;
const sq = S.status_quo, dlm = S.dl_matched, dl = S.dl, orc = S.oracle, naive = S.naive, ridge = S.ridge;
const wape = R.wape;

const wasteCutMatched = 1 - dlm.waste_retail / sq.waste_retail;
const wasteCutOpt = 1 - dl.waste_retail / sq.waste_retail;
const econSavedMatched = sq.econ_cost - dlm.econ_cost;
const econSavedOpt = sq.econ_cost - dl.econ_cost;
const gapClosed = (sq.econ_cost - dl.econ_cost) / (sq.econ_cost - orc.econ_cost);
const retailSavedMatched = sq.waste_retail - dlm.waste_retail;

// Palette (hex without '#'). Green dominates; orange is reserved for the status quo.
const C = {
  ink: '17211B', ink2: '4D5A52', muted: '7D8A81', paper: 'F8F9F6', card: 'FFFFFF',
  green: '1E7A46', greenSoft: 'E7F2EC', greenDeep: '124D2E', orange: 'EB6834', blue: '2A78D6',
  ctx: '9AA39C', grid: 'E3E7E2', border: 'D9DFD9', warnBg: 'FDF6E8', warnBr: 'E5CF9A',
  dark: '0F1A14', darkText: 'EEF1EC', darkMuted: '9FB0A5', mint: '7FD1A0',
};
const HEAD = 'Arial';
const BODY = 'Calibri';

const money = (x, k) => (k ? `$${Math.round(x / 1000)}K` : `$${Math.round(x).toLocaleString('en-US')}`);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const pct0 = (x) => `${Math.round(x * 100)}%`;

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625 in
pres.author = 'Fresh Forecast';
pres.title = 'Fresh Forecast · Proposal for Harris Teeter leadership';

const M = 0.6; // side margin
const CW = 10 - 2 * M; // content width

function baseSlide(dark = false) {
  const s = pres.addSlide();
  s.background = { color: dark ? C.dark : C.paper };
  return s;
}

function eyebrow(s, text, dark = false) {
  s.addText(text.toUpperCase(), {
    x: M, y: 0.38, w: CW, h: 0.28, fontFace: HEAD, fontSize: 10, bold: true, charSpacing: 2,
    color: dark ? C.mint : C.green, margin: 0, isTextBox: true,
  });
}

function title(s, text, opts = {}) {
  s.addText(text, {
    x: M, y: 0.62, w: opts.w || CW, h: opts.h || 0.9, fontFace: HEAD, fontSize: opts.size || (text.length > 46 ? 22 : 26), bold: true,
    color: C.ink, margin: 0, valign: 'top', isTextBox: true, fit: 'shrink',
  });
}

function para(s, text, x, y, w, h, opts = {}) {
  s.addText(text, {
    x, y, w, h, fontFace: BODY, fontSize: opts.size || 13, color: opts.color || C.ink,
    margin: 0, valign: opts.valign || 'top', isTextBox: true, paraSpaceAfter: opts.gap == null ? 6 : opts.gap,
    lineSpacingMultiple: 1.12, bold: !!opts.bold, italic: !!opts.italic,
  });
}

function footnote(s, text) {
  s.addText(text, {
    x: M, y: 4.95, w: CW - 0.9, h: 0.45, fontFace: BODY, fontSize: 9.5, color: C.muted,
    margin: 0, valign: 'bottom', isTextBox: true,
  });
}

function pageNo(s, n, dark = false) {
  s.addText(`${n}`, {
    x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontFace: HEAD, fontSize: 8, color: dark ? C.darkMuted : C.muted,
    align: 'right', margin: 0, isTextBox: true,
  });
}

function card(s, x, y, w, h, fill = C.card, line = C.border) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.75 }, rectRadius: 0.12,
  });
}

let n = 0;
const next = () => ++n;

// ------------------------------------------------------------------ 1 · Title
{
  const s = baseSlide(true);
  eyebrow(s, 'Harris Teeter · Prepared Foods · Proposal for leadership', true);
  s.addText('Fresh Forecast', {
    x: M, y: 1.5, w: CW, h: 1.1, fontFace: HEAD, fontSize: 60, bold: true, color: C.darkText, margin: 0, isTextBox: true,
  });
  s.addText('Measure, predict and reduce prepared-food waste, one store at a time, starting with a 90-day pilot.', {
    x: M, y: 2.7, w: 7.2, h: 0.9, fontFace: BODY, fontSize: 20, color: 'C9D4CC', margin: 0, isTextBox: true,
  });
  s.addText([
    { text: '●  ', options: { color: C.mint, fontSize: 7 } }, { text: 'Concept validated in a full-year simulation      ' },
    { text: '●  ', options: { color: C.mint, fontSize: 7 } }, { text: 'Built on zero company data      ' },
    { text: '●  ', options: { color: C.mint, fontSize: 7 } }, { text: 'September 2026' },
  ], { x: M, y: 3.75, w: CW, h: 0.35, fontFace: HEAD, fontSize: 11, color: C.darkMuted, margin: 0, isTextBox: true });
  s.addNotes('Open with the two-sided cost: every night we throw away food we made too much of, and every week we send customers home without food we made too little of. Same forecasting problem.');
  pageNo(s, next(), true);
}

// ------------------------------------------------------------------ 2 · One-slide version
{
  const s = baseSlide();
  eyebrow(s, 'The one-slide version');
  title(s, 'We can stop guessing how much to make');
  para(s, 'Bakery, pizza, hot foods and fresh cases are produced against gut-feel par sheets. Pars cannot see weather, holidays or trend, so they pad, and the padding becomes waste. A demand forecast can see all of those a day ahead.', M, 1.55, 4.3, 1.6, { size: 12 });
  para(s, 'A working forecasting system already exists. It was built and tested on a fully synthetic store so that no data-permission or IP question had to be answered first. The ask is one pilot store, read-only data access, and a modest number of hours.', M, 3.2, 4.3, 1.6, { size: 12 });
  const stats = [
    [`−${Math.round(wasteCutMatched * 100)}%`, "waste in the simulated pilot year with product availability held at today's level", false],
    [`${Math.round(gapClosed * 100)}%`, 'of the improvement that perfect knowledge of demand would allow', false],
    ['10–15%', 'the waste reduction this proposal actually promises for a real 90-day pilot', true],
  ];
  stats.forEach(([v, l, accent], i) => {
    const y = 1.55 + i * 1.05;
    card(s, 5.2, y, 4.2, 0.9, accent ? C.greenSoft : C.card, accent ? 'B9D8C6' : C.border);
    s.addText(v, { x: 5.35, y: y + 0.1, w: 1.55, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: C.green, margin: 0, valign: 'middle', isTextBox: true, fit: 'shrink' });
    s.addText(l, { x: 6.95, y: y + 0.1, w: 2.35, h: 0.7, fontFace: BODY, fontSize: 10.5, color: C.ink2, margin: 0, valign: 'middle', isTextBox: true });
  });
  footnote(s, `Simulation figures come from a synthetic three-year store, held-out year ${R.test_year}. They prove the method. The real dollar figure comes from the pilot's measured baseline.`);
  s.addNotes('If leadership reads one slide, this is it. Method proven; dollars to be measured; ask is small.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 3 · Heatmap
{
  const s = baseSlide();
  eyebrow(s, 'The problem');
  title(s, 'A par sheet is one number. Demand is not.');
  para(s, 'Make too much and we eat the cost of goods. Make too little and we lose the margin and disappoint the customer at an empty hot bar at 6 pm.', M, 1.55, 3.1, 1.15, { size: 11.5 });
  para(s, 'Pars solve this with a flat safety pad. But demand swings with the weekday, the weather, the season and the calendar, and all of those are known the day before.', M, 2.75, 3.1, 1.25, { size: 11.5 });
  para(s, 'Demand index by weekday and weather in the simulated store. 1.00 is an average day. The spread between a snowy Monday and a sunny Friday is almost 2×, and a single par number has to cover both.', M, 4.05, 3.1, 1.1, { size: 9, color: C.muted });
  const dows = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const weathers = ['sunny', 'cloudy', 'rain', 'snow'];
  const cells = R.charts.dow_weather;
  const lookup = {};
  cells.forEach((c) => { lookup[`${c.dow}|${c.weather}`] = c.index; });
  const vals = cells.map((c) => c.index);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x0 = 4.45, y0 = 1.85, cw = 0.7, ch = 0.72;
  dows.forEach((d, j) => s.addText(d, { x: x0 + j * cw, y: y0 - 0.32, w: cw, h: 0.3, fontFace: HEAD, fontSize: 10, bold: true, color: C.ink2, align: 'center', margin: 0, isTextBox: true }));
  weathers.forEach((w, i) => {
    s.addText(w[0].toUpperCase() + w.slice(1), { x: x0 - 0.75, y: y0 + i * ch, w: 0.7, h: ch, fontFace: HEAD, fontSize: 10, bold: true, color: C.ink2, align: 'right', valign: 'middle', margin: 0, isTextBox: true });
    dows.forEach((_, j) => {
      const v = lookup[`${j}|${w}`];
      const x = x0 + j * cw + 0.03, y = y0 + i * ch + 0.03;
      if (v == null) {
        s.addShape(pres.ShapeType.roundRect, { x, y, w: cw - 0.06, h: ch - 0.06, fill: { color: C.paper }, line: { color: C.grid, width: 0.75, dashType: 'dash' }, rectRadius: 0.06 });
        s.addText('n/a', { x, y, w: cw - 0.06, h: ch - 0.06, fontFace: HEAD, fontSize: 8, color: C.muted, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
        return;
      }
      const t = (v - lo) / (hi - lo);
      const a = [0xe7, 0xf2, 0xec], b = [0x12, 0x4d, 0x2e];
      const hex = a.map((av, k) => Math.round(av + (b[k] - av) * t).toString(16).padStart(2, '0')).join('').toUpperCase();
      s.addShape(pres.ShapeType.roundRect, { x, y, w: cw - 0.06, h: ch - 0.06, fill: { color: hex }, line: { color: hex, width: 0 }, rectRadius: 0.06 });
      s.addText(v.toFixed(2), { x, y, w: cw - 0.06, h: ch - 0.06, fontFace: HEAD, fontSize: 12, bold: true, color: t > 0.55 ? 'FFFFFF' : C.ink, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    });
  });
  s.addNotes('Industry-typical waste for these departments runs 15 to 25 percent of production. The heatmap is the simulated store, but the shape is the one every kitchen manager knows.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 4 · Seasonality small multiples
{
  const s = baseSlide();
  eyebrow(s, 'The problem, continued');
  title(s, 'Each department has its own year');
  para(s, 'Hot foods peak in winter, fresh foods in summer, bakery and pizza around the holidays. A trailing average lags each of these turns by weeks. A forecast that knows the calendar turns with them.', M, 1.55, 3.0, 1.6, { size: 12.5 });
  para(s, "Monthly demand index by department in the simulated store, relative to each department's own average.", M, 3.3, 3.0, 1.0, { size: 9.5, color: C.muted });
  const months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
  ['Bakery', 'Pizza', 'Hot Foods', 'Fresh Foods'].forEach((dept, k) => {
    const pts = R.charts.seasonality.filter((r) => r.dept === dept).sort((a, b) => a.month - b.month).map((r) => r.index);
    const x = 3.9 + (k % 2) * 2.85, y = 1.5 + Math.floor(k / 2) * 1.85;
    s.addText(dept, { x: x + 0.1, y, w: 2.6, h: 0.25, fontFace: HEAD, fontSize: 10.5, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addChart(pres.ChartType.line, [{ name: dept, labels: months, values: pts }], {
      x, y: y + 0.22, w: 2.75, h: 1.6,
      chartColors: [C.green], lineSize: 2, lineDataSymbol: 'none', showLegend: false, showTitle: false,
      valAxisMinVal: 0.8, valAxisMaxVal: 1.25, valAxisMajorUnit: 0.2, valAxisLabelFontSize: 8, valAxisLabelColor: C.muted,
      catAxisLabelFontSize: 8, catAxisLabelColor: C.muted, valGridLine: { color: C.grid, size: 0.5 }, catGridLine: { style: 'none' },
      valAxisLabelFormatCode: '0.0', valAxisLineShow: false, catAxisLineShow: true, catAxisLineColor: C.grid,
    });
  });
  s.addNotes('Point at hot foods: a swing of nearly 40 points between July and January. No single par survives that.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 5 · Pipeline
{
  const s = baseSlide();
  eyebrow(s, 'What already exists');
  title(s, 'The whole system, end to end');
  const boxes = [
    { x: 0.6, w: 1.7, t: 'Inputs', l: 'Sales history (sellouts flagged)\nWeather forecast\nCalendar & holidays', accent: false },
    { x: 2.967, w: 1.7, t: 'Demand model', l: 'one network,\nevery item', accent: true },
    { x: 5.333, w: 1.7, t: 'Decision rule', l: 'cost of one-too-many\nvs one-too-few', accent: true },
    { x: 7.7, w: 1.7, t: 'Morning sheet', l: 'printed in the kitchen\nmake 14 · thaw 2 cases', accent: false },
  ];
  const by = 1.75, bh = 1.35;
  boxes.forEach((b) => {
    card(s, b.x, by, b.w, bh, b.accent ? C.greenSoft : C.card, b.accent ? C.green : 'CFD6D0');
    s.addText(b.t, { x: b.x, y: by + 0.12, w: b.w, h: 0.35, fontFace: HEAD, fontSize: 13, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
    s.addText(b.l, { x: b.x + 0.08, y: by + 0.5, w: b.w - 0.16, h: 0.8, fontFace: BODY, fontSize: 10.5, color: C.ink2, align: 'center', margin: 0, isTextBox: true, lineSpacingMultiple: 1.1 });
  });
  [[2.3, 2.967, ''], [4.667, 5.333, 'demand as\na range'], [7.033, 7.7, 'quantity']].forEach(([x1, x2, lab]) => {
    s.addShape(pres.ShapeType.line, { x: x1 + 0.05, y: by + bh / 2, w: x2 - x1 - 0.1, h: 0, line: { color: C.ink2, width: 1.5, endArrowType: 'triangle' } });
    if (lab) s.addText(lab, { x: x1, y: by + bh / 2 + 0.06, w: x2 - x1, h: 0.35, fontFace: HEAD, fontSize: 7, color: C.muted, align: 'center', margin: 0, isTextBox: true });
  });
  // feedback loop
  s.addShape(pres.ShapeType.line, { x: 8.55, y: by + bh, w: 0, h: 0.35, line: { color: C.ink2, width: 1.25, dashType: 'dash' } });
  s.addShape(pres.ShapeType.line, { x: 1.45, y: by + bh + 0.35, w: 7.1, h: 0, line: { color: C.ink2, width: 1.25, dashType: 'dash' } });
  s.addShape(pres.ShapeType.line, { x: 1.45, y: by + bh + 0.35, w: 0, h: -0.35, line: { color: C.ink2, width: 1.25, dashType: 'dash', endArrowType: 'triangle' } });
  s.addText("each day's sales and waste feed back into the history", { x: 2.5, y: by + bh + 0.38, w: 5, h: 0.25, fontFace: HEAD, fontSize: 8.5, color: C.muted, align: 'center', margin: 0, isTextBox: true });
  const cols = [
    ['No integration needed for a pilot. ', 'The inputs are a sales export, a public weather forecast and a calendar. The output is a printed sheet in the kitchen each morning.'],
    ['One model, every item. ', 'A compact neural network of the family used industrially for retail demand forecasting, forecasting a demand range rather than a point.'],
    ['A decision rule, not a guess. ', 'Produce where the cost of one wasted unit equals the cost of one missed sale. Cheap, high-margin items pad; expensive, low-margin items run lean.'],
  ];
  cols.forEach(([b, t], i) => {
    s.addText([{ text: b, options: { bold: true, fontFace: HEAD, color: C.ink } }, { text: t, options: { color: C.ink2 } }], {
      x: M + i * 2.95, y: 3.85, w: 2.75, h: 1.2, fontFace: BODY, fontSize: 10.5, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1,
    });
  });
  s.addNotes('Emphasise how little has to change: the kitchen gets a piece of paper. The manager can override any line.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 6 · Cost by policy
{
  const s = baseSlide();
  eyebrow(s, 'Proof of concept');
  title(s, `A year replayed: the forecast closes ${Math.round(gapClosed * 100)}% of the gap to perfect knowledge`);
  para(s, "The model trained on two simulated years, seeing only what a real store sees, then ran in shadow against the untouched third year. For every day and item it recommended a quantity; the simulation's true demand settled what sold, what was wasted and what was missed.", M, 1.6, 3.4, 1.75, { size: 11.5 });
  para(s, 'Two benchmark forecasters ran alongside so the network had to earn its seat.', M, 3.4, 3.4, 0.6, { size: 11.5 });
  para(s, `Total cost of being wrong per simulated store-year: wasted units at cost of goods plus missed sales at lost margin. ${R.n_test_days} days.`, M, 4.05, 3.4, 0.8, { size: 9, color: C.muted });
  const rows = [
    ['Status quo (par sheet)', sq.econ_cost, C.orange],
    ['Naive trailing average', naive.econ_cost, C.ctx],
    ['Linear model', ridge.econ_cost, C.ctx],
    ['Forecast · availability held', dlm.econ_cost, C.green],
    ['Forecast · profit-optimal', dl.econ_cost, C.green],
    ['Oracle (perfect knowledge)', orc.econ_cost, C.blue],
  ];
  // Hand-drawn bars: one fill per policy is the point of the chart, and a single-series
  // native bar chart cannot color its points independently in every viewer.
  const x0 = 6.35, y0 = 1.65, bandH = 0.52, maxW = 2.55, vmax = sq.econ_cost * 1.08;
  rows.forEach(([lab, v, col], i) => {
    const y = y0 + i * bandH;
    s.addText(lab, { x: 3.95, y, w: 2.3, h: 0.36, fontFace: HEAD, fontSize: 10, bold: true, color: C.ink, align: 'right', valign: 'middle', margin: 0, isTextBox: true });
    const w = (v / vmax) * maxW;
    s.addShape(pres.ShapeType.roundRect, { x: x0, y: y + 0.06, w, h: 0.24, fill: { color: col }, line: { color: col, width: 0 }, rectRadius: 0.04 });
    s.addText(money(v, true), { x: x0 + w + 0.06, y, w: 0.7, h: 0.36, fontFace: HEAD, fontSize: 10, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  });
  s.addNotes(`Forecast accuracy: ${pct(wape.dl)} weighted error for the network against ${pct(wape.naive)} for the trailing average and ${pct(wape.ridge)} for the linear model.`);
  pageNo(s, next());
}

// ------------------------------------------------------------------ 7 · Results table
{
  const s = baseSlide();
  eyebrow(s, 'Proof of concept');
  title(s, 'Less waste at the same availability, or much less waste at slightly lower availability');
  const th = (t, al = 'right') => ({ text: t, options: { bold: true, fontFace: HEAD, fontSize: 8.5, color: C.ink2, align: al, fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const td = (t, al = 'right', hl = false, bold = false, color = C.ink) => ({ text: t, options: { fontFace: BODY, fontSize: 11.5, color, align: al, bold, fill: { color: hl ? C.greenSoft : C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const row = (name, r, delta, hl) => [
    td(name, 'left', hl, true),
    delta ? { text: [{ text: money(r.waste_retail) + '  ', options: { fontFace: BODY, fontSize: 11.5, color: C.ink } }, { text: `(${delta})`, options: { fontFace: HEAD, fontSize: 11, bold: true, color: C.green } }], options: td('', 'right', hl).options }
          : td(money(r.waste_retail), 'right', hl),
    td(pct(r.waste_pct_of_production), 'right', hl), td(pct(r.fill_rate), 'right', hl), td(money(r.econ_cost), 'right', hl),
  ];
  s.addTable([
    [th('HELD-OUT YEAR, ONE STORE', 'left'), th('WASTE (RETAIL VALUE)'), th('WASTE, % OF PRODUCTION'), th('AVAILABILITY (FILL RATE)'), th('TOTAL ECONOMIC COST')],
    row('Status quo (par sheet)', sq),
    row('Forecast, availability held', dlm, `−${Math.round(wasteCutMatched * 100)}%`, true),
    row('Forecast, profit-optimal', dl, `−${Math.round(wasteCutOpt * 100)}%`, true),
    row('Oracle (perfect knowledge)', orc),
  ], { x: M, y: 1.65, w: CW, colW: [2.4, 1.9, 1.6, 1.5, 1.4], rowH: 0.36, margin: 0.06 });
  const cols = [
    ['Availability held ', `is the pilot setting: ${money(retailSavedMatched)} less food thrown away per year at retail value, with customers seeing an empty case about as often as they do today.`],
    ['Profit-optimal ', `is where the decision rule would go on its own: waste falls by ${Math.round(wasteCutOpt * 100)}%, and availability slips from ${pct(sq.fill_rate)} to ${pct(dl.fill_rate)}. That is a business choice, item by item, not a model setting.`],
    ['Forecast accuracy ', `${pct(wape.dl)} weighted error, against ${pct(wape.naive)} for a trailing average and ${pct(wape.ridge)} for a linear model.`],
  ];
  cols.forEach(([b, t], i) => {
    s.addText([{ text: b, options: { bold: true, fontFace: HEAD, color: C.ink } }, { text: t, options: { color: C.ink2 } }], {
      x: M + i * 2.95, y: 3.75, w: 2.75, h: 1.3, fontFace: BODY, fontSize: 10, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1,
    });
  });
  s.addNotes('The two forecast rows are the same model with a different availability target. Leadership picks the row.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 8 · Day to day line chart
{
  const s = baseSlide();
  eyebrow(s, 'What it looks like day to day');
  title(s, 'Whole pizza, eight simulated weeks: the forecast hugs demand, the par sheet pads it');
  const ser = R.charts['series_pizza-whole'];
  const labels = ser.dates.map((d, i) => (i % 7 === 0 ? d.slice(5) : ''));
  s.addChart(pres.ChartType.line, [
    { name: 'Actual demand', labels, values: ser.true_demand },
    { name: 'Par sheet', labels, values: ser.status_quo },
    { name: 'Forecast', labels, values: ser.dl },
  ], {
    x: M, y: 1.6, w: CW, h: 3.25,
    chartColors: [C.ctx, C.orange, C.green], lineSize: 2, lineDataSymbol: 'none',
    showLegend: true, legendPos: 'r', legendFontSize: 10, legendFontFace: HEAD, legendColor: C.ink,
    valAxisLabelFontSize: 9, valAxisLabelColor: C.muted, catAxisLabelFontSize: 9, catAxisLabelColor: C.muted,
    valGridLine: { color: C.grid, size: 0.5 }, catGridLine: { style: 'none' }, valAxisMinVal: 0,
    valAxisLineShow: false, catAxisLineColor: C.grid, showTitle: false,
  });
  footnote(s, 'Units per day. Actual demand is what no method ever sees; both policies only see sales. Super Bowl Sunday falls on 02-09 and Valentine\'s Day on 02-14, both known to the forecast a day ahead.');
  s.addNotes('Super Bowl Sunday: the forecast lifts production ahead of the spike; the par sheet catches up a week late, then over-produces.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 9 · Cumulative savings
{
  const s = baseSlide();
  eyebrow(s, 'Proof of concept');
  title(s, 'Savings accrue every week, not at the end of a project');
  para(s, `Cumulative economic cost avoided by the profit-optimal forecast versus the par sheet across the simulated year: ${money(econSavedOpt)} per store, or about ${money(econSavedOpt / 52)} a week from the first week of live operation.`, M, 1.55, 3.3, 1.5, { size: 12 });
  para(s, `At the availability-held setting it is ${money(econSavedMatched)} per store-year in economic cost, and ${money(retailSavedMatched)} of food, at retail, not thrown away.`, M, 3.0, 3.3, 1.1, { size: 12 });
  para(s, 'Economic cost counts waste at cost of goods and missed sales at lost margin. Retail value of waste is the number a shrink report shows.', M, 4.05, 3.3, 0.9, { size: 9.5, color: C.muted });
  const cum = R.charts.cumulative_savings;
  const names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const labels = cum.dates.map((d) => (d.endsWith('-01') ? names[parseInt(d.slice(5, 7), 10) - 1] : ''));
  s.addChart(pres.ChartType.area, [{ name: 'Cumulative economic cost avoided', labels, values: cum.dollars }], {
    x: 4.1, y: 1.5, w: 5.3, h: 3.4,
    chartColors: [C.green], chartColorsOpacity: 18, lineSize: 2, showLegend: false, showTitle: false,
    valAxisLabelFontSize: 9, valAxisLabelColor: C.muted, catAxisLabelFontSize: 9, catAxisLabelColor: C.muted,
    valAxisLabelFormatCode: '$#,##0', valGridLine: { color: C.grid, size: 0.5 }, catGridLine: { style: 'none' },
    valAxisLineShow: false, catAxisLineColor: C.grid, valAxisMinVal: 0,
  });
  s.addNotes('Steady slope means no big-bang: the value shows up in the first shrink report after go-live.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 10 · Honesty
{
  const s = baseSlide();
  eyebrow(s, 'Read this slide carefully');
  title(s, 'What the simulation proves, and what it does not');
  const bullets = (items) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  card(s, M, 1.6, 4.25, 3.3, C.greenSoft, 'B9D8C6');
  s.addText('It proves the method', { x: M + 0.2, y: 1.72, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  s.addText(bullets([
    "Demand with realistic weekday, weather, holiday and seasonal structure is learnable from sales data alone, censored by sellouts as a real store's data is.",
    'Production driven by those forecasts beats a well-run par sheet by a wide margin, and beats two simpler forecasters too.',
    'Every number is reproducible from the repository with one command, and the inputs are frozen.',
  ]), { x: M + 0.2, y: 2.1, w: 3.9, h: 2.7, fontFace: BODY, fontSize: 11.5, color: C.ink, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1 });
  card(s, 5.15, 1.6, 4.25, 3.3, C.warnBg, C.warnBr);
  s.addText('It does not prove our dollar figure', { x: 5.35, y: 1.72, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  s.addText(bullets([
    'The simulated store is a plausible invention. Its simulated manager, though deliberately competent, is an invention too.',
    'No company data of any kind was used. Nothing here has seen a real export yet.',
    `That is why this proposal promises 10–15% in a real pilot, not ${Math.round(wasteCutMatched * 100)}–${Math.round(wasteCutOpt * 100)}%, and why the pilot's measured baseline, not the simulation, is the number it is judged against.`,
  ]), { x: 5.35, y: 2.1, w: 3.9, h: 2.7, fontFace: BODY, fontSize: 11.5, color: C.ink, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1 });
  s.addNotes('This slide is deliberate. Leadership funds measured numbers, and the credibility of the ask rests on not overclaiming.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 11 · Pilot timeline
{
  const s = baseSlide();
  eyebrow(s, 'The proposal');
  title(s, 'A 90-day pilot at one store, with a stop gate before anything in the kitchen changes');
  const phases = [
    ['Weeks 1–2', 'Baseline', 'Log every discarded prepared-food item across 5–10 pilot SKUs with the phone-based waste logger that already exists.', "Output: this store's real waste number, in dollars per year.", false],
    ['Weeks 3–6', 'Shadow mode', "With data access granted, the model retrains on this store's history and prints its morning sheet. The kitchen ignores it. We record what it would have done.", 'Output: forecast accuracy against reality, and five go/no-go criteria fixed in advance.', false],
    ['Weeks 7–13', 'Live', 'The kitchen follows the sheet. The manager can override any line on any day, and overrides are logged, not fought.', 'Output: measured waste and sellout change versus the baseline.', true],
  ];
  phases.forEach(([when, name, body, out, live], i) => {
    const x = M + i * 2.95, y = 1.6, w = 2.75, h = 2.35;
    card(s, x, y, w, h, live ? C.greenDeep : C.card, live ? C.greenDeep : C.border);
    s.addText(when.toUpperCase(), { x: x + 0.18, y: y + 0.14, w: w - 0.36, h: 0.22, fontFace: HEAD, fontSize: 8.5, bold: true, charSpacing: 1.5, color: live ? '9FE0B9' : C.green, margin: 0, isTextBox: true });
    s.addText(name, { x: x + 0.18, y: y + 0.36, w: w - 0.36, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: live ? C.darkText : C.ink, margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.18, y: y + 0.7, w: w - 0.36, h: 1.05, fontFace: BODY, fontSize: 10.5, color: live ? C.darkText : C.ink, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1 });
    s.addText(out, { x: x + 0.18, y: y + 1.72, w: w - 0.36, h: 0.55, fontFace: HEAD, fontSize: 9.5, bold: true, color: live ? 'C9E6D4' : C.ink2, margin: 0, valign: 'top', isTextBox: true });
  });
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 4.15, w: CW, h: 0.85, fill: { color: C.greenSoft }, line: { color: C.greenSoft, width: 0 }, rectRadius: 0.1 });
  s.addText([
    { text: 'Success criterion, agreed in advance: ', options: { bold: true, fontFace: HEAD } },
    { text: 'at least a 10% reduction in prepared-food waste dollars versus the baseline, with no material drop in on-shelf availability. If shadow mode shows the forecasts are not accurate enough, we stop at week 6 and nothing in the kitchen has changed.' },
  ], { x: M + 0.2, y: 4.2, w: CW - 0.4, h: 0.75, fontFace: BODY, fontSize: 11, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
  s.addNotes('The gate at week 6 is the point: the downside of the pilot is bounded to a few pallets of mis-produced food, and only after shadow mode passes.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 12 · Asks
{
  const s = baseSlide();
  eyebrow(s, 'The asks');
  title(s, 'Three decisions, all reversible');
  const asks = [
    ['Data access, in writing', 'Read-only access to daily sales and production counts for the pilot items at one store, for the duration of the pilot. Scoped, revocable, and requested before any real data is touched.'],
    ['40–80 paid hours', "Non-floor time to connect the store's export, run shadow mode, and operate the live phase. The importer, validator, training and scoring tools are already built and rehearsed on a mock export."],
    ['A sponsor and a recognition structure', 'A store or district leader who owns the pilot, and an agreed, written structure tied to the measured result, settled before the live phase, so that a successful pilot has somewhere to scale.'],
  ];
  asks.forEach(([h, b], i) => {
    const x = M + i * 2.95, y = 1.6, w = 2.75, h2 = 2.7;
    card(s, x, y, w, h2);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fill: { color: C.green }, line: { color: C.green, width: 0 } });
    s.addText(`${i + 1}`, { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fontFace: HEAD, fontSize: 13, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(h, { x: x + 0.2, y: y + 0.72, w: w - 0.4, h: 0.55, fontFace: HEAD, fontSize: 13.5, bold: true, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
    s.addText(b, { x: x + 0.2, y: y + 1.3, w: w - 0.4, h: 1.3, fontFace: BODY, fontSize: 10.5, color: C.ink, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1 });
  });
  footnote(s, 'Ownership of the work built so far was deliberately kept clean: nothing was built on company data, so IP and data questions can be settled on the way in rather than argued about on the way out.');
  s.addNotes('Ask three is what makes this a leadership decision rather than a store one.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 13 · If it works
{
  const s = baseSlide();
  eyebrow(s, 'If it works');
  title(s, 'One model serves every item, so the pilot is also the template');
  para(s, 'The same architecture scales from 9 SKUs at one store to every prepared-foods case in a district without redesign: retraining per store takes about a minute, and the morning sheet is the only interface the kitchen ever sees.', M, 1.6, 3.4, 1.5, { size: 11.5 });
  para(s, "The table is arithmetic on the simulated store, not a forecast of Harris Teeter's numbers. Its job is to show which column the pilot replaces with a measured fact. The last column is what this proposal actually promises, applied to a store whose waste looks like the simulated one.", M, 3.05, 3.4, 1.8, { size: 11.5 });
  const th = (t, al = 'right') => ({ text: t, options: { bold: true, fontFace: HEAD, fontSize: 8, color: C.ink2, align: al, valign: 'bottom', fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const td = (t, al = 'right', bold = false) => ({ text: t, options: { fontFace: BODY, fontSize: 10.5, color: C.ink, align: al, bold, fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const rows = [[th('PER YEAR', 'left'), th('FOOD NOT WASTED (RETAIL, SIMULATED)'), th('ECONOMIC COST AVOIDED (SIMULATED)'), th('PROMISE APPLIED (10–15% OF WASTE)')]];
  [1, 20, 50, 100].forEach((k) => rows.push([
    td(`${k} store${k > 1 ? 's' : ''}`, 'left', true), td(money(retailSavedMatched * k)), td(money(econSavedMatched * k)),
    td(`${money(sq.waste_retail * 0.10 * k)} – ${money(sq.waste_retail * 0.15 * k)}`),
  ]));
  s.addTable(rows, { x: 4.2, y: 1.6, w: 5.2, colW: [1.0, 1.2, 1.2, 1.8], rowH: [0.55, 0.36, 0.36, 0.36, 0.36], margin: 0.05 });
  para(s, `Simulated store: ${money(sq.waste_retail)} of prepared-food waste at retail per year under the par sheet; availability-held forecast setting. Replace the first row with the pilot's measured baseline and the rest follows.`, 4.2, 3.75, 5.2, 0.9, { size: 9, color: C.muted });
  s.addNotes('Do not read the 100-store row as a projection. Read it as: this is why a single-store pilot is worth 80 hours.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 14 · Risks
{
  const s = baseSlide();
  eyebrow(s, 'Risks, named');
  title(s, 'Every risk has a gate, a fallback, or a written answer');
  const th = (t) => ({ text: t, options: { bold: true, fontFace: HEAD, fontSize: 8.5, color: C.ink2, fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const td = (t, bold = false) => ({ text: t, options: { fontFace: BODY, fontSize: 11, color: C.ink, bold, fill: { color: C.paper }, valign: 'top', border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  s.addTable([
    [th('RISK'), th('MITIGATION')],
    [td('Forecasts do not transfer from simulation to this store', true), td('Shadow mode is the gate: four weeks of predictions scored against reality before production changes at all.')],
    [td('Availability suffers and customers notice', true), td('The pilot runs at the availability-held setting; sellouts are tracked daily and any item can revert to its par instantly.')],
    [td('The kitchen does not trust the sheet', true), td('Manager override is a feature, logged not fought; the pilot measures the blend, which is how it would really run.')],
    [td("The store's export is dirtier than expected", true), td('The importer has been rehearsed against a deliberately dirty mock export. The first real one will still surprise us, and the hours ask covers that.')],
    [td('Data access or IP questions', true), td('Nothing was built on company data; access is scoped, read-only, written and revocable. IP and recognition are settled in writing before the live phase.')],
  ], { x: M, y: 1.55, w: CW, colW: [3.1, 5.7], rowH: [0.3, 0.55, 0.55, 0.55, 0.55, 0.55], margin: 0.07 });
  s.addNotes('The fourth row is the honest one: the importer was tested against its own author\'s imagination.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 15 · Decision
{
  const s = baseSlide(true);
  eyebrow(s, 'The decision', true);
  s.addText('Approve one pilot store.', { x: M, y: 1.5, w: CW, h: 1.0, fontFace: HEAD, fontSize: 46, bold: true, color: C.darkText, margin: 0, isTextBox: true });
  s.addText('Week 1 begins the day the data-access letter is signed. By week 6 we know whether the forecasts are accurate here. By week 13 we have a measured number.', {
    x: M, y: 2.6, w: 7.4, h: 1.0, fontFace: BODY, fontSize: 18, color: 'C9D4CC', margin: 0, isTextBox: true,
  });
  s.addText([
    { text: '●  ', options: { color: C.mint, fontSize: 7 } }, { text: 'Downside: a few pallets of mis-produced food, after a stop gate      ' },
    { text: '●  ', options: { color: C.mint, fontSize: 7 } }, { text: 'Upside: a measured template for every store' },
  ], { x: M, y: 3.75, w: CW, h: 0.35, fontFace: HEAD, fontSize: 11, color: C.darkMuted, margin: 0, isTextBox: true });
  s.addNotes('Close by asking for the sponsor by name.');
  pageNo(s, next(), true);
}

// ------------------------------------------------------------------ 16 · Appendix per item
{
  const s = baseSlide();
  eyebrow(s, 'Appendix');
  title(s, 'Per item, simulated held-out year, profit-optimal setting');
  const th = (t, al = 'right') => ({ text: t, options: { bold: true, fontFace: HEAD, fontSize: 7.5, color: C.ink2, align: al, valign: 'bottom', fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const td = (t, al = 'right', bold = false) => ({ text: t, options: { fontFace: BODY, fontSize: 10, color: C.ink, align: al, bold, fill: { color: C.paper }, border: [{ type: 'none' }, { type: 'none' }, { pt: 0.75, color: C.grid }, { type: 'none' }] } });
  const items = Object.values(R.per_item).sort((a, b) => b.sq.waste_retail - a.sq.waste_retail);
  const rows = [[th('ITEM', 'left'), th('WASTE TODAY (RETAIL)'), th('WASTE % OF PRODUCTION'), th('WASTE WITH FORECAST'), th('CHANGE'), th('FILL RATE TODAY → FORECAST'), th('SERVICE TARGET')]];
  items.forEach((r) => rows.push([
    td(r.name, 'left', true), td(money(r.sq.waste_retail)), td(pct0(r.sq.waste_pct_of_production)), td(money(r.dl.waste_retail)),
    td(`−${Math.round((1 - r.dl.waste_retail / r.sq.waste_retail) * 100)}%`), td(`${pct(r.sq.fill_rate)} → ${pct(r.dl.fill_rate)}`), td(r.q_star.toFixed(2)),
  ]));
  s.addTable(rows, { x: M, y: 1.5, w: CW, colW: [1.7, 1.25, 1.25, 1.25, 0.9, 1.55, 0.9], rowH: 0.3, margin: 0.04 });
  footnote(s, "Service target is the newsvendor critical fractile each item's price, cost and salvage imply: the fraction of days the item should not sell out. Expensive, low-margin items get a lower target and run lean; cheap, high-margin items pad.");
  s.addNotes('Sushi is the item to watch: high cost, so the rule runs it lean and availability drops most. In a real pilot that item would get a floor.');
  pageNo(s, next());
}

// ------------------------------------------------------------------ 17 · Appendix method
{
  const s = baseSlide();
  eyebrow(s, 'Appendix');
  title(s, 'Method notes and provenance');
  const block = (x, y, w, h, head, body) => {
    s.addText(head, { x, y, w, h: 0.3, fontFace: HEAD, fontSize: 12.5, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(body, { x, y: y + 0.3, w, h: h - 0.3, fontFace: BODY, fontSize: 10, color: C.ink, margin: 0, valign: 'top', isTextBox: true, lineSpacingMultiple: 1.1 });
  };
  block(M, 1.55, 4.2, 1.75, 'Demand model', "A global quantile-regression neural network: a GRU encoder over each item's trailing 28 days, learned item embeddings, and calendar, weather and holiday covariates, trained with pinball loss across 11 quantiles. Sellout days are treated as censored, since sales only bound demand from below when the case sold out.");
  block(M, 3.35, 4.2, 1.5, 'Decision layer', "Per-item newsvendor critical fractile, rounded to batch sizes. The availability-held variant raises each item's target until the year's fill rate matches the par sheet's.");
  block(5.2, 1.55, 4.2, 0.95, 'Validation', `Trained on simulated 2023–24, tested untouched on ${R.test_year} against naive, linear and oracle policies. Weighted forecast error: network ${pct(wape.dl)}, trailing average ${pct(wape.naive)}, linear ${pct(wape.ridge)}.`);
  block(5.2, 2.5, 4.2, 1.55, 'Provenance', 'The synthetic dataset, the trained checkpoint and the results file are frozen and checked in. The backtest reproduces the results byte for byte, and the tools refuse to overwrite the frozen files by accident. This deck is generated from that results file by tools/build_deck_pptx.js.');
  block(5.2, 4.05, 4.2, 1.1, 'Real-data path', 'Importer, validation gate, training, evaluation and the morning-sheet loop have been run end to end on a mock export built from the synthetic store with realistic dirt added. None of it has seen real data.');
  s.addNotes('For the analyst in the room.');
  pageNo(s, next());
}

pres.writeFile({ fileName: OUT }).then((f) => console.log(`wrote ${f}`));
