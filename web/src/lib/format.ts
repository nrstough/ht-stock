// Rendering only. Nothing here computes a figure the server did not send: every number on
// screen has to trace to a field ht/serve.py emitted, or the page is quoting itself.

/**
 * Pieces are printed as pieces. Only a weighed item ever shows a decimal.
 *
 * The rule is model/shadow.py's `_fmt_qty` (:397), and it matters on paper: "make 12.0
 * doughnuts" reads as a machine talking, and a kitchen that has to decide what .0 of a
 * doughnut means stops trusting the sheet. A missing value prints as a dash rather than 0 --
 * an item the checkpoint never saw has no quantile, and a zero would say "make none".
 */
export function qty(value: number | null | undefined, unit = 'each'): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  if (unit === 'lb') return `${value.toFixed(1)} lb`
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

export function usd(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '-'
  return n.toLocaleString('en-US', { style: 'currency', currency: 'USD' })
}

export function pct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '-'
  return `${(n * 100).toFixed(digits)}%`
}

/** ISO date -> "Tue 30 Dec", the way the printed sheet heads a day. */
export function dayLabel(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric', month: 'short' })
}

export function longDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US',
    { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
}

/** Shift an ISO date by whole days without touching the local timezone. */
export function shiftDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

/**
 * PASS / FAIL / PENDING, and nothing in between.
 *
 * PENDING is not PASS. An unmeasured gate rendered in the same colour as a met one is the
 * single most misleading thing this page could do to a district manager, so an unknown
 * verdict falls back to pending rather than to anything reassuring.
 */
export function verdictClass(v: string | null | undefined): string {
  if (v === 'PASS') return 'pass'
  if (v === 'FAIL') return 'fail'
  return 'pending'
}

export const GATE_TITLES: Record<string, string> = {
  G1: 'Completeness',
  G2: 'Accuracy vs par',
  G3: 'Calibration',
  G4: 'Economics',
  G5: 'Item coverage',
}
