import { describe, expect, it } from 'vitest'
import { dayLabel, longDate, pct, qty, shiftDays, usd } from './format'

// Vectors mirror model/shadow.py's _fmt_qty (:397). Pieces are printed as pieces; only a
// weighed item ever shows a decimal, and a missing value is a dash, never a zero.

describe('qty', () => {
  it('prints pieces as pieces', () => {
    expect(qty(12)).toBe('12')
    expect(qty(0)).toBe('0')
    expect(qty(12.0)).toBe('12')
  })

  it('shows a decimal only for a weighed item', () => {
    expect(qty(12.5, 'lb')).toBe('12.5 lb')
    expect(qty(12, 'lb')).toBe('12.0 lb')
    expect(qty(12.5)).toBe('12.5')
  })

  it('prints a missing quantity as a dash, never 0', () => {
    // an item the checkpoint never saw has no quantile; "make none" would be a lie
    expect(qty(null)).toBe('-')
    expect(qty(undefined)).toBe('-')
    expect(qty(NaN)).toBe('-')
    expect(qty(Infinity)).toBe('-')
  })
})

describe('pct and usd', () => {
  it('formats what it has', () => {
    expect(pct(0.13)).toBe('13.0%')
    expect(pct(0.13, 0)).toBe('13%')
    expect(usd(1234.5)).toBe('$1,234.50')
  })

  it('does not invent a value it was not given', () => {
    expect(pct(null)).toBe('-')
    expect(pct(NaN)).toBe('-')
    expect(usd(null)).toBe('-')
    expect(usd(NaN)).toBe('-')
  })
})

describe('dates', () => {
  it('labels a day without drifting across a timezone', () => {
    // parsed as local midnight: a naive new Date('2025-12-30') is UTC and can render as the
    // 29th west of Greenwich, which would mislabel the sheet in a store's own back room
    expect(dayLabel('2025-12-30')).toBe('Tue, Dec 30')
    expect(longDate('2025-12-30')).toContain('Tuesday')
    expect(longDate('2025-12-30')).toContain('2025')
  })

  it('shifts whole days without touching the clock', () => {
    expect(shiftDays('2025-12-30', -1)).toBe('2025-12-29')
    expect(shiftDays('2025-12-31', 1)).toBe('2026-01-01')
    expect(shiftDays('2026-03-01', -1)).toBe('2026-02-28')
    expect(shiftDays('2024-03-01', -1)).toBe('2024-02-29')
  })

  it('passes an unreadable date through rather than printing Invalid Date', () => {
    expect(dayLabel('not-a-date')).toBe('not-a-date')
    expect(longDate('')).toBe('')
  })
})
