// A second implementation of model/shadow.py's parse_entries, in the browser.
//
// It exists so the entry form can show a mistake next to the box that caused it instead of
// after a round trip. It is NOT the authority: the server parses again and its answer is
// the one that decides whether anything is written. Two parsers can drift, and a client
// that accepts what the server refuses wastes a keystroke while one that refuses what the
// server accepts loses a returned sheet -- so both are tested against the SAME committed
// vector file, tests/fixtures/entry_vectors.json, generated from the Python side.

export interface ParsedRow {
  item: string
  actual_produced: number | ''
  sold_out_at: string
  note: string
}

export interface ParseResult {
  rows: ParsedRow[]
  errors: string[]
}

const AFFIRMATIVE = ['y', 'yes', 'soldout', 'out', 'circled']
const NEGATIVE = ['', '-', 'n', 'no', '0', 'na', 'n/a', 'none']
const SHEET_ITEM_WIDTH = 18

/**
 * The SOLD OUT AT cell as a person writes it: 14:30, 2:30pm, 2pm, 1430 -- or "yes".
 *
 * Returns '' for an empty cell or a written negative, 'HH:MM' for a time, 'yes' for a bare
 * affirmative, and null for anything unreadable, which the caller reports rather than
 * guessing at: a guessed sellout time is a fabricated observation.
 *
 * '0' is a written negative, not midnight. A prepared-foods case is not open at 00:00, and
 * somebody writing 0 in a box asking for a time means "it did not".
 */
export function parseTime(text: string): string | null {
  const s = String(text ?? '').trim().toLowerCase().replace(/\./g, '').replace(/ /g, '')
  if (NEGATIVE.includes(s)) return ''
  if (AFFIRMATIVE.includes(s)) return 'yes'
  const m = /^(\d{1,2}):?(\d{2})?(am|pm)?$/.exec(s)
  if (!m) return null
  let hour = parseInt(m[1], 10)
  const minute = parseInt(m[2] ?? '0', 10)
  const half = m[3]
  if (minute > 59) return null
  if (half && !(hour >= 1 && hour <= 12)) return null
  if (!half && hour > 23) return null
  if (half) hour = (hour % 12) + (half === 'pm' ? 12 : 0)
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`
}

export interface ItemLike { name: string }

/**
 * Config key or printed name -> config key. The person holding the sheet reads a name.
 *
 * The truncated printed name is registered too -- the sheet's ITEM column is 18 characters,
 * so an ordinary POS description reads back as a name the config does not contain, and
 * refusing it would lose a whole sheet over a column width. Only where the truncation stays
 * unambiguous: two names that shorten to the same 18 characters would otherwise file one
 * item's production number under the other.
 */
export function itemIndex(items: Record<string, ItemLike>): Map<string, string> {
  const idx = new Map<string, string>()
  for (const key of Object.keys(items)) idx.set(key.toLowerCase(), key)
  for (const [key, it] of Object.entries(items)) {
    const name = String(it.name).toLowerCase()
    if (!idx.has(name)) idx.set(name, key)
  }
  const counts = new Map<string, number>()
  for (const it of Object.values(items)) {
    const short = String(it.name).slice(0, SHEET_ITEM_WIDTH).trim().toLowerCase()
    counts.set(short, (counts.get(short) ?? 0) + 1)
  }
  for (const [key, it] of Object.entries(items)) {
    const short = String(it.name).slice(0, SHEET_ITEM_WIDTH).trim().toLowerCase()
    if (counts.get(short) === 1 && !idx.has(short)) idx.set(short, key)
  }
  return idx
}

/** `item, made, sold out at[, note]` per line. All of it parses, or none of it is written. */
export function parseEntries(
  lines: string[],
  items: Record<string, ItemLike>,
): ParseResult {
  const idx = itemIndex(items)
  const rows: ParsedRow[] = []
  const errors: string[] = []
  const seen = new Set<string>()

  lines.forEach((raw, i) => {
    const n = i + 1
    const line = String(raw ?? '').replace(/^﻿/, '').trim()
    if (!line || line.startsWith('#')) return
    const parts = line.split(',').map((p) => p.trim())
    if (['item', 'item_name'].includes(parts[0].toLowerCase())) return

    const key = idx.get(parts[0].toLowerCase())
    if (key === undefined) {
      const near = Object.entries(items).find(
        ([, it]) => parts[0] && String(it.name).toLowerCase().startsWith(parts[0].toLowerCase()),
      )
      const hint = near ? ` -- did you mean ${near[0]}?` : ''
      errors.push(`line ${n}: '${parts[0]}' is not an item in the items config${hint}`)
      return
    }
    if (seen.has(key)) {
      errors.push(`line ${n}: ${key} was already entered on an earlier line`)
      return
    }

    let made: number | '' = ''
    if (parts.length > 1 && parts[1]) {
      const value = Number(parts[1])
      if (!Number.isFinite(value) || value < 0 || parts[1].trim() === '') {
        errors.push(`line ${n}: made '${parts[1]}' is not a quantity`)
        return
      }
      made = value
    }

    const soldOut = parts.length > 2 ? parseTime(parts[2]) : ''
    if (soldOut === null) {
      errors.push(
        `line ${n}: sold out at '${parts[2]}' is not a time -- write it as 14:30, 2:30pm or ` +
        `1430, or 'yes' if it sold out and nobody wrote the time, or leave it blank (or 'no') ` +
        `if it did not sell out`,
      )
      return
    }

    seen.add(key)
    rows.push({
      item: key,
      actual_produced: made,
      sold_out_at: soldOut,
      note: parts.length > 3 ? parts.slice(3).join(', ') : '',
    })
  })

  return { rows, errors }
}

/** The form's fields back into the one line format both intakes are validated by. */
export function toLine(item: string, made: string, soldOut: string, note = ''): string {
  const base = `${item},${made.trim()},${soldOut.trim()}`
  return note.trim() ? `${base},${note.trim()}` : base
}
