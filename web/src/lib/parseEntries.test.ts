import { describe, expect, it } from 'vitest'
import vectors from '../../../tests/fixtures/entry_vectors.json'
import { parseEntries, parseTime, itemIndex, toLine } from './parseEntries'

// The fixture is generated from model.shadow.parse_entries, which is the authority. If this
// file goes red, the browser has started disagreeing with the server about what a returned
// sheet says -- which is how a store loses a day of the only sellout observation it gets.

const items = vectors.items as Record<string, { name: string }>

describe('agreement with the Python parser, vector by vector', () => {
  for (const c of vectors.cases) {
    it(c.label, () => {
      // some cases carry their own catalogue: no shipped item name is longer than the
      // sheet's 18-character ITEM column, so truncation cannot be exercised without one
      const catalogue = ((c as { items?: Record<string, { name: string }> }).items ?? items)
      const got = parseEntries(c.lines, catalogue)
      // the fixture stores '' for "nothing written there"; JSON widens the union, so the
      // expectation is rebuilt rather than compared field by field
      expect(got.rows).toEqual(
        (c.rows as { item: string; actual_produced: unknown; sold_out_at: string;
                     note: string }[]).map((r) => ({
          item: r.item,
          actual_produced: r.actual_produced === '' ? '' : Number(r.actual_produced),
          sold_out_at: r.sold_out_at,
          note: r.note,
        })),
      )
      expect(got.errors).toEqual(c.errors)
    })
  }
})

describe('the vector file is doing real work', () => {
  it('covers both outcomes', () => {
    expect(vectors.cases.length).toBeGreaterThanOrEqual(20)
    expect(vectors.cases.some((c) => c.errors.length)).toBe(true)
    expect(vectors.cases.some((c) => c.rows.length)).toBe(true)
  })

  it('one unreadable line still writes nothing', () => {
    const c = vectors.cases.find((x) => x.label === 'one bad line poisons the whole sheet')!
    expect(parseEntries(c.lines, items).errors.length).toBe(1)
    // the caller must send nothing when errors is non-empty; the server refuses regardless
  })
})

describe('parseTime', () => {
  it('reads 0 as a written negative, never midnight', () => {
    // a prepared-foods case is not open at 00:00; "0" means "it did not"
    expect(parseTime('0')).toBe('')
    expect(parseTime('00:00')).toBe('00:00')
  })

  it('refuses what it cannot read rather than guessing', () => {
    for (const bad of ['half past two', '14:75', '14:30pm', '25:00', 'lunchtime']) {
      expect(parseTime(bad)).toBeNull()
    }
  })

  it('accepts the ways a person writes a time', () => {
    expect(parseTime('2:30pm')).toBe('14:30')
    expect(parseTime('2pm')).toBe('14:00')
    expect(parseTime('1430')).toBe('14:30')
    expect(parseTime('2.30 PM')).toBe('14:30')
    expect(parseTime('11:15am')).toBe('11:15')
    expect(parseTime('12am')).toBe('00:00')
    expect(parseTime('12pm')).toBe('12:00')
  })

  it('treats a bare affirmative as one', () => {
    for (const yes of ['y', 'yes', 'circled', 'out', 'soldout']) {
      expect(parseTime(yes)).toBe('yes')
    }
  })
})

describe('itemIndex', () => {
  const long = {
    'rotisserie-lp': { name: 'Rotisserie Chicken Lemon Pepper' },
    'rotisserie-bbq': { name: 'Rotisserie Chicken Barbecue' },
    'cake-sheet': { name: 'Sheet Cake Half Vanilla' },
  }

  it('accepts the name as the 18-character ITEM column printed it', () => {
    // refusing it would lose a whole returned sheet over a column width
    expect(itemIndex(long).get('sheet cake half va')).toBe('cake-sheet')
  })

  it('still accepts the full name and the key', () => {
    const idx = itemIndex(long)
    expect(idx.get('sheet cake half vanilla')).toBe('cake-sheet')
    expect(idx.get('cake-sheet')).toBe('cake-sheet')
  })

  it('refuses a truncation two items share, rather than misfiling one under the other', () => {
    // both shorten to "Rotisserie Chicken"; guessing would file one item's production
    // number under the other, silently
    expect(itemIndex(long).get('rotisserie chicken')).toBeUndefined()
  })

  it('does not truncate a name that already fits', () => {
    expect(itemIndex(items).get('rotisserie chicke')).toBeUndefined()
    expect(itemIndex(items).get('rotisserie chicken')).toBe('rotisserie')
  })
})

describe('toLine', () => {
  it('round-trips the form back into the one format both intakes validate', () => {
    expect(toLine('bread', ' 20 ', ' 14:30 ')).toBe('bread,20,14:30')
    expect(toLine('bread', '20', '', 'ran out')).toBe('bread,20,,ran out')
    expect(parseEntries([toLine('bread', '20', '2:30pm')], items).rows[0].sold_out_at)
      .toBe('14:30')
  })
})
