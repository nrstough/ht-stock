import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Enter } from './Enter'
import type { EntryOrderPayload, StatusPayload } from '../api/types'
import { api } from '../api/client'

// The form walks the page in the order it PRINTED. That is not presentation: the person is
// reading down a piece of paper, and a different order is how one item's production number
// gets typed into another's.

afterEach(() => { cleanup(); vi.restoreAllMocks() })

const status: StatusPayload = {
  last_ingested_date: '2025-12-31', last_sheet_date: '2025-12-30', last_scored_date: null,
  missed_sheet_dates: [], unscored_dates: [], current_model_version: 'abc',
  current_artifacts_dir: 'a', sellout_source: 'produced_vs_sold', last_gates: {},
  today: '2025-12-31', store: 'Store 0123', new_shadow_dir: false, shadow_dir: '/srv/shadow',
}

const ROWS = [
  { item: 'pizza-slice', name: 'Pizza Slice', dept: 'Pizza', unit: 'each', said: 40, source: 'model' },
  { item: 'bread', name: 'Bread Loaf', dept: 'Bakery', unit: 'each', said: 32, source: 'model' },
  { item: 'hotbar-lb', name: 'Hot Bar (per lb)', dept: 'Hot Foods', unit: 'lb', said: 60, source: 'model' },
  { item: 'sushi', name: 'Sushi Roll', dept: 'Fresh', unit: 'each', said: 12, source: 'par_fallback' },
]

const order = (over: Partial<EntryOrderPayload> = {}): EntryOrderPayload => ({
  for_date: '2025-12-30', rows: ROWS, no_sheet_logged: false, entered_by: 'kmurphy', ...over,
})

const show = (payload: EntryOrderPayload) => {
  vi.spyOn(api, 'entryOrder').mockResolvedValue(payload)
  render(<Enter status={status} onChange={() => {}} />)
}

describe('the order the sheet printed', () => {
  it('renders the rows in the server’s order, not alphabetically', async () => {
    show(order())
    await screen.findByText('Pizza Slice')
    const names = Array.from(document.querySelectorAll('.entry-row .name'))
      .map((el) => el.textContent?.split('model')[0].split('your par')[0].trim())
    expect(names).toEqual(['Pizza Slice', 'Bread Loaf', 'Hot Bar (per lb)', 'Sushi Roll'])
    expect(names).not.toEqual([...names].sort())
  })

  it('shows what the sheet said beside each box, in that item’s own unit', async () => {
    show(order())
    expect(await screen.findByText(/model said 40/)).toBeInTheDocument()
    expect(screen.getByText(/model said 60\.0 lb/)).toBeInTheDocument()
  })

  it('does not call a trailing par a model prediction', async () => {
    show(order())
    // the sheet lists short-history items with a par and the reason they have no forecast;
    // calling that "model said" hands an untrained number the model's authority
    const said = await screen.findByText(/your par was 12/)
    // and asserted one-sidedly it would still pass if the row rendered BOTH labels
    expect(said.textContent).not.toMatch(/model said/)
  })

  it('says "not on the sheet" for an item the sheet did not carry', async () => {
    show(order({ rows: [{ ...ROWS[0], said: null, source: null }] }))
    expect(await screen.findByText('not on the sheet')).toBeInTheDocument()
  })
})

describe('the wrong-date signal', () => {
  it('warns, and does not quietly present an alphabetical page as the printed one', async () => {
    show(order({
      no_sheet_logged: true,
      warning: 'no sheet was logged for 2025-11-04; this page is in alphabetical order, '
        + 'not the order the sheet printed.',
    }))
    expect(await screen.findByText(/No sheet was logged for this date/)).toBeInTheDocument()
    expect(screen.getByText(/alphabetical order/)).toBeInTheDocument()
  })

  it('stays quiet on a date that does have a sheet', async () => {
    show(order())
    await screen.findByText('Bread Loaf')
    expect(screen.queryByText(/No sheet was logged/)).not.toBeInTheDocument()
  })
})

describe('nothing is sent until every line parses', () => {
  it('says what an empty sold-out box means, because it is a real observation', async () => {
    show(order())
    await screen.findByText('Bread Loaf')
    // the sentence is broken up by <strong> and <em>, so it is matched on the hint element
    const hint = document.querySelector('.card .hint')!
    expect(hint.textContent).toMatch(/empty\s+sold out at\s+means the item did\s*not\s*sell out/)
    expect(hint.textContent).toMatch(/the only one the pilot will get/)
  })

  it('disables the button until something has been typed', async () => {
    show(order())
    await screen.findByText('Bread Loaf')
    expect(screen.getByRole('button', { name: /Record/ })).toBeDisabled()
  })

  it('refuses to send a sheet the browser cannot read', async () => {
    const { default: userEvent } = await import('@testing-library/user-event')
    const send = vi.spyOn(api, 'enter')
    show(order())
    await screen.findByText('Bread Loaf')
    await userEvent.type(screen.getByLabelText('Bread Loaf made'), 'banana')
    await waitFor(() => expect(screen.getByText(/is not a quantity/)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Record/ })).toBeDisabled()
    expect(send).not.toHaveBeenCalled()
  })
})
