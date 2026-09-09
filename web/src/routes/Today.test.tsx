import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Today } from './Today'
import type { MorningPayload, StatusPayload } from '../api/types'
import { api } from '../api/client'

// The stamps the record carries have to be on the page, not in a tooltip: a sheet made after
// the day it is for proves nothing about what was known that morning, and a day with two
// sheets has had its morning forecast replaced. Both were rendered but neither was pinned.

afterEach(() => { cleanup(); vi.restoreAllMocks() })

const status: StatusPayload = {
  last_ingested_date: '2025-12-31', last_sheet_date: '2025-12-30', last_scored_date: null,
  missed_sheet_dates: [], unscored_dates: [], current_model_version: 'abc',
  current_artifacts_dir: 'model/artifacts', sellout_source: 'produced_vs_sold',
  last_gates: {}, today: '2025-12-30', store: 'Store 0123',
  new_shadow_dir: false, shadow_dir: '/srv/shadow',
}

const sheet = (over: Partial<MorningPayload> = {}): MorningPayload => ({
  for_date: '2025-12-30',
  rows: [{
    item: 'bread', item_name: 'Bread Loaf', dept: 'Bakery', rec_qty: 32, par_qty: 24,
    batch: 1, unit: 'each', source: 'model', fallback_reason: '', why_text: 'rain',
    backfilled: 0,
  }],
  runs: [{ run_id: 'aaa', made_at: '2025-12-30T05:30:00-06:00', backfilled: 0 }],
  superseded: false, backfilled: 0,
  conditions: { tmax_f: 60, weather: 'sunny', holiday: '', payday: false, snow_tomorrow: false },
  caveats: [], yesterday: [], day_source: '', staleness_days: 0,
  ...over,
})

const show = (payload: MorningPayload) => {
  vi.spyOn(api, 'morning').mockResolvedValue(payload)
  render(<Today status={status} onChange={() => {}} />)
}

describe('the stamps the record carries', () => {
  it('shows BACKFILLED at the top of a sheet made after its own day', async () => {
    show(sheet({ backfilled: 1 }))
    expect(await screen.findByText(/BACKFILLED/)).toBeInTheDocument()
    expect(screen.getByText(/proves nothing about what was known/)).toBeInTheDocument()
  })

  it('does not stamp a sheet that was made before its day', async () => {
    show(sheet())
    await screen.findByText('Bread Loaf')
    expect(screen.queryByText(/BACKFILLED/)).not.toBeInTheDocument()
  })

  it('says when a day has more than one sheet, and which one is live', async () => {
    show(sheet({
      superseded: true, live_run_id: 'bbb',
      runs: [
        { run_id: 'aaa', made_at: '2025-12-30T05:30:00-06:00', backfilled: 0 },
        { run_id: 'bbb', made_at: '2025-12-30T14:02:00-06:00', backfilled: 0 },
      ],
    }))
    expect(await screen.findByText(/more than one sheet/)).toBeInTheDocument()
    expect(screen.getByText('bbb')).toBeInTheDocument()
    expect(screen.getByText(/stay in the record underneath/)).toBeInTheDocument()
  })

  it('renders every caveat the sheet came with', async () => {
    show(sheet({ caveats: ['No row for today in the panel', 'NO SELLOUT SIGNAL'] }))
    expect(await screen.findByText(/No row for today in the panel/)).toBeInTheDocument()
    expect(screen.getByText(/NO SELLOUT SIGNAL/)).toBeInTheDocument()
  })

  it('survives a payload with no caveats field at all', async () => {
    // it crashed on this in the browser: a component that trusts a shape absolutely takes the
    // whole page down, and a blank page in a back room reads as "nothing to make today"
    const partial = sheet()
    delete (partial as Partial<MorningPayload>).caveats
    show(partial)
    expect(await screen.findByText('Bread Loaf')).toBeInTheDocument()
  })

  it('survives a payload with no runs field when it says it is superseded', async () => {
    const partial = sheet({ superseded: true, live_run_id: 'bbb' })
    delete (partial as Partial<MorningPayload>).runs
    show(partial)
    expect(await screen.findByText(/more than one sheet/)).toBeInTheDocument()
  })
})

describe('the sheet itself', () => {
  it('puts the store’s own par beside the suggestion', async () => {
    show(sheet())
    expect(await screen.findByText('Bread Loaf')).toBeInTheDocument()
    expect(screen.getByText('32')).toBeInTheDocument()
    expect(screen.getByText('24')).toBeInTheDocument()
  })

  it('marks a row the model could not forecast as not the model speaking', async () => {
    show(sheet({
      rows: [{
        item: 'sushi', item_name: 'Sushi Roll', dept: 'Fresh', rec_qty: 12, par_qty: 12,
        batch: 1, unit: 'each', source: 'par_fallback',
        fallback_reason: 'only 9 of 28 days', why_text: '', backfilled: 0,
      }],
    }))
    expect(await screen.findByText(/no forecast — only 9 of 28 days/)).toBeInTheDocument()
  })
})
