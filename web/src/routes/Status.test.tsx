import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Status } from './Status'
import type { StatusPayload } from '../api/types'
import { api } from '../api/client'

// Freezing a day's verdict is permanent. Whatever the server held back, and whatever it froze
// anyway, has to reach the person who pressed the button -- reporting it in JSON only is the
// same as not reporting it.

afterEach(() => { cleanup(); vi.restoreAllMocks() })

const status: StatusPayload = {
  last_ingested_date: '2025-12-31', last_sheet_date: '2025-12-30',
  last_scored_date: '2025-12-29', missed_sheet_dates: [],
  unscored_dates: ['2025-12-30'], current_model_version: 'abc',
  current_artifacts_dir: 'a', sellout_source: 'produced_vs_sold', last_gates: {},
  today: '2025-12-31', store: 'Store 0123', new_shadow_dir: false, shadow_dir: '/srv/shadow',
}

describe('catch up', () => {
  it('names the days it left alone and why', async () => {
    vi.spyOn(api, 'catchUp').mockResolvedValue({
      scored: ['2025-12-28'], unscored: ['2025-12-30'],
      waiting_on_data: { '2025-12-30': ['cake', 'sushi'] }, frozen_without_data: {},
    })
    render(<Status status={status} onChange={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Catch up' }))
    expect(await screen.findByText(/Left alone until their export lands/)).toBeInTheDocument()
    expect(screen.getByText(/cake, sushi/)).toBeInTheDocument()
  })

  it('names what it froze permanently, separately from what it skipped', async () => {
    vi.spyOn(api, 'catchUp').mockResolvedValue({
      scored: ['2025-12-30'], unscored: [],
      waiting_on_data: {}, frozen_without_data: { '2025-12-30': ['cake'] },
    })
    render(<Status status={status} onChange={() => {}} />)
    await userEvent.click(screen.getByRole('button', { name: 'Catch up' }))
    expect(await screen.findByText(/Recorded as missing, permanently/)).toBeInTheDocument()
  })

  it('describes itself accurately: complete days, not "days that have data"', () => {
    render(<Status status={status} onChange={() => {}} />)
    expect(screen.getByText(/whose sales data has fully landed/)).toBeInTheDocument()
  })
})

describe('score one day', () => {
  it('passes the server’s note through when a row was frozen without data', async () => {
    vi.spyOn(api, 'score').mockResolvedValue({
      for_date: '2025-12-30', rows: 9, counts: { scored: 8, missing_data: 1 },
      frozen_without_data: ['cake'],
      note: 'cake had no sales data and has not sold in the last 14 days, so it was '
        + 'recorded as missing rather than holding the day back.',
    })
    render(<Status status={status} onChange={() => {}} />)
    await userEvent.type(screen.getByLabelText('Date'), '2025-12-30')
    await userEvent.click(screen.getByRole('button', { name: 'Score it' }))
    expect(await screen.findByText(/recorded as missing rather than holding/))
      .toBeInTheDocument()
  })

  it('shows the refusal when the day is not fully in the panel', async () => {
    const { ApiFailure } = await import('../api/client')
    vi.spyOn(api, 'score').mockRejectedValue(
      new ApiFailure(409, { error: '2025-12-30 is not fully in the panel yet -- 1 item(s)' }))
    render(<Status status={status} onChange={() => {}} />)
    await userEvent.type(screen.getByLabelText('Date'), '2025-12-30')
    await userEvent.click(screen.getByRole('button', { name: 'Score it' }))
    expect(await screen.findByText('Refused')).toBeInTheDocument()
    expect(screen.getByText(/not fully in the panel yet/)).toBeInTheDocument()
  })
})
