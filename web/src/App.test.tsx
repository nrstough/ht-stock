import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'

// With no service behind it, the operator pages must say so. A blank sheet reads as
// "nothing to make today", which is the wrong answer to "the forecast service is down" and
// exactly the kind of thing a back room acts on before anyone notices.

const draw = () => render(<MemoryRouter><App /></MemoryRouter>)

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('when /api cannot be reached', () => {
  it('disables the app and names the command that starts it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed to fetch')))
    draw()
    expect(await screen.findByText('Not connected')).toBeInTheDocument()
    // the message says it and the copyable command repeats it; both are wanted
    expect(screen.getAllByText(/python -m ht\.serve/).length).toBeGreaterThan(0)
    // and none of the operator pages are offered
    expect(screen.queryByText('Today')).not.toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})

describe('when the service answers', () => {
  // route by URL: one body for every endpoint is not what a server does, and a test that
  // pretends otherwise hides the shape the page actually receives
  const ok = (status: unknown) => vi.fn().mockImplementation((url: string) => {
    const body = url.startsWith('/api/status') ? status
      : url.startsWith('/api/morning') ? { error: 'no morning sheet has been made' }
      : {}
    const okStatus = !url.startsWith('/api/morning')
    return Promise.resolve({
      ok: okStatus, status: okStatus ? 200 : 404, text: async () => JSON.stringify(body),
    })
  })

  const base = {
    last_ingested_date: '2025-12-31', last_sheet_date: '2026-01-01',
    last_scored_date: null, missed_sheet_dates: [], unscored_dates: [],
    current_model_version: '76dc4ae67cf6', current_artifacts_dir: 'model/artifacts',
    sellout_source: 'produced_vs_sold', last_gates: {}, today: '2026-01-01',
    store: 'Store 0123', new_shadow_dir: false, shadow_dir: '/srv/shadow',
  }

  it('shows the store and the store’s own today', async () => {
    vi.stubGlobal('fetch', ok(base))
    draw()
    expect(await screen.findByText(/Store 0123/)).toBeInTheDocument()
    expect(screen.getByText(/today is 2026-01-01/)).toBeInTheDocument()
  })

  it('warns loudly when the record is a brand new directory', async () => {
    // a mistyped --out looks exactly like a pilot that lost its record; the CLI refuses a
    // missing directory but a server has to create one to take its lock, so it says so
    vi.stubGlobal('fetch', ok({ ...base, new_shadow_dir: true, last_sheet_date: null }))
    draw()
    expect(await screen.findByText('This is a brand new pilot record')).toBeInTheDocument()
    expect(screen.getByText(/\/srv\/shadow/)).toBeInTheDocument()
  })

  it('does not warn about a record that already has sheets in it', async () => {
    vi.stubGlobal('fetch', ok(base))
    draw()
    await waitFor(() => expect(screen.getByText(/Store 0123/)).toBeInTheDocument())
    expect(screen.queryByText('This is a brand new pilot record')).not.toBeInTheDocument()
  })
})
