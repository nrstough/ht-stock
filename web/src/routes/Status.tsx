import { useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { StatusPayload } from '../api/types'
import { Banner } from '../components/Banner'
import { Gates } from '../components/Gates'

/**
 * What is behind, and the two ways to settle a day.
 *
 * Scoring freezes a verdict once, so the server refuses a day whose sales data has not
 * fully arrived -- scoring it then would record the missing items as missing for the rest of
 * the pilot, and catch-up cannot repair it. Catch-up is offered first because it applies the
 * same precondition per day: it scores the days that are complete and leaves the rest alone.
 *
 * Both routes can still freeze an item that has not sold in a fortnight, because waiting for
 * a row that is never coming would block the day forever. That is a permanent decision, so
 * the item is named in the result rather than left inside a count.
 */
export function Status({ status, onChange }: {
  status: StatusPayload | null
  onChange: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scoreDate, setScoreDate] = useState('')

  async function run(fn: () => Promise<string>) {
    setBusy(true); setError(null); setMessage(null)
    try {
      setMessage(await fn())
      onChange()
    } catch (err) {
      setError(err instanceof ApiFailure || err instanceof ApiUnreachable
        ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (!status) return <p className="hint">Loading…</p>

  return (
    <>
      {message ? <Banner kind="info">{message}</Banner> : null}
      {error ? <Banner kind="danger" title="Refused">{error}</Banner> : null}

      <div className="card">
        <h2>Where the pilot is</h2>
        <dl className="kv">
          <dt>Store</dt><dd>{status.store || '—'}</dd>
          <dt>Today</dt><dd>{status.today}</dd>
          <dt>Panel runs through</dt><dd>{status.last_ingested_date ?? '—'}</dd>
          <dt>Last sheet</dt><dd>{status.last_sheet_date ?? 'none yet'}</dd>
          <dt>Last scored</dt><dd>{status.last_scored_date ?? 'none yet'}</dd>
          <dt>Model</dt><dd>{status.current_model_version ?? '—'}</dd>
          <dt>Sellout signal</dt><dd>{status.sellout_source ?? '—'}</dd>
        </dl>
        {status.sellout_source === 'none' ? (
          <Banner kind="warn" title="This store has no sellout signal">
            The model fits the distribution of sales rather than demand, so recommended
            quantities run low on the busiest days. How low is not measured.
          </Banner>
        ) : null}
      </div>

      <div className="card">
        <h2>Waiting to be scored</h2>
        {status.unscored_dates.length ? (
          <p>{status.unscored_dates.join(', ')}</p>
        ) : (
          <p className="hint">Nothing is waiting.</p>
        )}
        <div className="row">
          <button className="primary" disabled={busy}
                  onClick={() => run(async () => {
                    const r = await api.catchUp()
                    const held = Object.entries(r.waiting_on_data ?? {})
                      .map(([d, items]) => `${d} (${items.join(', ')})`)
                    const froze = Object.entries(r.frozen_without_data ?? {})
                      .map(([d, items]) => `${d} (${items.join(', ')})`)
                    return [
                      `Scored: ${r.scored.join(', ') || 'nothing new'}.`,
                      held.length
                        ? `Left alone until their export lands: ${held.join('; ')}.`
                        : '',
                      froze.length
                        ? `Recorded as missing, permanently, because they have not sold `
                          + `recently: ${froze.join('; ')}.`
                        : '',
                    ].filter(Boolean).join(' ')
                  })}>
            {busy ? 'Working…' : 'Catch up'}
          </button>
          <span className="hint">
            Scores every day whose sales data has fully landed, and leaves the rest alone —
            a day&apos;s verdict is frozen once, so a day that is only partly in the panel is
            better left until it is complete.
          </span>
        </div>
      </div>

      <div className="card">
        <h2>Score one day</h2>
        <div className="row">
          <div className="field">
            <label htmlFor="score-date">Date</label>
            <input id="score-date" type="date" value={scoreDate}
                   onChange={(e) => setScoreDate(e.target.value)} />
          </div>
          <button disabled={busy || !scoreDate}
                  onClick={() => run(async () => {
                    const r = await api.score(scoreDate)
                    const counts = Object.entries(r.counts)
                      .map(([k, v]) => `${k} ${v}`).join(', ')
                    // frozen rows cannot be re-scored, so the items are named rather than
                    // left inside a missing_data count nobody expands
                    return `${r.for_date}: ${counts}.` + (r.note ? ` ${r.note}` : '')
                  })}>
            Score it
          </button>
        </div>
        <p className="hint">
          A day&apos;s verdict is frozen once. If its sales data has not arrived, this refuses
          rather than freezing an empty one — key the returned sheet in first, and let the
          export land.
        </p>
      </div>

      {status.missed_sheet_dates.length ? (
        <div className="card">
          <h2>Mornings with no sheet</h2>
          <p>{status.missed_sheet_dates.join(', ')}</p>
          <p className="hint">
            These count against completeness exactly like a day the export never explained.
          </p>
        </div>
      ) : null}

      {Object.keys(status.last_gates ?? {}).length ? (
        <div className="card">
          <h2>Gates, as of the last weekly report</h2>
          <Gates gates={status.last_gates} />
        </div>
      ) : null}
    </>
  )
}
