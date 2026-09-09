import { useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { StatusPayload } from '../api/types'
import { Banner } from '../components/Banner'
import { Gates } from '../components/Gates'

/**
 * What is behind, and the two ways to settle a day.
 *
 * Scoring freezes a verdict once, so the server refuses a day whose sales data has not
 * arrived -- scoring it then would freeze an empty verdict that catch-up cannot repair and
 * that counts against completeness for the rest of the pilot. Catch-up is offered first
 * because it does the safe thing by construction: it scores every day that has data and
 * skips the ones that do not.
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
                    return `Scored: ${r.scored.join(', ') || 'nothing new'}. `
                      + `Still waiting on data: ${r.unscored.join(', ') || 'none'}.`
                  })}>
            {busy ? 'Working…' : 'Catch up'}
          </button>
          <span className="hint">
            Scores every day that has sales data and skips the ones that do not.
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
                    return `${r.for_date}: `
                      + Object.entries(r.counts).map(([k, v]) => `${k} ${v}`).join(', ')
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
