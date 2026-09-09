import { useCallback, useEffect, useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { MorningPayload, StatusPayload } from '../api/types'
import { Banner } from '../components/Banner'
import { longDate, qty } from '../lib/format'

/**
 * The morning sheet: what to make, beside what the store's own par sheet says.
 *
 * Two behaviours here are not styling decisions.
 *
 * Reading a sheet and making one are different buttons. Opening this page GETs; it never
 * logs. Only "Make today's sheet" posts, because the prediction log is append-only and the
 * last run for a date becomes the live one -- so a page that logged on load would let a
 * refresh at 2pm quietly replace what the model said at 5:30am.
 *
 * A sheet that already exists is never silently replaced. The server answers 409 and this
 * page asks, naming the run it would supersede.
 */
export function Today({ status, onChange }: {
  status: StatusPayload | null
  onChange: () => void
}) {
  const [date, setDate] = useState('')
  const [sheet, setSheet] = useState<MorningPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [conflict, setConflict] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [needsBackfill, setNeedsBackfill] = useState(false)

  useEffect(() => { if (status?.today && !date) setDate(status.today) }, [status, date])

  const load = useCallback(async (d: string) => {
    if (!d) return
    setError(null); setConflict(null); setNeedsBackfill(false)
    try {
      setSheet(await api.morning(d))
    } catch (err) {
      setSheet(null)
      if (err instanceof ApiFailure && err.status === 404) setError(null)
      else if (err instanceof ApiFailure) setError(err.message)
      else if (err instanceof ApiUnreachable) setError(err.message)
    }
  }, [])

  useEffect(() => { void load(date) }, [date, load])

  async function make(opts: { backfill?: boolean; reforecast?: boolean } = {}) {
    setBusy(true); setError(null); setConflict(null)
    try {
      setSheet(await api.makeMorning(date, opts))
      setNeedsBackfill(false)
      onChange()
    } catch (err) {
      if (err instanceof ApiFailure && err.isConflict) setConflict(err.message)
      else if (err instanceof ApiFailure) {
        setError(err.message)
        // the forecaster's own words name the flag; offering it is not the same as taking it
        setNeedsBackfill(err.message.includes('--backfill'))
      } else if (err instanceof ApiUnreachable) setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const rows = sheet?.rows ?? []

  return (
    <>
      <div className="card no-print">
        <h2>Morning sheet</h2>
        <div className="row">
          <div className="field">
            <label htmlFor="today-date">Date on the sheet</label>
            <input id="today-date" type="date" value={date}
                   onChange={(e) => setDate(e.target.value)} />
          </div>
          {sheet ? (
            <button onClick={() => window.print()}>Print</button>
          ) : (
            <button className="primary" disabled={busy || !date} onClick={() => make()}>
              {busy ? 'Forecasting…' : "Make this day's sheet"}
            </button>
          )}
          {sheet ? (
            <a href={api.sheetUrl(date, 'txt')} target="_blank" rel="noreferrer">
              plain text
            </a>
          ) : null}
        </div>
        {status?.today && date !== status.today ? (
          <p className="hint">
            The store&apos;s today is {status.today}. You are looking at {date}.
          </p>
        ) : null}
      </div>

      {error ? <Banner kind="danger" title="The forecast was refused">{error}</Banner> : null}

      {needsBackfill ? (
        <div className="card no-print">
          <p>
            This date is already covered by the store&apos;s sales data, so a sheet for it
            would be made after the fact. That is allowed for a replay, and the record stamps
            every row <strong>backfilled</strong> so the weekly page leaves it out of the
            headline numbers.
          </p>
          <button onClick={() => make({ backfill: true })} disabled={busy}>
            Make it anyway, stamped backfilled
          </button>
        </div>
      ) : null}

      {conflict ? (
        <div className="card no-print">
          <Banner kind="warn" title="A sheet already exists for this day">{conflict}</Banner>
          <button onClick={() => make({ reforecast: true })} disabled={busy}>
            Replace it — I understand this supersedes the earlier sheet
          </button>
        </div>
      ) : null}

      {!sheet && !error ? (
        <p className="hint">No sheet has been made for {date || 'this day'} yet.</p>
      ) : null}

      {sheet ? (
        <>
          {sheet.backfilled ? (
            <Banner kind="warn" title="BACKFILLED — made after the day it is for">
              This sheet was not in anyone&apos;s hands that morning, so it proves nothing
              about what was known. Every row is stamped, and the weekly page excludes it
              unless it is explicitly asked to reconstruct.
            </Banner>
          ) : null}

          {sheet.superseded ? (
            <Banner kind="warn" title="This day has more than one sheet">
              {(sheet.runs ?? []).length} runs were logged for {sheet.for_date}. The live one
              is <code>{sheet.live_run_id}</code>, made{' '}
              {(sheet.runs ?? [])[(sheet.runs ?? []).length - 1]?.made_at}.
              The earlier ones stay in the record underneath it.
            </Banner>
          ) : null}

          {(sheet.caveats ?? []).map((c) => (
            <Banner key={c} kind="warn">{c}</Banner>
          ))}

          <div className="card">
            <h2>{longDate(sheet.for_date)}</h2>
            <p className="hint">
              {sheet.conditions?.weather}
              {sheet.conditions?.tmax_f != null ? `, high ${Math.round(sheet.conditions.tmax_f)}°F` : ''}
              {sheet.conditions?.holiday ? ` · ${sheet.conditions.holiday}` : ''}
              {sheet.conditions?.payday ? ' · payday' : ''}
              {sheet.conditions?.snow_tomorrow ? ' · snow expected tomorrow' : ''}
            </p>
            <Banner kind="info" title="Shadow mode">
              Do not change what you make today. Write what you actually made on this paper.
            </Banner>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Item</th>
                    <th className="num">Suggested</th>
                    <th className="num">Your par</th>
                    <th>Why</th>
                    <th className="num">Made</th>
                    <th>Sold out at</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.item}
                        className={`sheet-row ${r.source !== 'model' ? 'fallback' : ''}`}>
                      <td>{r.item_name}</td>
                      <td className="num">{qty(r.rec_qty, r.unit)}</td>
                      <td className="num">{qty(r.par_qty, r.unit)}</td>
                      <td className="why">
                        {r.source === 'model' ? r.why_text
                          : `no forecast — ${r.fallback_reason || 'your trailing par'}`}
                      </td>
                      <td className="num">&nbsp;</td>
                      <td>&nbsp;</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">
              Rows shaded amber had no model forecast and show your own trailing par instead —
              they are not the model speaking.
            </p>
          </div>
        </>
      ) : null}
    </>
  )
}
