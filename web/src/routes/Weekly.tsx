import { useEffect, useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { StatusPayload, WeeklyPayload } from '../api/types'
import { AccuracyChart } from '../components/AccuracyChart'
import { Banner } from '../components/Banner'
import { Gates } from '../components/Gates'
import { pct, shiftDays } from '../lib/format'

/**
 * The page a district manager reads, and the five criteria fixed in advance.
 *
 * Making a report writes it to disk, so it is a button rather than something this page does
 * on load. Opening the page lists what has already been written.
 */
export function Weekly({ status, onChange }: {
  status: StatusPayload | null
  onChange: () => void
}) {
  const [weekEnding, setWeekEnding] = useState('')
  const [includeBackfilled, setIncludeBackfilled] = useState(false)
  const [report, setReport] = useState<WeeklyPayload | null>(null)
  const [written, setWritten] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (status?.today && !weekEnding) {
      // the most recent Sunday on or before today
      const d = new Date(`${status.today}T00:00:00Z`)
      setWeekEnding(shiftDays(status.today, -(d.getUTCDay() % 7)))
    }
  }, [status, weekEnding])

  useEffect(() => {
    let live = true
    api.weeks().then((w) => { if (live) setWritten(w.weeks) }).catch(() => {})
    return () => { live = false }
  }, [report])

  async function make() {
    setBusy(true); setError(null)
    try {
      setReport(await api.makeWeekly(weekEnding, { include_backfilled: includeBackfilled }))
      onChange()
    } catch (err) {
      setError(err instanceof ApiFailure || err instanceof ApiUnreachable
        ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const weeks = (report?.accuracy?.weekly ?? []).map((w) => ({
    week: w.week, model: w.model, par: w.par,
  }))

  return (
    <>
      <div className="card no-print">
        <h2>Weekly report</h2>
        <div className="row">
          <div className="field">
            <label htmlFor="week-ending">Week ending</label>
            <input id="week-ending" type="date" value={weekEnding}
                   onChange={(e) => setWeekEnding(e.target.value)} />
          </div>
          <button className="primary" onClick={make} disabled={busy || !weekEnding}>
            {busy ? 'Building…' : 'Build the report'}
          </button>
          <label className="hint">
            <input type="checkbox" checked={includeBackfilled}
                   onChange={(e) => setIncludeBackfilled(e.target.checked)} />
            {' '}include backfilled sheets
          </label>
        </div>
        <p className="hint">
          Backfilled sheets are left out by default: one made after the day it is for proves
          nothing about what was known that morning. Including them stamps the page
          RECONSTRUCTED — that is the dress-rehearsal mode, not a pilot week.
        </p>
        {written.length ? (
          <p className="hint">Already written: {written.join(', ')}</p>
        ) : null}
      </div>

      {error ? <Banner kind="danger">{error}</Banner> : null}

      {report ? (
        <>
          {report.reconstructed ? (
            <Banner kind="warn" title="RECONSTRUCTED">
              This page folds in sheets that were replayed from history rather than printed on
              the morning they are for. It is not the mode a real pilot week is read in.
            </Banner>
          ) : null}

          {(report.caveats ?? []).map((c) => <Banner key={c} kind="warn">{c}</Banner>)}

          <div className="card">
            <h2>{report.week} · {report.week_start} to {report.week_end}</h2>
            <Gates gates={report.gates} />
            <p className="hint">
              PENDING is not a pass — it means the criterion could not be measured yet.
            </p>
          </div>

          <div className="card">
            <h2>Coverage</h2>
            <dl className="kv">
              <dt>Completeness</dt><dd>{pct(report.completeness)}</dd>
              <dt>Rows scored</dt>
              <dd>{report.n_rows_scored} of {report.n_rows_expected} expected</dd>
              <dt>Days covered</dt>
              <dd>{report.days_covered} of {report.days_expected}</dd>
              {report.backfilled_rows ? (
                <><dt>Backfilled rows</dt><dd>{report.backfilled_rows}</dd></>
              ) : null}
              {report.revisions ? (
                <><dt>Disclosed revisions</dt><dd>{report.revisions}</dd></>
              ) : null}
            </dl>
            {report.missing_sheets.length ? (
              <p className="hint">
                No sheet was printed on: {report.missing_sheets.join(', ')}
              </p>
            ) : null}
            {report.data_gaps.length ? (
              <p className="hint">
                The export never explained: {report.data_gaps.join(', ')}
              </p>
            ) : null}
          </div>

          {weeks.length ? (
            <div className="card">
              <AccuracyChart weeks={weeks} />
            </div>
          ) : (
            <div className="card">
              <p className="hint">
                No week has enough scored rows to compare the model against your par sheet yet.
              </p>
            </div>
          )}
        </>
      ) : null}
    </>
  )
}
