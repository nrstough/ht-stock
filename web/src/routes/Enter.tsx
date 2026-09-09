import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { EntryOrderPayload, StatusPayload } from '../api/types'
import { Banner } from '../components/Banner'
import { parseEntries, toLine } from '../lib/parseEntries'
import { qty, shiftDays } from '../lib/format'

/**
 * Keying the returned paper sheet back in.
 *
 * The form walks the page in the order it printed -- departments, then carry-over, then no
 * forecast -- and shows what the sheet said beside each box, because the person holding the
 * paper is reading down it. Presenting a different order is how one item's production number
 * gets typed into another's.
 *
 * Nothing is sent unless every line parses. The browser checks first so a mistake shows up
 * next to the box that caused it, but the server parses again and its answer decides: a
 * half-keyed day that looks entered is worse than one that obviously is not.
 *
 * An empty "sold out at" box means "it did not sell out" -- and only here. That is the whole
 * reason this page exists for a store whose export has no sellout column, and it is why the
 * label says so rather than leaving the blank ambiguous.
 */
export function Enter({ status, onChange }: {
  status: StatusPayload | null
  onChange: () => void
}) {
  const [date, setDate] = useState('')
  const [order, setOrder] = useState<EntryOrderPayload | null>(null)
  const [made, setMade] = useState<Record<string, string>>({})
  const [soldOut, setSoldOut] = useState<Record<string, string>>({})
  const [by, setBy] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [serverErrors, setServerErrors] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // yesterday, not today: the sheet in hand is the one that came back from the kitchen
  useEffect(() => {
    if (status?.today && !date) setDate(shiftDays(status.today, -1))
  }, [status, date])

  const load = useCallback(async (d: string) => {
    if (!d) return
    setError(null); setResult(null); setServerErrors([])
    try {
      const payload = await api.entryOrder(d)
      setOrder(payload)
      setBy((b) => b || payload.entered_by)
      setMade({}); setSoldOut({})
    } catch (err) {
      setOrder(null)
      setError(err instanceof ApiFailure || err instanceof ApiUnreachable
        ? err.message : String(err))
    }
  }, [])

  useEffect(() => { void load(date) }, [date, load])

  const items = useMemo(() => Object.fromEntries(
    (order?.rows ?? []).map((r) => [r.item, { name: r.name }]),
  ), [order])

  const lines = useMemo(() => (order?.rows ?? [])
    .filter((r) => (made[r.item] ?? '').trim() || (soldOut[r.item] ?? '').trim())
    .map((r) => toLine(r.item, made[r.item] ?? '', soldOut[r.item] ?? '')),
  [order, made, soldOut])

  // the client's read, shown inline; the server's read is the one that decides
  const local = useMemo(() => parseEntries(lines, items), [lines, items])
  const badItems = useMemo(() => {
    const bad = new Set<string>()
    local.errors.forEach((e) => {
      const m = /^line (\d+):/.exec(e)
      if (m) {
        const line = lines[Number(m[1]) - 1]
        if (line) bad.add(line.split(',')[0])
      }
    })
    return bad
  }, [local, lines])

  async function submit() {
    setBusy(true); setError(null); setServerErrors([]); setResult(null)
    try {
      const res = await api.enter(date, lines, by)
      setResult(`${res.written} item(s) recorded by ${res.entered_by}.`
        + (res.warning ? ` ${res.warning}` : ''))
      onChange()
    } catch (err) {
      if (err instanceof ApiFailure) {
        setServerErrors(err.detail.errors ?? [])
        if (!err.detail.errors?.length) setError(err.message)
      } else if (err instanceof ApiUnreachable) setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="card">
        <h2>Key in the returned sheet</h2>
        <div className="row">
          <div className="field">
            <label htmlFor="enter-date">Date printed on the paper</label>
            <input id="enter-date" type="date" value={date}
                   onChange={(e) => setDate(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="enter-by">Keyed in by</label>
            <input id="enter-by" value={by} onChange={(e) => setBy(e.target.value)}
                   placeholder="your name" />
          </div>
        </div>
      </div>

      {error ? <Banner kind="danger">{error}</Banner> : null}

      {order?.no_sheet_logged ? (
        <Banner kind="warn" title="No sheet was logged for this date">
          {order.warning}
        </Banner>
      ) : null}

      {result ? <Banner kind="info" title="Recorded">{result}</Banner> : null}

      {serverErrors.length ? (
        <Banner kind="danger" title="Nothing was written. Fix these and send it again:">
          <ul>{serverErrors.map((e) => <li key={e}>{e}</li>)}</ul>
        </Banner>
      ) : null}

      {order ? (
        <div className="card">
          <p className="hint">
            Leave a box empty if nothing was written there. An empty <strong>sold out at</strong>{' '}
            means the item did <em>not</em> sell out — that is a real observation, and on a
            row keyed in here it is the only one the pilot will get.
          </p>
          <div className="entry-grid">
            {order.rows.map((r) => (
              <div key={r.item}
                   className={`entry-row ${badItems.has(r.item) ? 'bad' : ''}`}>
                <div className="name">
                  {r.name}
                  <span className="said">
                    {r.said != null
                      ? `${r.source === 'model' ? 'model said' : 'your par was'} ${qty(r.said, r.unit)}`
                      : 'not on the sheet'}
                  </span>
                </div>
                <input aria-label={`${r.name} made`} inputMode="decimal" placeholder="made"
                       value={made[r.item] ?? ''}
                       onChange={(e) => setMade({ ...made, [r.item]: e.target.value })} />
                <input aria-label={`${r.name} sold out at`} placeholder="sold out at"
                       value={soldOut[r.item] ?? ''}
                       onChange={(e) => setSoldOut({ ...soldOut, [r.item]: e.target.value })} />
              </div>
            ))}
          </div>

          {local.errors.length ? (
            <Banner kind="danger" title="Fix these before sending:">
              <ul>{local.errors.map((e) => <li key={e}>{e}</li>)}</ul>
            </Banner>
          ) : null}

          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" disabled={busy || !lines.length || !!local.errors.length}
                    onClick={submit}>
              {busy ? 'Recording…' : `Record ${local.rows.length} item(s)`}
            </button>
            <span className="hint">
              {lines.length ? '' : 'Type what the kitchen wrote on at least one row.'}
            </span>
          </div>
        </div>
      ) : null}
    </>
  )
}
