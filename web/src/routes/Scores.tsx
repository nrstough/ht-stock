import { useEffect, useState } from 'react'
import { ApiFailure, ApiUnreachable, api } from '../api/client'
import type { StatusPayload } from '../api/types'
import { Banner } from '../components/Banner'
import { dayLabel, qty } from '../lib/format'

type Row = Record<string, unknown>

/** The frozen verdicts, day by day. Read-only: scoring happens on the Status page. */
export function Scores({ status }: { status: StatusPayload | null }) {
  const [rows, setRows] = useState<Row[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    api.scores()
      .then((r) => { if (live) { setRows(r.rows); setError(null) } })
      .catch((err) => {
        if (!live) return
        setError(err instanceof ApiFailure || err instanceof ApiUnreachable
          ? err.message : String(err))
      })
    return () => { live = false }
  }, [status?.last_scored_date])

  const byDate = new Map<string, Row[]>()
  for (const r of rows) {
    const d = String(r.for_date ?? '')
    byDate.set(d, [...(byDate.get(d) ?? []), r])
  }
  const dates = [...byDate.keys()].sort().reverse()

  return (
    <>
      {error ? <Banner kind="danger">{error}</Banner> : null}
      {!rows.length && !error ? (
        <p className="hint">Nothing has been scored yet.</p>
      ) : null}
      {dates.map((d) => {
        const day = byDate.get(d)!
        const counts = day.reduce<Record<string, number>>((acc, r) => {
          const s = String(r.status ?? '')
          acc[s] = (acc[s] ?? 0) + 1
          return acc
        }, {})
        return (
          <div className="card" key={d}>
            <h2>{dayLabel(d)}</h2>
            <p className="hint">
              {Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </p>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Item</th>
                    <th className="num">Said</th>
                    <th className="num">Sold</th>
                    <th className="num">Made</th>
                    <th className="num">Error</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {day.map((r) => (
                    <tr key={String(r.item)}>
                      <td>{String(r.item)}</td>
                      <td className="num">{qty(r.rec_qty as number | null)}</td>
                      <td className="num">{qty(r.sold as number | null)}</td>
                      <td className="num">{qty(r.produced as number | null)}</td>
                      <td className="num">{qty(r.abs_err as number | null)}</td>
                      <td>{String(r.status ?? '')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </>
  )
}
