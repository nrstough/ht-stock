import { pct } from '../lib/format'

export interface WeekPoint {
  week: string
  model: number | null
  par: number | null
}

/**
 * Model error against the store's own par sheet, one pair of columns per shadow week.
 *
 * This is the picture behind G2, which asks two things a table answers badly: is the model's
 * error below par's, and is it below in at least three weeks of four. Paired columns put
 * that comparison inside one glance per week.
 *
 * Deliberate choices, in the order the dataviz method takes them:
 *
 *   Form. Two series, at most four groups -- a grouped column chart. Not a line: four points
 *   is not a trend, and connecting them would imply a trajectory nobody has measured.
 *
 *   Colour. Two categorical slots, --series-model and --series-par, validated against this
 *   app's own light and dark surfaces (see styles/tokens.css). Never re-derived here.
 *
 *   Labels. Every column carries its value and there are no y-axis ticks: with eight marks
 *   the numbers ARE the axis, and a reader comparing 13.0% against 15.5% should not have to
 *   interpolate against gridlines. Identity is legend AND label, never colour alone.
 *
 * Lower is better, because this is error. The subtitle says so, because a chart where the
 * short bar wins is the one a reader most reliably gets backwards.
 */
export function AccuracyChart({ weeks }: { weeks: WeekPoint[] }) {
  const points = weeks.filter((w) => w.model !== null || w.par !== null)
  if (!points.length) {
    return (
      <p className="hint">
        No week has enough scored rows to compare accuracy yet.
      </p>
    )
  }

  const values = points.flatMap((w) => [w.model, w.par]).filter(
    (v): v is number => v !== null && Number.isFinite(v),
  )
  const max = Math.max(...values, 0.01)

  // geometry, in a viewBox so the chart scales without re-measuring
  const W = 640
  const H = 210
  const padL = 8
  const padR = 8
  const padTop = 26          // room for the value labels above the tallest column
  const baseline = H - 34    // room for the week label under it
  const groups = points.length
  const groupW = (W - padL - padR) / groups
  const barW = Math.min(46, (groupW - 18) / 2)
  const gap = 2              // the surface gap the method asks for between adjacent fills
  const scale = (v: number) => ((baseline - padTop) * v) / max

  return (
    <figure className="chart">
      <figcaption>
        <span className="chart-title">Forecast error by shadow week</span>
        <span className="chart-sub">
          WAPE on days demand was fully served. <strong>Lower is better.</strong>
        </span>
      </figcaption>

      <div className="legend" aria-hidden="true">
        <span><i style={{ background: 'var(--series-model)' }} /> Model</span>
        <span><i style={{ background: 'var(--series-par)' }} /> Your par sheet</span>
      </div>

      <div className="scroll-x">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" width="100%"
             aria-label={`Forecast error by week. ${points.map((w) => (
               `${w.week}: model ${pct(w.model)}, par ${pct(w.par)}`)).join('. ')}`}>
          <line x1={padL} y1={baseline} x2={W - padR} y2={baseline}
                stroke="var(--border)" strokeWidth="1" />
          {points.map((w, i) => {
            const cx = padL + groupW * i + groupW / 2
            const xModel = cx - barW - gap / 2
            const xPar = cx + gap / 2
            const series: [string, number | null, string][] = [
              ['model', w.model, xModel.toString()],
              ['par', w.par, xPar.toString()],
            ]
            return (
              <g key={w.week}>
                {series.map(([name, value, x]) => {
                  if (value === null || !Number.isFinite(value)) return null
                  const h = Math.max(scale(value), 2)
                  return (
                    <g key={name}>
                      <rect x={Number(x)} y={baseline - h} width={barW} height={h} rx={4}
                            fill={`var(--series-${name})`}>
                        <title>{`${w.week} ${name === 'model' ? 'model' : 'par'}: ${pct(value)}`}</title>
                      </rect>
                      <text x={Number(x) + barW / 2} y={baseline - h - 7}
                            textAnchor="middle" fontSize="12"
                            fill="var(--text)" fontWeight="600">
                        {pct(value, 0)}
                      </text>
                    </g>
                  )
                })}
                <text x={cx} y={baseline + 18} textAnchor="middle" fontSize="12"
                      fill="var(--muted)">
                  {w.week}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      <table className="visually-detailed">
        <caption className="hint">The same numbers, for a screen reader or a copy-paste.</caption>
        <thead>
          <tr><th>Week</th><th className="num">Model</th><th className="num">Par</th></tr>
        </thead>
        <tbody>
          {points.map((w) => (
            <tr key={w.week}>
              <td>{w.week}</td>
              <td className="num">{pct(w.model)}</td>
              <td className="num">{pct(w.par)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
