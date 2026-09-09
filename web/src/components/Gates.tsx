import type { Verdict } from '../api/types'
import { GATE_TITLES, verdictClass } from '../lib/format'

/**
 * The five go/no-go criteria, fixed in advance and printed from week one.
 *
 * PENDING is not PASS. A criterion nobody has measured yet gets its own neutral -- never a
 * paler green -- and the word is always shown beside the colour, so the state never depends
 * on hue alone.
 */
export function Gates({ gates }: { gates: Record<string, Verdict> }) {
  const keys = ['G1', 'G2', 'G3', 'G4', 'G5'].filter((k) => k in gates)
  if (!keys.length) return null
  return (
    <ul className="gates" aria-label="Go / no-go criteria">
      {keys.map((k) => (
        <li key={k}>
          <span className="gate-id">{k}</span>
          <span className="gate-title">{GATE_TITLES[k] ?? k}</span>
          <span className={`verdict ${verdictClass(gates[k])}`}>{gates[k]}</span>
        </li>
      ))}
    </ul>
  )
}
