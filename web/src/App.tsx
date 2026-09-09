import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { ApiUnreachable, api } from './api/client'
import type { StatusPayload } from './api/types'
import { Banner } from './components/Banner'
import { Today } from './routes/Today'
import { Enter } from './routes/Enter'
import { Scores } from './routes/Scores'
import { Weekly } from './routes/Weekly'
import { Status } from './routes/Status'

const TABS = [
  { to: '/today', label: 'Today' },
  { to: '/enter', label: 'Enter' },
  { to: '/scores', label: 'Scores' },
  { to: '/weekly', label: 'Weekly' },
  { to: '/status', label: 'Status' },
]

/**
 * The shell, and the one decision it makes: whether the operator routes work at all.
 *
 * Everything here needs `python -m ht.serve` running. If it is not, the pages are disabled
 * with the reason rather than rendered empty -- a blank sheet reads as "nothing to make
 * today", which is the wrong answer to "the service is down" and the kind of thing a store
 * acts on before anyone notices.
 */
export function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null)
  const [offline, setOffline] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let live = true
    api.status()
      .then((s) => { if (live) { setStatus(s); setOffline(null) } })
      .catch((err) => {
        if (!live) return
        // a 400 here means the shadow directory is missing, which is a real answer from a
        // live server; only an unreachable service disables the app
        if (err instanceof ApiUnreachable) setOffline(err.message)
        else setOffline(null)
      })
    return () => { live = false }
  }, [tick])

  const refresh = () => setTick((n) => n + 1)

  if (offline) {
    return (
      <div className="offline">
        <h1>Not connected</h1>
        <p>{offline}</p>
        <pre>python -m ht.serve --panel PANEL.csv --items ITEMS.json \
  --artifacts ARTIFACTS/ --timezone America/Chicago</pre>
        <button onClick={refresh}>Try again</button>
      </div>
    )
  }

  return (
    <>
      <header className="app">
        <h1>Shadow mode</h1>
        <span className="sub">
          {status?.store ? `${status.store} · ` : ''}
          {status?.today ? `today is ${status.today}` : 'connecting…'}
        </span>
      </header>

      <main className="wrap">
        {status?.new_shadow_dir ? (
          <Banner kind="warn" title="This is a brand new pilot record">
            {status.shadow_dir} was created just now and holds nothing yet. If you expected an
            existing pilot here, stop and check the path the service was started with — a
            mistyped one looks exactly like a pilot that lost its record.
          </Banner>
        ) : null}

        <Routes>
          <Route path="/" element={<Navigate to="/today" replace />} />
          <Route path="/today" element={<Today status={status} onChange={refresh} />} />
          <Route path="/enter" element={<Enter status={status} onChange={refresh} />} />
          <Route path="/scores" element={<Scores status={status} />} />
          <Route path="/weekly" element={<Weekly status={status} onChange={refresh} />} />
          <Route path="/status" element={<Status status={status} onChange={refresh} />} />
        </Routes>
      </main>

      <nav className="tabs">
        {TABS.map((t) => (
          <NavLink key={t.to} to={t.to}
                   className={({ isActive }) => (isActive ? 'active' : undefined)}>
            {t.label}
          </NavLink>
        ))}
      </nav>
    </>
  )
}
