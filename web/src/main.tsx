import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { App } from './App'
import './styles/tokens.css'
import './styles/app.css'

// StrictMode double-invokes effects in development. That is not a hazard here and it is
// worth saying why: no GET in this API writes, and the one action that logs a forecast is a
// POST the operator presses. An effect that fired twice would read twice, not log twice.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
