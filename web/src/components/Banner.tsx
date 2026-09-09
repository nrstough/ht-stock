import type { ReactNode } from 'react'

/**
 * A stamp the record carries is never a tooltip.
 *
 * If a sheet was made after the day it is for, replaced an earlier one, or rests on a
 * carried-forward calendar, the person reading it sees that before they read a quantity.
 */
export function Banner(
  { kind = 'info', title, children }:
  { kind?: 'info' | 'warn' | 'danger'; title?: string; children: ReactNode },
) {
  return (
    <div className={`banner ${kind}`} role={kind === 'info' ? undefined : 'alert'}>
      {title ? <strong>{title}</strong> : null}
      {children}
    </div>
  )
}
