// The one place that knows the endpoints. Everything else asks this.
//
// Two rules it enforces on behalf of the record, both of which exist because a browser can
// fire a request the person did not mean to send:
//
//   a GET never writes, so nothing here retries a POST automatically. A retried
//   POST /api/morning would be a second run for the same date, and read_predictions keeps
//   the LAST one -- which is how the morning's forecast quietly becomes an afternoon's.
//
//   a 409 is not a failure to paper over. It is the server saying "this date already has a
//   sheet", and the caller has to decide, in front of the person, whether to supersede it.

import type {
  ApiError, ConfigPayload, EntryOrderPayload, MorningPayload, StatusPayload, WeeklyPayload,
} from './types'

export class ApiFailure extends Error {
  status: number
  detail: ApiError

  constructor(status: number, detail: ApiError) {
    super(detail.error || `request failed (${status})`)
    this.status = status
    this.detail = detail
  }

  /** The server refused because the date already carries a logged sheet. */
  get isConflict(): boolean {
    return this.status === 409
  }
}

/** Raised when /api cannot be reached at all -- the operator routes then disable themselves. */
export class ApiUnreachable extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiUnreachable(
      'The forecast service is not running. Start it with python -m ht.serve.',
    )
  }
  const text = await res.text()
  let body: unknown = {}
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { error: text.slice(0, 400) }
    }
  }
  if (!res.ok) throw new ApiFailure(res.status, body as ApiError)
  return body as T
}

const get = <T>(path: string) => request<T>(path)
const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) })

const qs = (params: Record<string, string | number | undefined>) => {
  const out = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') out.set(k, String(v))
  const s = out.toString()
  return s ? `?${s}` : ''
}

export const api = {
  status: () => get<StatusPayload>('/api/status'),
  config: () => get<ConfigPayload>('/api/config'),

  /** Read the sheet that was logged for a date. Never logs one. */
  morning: (date: string) => get<MorningPayload>(`/api/morning${qs({ date })}`),

  /**
   * Make and log a sheet. `reforecast` is deliberate and never a retry: it replaces what the
   * morning said, and the API refuses without it once a date has a run.
   */
  makeMorning: (date: string, opts: { backfill?: boolean; reforecast?: boolean } = {}) =>
    post<MorningPayload>('/api/morning', { date, ...opts }),

  sheetUrl: (date: string, ext: 'txt' | 'html') => `/api/sheet/${date}.${ext}`,

  entryOrder: (date: string) => get<EntryOrderPayload>(`/api/entry-order${qs({ date })}`),

  enter: (date: string, lines: string[], by?: string) =>
    post<{ written: number; entered_by: string; warning: string | null }>(
      '/api/enter', { date, lines, by }),

  score: (date: string) =>
    post<{ for_date: string; counts: Record<string, number>; rows: number }>(
      '/api/score', { date }),

  catchUp: (since?: string) =>
    post<{ scored: string[]; unscored: string[] }>('/api/catch-up', { since }),

  makeWeekly: (week_ending: string, opts: { weeks?: number; include_backfilled?: boolean } = {}) =>
    post<WeeklyPayload>('/api/weekly', { week_ending, ...opts }),

  weeks: () => get<{ weeks: string[] }>('/api/weekly'),
  weekly: (week: string) => get<WeeklyPayload>(`/api/weekly${qs({ week })}`),

  scores: (from?: string, to?: string) =>
    get<{ rows: Record<string, unknown>[] }>(`/api/scores${qs({ from, to })}`),
}
