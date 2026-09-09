// Mirrors what ht/serve.py returns, which in turn mirrors model/shadow.py's own columns.
// Every numeric field is `number | null`: the server serialises NaN as null and never 0,
// because an item the checkpoint never saw carries NaN quantiles by design and a zero there
// would print "make none" against a real item on a real morning.

export type Verdict = 'PASS' | 'FAIL' | 'PENDING'

export interface Run {
  run_id: string
  made_at: string
  backfilled: number
}

export interface PredictionRow {
  item: string
  item_name: string
  dept: string
  rec_qty: number | null
  par_qty: number | null
  batch: number | null
  unit: string
  source: 'model' | 'par_fallback' | string
  fallback_reason: string
  why_text: string
  backfilled: number
  'q_0.20'?: number | null
  'q_0.50'?: number | null
  'q_0.90'?: number | null
}

export interface Conditions {
  tmax_f: number | null
  weather: string
  holiday: string
  payday: boolean
  snow_tomorrow: boolean
}

export interface MorningPayload {
  for_date: string
  rows: PredictionRow[]
  runs: Run[]
  superseded: boolean
  live_run_id?: string
  backfilled?: number
  conditions: Conditions
  caveats: string[]
  yesterday: Record<string, unknown>[]
  day_source: string
  staleness_days: number | null
}

export interface EntryRow {
  item: string
  name: string
  dept: string
  unit: string
  said: number | null
  source: string | null
}

export interface EntryOrderPayload {
  for_date: string
  rows: EntryRow[]
  no_sheet_logged: boolean
  entered_by: string
  warning?: string
}

export interface StatusPayload {
  last_ingested_date: string | null
  last_sheet_date: string | null
  last_scored_date: string | null
  missed_sheet_dates: string[]
  unscored_dates: string[]
  current_model_version: string | null
  current_artifacts_dir: string | null
  sellout_source: string | null
  last_gates: Record<string, Verdict>
  today: string
  store: string
  new_shadow_dir: boolean
  shadow_dir: string
}

export interface ConfigPayload {
  store: string
  today: string
  timezone: string
  entered_by: string
  items: Record<string, { name: string; dept: string; unit?: string; batch?: number }>
  panel_first_date: string
  panel_last_date: string
  model_version: string
  max_staleness: number
}

export interface WeeklyPayload {
  week: string
  week_start: string
  week_end: string
  store: string
  gates: Record<string, Verdict>
  completeness: number | null
  days_expected: number
  days_covered: number
  n_rows_expected: number
  n_rows_scored: number
  backfilled_rows: number
  reconstructed: boolean
  missing_sheets: string[]
  unscored: string[]
  data_gaps: string[]
  revisions: number
  caveats?: string[]
  accuracy?: {
    model?: { wape_uncensored: number | null }
    par?: { wape_uncensored: number | null }
    weekly?: { week: string; model: number | null; par: number | null }[]
  }
  exclusions?: Record<string, number>
}

export interface ApiError {
  error: string
  errors?: string[]
  runs?: Run[]
}
