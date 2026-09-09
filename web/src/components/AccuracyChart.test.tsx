import { cleanup, render, screen } from '@testing-library/react'
import { afterEach } from 'vitest'
import { describe, expect, it } from 'vitest'
import { AccuracyChart } from './AccuracyChart'

afterEach(cleanup)

describe('the accuracy chart', () => {
  const weeks = [
    { week: '2026-W01', model: 0.13, par: 0.155 },
    { week: '2026-W02', model: 0.142, par: 0.151 },
  ]

  it('says which direction is good, because a chart where the short bar wins gets misread', () => {
    render(<AccuracyChart weeks={weeks} />)
    expect(screen.getByText(/Lower is better/)).toBeInTheDocument()
  })

  it('carries a legend AND the values, so identity is never colour alone', () => {
    const { container } = render(<AccuracyChart weeks={weeks} />)
    const legend = container.querySelector('.legend')!
    expect(legend).toHaveTextContent('Model')
    expect(legend).toHaveTextContent('Your par sheet')
    // and the value is on the mark, so the reader never interpolates against a gridline
    expect(screen.getAllByText('13%').length).toBeGreaterThan(0)
  })

  it('offers the same numbers as a table for a screen reader', () => {
    render(<AccuracyChart weeks={weeks} />)
    const table = screen.getByRole('table')
    expect(table).toHaveTextContent('2026-W01')
    expect(table).toHaveTextContent('15.5%')
  })

  it('draws a bar only for a series that has a value', () => {
    const { container } = render(
      <AccuracyChart weeks={[{ week: '2026-W03', model: 0.12, par: null }]} />,
    )
    // one bar, not a par bar sitting at zero -- a missing measurement is not a good one
    expect(container.querySelectorAll('rect')).toHaveLength(1)
  })

  it('says so plainly when no week can be compared yet', () => {
    render(<AccuracyChart weeks={[{ week: '2026-W04', model: null, par: null }]} />)
    expect(screen.getByText(/No week has enough scored rows/)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })
})
