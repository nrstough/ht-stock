import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Gates } from './Gates'
import { verdictClass } from '../lib/format'

// PENDING is not PASS. A criterion nobody has measured yet, rendered as one that was met, is
// the single most misleading thing this app could put in front of a district manager -- so
// the three states are pinned here in every form they take: the class, the word, and the
// absence of any path from an unknown verdict to a passing one.

describe('the five go/no-go criteria', () => {
  const all = { G1: 'PASS', G2: 'FAIL', G3: 'PENDING', G4: 'PENDING', G5: 'PASS' } as const

  it('renders one row per gate, in order', () => {
    render(<Gates gates={all} />)
    const ids = screen.getAllByText(/^G[1-5]$/).map((el) => el.textContent)
    expect(ids).toEqual(['G1', 'G2', 'G3', 'G4', 'G5'])
  })

  it('shows the word beside the colour, so state never depends on hue alone', () => {
    render(<Gates gates={all} />)
    expect(screen.getAllByText('PENDING')).toHaveLength(2)
    expect(screen.getAllByText('PASS')).toHaveLength(2)
    expect(screen.getByText('FAIL')).toBeInTheDocument()
  })

  it('gives PENDING its own class, never the passing one', () => {
    render(<Gates gates={all} />)
    for (const el of screen.getAllByText('PENDING')) {
      expect(el).toHaveClass('pending')
      expect(el).not.toHaveClass('pass')
    }
  })

  it('names what each gate asks, so a verdict is not a bare letter', () => {
    render(<Gates gates={all} />)
    expect(screen.getByText('Completeness')).toBeInTheDocument()
    expect(screen.getByText('Accuracy vs par')).toBeInTheDocument()
    expect(screen.getByText('Economics')).toBeInTheDocument()
  })

  it('renders nothing when there are no gates rather than an empty shell', () => {
    const { container } = render(<Gates gates={{}} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('verdictClass', () => {
  it('maps only the three states the server sends', () => {
    expect(verdictClass('PASS')).toBe('pass')
    expect(verdictClass('FAIL')).toBe('fail')
    expect(verdictClass('PENDING')).toBe('pending')
  })

  it('falls back to pending, never to pass', () => {
    // an unknown verdict must never arrive at the reassuring colour
    for (const odd of [undefined, null, '', 'ok', 'true', 'PASSED', 'pass']) {
      expect(verdictClass(odd)).toBe('pending')
    }
  })
})
