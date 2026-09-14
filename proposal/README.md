# The pitch

Three documents, one set of numbers. Every dollar figure in all three comes from the frozen
`results/results.json`, which `python -m model.backtest` reproduces byte for byte.

| File | Audience | How it is made |
|---|---|---|
| `Fresh-Forecast-Leadership-Deck.pptx` | district and division leadership, presented in a room | generated: `node tools/build_deck_pptx.js` (needs `pptxgenjs` on `NODE_PATH`) |
| `deck.html` | the same deck, opened in a browser or printed to PDF with `P` | generated: `python tools/build_deck.py` |
| `executive-proposal.html` | the leave-behind: the memo leadership reads after the meeting | hand-written against the frozen results |
| `Fresh-Forecast-Leadership-Deck.pdf` | emailing the deck, or presenting where PowerPoint is not available | printed from `deck.html` at 1280x720 |
| `Fresh-Forecast-Executive-Proposal.pdf` | emailing or printing the memo | printed from `executive-proposal.html` on letter paper |
| `proposal.html` | the original store-level proposal, written to a store manager | hand-written against the frozen results |

Both PDFs are printed from the HTML with a headless browser, which is what the two print
stylesheets are written for. Any browser's own print-to-PDF produces the same thing:

```
deck.html                -> 1280x720 page size, backgrounds on   -> one slide per page
executive-proposal.html  -> letter portrait, backgrounds on      -> four pages
```

The generated files are checked in so the pitch can be opened without a build step. Both
generators read `results/results.json` at build time, so the slides cannot drift from the backtest
they describe; the two hand-written memos carry the same figures and are the thing to re-check if
the frozen results are ever regenerated with `--force-frozen`.

The deck is seventeen slides: a one-slide summary, the problem (demand structure the par sheet
cannot see), what already exists, the proof of concept and its results, an honesty slide on what
the simulation does and does not prove, the 90-day pilot with its week-6 stop gate, the three asks,
the scale table, risks, the decision, and two appendix slides for the analyst in the room. Speaker
notes are on every slide of the `.pptx`.

Placeholders left deliberately: the memo's `[name]`, `[role]` and `[store]`, and every dollar figure
that the pilot's measured baseline is meant to replace. Nothing in these documents has seen real
store data, and the documents say so.
