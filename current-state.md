# Current State — Predict-XI

**Status:** v8 shipped — squad market value, a corrected draw decision rule, and per-league
evaluation.

## Headline numbers (v8)

| | Argmax | Shipped rule | v7 |
|---|---|---|---|
| Accuracy (held-out) | 49.8% | **53.2%** | — |
| Macro F1 | 0.477 | **0.528** | 0.431 |
| Draw recall | 32.4% | **62.6%** | 23.3% |
| Accuracy (purged CV) | 46.1% | — | 46.0% |

68,228 matches · 30 leagues · 707 teams · 82 features · 40 MB model.

## What changed in v8

**The draw decision rule — the biggest single win, and it wasn't a feature.**
Argmax is the wrong rule for this problem: P(draw) never exceeds ~0.48 because a draw is rarely
the most likely *single* outcome even when it's the best call, so taking the largest probability
under-predicts draws regardless of calibration quality. Predicting Draw once P(draw) ≥ 0.35
gained +3.3pt accuracy and nearly doubled draw recall. Validated on 5 disjoint folds — every fold
improved.

Worth recording how this nearly went wrong: the first version tuned the threshold and scored it on
the *same* slice, showing a +3.6pt "gain" that was pure selection bias. Splitting tune/score
exposed it; the real effect then held up on a reversed split and a 5-fold check.

**Squad market value.** Teams carry Transfermarkt squad values, so signings move predictions —
with the weight learned from history, unlike the invented `CLUB_PRIOR` bonuses deleted in v3.
`squad_value_diff` is the 5th most important feature despite only 48% coverage.

**Per-league evaluation + calibration.** `evaluate.py` → `evaluation.json` → dashboard. Accuracy
ranges 61.9% (Bundesliga) to ~40% (Scottish Championship); the single global number was hiding an
~20-point spread.

## Data sources
- Match results: football-data.co.uk (main + "new leagues" feeds), footballcsv fallback.
- Squad values: `dcaribou/transfermarkt-datasets` (CC0-1.0, weekly refresh) — a published dataset,
  not scraped. Point-in-time for training, current for live predictions.

## Known limitations
- **Squad coverage 48%** — Transfermarkt covers first tiers only; our lower divisions (eng.2-5,
  es.2, de.2, it.2, fr.2, sco.2-4) have none. Flagged per-match via `has_squad_value`.
- **33 clubs unmapped**, explicitly listed in `club_mapping.KNOWN_ABSENT` (defunct, or never in the
  covered tier). Unmapped is deliberate: a wrong join would silently attach another club's value,
  so the residual was hand-verified rather than fuzzy-matched.
- **Transfer latency** — Transfermarkt revalues a few times a year, so a signing lands within
  weeks, not same-day.
- Two accuracy figures (CV vs held-out) measure different slices; both are published rather than
  only the flattering one.

## Next ideas
- Explainability on the Predict page — surface *why* (Elo edge, DC expected goals, squad gap)
  rather than only the probabilities.
- Per-league draw thresholds: the optimum may differ between a 62% league and a 40% one.
- Player-level features (lineups/injuries) still need a paid API; free tiers cannot backfill
  68k historical matches.
