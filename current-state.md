# Current State — Predict-XI

**Status:** Version 6 complete — full league coverage, honest failure reporting,
a tested-and-rejected Dixon-Coles blending idea, documented freshness process.

## Done (v6)
- **Backfilled the last 7 stale leagues** (Russia, Poland, Austria,
  Switzerland, Denmark, Romania, Mexico): found football-data.co.uk has a
  *second* feed (`football-data.co.uk/new/{CODE}.csv`, one file per country
  covering 2012/13 through 2025/26) that covers exactly the leagues the main
  feed doesn't. Added `FD_COUK_NEW_LEAGUE_MAP`, a fetch/cache pair, and a new
  parser branch to `csv_data_loader.py`; wired in as a third rung on the
  fallback chain (main football-data.co.uk feed → new-leagues feed →
  footballcsv). Every one of the 29 training leagues is now on either the
  richest feed or this one — none are still capped at 2023-24.
- **Extended the team-alias audit** beyond the 8 leagues covered last round.
  Findings, each handled differently on purpose:
  - 9 more leagues (FL2, PD2, BL2, ELC2, ELC3, SA2, PPL2, DED2) either 403
    (not on the free football-data.org tier — confirmed via direct API
    calls, same failure as FL2/BL2 last round) or currently have zero
    scheduled fixtures to audit against. Not an alias problem; nothing to
    fix without a paid API tier.
  - EC/WC (Euros, World Cup) are **national-team** tournaments — there's no
    club-level training data to resolve against by definition, not a bug.
  - CL (Champions League) *is* a club competition and does work on the free
    tier — audited its full 36-team roster (not just a fixture snapshot) via
    `/competitions/CL/teams`, found and fixed 4 real mismatches (Benfica,
    Sporting Lisbon, Union SG, FC Copenhagen). The other 5 CL mismatches are
    clubs from leagues we don't train on (Norway, Azerbaijan, Czech
    Republic, Kazakhstan, Cyprus) — same "no training data" class as BSA,
    not an alias problem either.
  - BSA/BSA2 (Brazil) still has zero training data — unchanged from last
    round, still honestly shows the "unknown team" warning.
- **QA pass** (scoped to two concrete findings, not an open-ended hunt):
  - `model_trainer.py::_fit_sklearn`'s holdout log-loss/Brier computation
    used to fail silently into the initialized `0.0` — which reads as a
    *suspiciously perfect* score, the same failure shape as the team-alias
    bug. Now fails into `None`/"unavailable", both in `get_metrics()` and
    the dashboard template, with a test that forces the failure path.
  - Tightened a bare `except:` in `csv_data_loader.py` to the specific
    expected exceptions, matching the pattern already used elsewhere in
    that file.
- **Dixon-Coles probability blending — tested, rejected, documented.**
  Hypothesis was that blending DC's own Home/Draw/Away probabilities with
  the ensemble's output (not just feeding them in as input features) might
  help further, mirroring literature on combining a generative Poisson
  model with a discriminative classifier. Ran the actual experiment
  post-hoc (no retrain needed — reused the trained pipeline + held-out
  slice): macro-F1 **drops** monotonically as blend weight increases (0.458
  at w=0 down to 0.397 at w=0.5) even though log-loss/Brier improve
  marginally. The classifier already extracts DC's signal better through
  the input features than a naive output-level blend can — blending redilutes
  the ensemble's more nuanced prediction with a much simpler standalone
  signal. `predict()` unchanged; full grid in `dc_blend_analysis.json`.
- **Freshness — documented, not automated**, per explicit choice: a
  "Keeping the model fresh" section in `README.md` states the retrain
  command and cadence (once per European close season) as a manual,
  reviewed step by design — no scheduled job pushes model updates on its
  own.
- Retrained on the now-fully-current dataset: 62,131 → **65,759 matches**,
  669 → **677 teams**, reusing round 2's proven, size-safe ensemble config
  (skips re-tuning, which also means no risk of re-discovering another
  oversized RF candidate). No regression: macro-F1 held flat at 0.430,
  draw recall 24.5% (was 24.6%) — expected, since the 7 backfilled leagues
  mostly extend *coverage*, not aggregate signal.

## Model numbers (temporal, purged 5-fold CV, 65,759 matches)
- Accuracy 45.6% · baseline 43.0% · macro-F1 0.430 · log-loss 1.024 ·
  Brier 0.205 · draw recall 24.5%. Essentially unchanged from round 2
  (0.430 / 24.6%) — the backfill's value is coverage, not aggregate lift.
- `model.joblib`: 65.3 MB (round 2 was 64.3 MB — stayed safely sized).

## Known limitations
- Brazil (BSA/BSA2) has no training data at all — would need a new data
  source investigated from scratch, not just an alias fix.
- FL2, BL2, PD2, ELC2, ELC3, SA2, PPL2, DED2 aren't reachable via the free
  football-data.org tier for live fixtures — Fixtures page will error or
  come back empty for these regardless of training-data coverage.
- Dixon-Coles blending doesn't help beyond its existing role as input
  features — confirmed, not just untried (see `dc_blend_analysis.json`).
- Draw recall (24.5%) has plateaued across the last two rounds despite
  meaningfully different work (new features, new data, Dixon-Coles) —
  worth treating as close to this feature set's ceiling rather than
  assuming the next tweak will move it further.

## Next ideas
- A genuinely different data source for Brazil, if that league matters to
  you specifically.
- SHAP-based explainability for individual predictions (mentioned since
  v3, still not done).
- If draw recall is worth pushing further, it likely needs a different
  lever than more features on the same architecture — e.g. an explicit
  draw-vs-decisive two-stage classifier, or reconsidering the loss
  function directly rather than macro-F1-driven hyperparameter selection.
