# Deep audit — findings & fixes

A 16-dimension multi-agent audit (81 agents: finders + adversarial verifiers +
synthesis) reviewed the codebase, focused on the large uncommitted changes.
**64 findings verified, 0 false positives** — 3 critical, 17 high, 19 medium,
13 low. Full machine-readable record in `AUDIT-FINDINGS.json`. This is the
human summary of the top 18 (de-duplicated) and their status.

## The one root cause behind the 3 criticals: evaluation data-leakage

The pipeline was refit on **all** data, then its own tail was used for (a) the
reported RPS/log-loss/Brier, (b) fitting the calibrator, and (c) `evaluate.py`'s
per-league + reliability numbers. All three were therefore **in-sample and
optimistic** — and recency weighting made it explode (a tail-holdout RPS of
0.175, better than bookmakers, which is how it was first caught).

**Fixed** by a leak-free honest-holdout step: an *eval pipeline* trained on
everything **before** the holdout produces the calibrator (on out-of-sample
probs) and the reported metrics (on a disjoint test slice); those OOS
predictions are persisted to `holdout_eval.json` and `evaluate.py`/dashboard
read them. The shipped pipeline still trains on all data (keeps recent matches).

## Fixed (top 18)

| # | Sev | Issue | Fix |
|---|-----|-------|-----|
| 1,2,4,5 | critical/high | Calibrator + reported holdout + evaluate.py all in-sample | Leak-free eval-pipeline holdout; `holdout_eval.json`; calibrated OOS metrics |
| 3 | critical | Flask `debug=True` on `0.0.0.0` (network RCE) | Env-gated `HOST`/`PORT`/`FLASK_DEBUG`, loopback + debug-off by default |
| 6 | high | **6th train/serve-skew bug**: season-gap Elo regression only in training | Baked the same >45-day regression into the served Elo in `compute_team_stats` |
| 7 | high | `SelectKBest(mutual_info_classif)` nondeterministic → model not reproducible | `random_state=42` via `functools.partial` |
| 8 | high | Argentina `ar.1` mixed season labels dropped ~35% of its matches | Match by trailing year (captures `2017` **and** `2016/2017`), no double-count → all 6,238 load |
| 9 | high | Squad values forfeited for 5 new leagues Transfermarkt covers | Added `NOR1/SWE1/JPN1/MLS/ARG1` → TM comp ids (safe auto-match only) |
| 10 | med | New-league clubs split across two spellings (`Ham-Kam`/`HamKam`) | Explicit `NEW_LEAGUE_NAME_CANON` (verified, not fuzzy) |
| 11 | high | Recency scaling silently cancelled the balanced class weights | `_blend_sample_weights` renormalizes **per class** |
| 12 | high | Hyperparameters tuned on a seasonally-skewed 8k recent tail | Widened tuning window to 25k (spans every league's full season) |
| 13 | high | `teams_json` `\|safe` over `json.dumps` → `</script>` XSS | `{{ teams\|tojson }}` |
| 14 | high | Unauthenticated `POST /train` clobbers the shipped model | Refuses once a model exists (403); retraining is an offline step |
| 15 | high | Duplicate `draw_threshold` key in `evaluate.py` silently dropped the scalar | Renamed to `draw_threshold_shipped` |
| 16 | high | Reworked code untested; guard test NaN-blind | Added finite-value guard + `TestDrawSignalFeatures` + RPS/recency tests |
| 17 | high | Docs falsely claim "pure standard library" + Git LFS | Removed both; reconciled (numeric tables refreshed post-retrain) |
| — | — | `MODEL_VERSION` stuck at 3.0.0 | Bumped to 4.0.0 |

## Second wave — the remaining findings, now fixed

After the first 18, the rest of the 64 were worked through too:

- **`dc_rho` threaded to serving** (#30/#45) — carried on `team_stats` and passed
  into `match_probabilities`, so a future rho retune can't skew serving.
- **Home-advantage Elo scaling** (#31) — now scales by the league's home/away
  goal *asymmetry* (ratio of ratios), not its total home scoring, which had
  conflated "high-scoring league" with "strong home edge". Validated by RPS on
  the retrain.
- **Dixon-Coles `tau` floored at 0** (#52/#57) — defensive; can't go negative.
- **`/predict` error handling + feature-count guard** (#32) — a stale artifact
  now yields a clear message and `predict()` fails loudly on a count mismatch.
- **`model_exists()`** now requires the `.joblib` too (#33); **`api_client`** no
  longer echoes raw upstream error bodies (#60); **recency** gives unparseable
  dates the *min* weight, is threaded through the tuning search, and its
  per-class renormalization is fixed (#35/#36/#11/#58); **`save()`** warns past
  90 MB (#47); the **API training path** now enriches squad values instead of
  wiping them (#41); **`_rps` is defined once** and imported by evaluate.py (#59);
  the **joblib trust boundary** is documented (#38).
- **Tests added** (#18/#19/#20/#42/#43/#44/#46): calibration fit/apply/round-trip,
  burn-in, the squad-value serving path, `test_evaluate.py`,
  `test_odds_benchmark.py`, the 8 new leagues + Argentina trailing-year parsing,
  and a finite-value guard. Suite: **131 → 169 tests**.

### Genuinely left as-is (by design or inherent)

- **FIN1 / IRL1 / CHN1 squad values** — no Transfermarkt counterpart; correctly
  unmapped with `has_squad_value=0`.
- **Both-sides `has_squad_value` rule** (#40) — deliberately voids the block when
  one club is uncovered (a one-sided value misleads); now covered by a test that
  asserts this *intended* behaviour.
- **Old-model calibration mismatch** (#24/#26) — only affects artifacts saved
  before this rework; the shipped v4 model is internally consistent.
- **`joblib.load` trust boundary** (#38) — a pickle can't be validated pre-load;
  mitigation is provenance (documented in code).
- **ClubElo cross-league feed** — the single highest genuine-signal *addition*
  (`RESEARCH-NOTES.md` rank 2). Deferred deliberately: the model already sits at
  the market RPS ceiling, so expected gain is marginal, and a cross-league
  name-mapping integration is the classic silent-degeneracy risk here — worth its
  own careful change, not a rushed one.
