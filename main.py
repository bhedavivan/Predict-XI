#!/usr/bin/env python3
"""
Soccer Sports Predictor
Uses football-data.org API and footballcsv/cache.footballdata CSV data
to train an ML model to predict match outcomes (Home Win, Draw, Away Win).
"""

import sys
import os
import argparse
import json
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from config import LEAGUE_CODES
from api_client import fetch_matches, MissingTokenError
from data_processor import (
    process_matches,
    add_form_features,
    prepare_training_data,
    prepare_prediction_features,
    save_data,
    load_data,
    compute_team_stats,
)
from model_trainer import MatchPredictorModel
from csv_data_loader import (
    load_and_process_data,
    load_multiple_seasons_leagues,
    AVAILABLE_SEASONS,
    LEAGUE_CODE_MAP,
    save_processed_data,
    load_processed_data,
)


class TrainingError(Exception):
    """Raised when training fails for a recoverable reason."""
    pass


def train_model_api(league_codes: list, season: str, cv_folds: int = 5,
                    model_type: str = "ensemble"):
    """Fetch data from API for given leagues, train model, return (model, metrics)."""
    print("Fetching match data from football-data.org API...")
    all_matches = []
    any_empty = False

    for code in league_codes:
        try:
            matches = fetch_matches(code, season)
            print(f"  {code} ({LEAGUE_CODES[code]}): {len(matches)} matches fetched")
            if not matches:
                any_empty = True
            all_matches.extend(matches)
        except MissingTokenError:
            raise
        except Exception as e:
            print(f"  Error fetching {code}: {e}")

    if not all_matches:
        raise TrainingError(
            "No matches fetched at all. Check your API token and league codes."
        )

    print(f"\nTotal matches fetched: {len(all_matches)}")
    print("Processing data...")
    rows = process_matches(all_matches)
    print(f"Finished matches: {len(rows)}")

    if not rows:
        msg = "No finished matches found in the data."
        if any_empty:
            msg += f" The season '{season}' may not have completed matches yet."
        msg += " Try passing --season with a completed season (e.g., --season 2023)."
        raise TrainingError(msg)

    # Squad market values, same optional enrichment as the CSV path — without
    # this the API path would build features with no squad value AND overwrite
    # team_stats.json with squad_value_eur=None for every club, silently wiping
    # the coverage a prior CSV train produced. Non-fatal on any failure.
    squad_values, club_map = None, None
    try:
        import transfermarkt_data
        import club_mapping
        # Tag each team with the most-recent Transfermarkt-COVERED league it
        # appeared in (fallback: its latest league). See the matching note in
        # csv_data_loader.load_and_process_data — this recovers promoted/
        # relegated clubs that have ever been top-flight.
        from collections import defaultdict as _dd
        _covered = set(club_mapping.LEAGUE_TO_TM_COMP)
        _seq = _dd(list)
        for r in rows:
            comp = r.get("competition", "")
            _seq[r["home_team"]].append(comp)
            _seq[r["away_team"]].append(comp)
        team_leagues = {}
        for _team, _cs in _seq.items():
            _cov = [c for c in _cs if c in _covered]
            team_leagues[_team] = {"league": _cov[-1] if _cov else _cs[-1]}
        club_map = club_mapping.build_club_mapping(
            team_leagues, transfermarkt_data.load_table("clubs"))
        squad_values = transfermarkt_data.get_index()
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"Squad values unavailable ({type(e).__name__}: {e}); continuing without them")
        squad_values, club_map = None, None

    clubelo = None
    try:
        import clubelo_data
        clubelo = clubelo_data.get_index()
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"ClubElo unavailable ({type(e).__name__}: {e}); continuing without it")
        clubelo = None

    pstats = None
    try:
        import player_stats
        pstats = player_stats.get_index()
    except Exception as e:  # noqa: BLE001 - optional enrichment, never fatal
        print(f"Player stats unavailable ({type(e).__name__}: {e}); continuing without them")
        pstats = None

    print("Adding form features...")
    rows = add_form_features(rows, squad_values=squad_values, club_map=club_map,
                             clubelo=clubelo, pstats=pstats)
    print(f"Features added. Total rows: {len(rows)}")

    save_data(rows)

    print("Preparing training data...")
    X, y, feature_names = prepare_training_data(rows)

    if not X or len(X) == 0:
        raise TrainingError(
            "Not enough data to train. Need more matches with form history. "
            "Try including more leagues or a different season."
        )

    print(f"Training samples: {len(X)}")
    print(f"Training model ({model_type})...")
    model = MatchPredictorModel(model_type=model_type)
    metrics = model.train(X, y, feature_names=feature_names, cv_folds=cv_folds)

    print_metrics(metrics)
    model.save()
    api_team_stats = compute_team_stats(rows, squad_values=squad_values, club_map=club_map,
                                        clubelo=clubelo, pstats=pstats)
    save_artifacts(metrics, api_team_stats, h2h_map=getattr(add_form_features, "last_h2h", {}))
    print("\nModel saved to model.json")
    return model, metrics


def train_model_csv(seasons: list, league_codes: list, cv_folds: int = 5,
                    model_type: str = "ensemble", recency_halflife_days=None,
                    burn_in_folds: int = 1, calibrate: bool = True):
    """Load data from CSV files, train model, return (model, metrics).

    One command reproduces the shipped model: calibration is folded in
    (`calibrate=True` fits it on the tail holdout inside train()), the coldest
    CV fold is burned in so the reported CV number isn't dragged down by
    warming-up ratings, and per-row dates enable optional recency weighting.
    Leagues, dates and head-to-head are all persisted for the app + evaluate.py.
    """
    print("Loading match data from football-data.co.uk / footballcsv...")

    X, y, feature_names, team_stats = load_and_process_data(
        seasons=seasons,
        league_codes=league_codes
    )
    leagues = getattr(load_and_process_data, "last_leagues", [])
    dates = getattr(load_and_process_data, "last_dates", [])
    h2h_map = getattr(load_and_process_data, "last_h2h", {})

    if not X or len(X) == 0:
        raise TrainingError(
            "Not enough data to train. Check seasons and league codes."
        )

    print(f"Training samples: {len(X)}")
    print(f"Training model ({model_type})...")
    model = MatchPredictorModel(model_type=model_type)
    metrics = model.train(X, y, feature_names=feature_names, cv_folds=cv_folds,
                          sample_dates=dates, recency_halflife_days=recency_halflife_days,
                          burn_in_folds=burn_in_folds, calibrate=calibrate)

    print_metrics(metrics)

    model.save()
    save_processed_data(X, y, feature_names, team_stats, leagues=leagues, dates=dates)
    save_artifacts(metrics, team_stats, h2h_map=h2h_map)
    print("\nModel saved to model.json")
    print("Processed data saved to processed_data.json")
    return model, metrics


def save_artifacts(metrics: dict, team_stats: dict, h2h_map: dict = None):
    """Persist lightweight, committable artifacts the web app reads."""
    metrics_out = {k: v for k, v in metrics.items() if k != "cv_scores"}
    metrics_out["trained_at"] = datetime.utcnow().isoformat() + "Z"
    # test_samples is a tail slice OF train_samples (see MatchPredictorModel
    # docs), not additional data, so the total is just train_samples.
    metrics_out["training_samples"] = metrics.get("train_samples", 0)
    metrics_out["n_teams"] = len(team_stats)
    with open(os.path.join(_SCRIPT_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    with open(os.path.join(_SCRIPT_DIR, "team_stats.json"), "w", encoding="utf-8") as f:
        json.dump(team_stats, f, indent=2)
    # Head-to-head is a property of a PAIR, so it cannot live on team_stats.
    # Persisted separately and loaded by the app; without it every live
    # prediction silently received h2h=0.
    if h2h_map:
        with open(os.path.join(_SCRIPT_DIR, "h2h_stats.json"), "w", encoding="utf-8") as f:
            json.dump(h2h_map, f)


def print_metrics(metrics: dict):
    """Print training metrics in a formatted way."""
    eval_method = metrics.get("eval_method", "shuffled")
    print(f"\nModel Accuracy ({eval_method} holdout): {metrics['accuracy']:.3f}")
    if "baseline_accuracy" in metrics:
        lift = metrics["accuracy"] - metrics["baseline_accuracy"]
        print(f"Majority-class baseline: {metrics['baseline_accuracy']:.3f}  (lift: {lift:+.3f})")
    if metrics.get("log_loss") is not None:
        print(f"Log-loss: {metrics['log_loss']:.3f}   Brier: {metrics['brier']:.3f}  (lower is better)")
    elif "log_loss" in metrics:
        print("Log-loss/Brier: unavailable (holdout probability computation failed)")
    if metrics.get("rps") is not None:
        print(f"RPS (holdout): {metrics['rps']:.4f}   RPS (warm CV): "
              f"{metrics.get('cv_rps') if metrics.get('cv_rps') is not None else float('nan'):.4f}  "
              f"(lower is better; ~0.19-0.21 is published no-odds SOTA)")
    print(f"Train samples: {metrics.get('train_samples', 0)}")
    if "cv_mean" in metrics:
        print(f"CV accuracy: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
    print("\nClassification Report:")
    for label, scores in metrics.get("classification_report", {}).items():
        label_name = {-1: "Away Win", 0: "Draw", 1: "Home Win"}.get(int(label), label)
        print(f"  {label_name}: precision={scores['precision']:.3f}, "
              f"recall={scores['recall']:.3f}, f1={scores['f1-score']:.3f}, "
              f"support={scores['support']}")

    print("\nConfusion Matrix:")
    print("  Predicted ->  Away Win  Draw  Home Win")
    classes = [-1, 0, 1]
    cm = metrics.get("confusion_matrix", [[0]*3 for _ in range(3)])
    for i, row in enumerate(cm):
        actual = {-1: "Away Win", 0: "Draw", 1: "Home Win"}.get(classes[i], classes[i])
        print(f"  Actual {actual:8s}  {row[0]:3d}    {row[1]:3d}    {row[2]:3d}")

    print("\nClass Distribution (training):")
    for cls, count in metrics.get("class_distribution", {}).items():
        label_name = {-1: "Away Win", 0: "Draw", 1: "Home Win"}.get(cls, cls)
        print(f"  {label_name}: {count}")

    if metrics.get("feature_importances"):
        print("\nTop features:")
        names = metrics.get("feature_names", [])
        imps = metrics["feature_importances"]
        pairs = sorted(zip(names, imps), key=lambda t: -t[1])
        for name, imp in pairs[:10]:
            print(f"  {name:<28} {imp:.4f}")


def interactive_predict(model: MatchPredictorModel, use_csv: bool = False):
    """Interactive prediction mode."""
    print("\n=== Interactive Predictor ===")
    print("Enter team names to predict match outcome.")
    print("Type 'quit' to exit.\n")

    if use_csv:
        data = load_processed_data()
        team_stats = data.get("team_stats", {}) if data else {}
    else:
        rows = load_data()
        team_stats = compute_team_stats(rows) if rows else {}

    while True:
        try:
            home = input("Home team: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if home.lower() == "quit":
            break
        try:
            away = input("Away team: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if away.lower() == "quit":
            break
        if not home or not away:
            print("Please enter both team names.")
            continue

        features = prepare_prediction_features(home, away, team_stats)
        result = model.predict(features)

        print(f"\nPrediction: {result['prediction']}")
        print("Probabilities:")
        for outcome, prob in result["probabilities"].items():
            print(f"  {outcome}: {prob:.1%}")
        print()


def list_csv_leagues():
    """List available CSV leagues."""
    print("Available CSV league codes (footballcsv/cache.footballdata):")
    for code, name in sorted(LEAGUE_CODE_MAP.items()):
        print(f"  {code}: {name}")
    print(f"\nAvailable seasons: {', '.join(AVAILABLE_SEASONS)}")


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Sports Predictor - ML-based match outcome prediction"
    )

    parser.add_argument(
        "--source",
        choices=["api", "csv"],
        default="csv",
        help="Data source: 'api' (football-data.org) or 'csv' (footballcsv/cache.footballdata). Default: csv"
    )
    parser.add_argument(
        "--train",
        nargs="+",
        default=None,
        help="League codes to train on (e.g., --train eng.1 esp.1 de.1). "
             "If omitted, uses default leagues when training is needed.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2023",
        help="Season year for API source (e.g., 2023). Default: 2023",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons for CSV source (e.g., --seasons 2022-23 2023-24). Default: ALL available",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Use ALL available CSV seasons for maximum training data.",
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Use ALL available CSV leagues for maximum training data.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds. Default: 5",
    )
    parser.add_argument(
        "--model-type",
        choices=["logreg", "nb", "histgb", "rf", "ensemble", "stacking"],
        default="ensemble",
        help="Model: 'ensemble' (default), 'stacking', 'histgb', 'rf', 'logreg', 'nb' (legacy).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if a saved model exists",
    )

    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME", "AWAY"),
        help="Predict a single match: HOME_TEAM AWAY_TEAM",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prediction mode",
    )
    parser.add_argument(
        "--list-leagues",
        action="store_true",
        help="List all available league codes for the selected source",
    )
    parser.add_argument(
        "--list-seasons",
        action="store_true",
        help="List available seasons (CSV source only)",
    )

    args = parser.parse_args()

    if args.list_leagues:
        if args.source == "api":
            print("Available API league codes (football-data.org):")
            for code, name in LEAGUE_CODES.items():
                print(f"  {code}: {name}")
        else:
            list_csv_leagues()
        return

    if args.list_seasons:
        if args.source == "csv":
            print(f"Available seasons: {', '.join(AVAILABLE_SEASONS)}")
        else:
            print("Seasons not applicable for API source. Use --season with year.")
        return

    if args.source == "api":
        if args.train:
            train_codes = args.train
        else:
            train_codes = list(LEAGUE_CODES.keys())
        seasons_arg = args.season
    else:
        if args.train:
            train_codes = args.train
        else:
            if args.all_leagues:
                # Top flights (scored) + second divisions (rating warm-up only:
                # loaded and fed through the feature loop, excluded from scoring
                # so promoted clubs carry real form into the top flight).
                import csv_data_loader as _cdl
                import data_processor as _dp
                train_codes = list(LEAGUE_CODE_MAP.keys()) + _cdl.WARMUP_CSV_CODES
                _dp.WARMUP_COMPETITION_CODES = set(_cdl.WARMUP_CODE_MAP.values())
                print(f"Warm-up leagues (unscored): {_cdl.WARMUP_CSV_CODES}")
            else:
                # Cold-start default: the strongest handful of top flights.
                import leagues as _lg
                train_codes = _lg.TRAIN_CSV_CODES[:5]

        if args.seasons:
            seasons_arg = args.seasons
        elif args.all_seasons:
            seasons_arg = AVAILABLE_SEASONS
        else:
            seasons_arg = AVAILABLE_SEASONS[-3:]  # Last 3 seasons

    model = MatchPredictorModel()
    model_loaded = model.load()

    should_train = False
    if args.force_retrain:
        should_train = True
        print("Force retrain requested.")
    elif not model_loaded:
        should_train = True
        print("No saved model found. Training required.")
    elif args.train is not None or args.all_leagues or args.all_seasons:
        should_train = True
        print("Explicit data selection passed. Retraining model.")
    else:
        print("Loaded existing model from model.json")

    if should_train:
        print("Training new model...")
        try:
            if args.source == "api":
                model, _ = train_model_api(train_codes, seasons_arg, args.cv_folds,
                                           model_type=args.model_type)
            else:
                model, _ = train_model_csv(seasons_arg, train_codes, args.cv_folds,
                                           model_type=args.model_type)
        except (TrainingError, MissingTokenError) as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    if not model.trained:
        print("\nERROR: No trained model available. Train first with --train or ensure model.json exists.")
        sys.exit(1)

    if args.predict:
        home, away = args.predict
        if args.source == "csv":
            data = load_processed_data()
            team_stats = data.get("team_stats", {}) if data else {}
        else:
            rows = load_data()
            team_stats = compute_team_stats(rows) if rows else {}
        features = prepare_prediction_features(home, away, team_stats)
        result = model.predict(features)
        print(f"\nMatch: {home} vs {away}")
        print(f"Prediction: {result['prediction']}")
        print("Probabilities:")
        for outcome, prob in result["probabilities"].items():
            print(f"  {outcome}: {prob:.1%}")

    if args.interactive:
        interactive_predict(model, use_csv=(args.source == "csv"))

    if not any([args.predict, args.interactive, args.list_leagues, args.list_seasons,
                args.force_retrain, args.train is not None, args.all_leagues, args.all_seasons]):
        print("\nNo prediction mode selected. Use --predict or --interactive.")
        print("Example: python main.py --predict 'Manchester City' 'Arsenal'")
        print("Or: python main.py --interactive")
        print("For max training data: python main.py --all-seasons --all-leagues --force-retrain")


if __name__ == "__main__":
    main()
