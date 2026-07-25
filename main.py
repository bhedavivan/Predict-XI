#!/usr/bin/env python3
"""
Predict-XI: Soccer Match Outcome Predictor

Uses football-data.org API to fetch match data and a Gaussian Naive Bayes
classifier to predict match outcomes (Home Win, Draw, Away Win).

No external ML dependencies — the model is built from scratch using only
the Python standard library.
"""

import sys
import os
import argparse
from config import LEAGUE_CODES
from api_client import fetch_matches, MissingTokenError
from data_processor import (
    process_matches,
    add_form_features,
    prepare_training_data,
    prepare_prediction_features,
    save_data,
    load_data,
)
from model_trainer import MatchPredictorModel


class TrainingError(Exception):
    """Raised when training fails for a recoverable reason."""
    pass


def train_model(league_codes: list, season: str):
    """Fetch data for given leagues, train model, return (model, metrics).

    Raises TrainingError on recoverable failures (no data, no finished matches).
    Raises MissingTokenError on auth issues.
    """
    print("Fetching match data from API...")
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

    print("Adding form features...")
    rows = add_form_features(rows)
    print(f"Features added. Total rows: {len(rows)}")

    # Save processed data for later use
    save_data(rows)

    print("Preparing training data...")
    X, y = prepare_training_data(rows)

    if not X or len(X) == 0:
        raise TrainingError(
            "Not enough data to train. Need more matches with form history. "
            "Try including more leagues or a different season."
        )

    print(f"Training samples: {len(X)}")
    print("Training model (Gaussian Naive Bayes)...")
    model = MatchPredictorModel()
    metrics = model.train(X, y)

    # Add train_samples to metrics for convenience
    metrics["train_samples"] = len(X) - metrics.get("test_samples", 0)

    print(f"\nModel Accuracy: {metrics['accuracy']:.3f}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Train samples: {metrics['train_samples']}")
    print("\nClassification Report:")
    for label, scores in metrics["classification_report"].items():
        label_name = {-1: "Away Win", 0: "Draw", 1: "Home Win"}.get(int(label), label)
        print(f"  {label_name}: precision={scores['precision']:.3f}, "
              f"recall={scores['recall']:.3f}, f1={scores['f1-score']:.3f}, "
              f"support={scores['support']}")

    model.save()
    print("\nModel saved to model.json")
    return model, metrics


def interactive_predict(model: MatchPredictorModel):
    """Interactive prediction mode."""
    print("\n=== Interactive Predictor ===")
    print("Enter team names to predict match outcome.")
    print("Type 'quit' to exit.\n")

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

        team_stats = {}
        features = prepare_prediction_features(home, away, team_stats)
        result = model.predict(features)

        print(f"\nPrediction: {result['prediction']}")
        print("Probabilities:")
        for outcome, prob in result["probabilities"].items():
            print(f"  {outcome}: {prob:.1%}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Predict-XI — Soccer Match Outcome Predictor"
    )
    parser.add_argument(
        "--train",
        nargs="+",
        choices=list(LEAGUE_CODES.keys()),
        default=None,
        help="League codes to train on (e.g., --train PL SA BL1). "
             "If omitted, uses default leagues when training is needed.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2023",
        help="Season year (e.g., 2023). Default: 2023 (completed season)",
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
        help="List all available league codes",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if a saved model exists",
    )

    args = parser.parse_args()

    if args.list_leagues:
        print("Available league codes:")
        for code, name in LEAGUE_CODES.items():
            print(f"  {code}: {name}")
        return

    # Determine if --train was explicitly passed by the user
    train_explicitly_passed = "--train" in sys.argv

    # Determine the league codes to use for training
    if train_explicitly_passed:
        train_codes = args.train
    else:
        train_codes = ["PL", "SA", "BL1", "PD", "FL1"]

    # Try to load existing model first
    model = MatchPredictorModel()
    model_loaded = model.load()

    # Decide whether to train
    should_train = False
    if args.force_retrain:
        should_train = True
        print("Force retrain requested.")
    elif not model_loaded:
        should_train = True
        print("No saved model found. Training required.")
    elif train_explicitly_passed:
        should_train = True
        print("--train flag passed. Retraining model.")
    else:
        print("Loaded existing model from model.json")

    if should_train:
        print("Training new model...")
        try:
            model, _ = train_model(train_codes, args.season)
        except (TrainingError, MissingTokenError) as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    # Verify model is trained before predicting
    if not model.trained:
        print("\nERROR: No trained model available. Train first with --train or ensure model.json exists.")
        sys.exit(1)

    if args.predict:
        home, away = args.predict
        team_stats = {}
        features = prepare_prediction_features(home, away, team_stats)
        result = model.predict(features)
        print(f"\nMatch: {home} vs {away}")
        print(f"Prediction: {result['prediction']}")
        print("Probabilities:")
        for outcome, prob in result["probabilities"].items():
            print(f"  {outcome}: {prob:.1%}")

    if args.interactive:
        interactive_predict(model)

    if not any([args.predict, args.interactive, args.list_leagues, args.force_retrain,
                train_explicitly_passed]):
        print("\nNo prediction mode selected. Use --predict or --interactive.")
        print("Example: python main.py --predict 'Manchester City' 'Arsenal'")
        print("Or: python main.py --interactive")


if __name__ == "__main__":
    main()