"""Train and evaluate the production stayed-dirty classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from stepclean.ml.training_data import (
    DEFAULT_FACE_CSV,
    DEFAULT_FORM_CSV,
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_NEIGHBOR_CSV,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_training_frame,
    limit_faces_per_object,
    prepare_frame,
)


DIRTY_COLUMN = "stayed_dirty"
PROBABILITY_COLUMN = "dirty_probability"
PREDICTION_COLUMN = "predicted_stayed_dirty"
DEFAULT_PREDICTIONS_OUTPUT = Path("cache") / "hotspot_dirty_classifier_predictions.csv"
DEFAULT_HOLDOUT_PREDICTIONS_OUTPUT = Path("cache") / "hotspot_dirty_classifier_holdout_predictions.csv"
DEFAULT_IMPORTANCE_OUTPUT = Path("cache") / "hotspot_dirty_classifier_feature_importance.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stayed-dirty LightGBM classifier.")
    parser.add_argument("--face-csv", type=Path, default=DEFAULT_FACE_CSV)
    parser.add_argument("--neighbor-csv", type=Path, default=DEFAULT_NEIGHBOR_CSV)
    parser.add_argument("--form-csv", type=Path, default=DEFAULT_FORM_CSV)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--holdout-predictions-output", type=Path, default=DEFAULT_HOLDOUT_PREDICTIONS_OUTPUT)
    parser.add_argument("--importance-output", type=Path, default=DEFAULT_IMPORTANCE_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-test-size", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-faces-per-object", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    frame = build_training_frame(args.face_csv, args.neighbor_csv, args.form_csv)
    frame = limit_faces_per_object(
        frame,
        max_faces_per_object=args.max_faces_per_object,
        random_state=args.random_state,
    )
    frame = prepare_classifier_frame(frame)

    if args.holdout_test_size:
        predictions = holdout_predictions(frame, args.holdout_test_size, args.threshold, args.random_state)
        predictions_output = args.holdout_predictions_output
        evaluation_name = "grouped holdout"
    else:
        predictions = cross_val_predictions(frame, args.folds, args.threshold, args.random_state)
        predictions_output = args.predictions_output
        evaluation_name = f"grouped {min(args.folds, frame['dataset'].nunique())}-fold cross-validation"

    model = make_classifier(args.random_state)
    model.fit(frame[FEATURE_COLUMNS], frame[DIRTY_COLUMN])
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))

    write_predictions(predictions, predictions_output)
    write_importance(model, args.importance_output)
    print_summary(frame, evaluate_predictions(predictions, threshold=args.threshold), evaluation_name, args, predictions_output)
    return 0


def prepare_classifier_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_frame(frame)
    frame[DIRTY_COLUMN] = (frame[TARGET_COLUMN] > 0).astype(int)
    return frame


def cross_val_predictions(frame: pd.DataFrame, folds: int, threshold: float, random_state: int) -> pd.DataFrame:
    groups = frame["dataset"].astype(str)
    predictions = prediction_frame(frame)
    splitter = GroupKFold(n_splits=min(max(2, folds), groups.nunique()))

    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame[FEATURE_COLUMNS], frame[DIRTY_COLUMN], groups),
        start=1,
    ):
        model = make_classifier(random_state + fold)
        model.fit(frame.iloc[train_index][FEATURE_COLUMNS], frame.iloc[train_index][DIRTY_COLUMN])
        add_probabilities(predictions, test_index, model, frame.iloc[test_index], threshold)
        predictions.loc[predictions.index[test_index], "fold"] = fold
    return predictions


def holdout_predictions(frame: pd.DataFrame, test_size: float, threshold: float, random_state: int) -> pd.DataFrame:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--holdout-test-size must be greater than 0 and less than 1.")

    groups = frame["dataset"].astype(str)
    train_index, test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(
            frame[FEATURE_COLUMNS],
            frame[DIRTY_COLUMN],
            groups,
        )
    )
    model = make_classifier(random_state)
    model.fit(frame.iloc[train_index][FEATURE_COLUMNS], frame.iloc[train_index][DIRTY_COLUMN])

    predictions = prediction_frame(frame.iloc[test_index])
    probability = model.predict_proba(frame.iloc[test_index][FEATURE_COLUMNS])[:, 1]
    predictions[PROBABILITY_COLUMN] = probability
    predictions[PREDICTION_COLUMN] = (probability >= threshold).astype(int)
    predictions["split"] = "test"
    return predictions


def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    predictions = frame[["dataset", "face_id", "form_type", TARGET_COLUMN, DIRTY_COLUMN]].copy()
    predictions[PROBABILITY_COLUMN] = np.nan
    predictions[PREDICTION_COLUMN] = -1
    return predictions


def add_probabilities(
    predictions: pd.DataFrame,
    row_indexes: np.ndarray,
    model: LGBMClassifier,
    frame: pd.DataFrame,
    threshold: float,
) -> None:
    probability = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    predictions.loc[predictions.index[row_indexes], PROBABILITY_COLUMN] = probability
    predictions.loc[predictions.index[row_indexes], PREDICTION_COLUMN] = (probability >= threshold).astype(int)


def make_classifier(random_state: int, *, class_weight: str = "none") -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=12,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        class_weight="balanced" if class_weight == "balanced" else None,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate_predictions(predictions: pd.DataFrame, *, threshold: float) -> dict[str, float]:
    y_true = predictions[DIRTY_COLUMN].astype(int).to_numpy()
    probability = predictions[PROBABILITY_COLUMN].astype(float).to_numpy()
    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "dirty_precision": float(precision),
        "dirty_recall": float(recall),
        "dirty_f1": float(f1),
        "clean_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "dirty_rate": float(y_true.mean()),
        "predicted_dirty_rate": float(y_pred.mean()),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def write_predictions(predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(["dataset", PROBABILITY_COLUMN], ascending=[True, False]).to_csv(path, index=False)


def write_importance(model: LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": model.booster_.feature_name(),
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain_importance", "split_importance"], ascending=False).to_csv(path, index=False)


def print_summary(
    frame: pd.DataFrame,
    metrics: dict[str, float],
    evaluation_name: str,
    args: argparse.Namespace,
    predictions_output: Path,
) -> None:
    print("LightGBM stayed-dirty classifier")
    print(f"Rows: {len(frame)}")
    print(f"Datasets: {frame['dataset'].nunique()}")
    print(f"Target: {DIRTY_COLUMN} = {TARGET_COLUMN} > 0")
    print(f"Dirty faces: {int(frame[DIRTY_COLUMN].sum())}/{len(frame)} ({frame[DIRTY_COLUMN].mean():.4f})")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Threshold: {args.threshold:.3f}")
    print()
    print(f"Evaluation: {evaluation_name}")
    for label, key in (
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Dirty precision", "dirty_precision"),
        ("Dirty recall", "dirty_recall"),
        ("Dirty F1", "dirty_f1"),
        ("Clean recall", "clean_recall"),
        ("ROC AUC", "roc_auc"),
        ("PR AUC", "average_precision"),
        ("Dirty rate", "dirty_rate"),
        ("Pred dirty rate", "predicted_dirty_rate"),
    ):
        print(f"  {label:<18} {metrics[key]:.4f}")
    print(
        "  Confusion matrix: "
        f"TN={int(metrics['true_negative'])}, FP={int(metrics['false_positive'])}, "
        f"FN={int(metrics['false_negative'])}, TP={int(metrics['true_positive'])}"
    )
    print()
    print(f"Wrote model: {args.model_output.resolve()}")
    print(f"Wrote predictions: {predictions_output.resolve()}")
    print(f"Wrote feature importance: {args.importance_output.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
