"""Feature diagnostics for the stayed-dirty classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from stepclean.ml.classifier import (
    DIRTY_COLUMN,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    evaluate_predictions,
    make_classifier,
    prepare_classifier_frame,
)
from stepclean.ml.training_data import (
    DEFAULT_FACE_CSV,
    DEFAULT_FORM_CSV,
    DEFAULT_NEIGHBOR_CSV,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_training_frame,
    limit_faces_per_object,
)


DEFAULT_OUTPUT_DIR = Path("cache") / "classifier_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact diagnostics for classifier features.")
    parser.add_argument("--face-csv", type=Path, default=DEFAULT_FACE_CSV)
    parser.add_argument("--neighbor-csv", type=Path, default=DEFAULT_NEIGHBOR_CSV)
    parser.add_argument("--form-csv", type=Path, default=DEFAULT_FORM_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--holdout-test-size", type=float, default=0.2)
    parser.add_argument("--max-faces-per-object", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--ablation-candidate-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_training_frame(args.face_csv, args.neighbor_csv, args.form_csv)
    frame = limit_faces_per_object(
        frame,
        max_faces_per_object=args.max_faces_per_object,
        random_state=args.random_state,
    )
    prepared = prepare_classifier_frame(frame)
    train_frame, test_frame = grouped_holdout(
        prepared,
        test_size=args.holdout_test_size,
        random_state=args.random_state,
    )

    model = fit_classifier(train_frame, FEATURE_COLUMNS, random_state=args.random_state)
    predictions = predict_classifier(model, test_frame, FEATURE_COLUMNS, threshold=args.threshold)
    metrics = evaluate_predictions(predictions, threshold=args.threshold)

    stats = feature_statistics(prepared, FEATURE_COLUMNS)
    importance = model_importance(model)
    pca_components, pca_scores = pca_analysis(prepared, FEATURE_COLUMNS, args.pca_components)
    mutual_info = mutual_information(prepared, FEATURE_COLUMNS, random_state=args.random_state)
    redundant_pairs = redundancy_analysis(prepared, FEATURE_COLUMNS, args.correlation_threshold)
    permutation = permutation_importance(
        model,
        test_frame,
        FEATURE_COLUMNS,
        metrics,
        threshold=args.threshold,
        repeats=args.permutation_repeats,
        random_state=args.random_state,
    )
    candidates = choose_ablation_candidates(importance, permutation, mutual_info, args.ablation_candidate_count)
    ablation = ablation_analysis(
        train_frame,
        test_frame,
        FEATURE_COLUMNS,
        candidates,
        metrics,
        threshold=args.threshold,
        random_state=args.random_state,
    )
    diagnostics = combine_diagnostics(stats, importance, pca_scores, mutual_info, permutation, ablation)

    write_json(
        args.output_dir / "baseline_metrics.json",
        {
            **metrics,
            "rows": int(len(prepared)),
            "datasets": int(prepared["dataset"].nunique()),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "features": int(len(FEATURE_COLUMNS)),
            "threshold": float(args.threshold),
            "target": f"{DIRTY_COLUMN} = {TARGET_COLUMN} > 0",
        },
    )
    diagnostics.to_csv(args.output_dir / "feature_diagnostics.csv", index=False)
    pca_components.to_csv(args.output_dir / "pca_components.csv", index=False)
    redundant_pairs.to_csv(args.output_dir / "redundant_pairs.csv", index=False)
    permutation.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    ablation.to_csv(args.output_dir / "ablation_results.csv", index=False)
    write_summary(args.output_dir / "summary.md", diagnostics, metrics, args)

    print("Stayed-dirty classifier diagnostics")
    print(f"Rows: {len(prepared)}")
    print(f"Datasets: {prepared['dataset'].nunique()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Wrote diagnostics: {(args.output_dir / 'feature_diagnostics.csv').resolve()}")
    return 0


def grouped_holdout(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame["dataset"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame[FEATURE_COLUMNS], frame[DIRTY_COLUMN], groups))
    return frame.iloc[train_index].copy(), frame.iloc[test_index].copy()


def fit_classifier(frame: pd.DataFrame, feature_columns: list[str], *, random_state: int):
    model = make_classifier(random_state, class_weight="none")
    model.fit(frame[feature_columns], frame[DIRTY_COLUMN])
    return model


def predict_classifier(model, frame: pd.DataFrame, feature_columns: list[str], *, threshold: float) -> pd.DataFrame:
    predictions = frame[["dataset", "face_id", "form_type", TARGET_COLUMN, DIRTY_COLUMN]].copy()
    probability = model.predict_proba(frame[feature_columns])[:, 1]
    predictions[PROBABILITY_COLUMN] = probability
    predictions[PREDICTION_COLUMN] = (probability >= threshold).astype(int)
    return predictions


def feature_statistics(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    target = frame[DIRTY_COLUMN].astype(int)
    rows = []
    for feature in feature_columns:
        values = frame[feature]
        if feature == "form_type":
            rows.append(
                {
                    "feature": feature,
                    "kind": "categorical",
                    "zero_pct": np.nan,
                    "unique_count": int(values.astype(str).nunique(dropna=False)),
                    "target_corr": np.nan,
                }
            )
            continue

        clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
        rows.append(
            {
                "feature": feature,
                "kind": "numeric",
                "zero_pct": float((clean == 0.0).mean()),
                "unique_count": int(clean.nunique(dropna=False)),
                "target_corr": safe_corr(clean, target),
            }
        )
    return pd.DataFrame(rows)


def model_importance(model) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": model.booster_.feature_name(),
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    )
    gain_total = float(importance["gain_importance"].sum())
    importance["gain_importance_pct"] = (
        importance["gain_importance"] / gain_total if gain_total > 0.0 else 0.0
    )
    return importance.sort_values(["gain_importance", "split_importance"], ascending=False)


def pca_analysis(
    frame: pd.DataFrame,
    feature_columns: list[str],
    component_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded, owner = encoded_features(frame, feature_columns)
    if encoded.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_components = min(component_count, encoded.shape[0], encoded.shape[1])
    scaled = StandardScaler().fit_transform(encoded)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(scaled)

    components = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    loadings = pd.DataFrame(np.abs(pca.components_), columns=encoded.columns)
    weighted = loadings.mul(pca.explained_variance_ratio_, axis=0).sum(axis=0)
    scores = aggregate_encoded_scores(weighted, owner, "pca_loading_score")
    return components, scores


def mutual_information(frame: pd.DataFrame, feature_columns: list[str], *, random_state: int) -> pd.DataFrame:
    encoded, owner = encoded_features(frame, feature_columns)
    target = frame[DIRTY_COLUMN].astype(int).to_numpy()
    scores = mutual_info_classif(encoded, target, discrete_features="auto", random_state=random_state)
    return aggregate_encoded_scores(pd.Series(scores, index=encoded.columns), owner, "mutual_info")


def redundancy_analysis(
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    encoded, owner = encoded_features(frame, feature_columns)
    numeric = encoded.loc[:, encoded.std(ddof=0) > 1e-12]
    corr = numeric.corr().abs().fillna(0.0)
    pairs = []
    columns = list(corr.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(corr.loc[left, right])
            if value >= threshold and owner[left] != owner[right]:
                pairs.append(
                    {
                        "left_feature": owner[left],
                        "right_feature": owner[right],
                        "correlation": value,
                    }
                )
    if not pairs:
        return pd.DataFrame(columns=["left_feature", "right_feature", "correlation"])
    return pd.DataFrame(pairs).drop_duplicates().sort_values("correlation", ascending=False)


def permutation_importance(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    baseline_metrics: dict[str, float],
    *,
    threshold: float,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_state)
    for feature in feature_columns:
        accuracy_drops = []
        f1_drops = []
        for _ in range(repeats):
            permuted = frame.copy()
            values = rng.permutation(permuted[feature].to_numpy())
            if feature == "form_type":
                permuted[feature] = pd.Categorical(values, categories=frame[feature].cat.categories)
            else:
                permuted[feature] = values
            predictions = predict_classifier(model, permuted, feature_columns, threshold=threshold)
            metrics = evaluate_predictions(predictions, threshold=threshold)
            accuracy_drops.append(baseline_metrics["accuracy"] - metrics["accuracy"])
            f1_drops.append(baseline_metrics["dirty_f1"] - metrics["dirty_f1"])
        rows.append(
            {
                "feature": feature,
                "permutation_accuracy_drop": float(np.mean(accuracy_drops)),
                "permutation_f1_drop": float(np.mean(f1_drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_accuracy_drop", ascending=False)


def ablation_analysis(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    candidates: list[str],
    baseline_metrics: dict[str, float],
    *,
    threshold: float,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for feature in candidates:
        kept = [column for column in feature_columns if column != feature]
        model = fit_classifier(train_frame, kept, random_state=random_state)
        predictions = predict_classifier(model, test_frame, kept, threshold=threshold)
        metrics = evaluate_predictions(predictions, threshold=threshold)
        rows.append(
            {
                "feature": feature,
                "ablation_accuracy_change": metrics["accuracy"] - baseline_metrics["accuracy"],
                "ablation_f1_change": metrics["dirty_f1"] - baseline_metrics["dirty_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("ablation_accuracy_change", ascending=False)


def choose_ablation_candidates(
    importance: pd.DataFrame,
    permutation: pd.DataFrame,
    mutual_info: pd.DataFrame,
    count: int,
) -> list[str]:
    merged = (
        importance[["feature", "gain_importance_pct"]]
        .merge(permutation, on="feature", how="outer")
        .merge(mutual_info, on="feature", how="outer")
        .fillna(0.0)
    )
    merged["weak_signal_score"] = (
        merged["gain_importance_pct"].rank(ascending=True)
        + merged["permutation_accuracy_drop"].rank(ascending=True)
        + merged["mutual_info"].rank(ascending=True)
    )
    return merged.sort_values("weak_signal_score")["feature"].head(count).tolist()


def combine_diagnostics(
    stats: pd.DataFrame,
    importance: pd.DataFrame,
    pca_scores: pd.DataFrame,
    mutual_info: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = stats.merge(importance, on="feature", how="left")
    for extra in (pca_scores, mutual_info, permutation, ablation):
        diagnostics = diagnostics.merge(extra, on="feature", how="left")
    diagnostics = diagnostics.fillna(0.0)
    diagnostics["recommendation"] = diagnostics.apply(recommend_feature, axis=1)
    return diagnostics.sort_values(
        ["recommendation", "gain_importance", "permutation_accuracy_drop"],
        ascending=[True, False, False],
    )


def recommend_feature(row: pd.Series) -> str:
    if row["unique_count"] <= 1:
        return "remove_constant"
    if row.get("ablation_accuracy_change", 0.0) > 0.001:
        return "remove_candidate"
    if row.get("gain_importance_pct", 0.0) < 0.001 and row.get("mutual_info", 0.0) < 0.001:
        return "low_signal"
    return "keep"


def encoded_features(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    encoded_parts = []
    owner: dict[str, str] = {}
    for feature in feature_columns:
        if feature == "form_type":
            dummies = pd.get_dummies(frame[feature].astype(str), prefix=feature, dtype=float)
            encoded_parts.append(dummies)
            owner.update({column: feature for column in dummies.columns})
        else:
            encoded_parts.append(
                pd.DataFrame(
                    {feature: pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)},
                    index=frame.index,
                )
            )
            owner[feature] = feature
    return pd.concat(encoded_parts, axis=1), owner


def aggregate_encoded_scores(scores: pd.Series, owner: dict[str, str], column_name: str) -> pd.DataFrame:
    rows = {}
    for encoded_feature, value in scores.items():
        feature = owner[encoded_feature]
        rows[feature] = max(float(value), rows.get(feature, 0.0))
    return pd.DataFrame({"feature": list(rows), column_name: list(rows.values())})


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return 0.0
    value = left.corr(right)
    return 0.0 if pd.isna(value) else float(value)


def write_json(path: Path, values: dict[str, float | int | str]) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def write_summary(path: Path, diagnostics: pd.DataFrame, metrics: dict[str, float], args: argparse.Namespace) -> None:
    removable = diagnostics[diagnostics["recommendation"].isin(["remove_constant", "remove_candidate", "low_signal"])]
    lines = [
        "# Classifier Feature Diagnostics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Dirty F1: {metrics['dirty_f1']:.4f}",
        f"- Threshold: {args.threshold:.3f}",
        "",
        "## Weakest Features",
    ]
    if removable.empty:
        lines.append("- None flagged by the default thresholds.")
    else:
        for _, row in removable.head(15).iterrows():
            lines.append(f"- `{row['feature']}`: {row['recommendation']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
