"""Compare two runs on identical OOF folds with paired tail-risk bootstraps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boneage.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/manifest_standardized_v2.parquet"),
    )
    parser.add_argument(
        "--quality-metrics",
        type=Path,
        default=Path("reports/eda-quality/quality_metrics.csv"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_folds(run_dir: Path, folds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in folds:
        path = run_dir / f"fold_{fold}" / "oof.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed fold predictions: {path}")
        frame = pd.read_csv(path)
        if set(frame["fold"].astype(int)) != {fold}:
            raise ValueError(f"Unexpected fold values in {path}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["image_id"].duplicated().any():
        raise ValueError(f"Duplicate OOF image IDs in {run_dir}")
    return combined.sort_values("image_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        **compute_metrics(frame["target_months"], frame[prediction_column]),
    }


def paired_bootstrap(
    baseline_absolute_error: np.ndarray,
    candidate_absolute_error: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    rng = np.random.default_rng(seed)
    rows = len(baseline_absolute_error)
    mae_deltas: list[np.ndarray] = []
    p95_deltas: list[np.ndarray] = []
    tail_deltas: list[np.ndarray] = []
    batch_size = 100
    for start in range(0, samples, batch_size):
        current = min(batch_size, samples - start)
        indices = rng.integers(0, rows, size=(current, rows))
        baseline_sample = baseline_absolute_error[indices]
        candidate_sample = candidate_absolute_error[indices]
        mae_deltas.append(np.mean(candidate_sample - baseline_sample, axis=1))
        p95_deltas.append(
            np.quantile(candidate_sample, 0.95, axis=1)
            - np.quantile(baseline_sample, 0.95, axis=1)
        )
        tail_deltas.append(
            np.mean(
                (candidate_sample > 24.0).astype(np.float32)
                - (baseline_sample > 24.0).astype(np.float32),
                axis=1,
            )
        )

    def interval(parts: list[np.ndarray]) -> list[float]:
        values = np.concatenate(parts)
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "mae_delta_ci_95": interval(mae_deltas),
        "p95_delta_ci_95": interval(p95_deltas),
        "over_24_rate_delta_ci_95": interval(tail_deltas),
    }


def tail_transition_labels(
    baseline_absolute_error: np.ndarray,
    candidate_absolute_error: np.ndarray,
    *,
    threshold: float = 24.0,
) -> np.ndarray:
    """Classify paired samples by strict catastrophic-error threshold transition."""
    baseline = np.asarray(baseline_absolute_error, dtype=float)
    candidate = np.asarray(candidate_absolute_error, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("Paired absolute-error arrays must be one-dimensional and equal length")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    baseline_tail = baseline > threshold
    candidate_tail = candidate > threshold
    return np.select(
        [
            baseline_tail & candidate_tail,
            baseline_tail & ~candidate_tail,
            ~baseline_tail & candidate_tail,
        ],
        ["both_over_24", "fixed_below_24", "new_over_24"],
        default="both_within_24",
    )


def subgroup_comparison(data: pd.DataFrame, group: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for value, subset in data.groupby(group):
        baseline = metrics(subset, "baseline_prediction")
        candidate = metrics(subset, "candidate_prediction")
        rows.append(
            {
                group: int(value),
                "rows": int(len(subset)),
                "baseline_mae": baseline["mae"],
                "candidate_mae": candidate["mae"],
                "mae_delta": float(candidate["mae"] - baseline["mae"]),
                "baseline_p95": baseline["p95_ae"],
                "candidate_p95": candidate["p95_ae"],
                "baseline_over_24_count": baseline["over_24_count"],
                "candidate_over_24_count": candidate["over_24_count"],
                "baseline_over_24_rate": baseline["over_24_rate"],
                "candidate_over_24_rate": candidate["over_24_rate"],
                "baseline_over_24_rate_upper_95": baseline["over_24_rate_upper_95"],
                "candidate_over_24_rate_upper_95": candidate["over_24_rate_upper_95"],
                "baseline_max_ae": baseline["max_ae"],
                "candidate_max_ae": candidate["max_ae"],
            }
        )
    return pd.DataFrame(rows)


def transition_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Summarize threshold crossings within each demographic subgroup."""
    required = {
        group,
        "tail_transition",
        "target_months",
        "baseline_absolute_error",
        "candidate_absolute_error",
        "absolute_error_delta",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing transition-summary columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for (value, transition), subset in data.groupby([group, "tail_transition"]):
        rows.append(
            {
                group: int(value),
                "tail_transition": str(transition),
                "rows": int(len(subset)),
                "mean_target_months": float(subset["target_months"].mean()),
                "mean_baseline_ae": float(subset["baseline_absolute_error"].mean()),
                "mean_candidate_ae": float(subset["candidate_absolute_error"].mean()),
                "mean_ae_delta": float(subset["absolute_error_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def transition_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Report raw-image QC prevalence for each tail-threshold transition."""
    if "tail_transition" not in data:
        raise ValueError("Missing transition-summary column: tail_transition")
    flag_columns = sorted(column for column in data if column.startswith("flag_"))
    if not flag_columns:
        raise ValueError("No flag_* quality columns were provided")

    rows: list[dict[str, float | int | str]] = []
    for transition, subset in data.groupby("tail_transition"):
        row: dict[str, float | int | str] = {
            "tail_transition": str(transition),
            "rows": int(len(subset)),
        }
        for flag in flag_columns:
            values = subset[flag].fillna(False).astype(bool)
            row[f"{flag}_count"] = int(values.sum())
            row[f"{flag}_rate"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_decision(
    deltas: dict[str, float | int],
    bootstrap: dict[str, list[float]],
    *,
    ambiguity_margin_months: float = 0.25,
) -> dict[str, bool | float]:
    """Apply promotion gates and flag comparisons needing a current-code control."""
    mae_delta = float(deltas["mae"])
    mae_ci = bootstrap["mae_delta_ci_95"]
    if len(mae_ci) != 2 or mae_ci[0] > mae_ci[1]:
        raise ValueError("mae_delta_ci_95 must contain ordered lower and upper bounds")
    if ambiguity_margin_months <= 0:
        raise ValueError("ambiguity_margin_months must be positive")

    mae_ci_includes_zero = mae_ci[0] <= 0.0 <= mae_ci[1]
    p95_significant_regression = bootstrap["p95_delta_ci_95"][0] > 0.0
    tail_significant_regression = bootstrap["over_24_rate_delta_ci_95"][0] > 0.0
    return {
        "ambiguity_margin_months": ambiguity_margin_months,
        "mae_improves": mae_delta < 0.0,
        "mae_improvement_ci_excludes_zero": mae_ci[1] < 0.0,
        "mae_ci_includes_zero": mae_ci_includes_zero,
        "requires_fresh_control": (
            abs(mae_delta) < ambiguity_margin_months or mae_ci_includes_zero
        ),
        "p95_significant_regression": p95_significant_regression,
        "over_24_rate_significant_regression": tail_significant_regression,
        "promote_screen": (
            mae_delta < 0.0
            and not p95_significant_regression
            and not tail_significant_regression
        ),
    }


def main() -> None:
    args = parse_args()
    folds = sorted(set(args.folds))
    baseline = load_folds(args.baseline_run, folds).rename(
        columns={"prediction_months": "baseline_prediction"}
    )
    candidate = load_folds(args.candidate_run, folds).rename(
        columns={"prediction_months": "candidate_prediction"}
    )
    paired = baseline.merge(
        candidate[["image_id", "target_months", "fold", "candidate_prediction"]],
        on=["image_id", "target_months", "fold"],
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("Baseline and candidate do not cover identical OOF rows")

    manifest = pd.read_parquet(args.manifest)[["image_id", "sex"]]
    paired = paired.merge(manifest, on="image_id", validate="one_to_one")
    quality = pd.read_csv(args.quality_metrics).rename(columns={"id": "image_id"})
    if quality["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image IDs in {args.quality_metrics}")
    quality_flags = sorted(column for column in quality if column.startswith("flag_"))
    if not quality_flags:
        raise ValueError(f"No flag_* columns in {args.quality_metrics}")
    paired = paired.merge(
        quality[["image_id", *quality_flags]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    paired["age_band_start"] = (paired["target_months"] // 24 * 24).astype(int)
    paired["baseline_absolute_error"] = (
        paired["baseline_prediction"] - paired["target_months"]
    ).abs()
    paired["candidate_absolute_error"] = (
        paired["candidate_prediction"] - paired["target_months"]
    ).abs()
    paired["absolute_error_delta"] = (
        paired["candidate_absolute_error"] - paired["baseline_absolute_error"]
    )
    paired["tail_transition"] = tail_transition_labels(
        paired["baseline_absolute_error"].to_numpy(),
        paired["candidate_absolute_error"].to_numpy(),
    )

    baseline_metrics = metrics(paired, "baseline_prediction")
    candidate_metrics = metrics(paired, "candidate_prediction")
    bootstrap = paired_bootstrap(
        paired["baseline_absolute_error"].to_numpy(),
        paired["candidate_absolute_error"].to_numpy(),
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    deltas = {
        "mae": float(candidate_metrics["mae"] - baseline_metrics["mae"]),
        "p95_ae": float(candidate_metrics["p95_ae"] - baseline_metrics["p95_ae"]),
        "over_24_count": int(
            candidate_metrics["over_24_count"] - baseline_metrics["over_24_count"]
        ),
        "over_24_rate": float(
            candidate_metrics["over_24_rate"] - baseline_metrics["over_24_rate"]
        ),
        "max_ae": float(candidate_metrics["max_ae"] - baseline_metrics["max_ae"]),
    }
    decision = comparison_decision(deltas, bootstrap)
    transition_counts = paired["tail_transition"].value_counts()
    result = {
        "folds": folds,
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_baseline": deltas,
        "tail_transitions": {
            label: int(transition_counts.get(label, 0))
            for label in (
                "both_within_24",
                "fixed_below_24",
                "new_over_24",
                "both_over_24",
            )
        },
        "paired_bootstrap": bootstrap,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    paired.sort_values("absolute_error_delta", ascending=False).to_csv(
        args.output_dir / "per_image_deltas.csv", index=False
    )
    paired.loc[paired["candidate_absolute_error"] > 24.0].sort_values(
        "candidate_absolute_error", ascending=False
    ).to_csv(args.output_dir / "candidate_over_24.csv", index=False)
    for transition in ("new_over_24", "fixed_below_24"):
        ascending = transition == "fixed_below_24"
        paired.loc[paired["tail_transition"] == transition].sort_values(
            "absolute_error_delta", ascending=ascending
        ).to_csv(args.output_dir / f"{transition}.csv", index=False)
    subgroup_comparison(paired, "fold").to_csv(
        args.output_dir / "by_fold.csv", index=False
    )
    subgroup_comparison(paired, "sex").to_csv(
        args.output_dir / "by_sex.csv", index=False
    )
    subgroup_comparison(paired, "age_band_start").to_csv(
        args.output_dir / "by_age_band.csv", index=False
    )
    transition_summary(paired, "sex").to_csv(
        args.output_dir / "tail_transitions_by_sex.csv", index=False
    )
    transition_summary(paired, "age_band_start").to_csv(
        args.output_dir / "tail_transitions_by_age_band.csv", index=False
    )
    transition_quality_summary(paired).to_csv(
        args.output_dir / "tail_transition_quality.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
