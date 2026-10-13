#!/usr/bin/env python3
"""Audit ROGII inputs and evaluate a reproducible, whole-well baseline.

Usage: python scripts/audit_data.py --data-dir /path/to/competition --output reports/data_audit.json

This audit deliberately records train/test feature overlap without using it to
recover prediction targets. Its baseline never accesses training-only columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rogii.baseline import INFERENCE_COLUMNS, predict_well


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {label: float(np.quantile(a, q)) for label, q in (
        ("min", 0.0), ("median", 0.5), ("p90", 0.9), ("max", 1.0)
    )}


def feature_hash(df: pd.DataFrame) -> str:
    """Hash only public prediction features, not labels or formation columns."""
    data = pd.util.hash_pandas_object(df.loc[:, list(INFERENCE_COLUMNS)], index=False)
    return hashlib.sha256(data.to_numpy().tobytes()).hexdigest()


def audit(data_dir: Path, seed: int = 42, validation_wells: int = 100, development_wells: int = 20) -> dict:
    paths = sorted((data_dir / "train").glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No training wells in {data_dir / 'train'}")
    heldout = paths.copy()
    random.Random(seed).shuffle(heldout)
    development_ids = {p.name.split("__")[0] for p in heldout[:development_wells]}
    evaluation_ids = {p.name.split("__")[0] for p in heldout[development_wells:development_wells + validation_wells]}
    heldout_ids = development_ids | evaluation_ids
    distributions = defaultdict(list)
    typewell_groups = defaultdict(list)
    feature_groups = defaultdict(list)
    missing = Counter()
    total_rows = total_prediction_rows = total_missing_gr = 0
    prefix_violations = []
    md_violations = []
    validation = {"development": [], "evaluation": []}
    per_baseline_sse = {"development": Counter(), "evaluation": Counter()}
    validation_rows = Counter()
    for path in paths:
        well_id = path.name.split("__")[0]
        df = pd.read_csv(path)
        known = df.TVT_input.notna().to_numpy()
        suffix = ~known
        k = int(known.sum())
        total_rows += len(df)
        total_prediction_rows += int(suffix.sum())
        total_missing_gr += int(df.GR.isna().sum())
        if not known[:k].all() or known[k:].any():
            prefix_violations.append(well_id)
        if not np.all(np.diff(df.MD.to_numpy()) > 0):
            md_violations.append(well_id)
        for col in df.columns:
            missing[col] += int(df[col].isna().sum())
        distributions["rows_per_well"].append(len(df))
        distributions["known_rows_per_well"].append(k)
        distributions["missing_gr_fraction_per_well"].append(float(df.GR.isna().mean()))
        distributions["start_x"].append(float(df.X.iloc[0]))
        distributions["start_y"].append(float(df.Y.iloc[0]))
        distributions["suffix_tvt_std"].append(float(df.loc[suffix, "TVT"].std(ddof=0)))
        distributions["suffix_end_minus_last_known_tvt"].append(float(df.TVT.iloc[-1] - df.TVT_input.dropna().iloc[-1]))
        structure = df.TVT.to_numpy() + df.Z.to_numpy()
        distributions["maximum_structure_step"].append(float(np.max(np.abs(np.diff(structure)))))
        if "ANCC" in df and df.ANCC.notna().any():
            residual = structure - df.ANCC.to_numpy()
            distributions["tvt_plus_z_minus_ancc_within_well_range"].append(float(np.nanmax(residual) - np.nanmin(residual)))
        typewell_path = path.with_name(f"{well_id}__typewell.csv")
        typewell_groups[hashlib.sha256(typewell_path.read_bytes()).hexdigest()].append(well_id)
        feature_groups[feature_hash(df)].append(well_id)
        if well_id in heldout_ids:
            # The predictor receives a copy of only the public columns.
            features = df.loc[:, list(INFERENCE_COLUMNS)].copy()
            target = df.loc[suffix, "TVT"].to_numpy()
            split = "development" if well_id in development_ids else "evaluation"
            record = {"well_id": well_id, "prediction_rows": len(target)}
            for name, weight in (("last_known_tvt", 0.0), ("shrunk_geometry", 0.25)):
                pred = predict_well(features, None, geometry_weight=weight)[suffix]
                sse = float(np.sum((pred - target) ** 2))
                per_baseline_sse[split][name] += sse
                record[f"{name}_rmse"] = float(np.sqrt(sse / len(target)))
            validation_rows[split] += len(target)
            validation[split].append(record)
    overlap = []
    test_count = 0
    for path in sorted((data_dir / "test").glob("*__horizontal_well.csv")):
        test_count += 1
        well_id = path.name.split("__")[0]
        matched = feature_groups.get(feature_hash(pd.read_csv(path)), [])
        if matched:
            overlap.append({"test_well_id": well_id, "matching_train_well_ids": matched})
    return {
        "dataset_path": str(data_dir),
        "training_wells": len(paths),
        "test_wells": test_count,
        "training_rows": total_rows,
        "training_prediction_rows": total_prediction_rows,
        "overall_missing_gr_fraction": total_missing_gr / total_rows,
        "missing_values_by_training_column": dict(missing),
        "known_prefix_violations": prefix_violations,
        "nonincreasing_md_wells": md_violations,
        "distributions": {name: summary(values) for name, values in sorted(distributions.items())},
        "unique_typewell_files": len(typewell_groups),
        "duplicate_typewell_groups": sorted((v for v in typewell_groups.values() if len(v) > 1), key=lambda v: (-len(v), v)),
        "duplicate_horizontal_feature_groups": [v for v in feature_groups.values() if len(v) > 1],
        "train_test_public_feature_overlap": overlap,
        "overlap_policy": "Overlap is diagnostic only. Never copy train labels to test. Evaluate complete held-out wells; group duplicates when fitting models.",
        "validation": {
            "split": "sorted well files shuffled with Python random.Random(seed); first development_wells for development, next validation_wells for independent evaluation",
            "seed": seed,
            "development": {
                "wells": len(validation["development"]),
                "well_ids": sorted(development_ids),
                "prediction_rows": validation_rows["development"],
                "pooled_rmse": {name: float(np.sqrt(sse / validation_rows["development"])) for name, sse in per_baseline_sse["development"].items()},
                "per_well": validation["development"],
            },
            "evaluation": {
                "wells": len(validation["evaluation"]),
                "well_ids": sorted(evaluation_ids),
                "prediction_rows": validation_rows["evaluation"],
                "pooled_rmse": {name: float(np.sqrt(sse / validation_rows["evaluation"])) for name, sse in per_baseline_sse["evaluation"].items()},
                "per_well": validation["evaluation"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-wells", type=int, default=100)
    parser.add_argument("--development-wells", type=int, default=20)
    args = parser.parse_args()
    if args.validation_wells < 1 or args.development_wells < 1:
        parser.error("Both development and validation well counts must be positive")
    report = audit(args.data_dir, args.seed, args.validation_wells, args.development_wells)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
        print(json.dumps({split: report["validation"][split]["pooled_rmse"] for split in ("development", "evaluation")}, indent=2))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
