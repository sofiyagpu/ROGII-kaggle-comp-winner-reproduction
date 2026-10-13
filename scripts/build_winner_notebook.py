#!/usr/bin/env python3
"""Build the standalone, attributed Kaggle six-family inference notebook."""
from pathlib import Path
import argparse
import ast
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("0719_V1", "w5833946/rogii-0719-v1-cv509"),
    ("0724_V1", "w5833946/rogii-0724-v1-cv486"),
    ("0729_V3", "w5833946/rogii-0729-v3-cv553"),
    ("0801_V1", "w5833946/rogii-0801-v1-cv516"),
    ("0801_V2", "w5833946/rogii-0801-v2-cv480"),
    ("0803_V2", "w5833946/rogii-0803-v2-cv500"),
]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    if kind == "code":
        ast.parse(source)
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def build(username):
    cells = [cell("markdown", """
        # ROGII · First-place reproduction + insights

        This notebook adapts Ruby / **w5833946**'s public
        [submission notebook](https://www.kaggle.com/code/w5833946/submit-reproduce)
        and [Apache-2.0 source repository](https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby).
        The model designs, trained weights, geographic gate, and ensemble weights are the original author's work.
        This is an attributed late-submission reproduction, not a claim of an original competition win or a newly achieved score.
        The [technical write-up](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution)
        describes the method.

        **Run:** attach the competition and the six datasets listed below; select a CUDA GPU and turn Internet **off**.
        Save & Run All, then submit the generated `/kaggle/working/submission.csv`.
        It predicts the test files mounted at execution time, including Kaggle's hidden test rerun.
        The public sample test is not the hidden leaderboard test.

        The six model families each average 15 folds. The upstream notebook also runs `0802_V2`,
        but that family has zero contribution to both final blends and is omitted here.
        We preserve the original float32 inference fix, geographic thresholds, model weights, and blending order.
        Changes to orchestration: checked subprocesses, scoped temporary directories, read-only weight symlinks,
        runtime path discovery, exact prediction-key checks, and a verified final row order.
        Auxiliary caches use this run's working directory rather than the author's local filesystem paths.

        Model/config pickles and code are loaded only from the explicitly listed author-owned datasets.
        These inputs are not vendored into this repository. The original notebook uses Kaggle's installed
        scientific packages and has **no additional wheel datasets**. This notebook does not install or download packages.
        The six inputs occupy roughly 20 GB uncompressed in Kaggle's read-only input mount; weights are not copied into working storage.
        Full hidden-test runtime and a fresh leaderboard score must be measured by an actual Kaggle run.
    """), cell("code", """
        import gc
        import hashlib
        import importlib.metadata
        import json
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys
        import tempfile
        import time

        import numpy as np
        import pandas as pd
        import torch
        import timm
        import pyarrow
        import scipy
        import sklearn
        import numba
        from tqdm.auto import tqdm

        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle CUDA GPU before running this six-family ensemble.")
        WORK = Path("/kaggle/working")
        WORK.mkdir(parents=True, exist_ok=True)
        RUN_ROOT = Path(tempfile.mkdtemp(prefix="rogii_reproduction_", dir=WORK))
        RUN_ARCHIVED_OOF = True  # Diagnostic on author-provided OOF; does not fit or tune any model.
        started_at = time.time()
        versions = {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "torch", "torchvision", "timm", "pyarrow", "scipy", "scikit-learn", "numba"
        )}
        print("GPU:", torch.cuda.get_device_name(0))
        print("Packages:", versions)
        print("Working directory:", RUN_ROOT)
    """), cell("code", """
        DATASET_REFS = [
            ("0719_V1", "w5833946/rogii-0719-v1-cv509"),
            ("0724_V1", "w5833946/rogii-0724-v1-cv486"),
            ("0729_V3", "w5833946/rogii-0729-v3-cv553"),
            ("0801_V1", "w5833946/rogii-0801-v1-cv516"),
            ("0801_V2", "w5833946/rogii-0801-v2-cv480"),
            ("0803_V2", "w5833946/rogii-0803-v2-cv500"),
        ]
        COMPETITION = "rogii-wellbore-geology-prediction"

        def locate_mount(ref, competition=False):
            owner, slug = (None, ref) if competition else ref.split("/", 1)
            candidates = ([Path("/kaggle/input/competitions") / slug] if competition else
                          [Path("/kaggle/input/datasets") / owner / slug])
            candidates.append(Path("/kaggle/input") / slug)
            for candidate in candidates:
                if candidate.is_dir():
                    return candidate
            raise FileNotFoundError(f"Attach {ref!r} as a Kaggle input. Tried: {candidates}")

        DATA = locate_mount(COMPETITION, competition=True)
        FAMILIES = [(marker, locate_mount(ref)) for marker, ref in DATASET_REFS]
        for marker, source in FAMILIES:
            for required in ("seq_NN_main.py", "seq_NN_cfg.py", "cfg.pkl", "models.pkl"):
                if not (source / required).is_file():
                    raise FileNotFoundError(f"{marker}: missing {source / required}")
        if not (DATA / "train").is_dir() or not (DATA / "test").is_dir():
            raise FileNotFoundError(f"Expected train and test folders under {DATA}")
        print("Competition:", DATA)
        print("Model families:", {marker: str(path) for marker, path in FAMILIES})
    """), cell("markdown", """
        ## Two findings from the training wells

        **1. TVT is a structural depth.** Within every training well, `TVT + Z - ANCC` is constant to
        within 0.07 ft. `ANCC` is a training-only formation top, so predicting TVT along the lateral
        means predicting where the formation surface sits relative to the bit, not a free-form curve.

        **2. The naive answer is far off.** Holding the last known `TVT_input` flat over the prediction
        interval gives about 16 ft RMSE on the training suffixes. The first-place families reach
        4.80-5.54 ft archived OOF RMSE, so almost all of the score comes from reading GR against the typewell.

        This cell recomputes both from the mounted training data. It only reads training labels and never touches test targets.
    """), cell("code", """
        rows = []
        for path in sorted((DATA / "train").glob("*__horizontal_well.csv")):
            df = pd.read_csv(path, usecols=["Z", "ANCC", "TVT", "TVT_input"])
            surface = df["TVT"] + df["Z"] - df["ANCC"]
            suffix = df["TVT_input"].isna()
            last_known = df["TVT_input"].dropna().iloc[-1]
            rows.append({"surface_range": surface.max() - surface.min(), "rows": int(suffix.sum()),
                         "sse_last": float(((df.loc[suffix, "TVT"] - last_known) ** 2).sum())})
        insights = pd.DataFrame(rows)
        print(f"Wells: {len(insights)} ({insights['surface_range'].isna().sum()} without ANCC)")
        print(f"Max within-well range of TVT + Z - ANCC: {insights['surface_range'].max():.2f} ft")
        print(f"Last-known-TVT RMSE over all training suffixes: "
              f"{np.sqrt(insights['sse_last'].sum() / insights['rows'].sum()):.3f} ft")
        del rows, insights
    """), cell("markdown", """
        ## Inference with the released models

        Each family runs in a fresh Python process so similarly named source modules cannot leak across families.
        Checkpoints remain in the read-only input mount. Only the configuration and Python source are copied.
        A subprocess failure stops execution; it cannot silently reuse an old submission.
        All models see the complete runtime test set. Temporary sources and outputs are removed only inside
        the directory created for that family; the notebook never broadly deletes `/kaggle/working`.
    """), cell("code", r'''
        # Run in a separate process so each pickle resolves its own seq_NN_cfg.CFG class.
        PATCH_CONFIG = "\n".join([
            "import sys",
            "from pathlib import Path",
            "import pandas as pd",
            "cfg = pd.read_pickle('cfg.pkl')",
            "cfg.amp_dtype = 'float32'",  # Exactly the upstream Kaggle inference fix.
            "cfg.well_limit = None",     # Never truncate hidden-test wells.
            "cfg.project_root = Path(sys.argv[1])",
            "cfg.PF_heatmap_cache_dir = cfg.project_root / 'PF_cache'",
            "cfg.PF_sample_cache_dir = cfg.project_root / 'PF_sample_cache'",
            "pd.to_pickle(cfg, 'cfg.pkl')",
        ])

        KEY_COLS = ["well_id", "submit_index"]
        GEO_COLS = ["geo_nbr_distance", "geo_radial_extrap_score", "geo_nbr_path_alignment",
                    "geo_nbr_distance_q10", "geo_prefix_weight_ratio"]
        XY_MARKER = "0801_V2"
        test_predictions = {}
        provenance = {}

        for marker, source in FAMILIES:
            family_started = time.time()
            print(f"\n=== {marker} ===", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"{marker}_", dir=RUN_ROOT) as directory:
                workspace = Path(directory)
                code_files = sorted(source.glob("*.py"))
                for item in code_files:
                    shutil.copy2(item, workspace / item.name)
                shutil.copy2(source / "cfg.pkl", workspace / "cfg.pkl")
                (workspace / "models.pkl").symlink_to(source / "models.pkl")
                provenance[marker] = {
                    "dataset": dict(DATASET_REFS)[marker],
                    "model_bytes": (source / "models.pkl").stat().st_size,
                    "source_sha256": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                      for item in code_files + [source / "cfg.pkl"]},
                }
                subprocess.run([sys.executable, "-c", PATCH_CONFIG, str(RUN_ROOT)],
                               cwd=workspace, check=True)
                output = workspace / "inference"
                subprocess.run([
                    sys.executable, "seq_NN_main.py", "--submit-mode",
                    "--data-dir", str(DATA), "--train-output-dir", str(workspace),
                    "--output-dir", str(output), "--device", "cuda",
                ], cwd=workspace, check=True)
                result_path = output / "submission_details.pqt"
                if not result_path.is_file():
                    raise FileNotFoundError(f"{marker}: inference did not produce {result_path}")
                columns = KEY_COLS + ["TVT_pred"] + (GEO_COLS if marker == XY_MARKER else [])
                test_predictions[marker] = pd.read_parquet(result_path, columns=columns)
            provenance[marker]["inference_seconds"] = round(time.time() - family_started, 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{marker}: {len(test_predictions[marker]):,} rows, "
                  f"{provenance[marker]['inference_seconds']:.1f} seconds", flush=True)
    '''), cell("markdown", """
        ## Author's conditional ensemble

        The geographic blend is used only when the reference family's mean well-level diagnostics pass
        all five original thresholds. Otherwise the GR-based blend is used. No weights or thresholds are
        tuned against this test set. Missing/non-finite geographic diagnostics naturally fail the gate.
        A missing model prediction is an error, not a reason to silently substitute a weaker fallback.
    """), cell("code", """
        GR_ONLY = [("0719_V1", 1.0), ("0729_V3", 0.5), ("0801_V1", 1.0)]
        XY_BASED = [("0719_V1", 0.25), ("0801_V1", 0.25), ("0724_V1", 1.0),
                    ("0801_V2", 1.0), ("0803_V2", 1.0)]

        def blend_predictions(frames):
            ordered = {}
            for marker, frame in frames.items():
                frame = frame.sort_values(KEY_COLS).reset_index(drop=True).copy()
                if frame.empty or frame[KEY_COLS].isna().any().any():
                    raise ValueError(f"{marker}: empty predictions or missing keys")
                if frame.duplicated(KEY_COLS).any():
                    raise ValueError(f"{marker}: duplicate prediction keys")
                if not np.isfinite(frame["TVT_pred"].to_numpy()).all():
                    raise ValueError(f"{marker}: non-finite predictions")
                ordered[marker] = frame
            template = ordered[XY_MARKER].copy()
            for marker, frame in ordered.items():
                if not frame[KEY_COLS].equals(template[KEY_COLS]):
                    raise ValueError(f"{marker}: prediction keys do not match {XY_MARKER}")
            xy_stats = template.groupby("well_id")[GEO_COLS].mean()
            template["final_pred"] = 0.0
            selected_gate = {}
            for well_id, sub_df in tqdm(template.groupby("well_id"), desc="Blending wells"):
                stat = xy_stats.loc[well_id]
                xy_safe = bool(
                    (stat["geo_nbr_distance"] < 2459)
                    & (stat["geo_nbr_distance_q10"] < 1426)
                    & (stat["geo_nbr_path_alignment"] > 0.96)
                    & (stat["geo_radial_extrap_score"] < 1.45)
                    & (stat["geo_prefix_weight_ratio"] < 0.315)
                )
                selected_gate[str(well_id)] = "xy" if xy_safe else "gr"
                used_combs = XY_BASED if xy_safe else GR_ONLY
                pred = 0.0
                weight_sum = 0.0
                for marker, weight in used_combs:
                    pred += ordered[marker].loc[sub_df.index, "TVT_pred"].values * weight
                    weight_sum += weight
                pred /= weight_sum
                template.loc[sub_df.index, "final_pred"] = pred
            return template, selected_gate

        blended, selected_gate = blend_predictions(test_predictions)
        del test_predictions
        gc.collect()
        print("Geographic blend wells:", sum(value == "xy" for value in selected_gate.values()))
        print("GR blend wells:", sum(value == "gr" for value in selected_gate.values()))
    """), cell("markdown", """
        ## Validate every runtime test row and write the submission

        IDs are reconstructed from every missing `TVT_input` row in the actual mounted test CSVs.
        The sample-submission order is preserved when its IDs match that runtime test set. If Kaggle
        retains a public-only sample during the hidden rerun, the complete hidden-test IDs are used instead.
        This avoids fixing the submission length to the public example.
    """), cell("code", """
        expected_ids = []
        for horizontal in sorted((DATA / "test").glob("*__horizontal_well.csv")):
            well_id = horizontal.name.removesuffix("__horizontal_well.csv")
            known_tvt = pd.read_csv(horizontal, usecols=["TVT_input"])["TVT_input"]
            missing_rows = np.flatnonzero(known_tvt.isna().to_numpy())
            expected_ids.extend(f"{well_id}_{int(row)}" for row in missing_rows)
        expected_index = pd.Index(expected_ids, name="id")
        if expected_index.empty or not expected_index.is_unique:
            raise ValueError("Runtime test files produce empty or duplicate submission IDs")
        submit_index = blended["submit_index"].to_numpy()
        if not np.equal(submit_index, submit_index.astype(np.int64)).all():
            raise ValueError("Non-integer prediction row index")
        blended["id"] = blended["well_id"].astype(str) + "_" + blended["submit_index"].astype(np.int64).astype(str)
        predictions = blended.set_index("id")["final_pred"]
        if not predictions.index.is_unique:
            raise ValueError("Duplicate predicted IDs")
        missing = expected_index.difference(predictions.index)
        extra = predictions.index.difference(expected_index)
        if len(missing) or len(extra):
            raise ValueError(f"Runtime ID mismatch: {len(missing)} missing, {len(extra)} extra")

        output_order = expected_index
        order_source = "runtime_test_files"
        sample_path = DATA / "sample_submission.csv"
        if sample_path.is_file():
            sample = pd.read_csv(sample_path, dtype={"id": str})
            sample_index = pd.Index(sample["id"], name="id")
            if (sample_index.is_unique and len(sample_index) == len(expected_index)
                    and sample_index.difference(expected_index).empty):
                output_order = sample_index
                order_source = "sample_submission"
        submission = pd.DataFrame({"id": output_order, "tvt": predictions.reindex(output_order).to_numpy()})
        if not np.isfinite(submission["tvt"].to_numpy()).all():
            raise ValueError("Submission contains non-finite predictions")
        submission_path = WORK / "submission.csv"
        submission.to_csv(submission_path, index=False)
        written = pd.read_csv(submission_path, dtype={"id": str})
        if list(written.columns) != ["id", "tvt"] or written["id"].tolist() != output_order.tolist():
            raise ValueError("CSV schema or row order changed during writing")
        if not np.isfinite(written["tvt"].to_numpy()).all():
            raise ValueError("Written CSV contains non-finite predictions")
        print(f"Saved {submission_path}: {len(submission):,} rows; order={order_source}")
        display(submission.head())
    """), cell("markdown", """
        ## Archived OOF diagnostic and provenance

        This optional score blends the original author's archived out-of-fold predictions. It is an audit
        of the published ensemble, not a new training/validation experiment or a hidden-test score.
        The actual late-submission result is whatever Kaggle reports after rerunning this notebook.
    """), cell("code", """
        archived_oof_rmse = None
        if RUN_ARCHIVED_OOF:
            oof_frames = {}
            for marker, source in FAMILIES:
                columns = KEY_COLS + ["TVT_pred", "TVT"] + (GEO_COLS if marker == XY_MARKER else [])
                oof_frames[marker] = pd.read_parquet(source / "oof_df.pqt", columns=columns)
            oof_blended, _ = blend_predictions(oof_frames)
            if not np.isfinite(oof_blended["TVT"].to_numpy()).all():
                raise ValueError("Archived OOF targets contain non-finite values")
            archived_oof_rmse = float(np.sqrt(np.mean((oof_blended["TVT"] - oof_blended["final_pred"]) ** 2)))
            print(f"Author-provided archived OOF ensemble RMSE: {archived_oof_rmse:.6f} ft")
            del oof_frames, oof_blended
            gc.collect()

        report = {
            "method": "Attributed reproduction of Ruby/w5833946 first-place ensemble",
            "upstream_notebook": "https://www.kaggle.com/code/w5833946/submit-reproduce",
            "upstream_repository": "https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby",
            "upstream_repository_commit_reviewed": "fade6112ffebd9e27c08ac7f053e8d4c3d5c319a",
            "runtime_versions": versions,
            "gpu": torch.cuda.get_device_name(0),
            "families": provenance,
            "test_wells": len(selected_gate),
            "xy_gate_wells": sum(value == "xy" for value in selected_gate.values()),
            "submission_rows": len(submission),
            "submission_order": order_source,
            "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
            "archived_author_oof_rmse": archived_oof_rmse,
            "leaderboard_score": None,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        (WORK / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\\n")
        print("Ready to submit:", submission_path)
        print("Actual leaderboard score: available only after Kaggle scoring completes.")
    """)]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": False, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    notebook_path = ROOT / "notebooks/rogii_winner_reproduction.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    metadata = {
        "id": f"{username}/rogii-1st-place-reproduction-insights",
        "title": "ROGII | 1st-Place Reproduction + Insights",
        "code_file": "../notebooks/rogii_winner_reproduction.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [ref for _, ref in DATASETS],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [], "model_sources": [],
    }
    metadata_path = ROOT / "kaggle/kernel-metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(notebook_path)
    print(metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="YOUR_KAGGLE_USERNAME")
    args = parser.parse_args()
    build(args.username)
