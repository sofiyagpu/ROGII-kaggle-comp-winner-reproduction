"""Portable inference adapter for Bilzard's released AnchorCNN ensemble.

The model and checkpoint files remain in the separately downloaded Apache-2.0
artifact. This adapter follows the author's public Kaggle inference notebook:
https://www.kaggle.com/code/tatamikenn/rogii-2nd-place-inference
It does not train models or use the training wells' target values.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

SOURCE_URL = "https://www.kaggle.com/datasets/tatamikenn/rogii-2nd-place-weights"
PHASES = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0)


def verify_artifacts(root: Path) -> dict:
    """Verify the upstream manifest before importing code or loading weights."""
    checked, missing_metadata = 0, []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest path escapes artifact directory: {name}")
        if name == "dataset-metadata.json" and not path.exists():
            # Kaggle consumes this upload metadata instead of distributing it.
            missing_metadata.append(name)
            continue
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        checked += 1
    return {"verified_files": checked, "upload_metadata_not_distributed": missing_metadata}


def predict(
    data_dir: Path,
    weights_dir: Path,
    output_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 2,
    checkpoint_limit: int | None = None,
    phase_limit: int | None = None,
    threads: int = 6,
) -> dict:
    """Run both checkpoint families and preserve sample-submission ID order.

    Limits are diagnostic options only; the default runs all 30 checkpoints and
    all eight phase shifts. No leaderboard score is implied by this local run.
    """
    import numpy as np
    import polars as pl
    import torch

    start = time.monotonic()
    data_dir, weights_dir = Path(data_dir).resolve(), Path(weights_dir).resolve()
    output_path = Path(output_path).resolve()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    if checkpoint_limit is not None and not 1 <= checkpoint_limit <= 15:
        raise ValueError("checkpoint_limit must be between 1 and 15")
    if phase_limit is not None and not 1 <= phase_limit <= 8:
        raise ValueError("phase_limit must be between 1 and 8")
    verification = verify_artifacts(weights_dir)
    torch.set_num_threads(min(threads, os.cpu_count() or threads))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError("device must be auto, cuda, mps, or cpu")
    torch.ones(2, device=device).sum().item()
    sys.path.insert(0, str(weights_dir / "src"))
    zlayer = importlib.import_module("zlayer_mlp")
    zlayer.DEVICE = device
    ad = importlib.import_module("anchor_deploy")
    gd = importlib.import_module("gr2tvt_data")
    gm = importlib.import_module("gr2tvt_model")
    ad.PRE_PS_GR, ad.Z_DIP = True, True
    test_dir = data_dir / "test"
    files = sorted(test_dir.glob("*__horizontal_well.csv"))
    if not files:
        raise FileNotFoundError(f"No horizontal test wells found in {test_dir}")

    def read_csv(path):
        df = pl.read_csv(path, null_values=["nan", "NaN", "NA", ""])
        columns = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
        return df.with_columns([pl.col(c).fill_nan(None) for c in columns])

    raw, typewells, all_known = {}, {}, []
    for path in files:
        name = path.name.removesuffix("__horizontal_well.csv")
        hw = read_csv(path).with_row_index("_row_idx").sort("MD")
        md, z, ti, gr = [hw[c].cast(pl.Float64).to_numpy() for c in ("MD", "Z", "TVT_input", "GR")]
        ev = ~np.isfinite(ti)
        all_known.extend(ti[np.isfinite(ti)])
        if not ev.any():
            continue
        raw[name] = (md, z, ti, gr, ev, hw["_row_idx"].to_numpy()[ev])
        tw_path = test_dir / f"{name}__typewell.csv"
        if tw_path.exists():
            tw = read_csv(tw_path).drop_nulls(["TVT", "GR"]).sort("TVT")
            typewells[name] = tuple(tw[c].to_numpy().astype(np.float64) for c in ("TVT", "GR"))
        else:
            typewells[name] = (None, None)
    if not raw or not all_known:
        raise ValueError("Test data must contain prediction rows and a known TVT prefix")
    global_tvt = float(np.median(all_known))
    phases = PHASES[:phase_limit]
    families = [
        ("dzl", "model/plan147/dzl_w1", 32.0, 320, 16,
         dict(stem_stride=1, fuse_div=4, ps_col=16, in_chans=6, anchor_m=336)),
        ("r16", "model/plan148/r16", 16.0, 640, 32,
         dict(stem_stride=1, fuse_div=8, ps_col=32, in_chans=6, anchor_m=672, n_move=5)),
    ]
    means, eligible, checkpoint_counts = {}, set(raw), {}
    print(f"AnchorCNN: {len(raw)} wells, device={device}, phases={len(phases)}, batch={batch_size}", flush=True)
    for family, relative, colw, height, ps, kwargs in families:
        checkpoints = sorted((weights_dir / relative).glob("ep119_seed*_fold*.pt"))
        if len(checkpoints) != 15:
            raise ValueError(f"Expected 15 {family} checkpoints, found {len(checkpoints)}")
        checkpoints = checkpoints[:checkpoint_limit]
        checkpoint_counts[family] = len(checkpoints)
        gd.COLW, gd.H, ad.PS_COL, ad.ANCHOR_KW = colw, height, ps, kwargs
        samples = {}
        for name, (md, z, ti, gr, ev, _) in raw.items():
            views = {}
            for phase in phases:
                sample = ad.build_sample(md, gr, ti, *typewells[name], np.zeros(int(ev.sum())), ev,
                                         z=z, md_phase=phase)
                if sample is None:
                    break
                views[phase] = sample
            if len(views) == len(phases):
                samples[name] = views
        eligible &= set(samples)
        names, accumulated = sorted(samples), {}
        for ci, checkpoint in enumerate(checkpoints):
            model = ad.load_anchor_models([checkpoint], device)[0]
            with torch.inference_mode():
                for phase in phases:
                    for i in range(0, len(names), batch_size):
                        batch = names[i:i + batch_size]
                        x, _ = gm.items_to_x([samples[n][phase]["item"] for n in batch], device, model)
                        if x.shape[1] != 9:
                            raise ValueError(f"Expected 9 input channels, found {x.shape[1]}")
                        if device == "mps":
                            # Hundreds of tiny DP steps are faster on CPU than MPS.
                            # Only the execution device changes; decoding is identical.
                            logits, offsets = model(x)
                            lv = gm.dp_expected_level(logits.float().cpu(), offsets.float().cpu(),
                                                      model.win, start_col=model.ps_col).numpy()
                        else:
                            lv = model.predict_level(x).float().cpu().numpy()
                        for j, name in enumerate(batch):
                            p = gd.cols_to_md(lv[j], samples[name][phase]["meta"], raw[name][0][raw[name][4]])
                            accumulated[name] = p if name not in accumulated else accumulated[name] + p
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            print(f"  {family}: {ci + 1}/{len(checkpoints)} checkpoints ({time.monotonic() - start:.1f}s)", flush=True)
        means[family] = {n: accumulated[n] / (len(checkpoints) * len(phases)) for n in names}

    prediction_by_id, fallback_wells = {}, []
    for name, (_, _, ti, _, ev, rows) in raw.items():
        if name in eligible:
            pred = 0.5 * (means["dzl"][name] + means["r16"][name])
        else:
            known = ti[np.isfinite(ti)]
            pred = np.full(int(ev.sum()), float(known[-1]) if len(known) else global_tvt)
            fallback_wells.append(name)
        for row, value in zip(rows, pred, strict=True):
            prediction_by_id[f"{name}_{int(row)}"] = float(value)
    template = pl.read_csv(data_dir / "sample_submission.csv")
    ids = template["id"].to_list()
    if len(ids) != len(set(ids)) or set(ids) != set(prediction_by_id):
        raise ValueError("Sample-submission IDs differ from test evaluation rows or contain duplicates")
    values = np.array([prediction_by_id[i] for i in ids])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids, "tvt": values}).write_csv(output_path)
    report = {
        "model": "Bilzard AnchorCNN second-place reproduction",
        "source": SOURCE_URL,
        "published_private_rmse": 5.802,
        "score_note": "Published upstream result; this local run has no Kaggle score.",
        "device": device,
        "checkpoints": checkpoint_counts,
        "phase_shifts": list(phases),
        "full_ensemble": checkpoint_limit in (None, 15) and phase_limit in (None, 8),
        "wells": len(raw), "rows": len(ids), "fallback_wells": fallback_wells,
        "finite": True, "sample_order_preserved": True,
        "prediction_min": float(values.min()), "prediction_max": float(values.max()),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "artifact_verification": verification,
    }
    output_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
