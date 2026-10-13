# Reproducibility and provenance

## Sources

| Component | Upstream | Version reviewed |
| :--- | :--- | :--- |
| First-place models and blend | [w5833946/submit-reproduce](https://www.kaggle.com/code/w5833946/submit-reproduce), [GitHub](https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby) | Notebook API version 2; commit `fade6112ffebd9e27c08ac7f053e8d4c3d5c319a` |
| First-place weights | Six `w5833946/rogii-*` Kaggle datasets, about 20 GB in total | As attached at run time |
| Second-place models | [tatamikenn/rogii-2nd-place-weights](https://www.kaggle.com/datasets/tatamikenn/rogii-2nd-place-weights) | Pinned by `third_party/anchorcnn/SHA256SUMS` |

The upstream notebooks are not vendored. Their logic is reproduced in `scripts/build_winner_notebook.py` and `src/rogii/second_place.py`. `THIRD_PARTY_NOTICES.md` lists every change.

## Kaggle notebook

`build_winner_notebook.py` writes both the notebook and the Kaggle metadata. CI fails if either committed file differs from the generator's output. Set `--username` before running `kaggle kernels push -p kaggle`.

| Setting | Value |
| :--- | :--- |
| Accelerator | GPU, required; the notebook stops without CUDA |
| Internet | Off; nothing is installed or downloaded |
| Families | `0719_V1`, `0724_V1`, `0729_V3`, `0801_V1`, `0801_V2`, `0803_V2` |
| Omitted | `0802_V2`, which has zero weight in both final blends |
| `RUN_ARCHIVED_OOF` | `True`: prints the blend RMSE on the author's archived OOF |

Outputs in `/kaggle/working`:

| Artifact | Content |
| :--- | :--- |
| `submission.csv` | `id,tvt` in sample order, or runtime-test order if the sample does not match |
| `reproduction_report.json` | Package versions, GPU, per-family source hashes and timings, gate counts, submission hash |

## Local second-place run

| CLI flag | Purpose |
| :--- | :--- |
| `--data-dir PATH` | Competition directory with `test/` and `sample_submission.csv` |
| `--weights-dir PATH` | Downloaded weights; default `models/second_place` |
| `--output PATH` | Default `outputs/second_place/submission.csv` |
| `--device` | `auto`, `cuda`, `mps`, or `cpu` |
| `--checkpoint-limit N` | Diagnostic only: first N of 15 checkpoints per family |
| `--phase-limit N` | Diagnostic only: first N of 8 MD phase shifts |

The full run uses 30 checkpoints × 8 phases. A `.report.json` next to the submission records the device, counts, fallback wells, prediction range, and output hash. A run with limits sets `full_ensemble: false`.

## Validation scope

- Baselines in `outputs/data_audit.json` use 20 development and 100 evaluation wells chosen with seed 42. Each baseline sees only `MD, X, Y, Z, GR, TVT_input`. The geometry weight was not tuned on the evaluation wells.
- The 15.91 ft and 0.07 ft insight figures are computed over every training well by the notebook's insights cell.
- Upstream OOF and leaderboard figures are quoted, not recomputed. The first-place OOF uses the author's geographic CV splits.
- The archived-OOF blend score evaluates the author's released predictions. It is not new training or a hidden-test score.
- Requirements are compatibility ranges, not a lockfile. The Kaggle notebook uses the packages installed in Kaggle's image and prints their versions.

## Repository checks

`scripts/validate.py` compiles every script and notebook cell, regenerates the notebook to confirm it is current, and checks the baseline on a synthetic well. It does not run inference, load weights, or verify leaderboard scores.
