#!/usr/bin/env python3
"""Repository checks: compile code, confirm the notebook matches its builder, sanity-check the baseline.

Does not run Kaggle inference, load weights, or verify leaderboard scores.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rogii.baseline import predict_well

NOTEBOOK = ROOT / "notebooks/rogii_winner_reproduction.ipynb"
METADATA = ROOT / "kaggle/kernel-metadata.json"

for path in sorted(p for p in ROOT.glob("*/**/*.py") if ".venv" not in p.parts and "models" not in p.parts):
    compile(path.read_text(), str(path), "exec")
for i, c in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"{NOTEBOOK.name}[{i}]", "exec")

before = NOTEBOOK.read_bytes(), METADATA.read_bytes()
subprocess.run([sys.executable, str(ROOT / "scripts/build_winner_notebook.py")], check=True, capture_output=True)
if (NOTEBOOK.read_bytes(), METADATA.read_bytes()) != before:
    NOTEBOOK.write_bytes(before[0])
    METADATA.write_bytes(before[1])
    sys.exit("Notebook or Kaggle metadata is stale: run scripts/build_winner_notebook.py and commit the result")

md = np.arange(10.0)
well = pd.DataFrame({"MD": md, "Z": -md, "TVT_input": [5.0, 6.0, 7.0] + [np.nan] * 7})
flat = predict_well(well)
assert flat[:3].tolist() == [5.0, 6.0, 7.0] and (flat[3:] == 7.0).all()
# TVT + Z is flat in the prefix, so full geometry keeps it flat: TVT = 5 - Z.
trend = predict_well(well, geometry_weight=1.0)
assert np.allclose(trend[3:], 5.0 - well["Z"].to_numpy()[3:]), trend
print("Validation passed")
