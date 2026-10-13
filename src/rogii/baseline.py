"""A conservative, training-free baseline using only inference-time columns.

TVT + Z describes the local structural elevation, up to a well-specific
constant. Its recent trend supplies an optional geometry prior. Horizontal
drilling often follows a layer, so the default predicts the final observed TVT.
The reference typewell is accepted for a common model interface but is not
required by this baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INFERENCE_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")


def predict_well(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame | None = None,
    *,
    geometry_weight: float = 0.0,
    window: int = 600,
) -> np.ndarray:
    """Return TVT for every input row, preserving supplied TVT_input values.

    No labels, formation tops, well IDs, or saved training targets are accessed.
    Rows are expected in the supplied measured-depth order. ``window`` counts
    observed rows; ``geometry_weight=0`` is the last-value baseline.
    """
    if not 0.0 <= geometry_weight <= 1.0:
        raise ValueError("geometry_weight must be between 0 and 1")
    if window < 2:
        raise ValueError("window must be at least 2")
    observed = horizontal_df["TVT_input"].to_numpy(dtype=float)
    indices = np.flatnonzero(np.isfinite(observed))
    if not len(indices):
        raise ValueError("A well must contain at least one known TVT_input value")
    anchor = int(indices[-1])
    prediction = np.full(len(horizontal_df), observed[anchor], dtype=float)
    if geometry_weight and "MD" in horizontal_df and "Z" in horizontal_df:
        md = horizontal_df["MD"].to_numpy(dtype=float)
        z = horizontal_df["Z"].to_numpy(dtype=float)
        usable = indices[np.isfinite(md[indices]) & np.isfinite(z[indices])]
        recent = usable[-window:]
        if len(recent) >= 2 and np.isfinite(md[anchor]) and np.isfinite(z[anchor]):
            x = md[recent] - np.mean(md[recent])
            structure = observed[recent] + z[recent]
            denominator = np.dot(x, x)
            if denominator > 0:
                slope = float(np.dot(x, structure - structure.mean()) / denominator)
                geometry = observed[anchor] + z[anchor] + slope * (md - md[anchor]) - z
                valid = np.isfinite(geometry)
                prediction[valid] += geometry_weight * (geometry[valid] - prediction[valid])
    prediction[indices] = observed[indices]
    return prediction
