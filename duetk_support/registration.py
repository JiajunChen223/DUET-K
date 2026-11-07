"""Frozen WHU RR registration protocol; no model or scoring imports."""
from __future__ import annotations

import numpy as np

METHODS = ("duet_k", "psrt", "mhfunet", "mcifnet", "sfignet", "emr_diff", "s3rnet", "hysure", "cstf", "sdp")
NEURAL = METHODS[:7]
DIRECTIONS = {"xp": (0, 1), "xm": (0, -1), "yp": (1, 1), "ym": (1, -1)}
MAGNITUDES = {"GEO1": 0.5, "GEO2": 1.0, "GEO3": 2.0}
CASES = {"GEO0": {"dy": 0.0, "dx": 0.0, "magnitude": 0.0, "direction": "zero"}}
for level, magnitude in MAGNITUDES.items():
    for direction, (axis, sign) in DIRECTIONS.items():
        CASES[f"{level}_{direction}"] = {
            "dy": sign * magnitude if axis == 1 else 0.0,
            "dx": sign * magnitude if axis == 0 else 0.0,
            "magnitude": magnitude,
            "direction": direction,
        }


def shift_msi(msi: np.ndarray, *, dy: float, dx: float) -> np.ndarray:
    """Translate image content by (dy,dx): out[y,x]=in[y-dy,x-dx].

    Explicit linear interpolation, edge replication; axes are row y and column x.
    Inputs are float32 HWC. Only the 13 preregistered cases are allowed.
    """
    if msi.shape != (192, 192, 8) or msi.dtype != np.float32 or not np.isfinite(msi).all():
        raise ValueError("expected finite float32 192x192x8 MSI")
    if (float(dy), float(dx)) not in {(v["dy"], v["dx"]) for v in CASES.values()}:
        raise ValueError("unregistered translation")
    if dy == 0 and dx == 0:
        return msi.copy()
    axis, offset = (0, dy) if dy else (1, dx)
    coordinate = np.arange(msi.shape[axis], dtype=np.float64) - offset
    lower = np.floor(coordinate).astype(np.int64)
    frac = (coordinate - lower).astype(np.float32)
    lo = np.take(msi, np.clip(lower, 0, msi.shape[axis] - 1), axis=axis)
    if not np.any(frac):
        return np.ascontiguousarray(lo)
    hi = np.take(msi, np.clip(lower + 1, 0, msi.shape[axis] - 1), axis=axis)
    weight = frac[:, None, None] if axis == 0 else frac[None, :, None]
    return np.ascontiguousarray(lo * (np.float32(1.0) - weight) + hi * weight, dtype=np.float32)


def scoring_mask(original: np.ndarray) -> np.ndarray:
    """Same 2-pixel border removal for all shifts, methods and GEO0."""
    if original.shape != (192, 192) or original.dtype != np.bool_:
        raise ValueError("expected boolean original 192x192 scoring mask")
    support = np.zeros_like(original)
    support[2:-2, 2:-2] = True
    result = original & support
    if not result.any():
        raise ValueError("empty common support")
    return result
