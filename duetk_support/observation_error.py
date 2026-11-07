"""Native ×3 observation-domain residual metrics from the DUET-K scorer.

This module evaluates fixed predictions; it does not run or reconstruct a model.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve1d, minimum_filter

def score_scene(prediction, low, high, low_mask, high_mask, response, bands, gains):
    if (prediction.dtype != np.float32 or prediction.ndim != 3 or low.ndim != 3 or high.ndim != 3
            or prediction.shape != (*high.shape[:2], low.shape[2])
            or high.shape[:2] != (3 * low.shape[0], 3 * low.shape[1])
            or response.shape != (high.shape[2], low.shape[2])
            or low_mask.dtype != np.bool_ or high_mask.dtype != np.bool_
            or low_mask.shape != low.shape[:2] or high_mask.shape != high.shape[:2]
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or not np.isfinite(response).all() or not np.isfinite(gains).all()
            or (response < 0).any() or (gains <= 0).any()
            or not np.allclose(response.sum(axis=1), 1.0, atol=1e-5, rtol=0)
            or any(i < 0 or i >= low.shape[2] for i in bands) or len(bands) != len(set(bands))):
        raise ValueError("invalid grid, channels, observation values, masks or response")
    if min(float(low.min()), float(high.min())) < -1e-7 or max(float(low.max()), float(high.max())) > 1+1e-7:
        raise ValueError("observations outside frozen scoring units; never clip observations")
    # The complete circular 9x9 support must be valid, in addition to the
    # original LR observation mask. The actual eight E regions are all valid.
    hmask = low_mask & (minimum_filter(high_mask.astype(np.uint8), size=9, mode="wrap")[1::3, 1::3] > 0)
    if not hmask.any() or not high_mask.any():
        raise ValueError("empty fixed observation support")
    taps = np.exp(-(np.arange(9, dtype=np.float64) - 4) ** 2 / 8.0)
    taps /= taps.sum()
    band_set = set(bands)
    m_projection = np.zeros(high.shape, dtype=np.float64)
    h_error_ss = h_observation_ss = 0.0
    n_clipped = 0
    raw_min, raw_max = float("inf"), -float("inf")
    # Chunked channels bound peak memory; scipy's float32 output matches the
    # two-pass convolve1d registered by materialize.py, then sums use float64.
    for start in range(0, low.shape[2], 8):
        stop = min(start + 8, low.shape[2])
        x = np.asarray(prediction[..., start:stop], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("prediction contains nonfinite values")
        raw_min, raw_max = min(raw_min, float(x.min())), max(raw_max, float(x.max()))
        n_clipped += int(np.count_nonzero((x < 0) | (x > 1)))
        x = np.clip(x, 0, 1)
        m_projection += np.asarray(x, dtype=np.float64) @ np.asarray(response[:, start:stop].T, dtype=np.float64)
        active = [i-start for i in range(start, stop) if i in band_set]
        if active:
            q = x[..., active]
            reduced = convolve1d(convolve1d(q, taps, axis=0, mode="wrap"), taps, axis=1, mode="wrap")[1::3, 1::3]
            observed = np.asarray(low[..., start:stop][..., active][hmask], dtype=np.float64)
            residual = np.asarray(reduced[hmask], dtype=np.float64) - observed
            h_error_ss += float(np.sum(residual * residual, dtype=np.float64))
            h_observation_ss += float(np.sum(observed * observed, dtype=np.float64))
    m_projection *= gains.reshape(1, 1, -1)
    observed_m = np.asarray(high[high_mask], dtype=np.float64)
    residual_m = m_projection[high_mask] - observed_m
    m_error_ss = float(np.sum(residual_m * residual_m, dtype=np.float64))
    m_observation_ss = float(np.sum(observed_m * observed_m, dtype=np.float64))
    if not all(np.isfinite(v) for v in (h_error_ss, h_observation_ss, m_error_ss, m_observation_ss)):
        raise ValueError("nonfinite score accumulator")
    if h_observation_ss <= 0 or m_observation_ss <= 0:
        raise ZeroDivisionError("an observation norm is zero")
    count_h, count_m = int(hmask.sum()) * len(bands), int(high_mask.sum()) * high.shape[2]
    return {"E_H_relative": float(np.sqrt(h_error_ss / h_observation_ss)),
            "E_M_relative": float(np.sqrt(m_error_ss / m_observation_ss)),
            "RMSE_H_scoring_units": float(np.sqrt(h_error_ss / count_h)),
            "RMSE_M_scoring_units": float(np.sqrt(m_error_ss / count_m)),
            "HSI_residual_sum_squares": h_error_ss, "HSI_observation_sum_squares": h_observation_ss,
            "MSI_residual_sum_squares": m_error_ss, "MSI_observation_sum_squares": m_observation_ss,
            "HSI_spatial_count": int(hmask.sum()), "MSI_spatial_count": int(high_mask.sum()),
            "HSI_element_count": count_h, "MSI_element_count": count_m,
            "prediction_min_before_clip": raw_min, "prediction_max_before_clip": raw_max,
            "prediction_clipped_fraction_all_channels": n_clipped / prediction.size}

def score_observation_error(prediction, low, high, low_mask, high_mask, response, bands, gains):
    """Validate public inputs and invoke the source scorer on one scene.

    `prediction`, `low`, and `high` are HWC; masks are fixed H×W booleans.
    `response` is [MSI bands, HSI bands], and `gains` has one positive value
    per MSI band. `bands` is a nonempty list of zero-based HSI indices.
    Dataset-specific calibration and scoring-domain conversion are caller-owned.
    """
    prediction, low, high = map(np.asarray, (prediction, low, high))
    response, gains = np.asarray(response), np.asarray(gains)
    low_mask, high_mask = np.asarray(low_mask), np.asarray(high_mask)
    if high.ndim != 3 or gains.shape != (high.shape[2],):
        raise ValueError("one gain per MSI channel is required")
    if not isinstance(bands, (list, tuple)) or not bands or any(type(i) is not int for i in bands):
        raise ValueError("bands must be a nonempty list of integer channel indices")
    return score_scene(prediction, low, high, low_mask, high_mask, response, bands, gains)
