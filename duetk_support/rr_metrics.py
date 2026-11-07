"""Four mainstream-compatible fusion metrics; no model or repository imports.

The primary estimator is the mean of complete-scene scores. This module is
independent of historical evaluator identities. Its caller owns all data-access
and test-unsealing authorization.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

METRIC_ID = "external15.mainstream_four_metrics.v1"
COLUMNS = ("MPSNR_dB", "MSSIM_Wang", "SAM_degrees", "ERGAS")
EPS = 1e-8
MSE_FLOOR = 1e-12


def scene_scores(reference, prediction, *, scale, region_mask=None):
    """Score one HWC scene. Only a fixed, reference-defined mask is permitted."""
    if type(scale) is not int or scale < 1:
        raise ValueError("scale must be the explicit positive HR/LR ratio")
    if np.iscomplexobj(reference) or np.iscomplexobj(prediction):
        raise ValueError("real-valued HWC arrays required")
    ref = np.asarray(reference, dtype=np.float64)
    raw = np.asarray(prediction, dtype=np.float64)
    if ref.ndim != 3 or ref.shape != raw.shape or min(ref.shape[:2]) < 11:
        raise ValueError("matching HWC grids with at least an 11x11 window required")
    if not np.isfinite(ref).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite predictions are errors, never repaired")
    if np.any(ref < 0) or np.any(ref > 1):
        raise ValueError("reference must retain its declared [0,1] normalization")
    if region_mask is None:
        region = np.ones(ref.shape[:2], dtype=bool)
    else:
        region = np.asarray(region_mask)
        if region.dtype != np.bool_ or region.shape != ref.shape[:2]:
            raise ValueError("fixed boolean HxW reference region required")
    valid = region & (np.linalg.norm(ref, axis=-1) > EPS)
    if not valid.any():
        raise ValueError("empty reference support")
    centers = binary_erosion(valid, structure=np.ones((11, 11), dtype=bool), border_value=0)
    if not centers.any():
        raise ValueError("no complete valid SSIM window")
    est = np.clip(raw, 0.0, 1.0)
    diff = est[valid] - ref[valid]
    mse_band = np.mean(diff * diff, axis=0)
    mpsnr = float(np.mean(10.0 * np.log10(1.0 / np.maximum(mse_band, MSE_FLOOR))))

    # sigma=1.5, radius5 => truncate=5/1.5. Population covariance matches Wang.
    kernel = dict(sigma=(1.5, 1.5, 0.0), truncate=5.0 / 1.5, mode="reflect")
    mu_x = gaussian_filter(ref, **kernel)
    mu_y = gaussian_filter(est, **kernel)
    var_x = np.maximum(gaussian_filter(ref * ref, **kernel) - mu_x * mu_x, 0.0)
    var_y = np.maximum(gaussian_filter(est * est, **kernel) - mu_y * mu_y, 0.0)
    cov = gaussian_filter(ref * est, **kernel) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    local = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / np.maximum(
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2), 1e-18)
    mssim = float(local[centers].mean())

    x, y = ref[valid], est[valid]
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    nonzero_prediction = ny > EPS
    angle = np.full(nx.shape, 90.0, dtype=np.float64)
    # Do not floor every nonzero norm product at EPS; that biases dark spectra.
    denominator = nx[nonzero_prediction] * ny[nonzero_prediction]
    cosine = np.sum(x[nonzero_prediction] * y[nonzero_prediction], axis=1) / denominator
    angle[nonzero_prediction] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    sam = float(angle.mean())
    mu = ref[valid].mean(axis=0)
    ergas = float(100.0 / scale * math.sqrt(np.mean(
        mse_band / np.maximum(np.abs(mu), EPS)**2)))
    return {
        "metrics": dict(zip(COLUMNS, (mpsnr, mssim, sam, ergas))),
        "diagnostics": {
            "valid_pixels": int(valid.sum()), "valid_SSIM_centers": int(centers.sum()),
            "zero_prediction_pixels": int((~nonzero_prediction).sum()),
            "clipped_fraction": float(np.mean((raw[valid] < 0) | (raw[valid] > 1))),
            "floored_MSE_bands": int((mse_band < MSE_FLOOR).sum()),
            "near_zero_reference_mean_bands": int((np.abs(mu) < EPS).sum()),
            "raw_min": float(raw[valid].min()), "raw_max": float(raw[valid].max()),
        },
    }


def evaluate_scenes(scenes, *, scale, source_manifest_sha256, evaluation_split):
    """Pure array scorer; authorization must occur before its caller loads arrays."""
    if evaluation_split not in {"V", "E"} or not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64:
        raise ValueError("explicit split and byte-bound source manifest required")
    rows, identifiers = {}, set()
    for scene in scenes:
        identifier = scene["scene_id"]
        if identifier in identifiers or scene["role"] != evaluation_split:
            raise ValueError("duplicate scene or mixed evaluation role")
        identifiers.add(identifier)
        rows[identifier] = scene_scores(scene["target"], scene["prediction"],
            scale=scale, region_mask=scene.get("region_mask"))
    if not rows:
        raise ValueError("complete nonempty scene collection required")
    macro = {key: float(np.mean([r["metrics"][key] for r in rows.values()])) for key in COLUMNS}
    return {"schema": METRIC_ID, "primary_columns": list(COLUMNS), "aggregation": "scene_macro",
        "evaluation_split": evaluation_split, "source_manifest_sha256": source_manifest_sha256,
        "scene_count": len(rows), "macro": macro, "per_scene": rows,
        "prediction_postprocess": "clip_0_1", "data_range": 1.0,
        "metric_policy": {"SSIM_covariance": "population", "SSIM_window": 11,
            "SSIM_sigma": 1.5, "SSIM_centers": "complete_valid_windows",
            "SAM_zero_reference": "reference_mask_excluded", "SAM_zero_prediction": "90_degrees",
            "MPSNR_MSE_floor": MSE_FLOOR, "ERGAS_scale": scale}}
