"""Circular Gaussian degradation utilities for HWC observation arrays.

`downsample` preserves the numerical implementation of the source snapshot.
`gaussian_degrade` adds explicit shape and parameter checks for public use.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve1d

def downsample(array, settings, scale):
    k = settings["kernel_size"]
    axis = np.arange(k) - (k - 1) / 2
    taps = np.exp(-axis**2 / (2 * settings["sigma"]**2))
    if k % 2 == 0:
        # Match the established OpenCV even-kernel convention.
        taps[k//2-1:k//2+1] = 1.0
    taps /= taps.sum()
    filtered = convolve1d(convolve1d(array, taps, axis=0, mode="wrap"), taps, axis=1, mode="wrap")
    y, x = settings["phase"]
    return np.ascontiguousarray(filtered[y::scale, x::scale], dtype=np.float32)

def gaussian_degrade(image, *, scale=3, kernel_size=9, sigma=2.0, phase=(1, 1)):
    """Return the circularly blurred, point-decimated HWC observation.

    This is the registered numerical proxy, not a measured sensor PSF.
    No normalization, noise or validity-mask generation is performed.
    """
    x = np.asarray(image)
    if x.ndim != 3 or x.dtype not in (np.float32, np.float64) or not np.isfinite(x).all():
        raise ValueError("finite float32 or float64 HWC image required")
    if type(scale) is not int or scale < 1 or any(n % scale for n in x.shape[:2]):
        raise ValueError("spatial dimensions must be divisible by a positive scale")
    if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("positive odd kernel size required")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma required")
    if len(phase) != 2 or any(type(v) is not int or not 0 <= v < scale for v in phase):
        raise ValueError("phase must contain two integer offsets within scale")
    return downsample(x, {"kernel_size": kernel_size, "sigma": float(sigma), "phase": phase}, scale)
