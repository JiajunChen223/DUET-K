# DUET-K

**DUET-K: Dual-Observation-Conditioned Joint Kernels for Hyperspectral and Multispectral Image Fusion**

Supporting code for DUET-K. The DUET-K model implementation will be released after paper acceptance.

## Contents

- `duetk_support/preprocessing.py`: observation preprocessing.
- `duetk_support/registration.py`: image registration and scoring masks.
- `duetk_support/rr_metrics.py`: reduced-resolution fusion metrics.
- `duetk_support/observation_error.py`: observation-domain errors.
- `tests/`: checks for these utilities.

## Installation and tests

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Datasets and model weights are not included. This release alone cannot reproduce the paper's results. Observation-domain errors are not QNR.

MIT License.
