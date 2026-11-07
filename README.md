# DUET-K: Experimental Support Code

DUET-K: Dual-Observation-Conditioned Joint Kernels for Hyperspectral and Multispectral Image Fusion.

This pre-acceptance release provides experimental settings and selected, runnable support utilities. The **DUET-K model implementation, training and inference entry points, and model weights are not included**; the core implementation is planned for release after paper acceptance.

## Included

| Path | Purpose |
| --- | --- |
| `duetk_support/rr_metrics.py` | Reduced-resolution MPSNR, Gaussian-window SSIM, SAM and ERGAS for HWC arrays |
| `duetk_support/preprocessing.py` | Circular Gaussian blur and point decimation used by the supporting protocol |
| `duetk_support/observation_error.py` | Native-resolution HSI/MSI observation-domain relative errors |
| `duetk_support/registration.py` | WHU-MHF translation and common scoring-mask protocol |
| `configs/datasets.json` | Dataset dimensions, band selections and public protocol metadata |
| `configs/whu_training.json` | Public WHU-MHF training settings extracted from the fixed-last snapshot |
| `baselines/README.md` | Scope and third-party baseline notice |
| `tests/` | Synthetic numerical and interface checks |
| `SOURCE_PROVENANCE.json` | Source identity and release adaptations |

The package contains **no datasets, predictions, fitted sensor responses, evaluation masks, baseline implementations, model checkpoints or DUET-K core**. It is not a complete reproduction package and cannot regenerate the paper tables by itself. Baselines must be obtained separately under their respective terms. The RR metric module is the preserved accompanying scoring snapshot; the FR observation-error and registration functions are extracted from the native DUET-K experiment code. See `SOURCE_PROVENANCE.json` for source hashes and limitations.

Native FR QNR/Dλ/Ds evaluation is not included in this release: the project snapshot identifies [NUD-PAF](https://github.com/jeline0110/NUD_PAF) as a source reference, and no redistribution license was identified in that upstream repository during the release check. The included observation-domain errors are **not** QNR and are not errors against a measured high-resolution HSI reference.

## Install and check

Requires Python 3.10 or later, NumPy and SciPy:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests use generated arrays only. They do not run training, model inference, or paper scoring.

## Reduced-resolution metrics

```python
import numpy as np
from duetk_support.rr_metrics import scene_scores

reference = np.full((16, 16, 4), 0.5, dtype=np.float32)
prediction = np.full_like(reference, 0.6)
result = scene_scores(reference, prediction, scale=3)
print(result["metrics"])
```

Arrays are HWC in declared `[0, 1]` working units. `scene_scores` clips predictions but not observations or references. The region mask is fixed from the reference, not selected per prediction. Scores are for an individual scene; aggregate scenes equally only when that matches the evaluation protocol.

## Native-resolution observation errors

`duetk_support.observation_error.score_observation_error` accepts a predicted HR-HSI, observed LR-HSI and HR-MSI, fixed HSI/MSI masks, a row-normalized response matrix and MSI gains. It uses the registered 9×9, σ=2 circular spatial proxy, ×3 sampling and phase `(1,1)`; `bands` selects the HSI channels for the HSI residual while the MSI projection uses all HSI channels. Input arrays and responses must already be in the correct, dataset-specific scoring domain. The response is a training-fitted effective proxy, not a measured sensor response. The function cannot infer a dataset's calibration or fixed support.

## Citation and license

Paper citation details will be added when available. This repository's original support code is released under the MIT License; it does not relicense datasets, external baselines, or third-party code.
