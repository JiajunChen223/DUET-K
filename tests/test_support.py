import unittest
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
from duetk_support.rr_metrics import scene_scores
from duetk_support.preprocessing import gaussian_degrade
from duetk_support.observation_error import score_observation_error
from duetk_support.registration import shift_msi, scoring_mask


class ReducedResolutionTests(unittest.TestCase):
    def test_identity_and_known_constant_error(self):
        ref = np.full((16, 16, 4), 0.5, np.float32)
        identity = scene_scores(ref, ref, scale=3)["metrics"]
        self.assertAlmostEqual(identity["MPSNR_dB"], 120.0, places=8)
        self.assertAlmostEqual(identity["MSSIM_Wang"], 1.0, places=8)
        self.assertAlmostEqual(identity["ERGAS"], 0.0, places=8)
        self.assertAlmostEqual(identity["SAM_degrees"], 0.0, places=5)
        changed = scene_scores(ref, np.full_like(ref, 0.6), scale=3)["metrics"]
        self.assertAlmostEqual(changed["MPSNR_dB"], 20.0, places=4)
        self.assertAlmostEqual(changed["ERGAS"], 20.0 / 3.0, places=4)

    def test_prediction_nonfinite_and_empty_mask_rejected(self):
        ref = np.full((16, 16, 3), 0.5, np.float32)
        bad = ref.copy(); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): scene_scores(ref, bad, scale=3)
        with self.assertRaises(ValueError): scene_scores(ref, ref, scale=3, region_mask=np.zeros((16,16), bool))


class PreprocessingTests(unittest.TestCase):
    def test_matches_independent_periodic_convolution(self):
        rng = np.random.default_rng(7)
        data = rng.uniform(0.1, 0.9, (12, 15, 2)).astype(np.float32)
        actual = gaussian_degrade(data, scale=3, kernel_size=9, sigma=2, phase=(1,1))
        axis = np.arange(9, dtype=np.float64) - 4
        taps = np.exp(-axis * axis / 8); taps /= taps.sum()
        kernel = np.outer(taps, taps)
        expected = np.stack([convolve2d(data[..., i], kernel, mode="same", boundary="wrap") for i in range(2)], axis=-1)[1::3,1::3]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(actual.shape, (4,5,2))

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError): gaussian_degrade(np.zeros((11,12,2),np.float32),scale=3)
        with self.assertRaises(ValueError): gaussian_degrade(np.zeros((12,12,2),np.float32),scale=3,kernel_size=8)


class ObservationErrorTests(unittest.TestCase):
    def test_exact_proxy_observations_have_zero_residual(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(.1,.9,(36,36,4)).astype(np.float32)
        axis = np.arange(9, dtype=np.float64)-4; taps=np.exp(-axis**2/8);taps/=taps.sum()
        low=convolve1d(convolve1d(x,taps,axis=0,mode="wrap"),taps,axis=1,mode="wrap")[1::3,1::3]
        response=np.eye(4,dtype=np.float64)[:3];high=x[...,:3].copy()
        result=score_observation_error(x,low,high,np.ones(low.shape[:2],bool),np.ones(high.shape[:2],bool),response,[0,1,2,3],np.ones(3))
        self.assertAlmostEqual(result["E_H_relative"],0.0,places=8)
        self.assertAlmostEqual(result["E_M_relative"],0.0,places=8)

    def test_gain_shape_rejected(self):
        with self.assertRaises(ValueError):
            score_observation_error(np.zeros((6,6,2),np.float32),np.zeros((2,2,2)),np.zeros((6,6,3)),
                np.ones((2,2),bool),np.ones((6,6),bool),np.ones((3,2))/2,[0,1],np.ones(2))


class RegistrationTests(unittest.TestCase):
    def test_shift_and_common_mask(self):
        x=np.zeros((192,192,8),np.float32);x[80,90,0]=1
        self.assertEqual(shift_msi(x,dy=0,dx=1)[80,91,0],1)
        self.assertEqual(shift_msi(x,dy=0,dx=.5)[80,90,0],.5)
        self.assertEqual(shift_msi(x,dy=0,dx=.5)[80,91,0],.5)
        mask=scoring_mask(np.ones((192,192),bool));self.assertEqual(int(mask.sum()),188*188)


if __name__ == "__main__": unittest.main()
