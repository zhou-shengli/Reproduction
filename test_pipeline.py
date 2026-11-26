"""验证数据划分、标签泄漏、物理换算和网络输入输出。"""

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import (
    decode_targets,
    encode_targets,
    fit_scaler,
    load_csv,
    make_features,
    make_split,
    targets_for,
)
from src.models import WellTestCNN
from src.source import DATA_PATH


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.audit = load_csv(Path(os.environ.get("WELLTEST_DATA", DATA_PATH)))

    def test_unknown_targets_never_enter_inputs(self):
        a = self.df.iloc[:12].copy()
        b = a.copy()
        for col in ["k", "skin", "stor", "L1", "L2", "L3", "L4"]:
            b[col] = b[col] * 3.2 + 17
        b["model"] = "not a label"
        for protocol in ["physical", "paper_style"]:
            np.testing.assert_array_equal(
                make_features(a, protocol), make_features(b, protocol)
            )

    def test_splits_are_disjoint_complete_and_reproducible(self):
        s = make_split(self.df)
        self.assertEqual(
            [len(s[k]) for k in ["train", "val", "test"]], [14000, 3000, 3000]
        )
        self.assertEqual(len(set(np.concatenate(list(s.values())))), len(self.df))
        for k, idx in s.items():
            np.testing.assert_array_equal(idx, make_split(self.df)[k])
            self.assertEqual(self.df.model.iloc[idx].nunique(), 7)

    def test_scaler_ignores_validation_and_test_values(self):
        x = make_features(self.df.iloc[:10])
        mean, std = fit_scaler(x, np.arange(5))
        x[5:] += 1e6
        mean2, std2 = fit_scaler(x, np.arange(5))
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)

    def test_derivative_restores_source_reference_rate(self):
        d = self.df.iloc[:1]
        x = make_features(d, "physical")
        factor = float((d.h / (19.03 * d.q * d.Bo * d.visc)).iloc[0])
        expected = np.log10(float(d.derp0.iloc[0]) * 15.9 * factor)
        self.assertAlmostEqual(float(x[0, 2, 0]), expected, places=5)

    def test_target_round_trip_including_negative_skin(self):
        d = self.df.iloc[:20]
        names = ["k", "skin", "stor"]
        np.testing.assert_allclose(
            decode_targets(encode_targets(d, names), names),
            d[names],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_irregular_observations_use_actual_log_time(self):
        sample = self.df.iloc[:1].copy()
        sample["t1"] = sample.t0 + 0.15 * (sample.t2 - sample.t0)
        time = np.log10(sample[[f"t{i}" for i in range(20)]].to_numpy()[0])
        pressure = sample[[f"dp{i}" for i in range(20)]].to_numpy()[0]
        factor = (sample.h / (19.03 * sample.q * sample.Bo * sample.visc)).iloc[0]
        expected = np.interp(
            np.linspace(time[0], time[-1], 128), time, np.log10(pressure * factor)
        )
        np.testing.assert_allclose(make_features(sample)[0, 1], expected, atol=1e-6)

    def test_storage_target_only_applies_to_storage_model(self):
        for cls in self.df.model.unique():
            self.assertEqual(targets_for(cls), ["k", "skin"])
            self.assertEqual(
                "stor" in targets_for(cls, "ks_storage"), cls == "radial storage"
            )

    def test_all_classes_and_input_shapes(self):
        self.assertEqual(self.audit["invalid_rows"], 0)
        for channels in [2, 3]:
            model = WellTestCNN(channels, 7).eval()
            with torch.no_grad():
                y = model(torch.zeros(3, channels, 128))
            self.assertEqual(tuple(y.shape), (3, 7))
            self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
