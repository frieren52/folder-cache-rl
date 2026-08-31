from __future__ import annotations

import unittest

import numpy as np

from common import MODULE_ROOT
from src.config import load_config_set
from src.features import (
    CANDIDATE_LOG1P_INDICES,
    CandidateFeatureInput,
    FeatureNormalizer,
    build_candidate_feature,
    completion_probabilities,
    expected_access_count_5m,
)


class ConfigFeatureTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config_set(MODULE_ROOT / "config")
        self.assertEqual(config["environment"]["environment"]["channel_count"], 30)

    def test_completion_probability_splits_current_bin(self) -> None:
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[0] = 0.5
        probabilities[1] = 0.3
        probabilities[9] = 0.2
        before, after = completion_probabilities(probabilities, 7.5)
        self.assertAlmostEqual(before, 0.65, places=5)
        self.assertAlmostEqual(after, 0.15, places=5)

    def test_candidate_feature_dimension_and_single_log_transform(self) -> None:
        feature = build_candidate_feature(
            CandidateFeatureInput(
                np.ones(128, dtype=np.float32),
                np.zeros(128, dtype=np.float32),
                False,
                0.1,
                0.0,
                0.1,
                0.1,
                2,
                0.2,
                3,
                10.0,
                0.0,
                True,
                1.0,
                2.0,
                3.0,
                0.2,
                0.3,
                False,
                4.0,
            )
        )
        self.assertEqual(feature.shape, (274,))
        normalizer = FeatureNormalizer(None, 274, CANDIDATE_LOG1P_INDICES)
        normalizer.fit(feature[None])
        transformed = normalizer.transform(feature[None])
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_expected_count_is_scaled_to_first_five_minutes(self) -> None:
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[0] = 0.2
        probabilities[6] = 0.3
        probabilities[9] = 0.5
        self.assertAlmostEqual(expected_access_count_5m(probabilities, 10.0), 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
