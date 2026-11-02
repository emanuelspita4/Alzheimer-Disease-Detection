import unittest
import numpy as np
from helpers import make_table
from modelling import Classifier, FeatureFusion


class ClassifierTest(unittest.TestCase):
    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Classifier("nope")

    def test_scaler_is_fit_on_training_data_only(self):
        t = make_table()
        clf = Classifier("logreg").fit(t.eeg, t.y)
        mean_after_fit = clf.model.named_steps["scale"].mean_.copy()
        clf.predict_proba(t.eeg + 100)             # very different data must not change the scaler
        np.testing.assert_allclose(clf.model.named_steps["scale"].mean_, mean_after_fit)

    def test_probabilities_are_in_unit_interval(self):
        t = make_table()
        p = Classifier("svm").fit(t.eeg, t.y).predict_proba(t.eeg)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class FeatureFusionTest(unittest.TestCase):
    def test_every_mode_fits_and_predicts_one_probability_per_window(self):
        t = make_table()
        for mode in FeatureFusion.MODES:
            p = FeatureFusion(mode).fit(t).predict_proba(t)
            self.assertEqual(p.shape, (len(t),), mode)
            self.assertTrue(((p >= 0) & (p <= 1)).all(), mode)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureFusion("middle")

    def test_early_fusion_concatenates_both_feature_sets(self):
        t = make_table()
        self.assertEqual(FeatureFusion("early")._X(t).shape, (len(t), t.eeg.shape[1] + t.audio.shape[1]))

    def test_late_fusion_with_all_weight_on_eeg_equals_the_eeg_model(self):
        t = make_table()
        late = FeatureFusion("late", weights=(1, 0)).fit(t).predict_proba(t)
        eeg = FeatureFusion("eeg").fit(t).predict_proba(t)
        np.testing.assert_allclose(late, eeg)

    def test_late_fusion_weights_are_normalised(self):
        np.testing.assert_allclose(FeatureFusion("late", weights=(3, 1)).weights, [0.75, 0.25])

    def test_fusion_learns_a_separable_signal(self):
        t = make_table(separable=True)
        pred = FeatureFusion("early").fit(t).predict(t)
        self.assertGreater((pred == t.y).mean(), 0.85)


if __name__ == "__main__":
    unittest.main()
