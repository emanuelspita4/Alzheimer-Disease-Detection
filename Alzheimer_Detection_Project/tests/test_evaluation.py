import unittest
import numpy as np
from evaluation import Evaluator, _metrics
from helpers import make_table
from modelling import FeatureFusion


class MetricsTest(unittest.TestCase):
    def test_known_values(self):
        m = _metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1]), 0.5)
        self.assertEqual((m["accuracy"], m["sensitivity"], m["specificity"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(m["auc"], 0.75)

    def test_auc_is_nan_when_only_one_class_is_present(self):
        self.assertTrue(np.isnan(_metrics(np.array([1, 1]), np.array([0.2, 0.8]), 0.5)["auc"]))


class EvaluatorTest(unittest.TestCase):
    def test_separable_data_scores_high_at_participant_level(self):
        res = Evaluator(n_splits=5).evaluate(make_table(separable=True), lambda: FeatureFusion("early"))
        self.assertGreater(res.participant["auc"], 0.9)

    def test_data_without_signal_scores_near_chance(self):
        res = Evaluator(n_splits=5).evaluate(make_table(n_participants=30, separable=False),
                                             lambda: FeatureFusion("early"))
        self.assertLess(res.participant["auc"], 0.8)
        self.assertGreater(res.participant["auc"], 0.2)

    def _spy(self):
        log = []

        class Spy(FeatureFusion):
            def fit(self, t):
                log.append(("train", set(t.groups)))
                return super().fit(t)

            def predict_proba(self, t):
                log.append(("test", set(t.groups)))
                return super().predict_proba(t)

        return Spy, log

    def test_no_participant_appears_in_both_train_and_test(self):
        Spy, log = self._spy()
        Evaluator(n_splits=5).evaluate(make_table(), lambda: Spy("eeg"))
        self.assertEqual(len(log), 10)
        for (k1, train), (k2, test) in zip(log[0::2], log[1::2]):
            self.assertEqual((k1, k2), ("train", "test"))
            self.assertTrue(train.isdisjoint(test))

    def test_leave_one_subject_out_trains_one_model_per_participant(self):
        Spy, log = self._spy()
        Evaluator(loso=True).evaluate(make_table(n_participants=8), lambda: Spy("eeg"))
        self.assertEqual(sum(1 for kind, _ in log if kind == "train"), 8)

    def test_predictions_have_one_row_per_participant(self):
        res = Evaluator(n_splits=5).evaluate(make_table(), lambda: FeatureFusion("late"))
        self.assertEqual(len(res.predictions), 20)
        self.assertTrue(res.predictions["p"].between(0, 1).all())
        self.assertEqual(set(res.predictions.columns), {"y", "p"})


if __name__ == "__main__":
    unittest.main()
