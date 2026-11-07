import unittest
import numpy as np

from run_project import metrics


class EvaluationTests(unittest.TestCase):
    def test_price_error_and_tolerance_metrics_have_correct_units(self):
        result = metrics(np.array([100.,200.]),np.array([110.,150.]))
        self.assertIn('median_ape',result,'Relative accuracy must be reported with absolute price errors')
        self.assertAlmostEqual(result['mae_kzt'],30.)
        self.assertAlmostEqual(result['median_absolute_error_kzt'],30.)
        self.assertAlmostEqual(result['median_ape'],.175)
        self.assertAlmostEqual(result['within_10pct'],.5)
        self.assertAlmostEqual(result['within_20pct'],.5)
        self.assertAlmostEqual(result['within_30pct'],1.)

    def test_nonfinite_predictions_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            metrics(np.array([100.,200.]),np.array([np.nan,150.]))


if __name__=='__main__':
    unittest.main()
