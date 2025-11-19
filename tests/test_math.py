import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.epistemic_math import EpistemicMath

class TestEpistemicMath(unittest.TestCase):
    def test_brier_score(self):
        score = EpistemicMath.brier_score(0.8, True)
        self.assertAlmostEqual(score, 0.04)

    def test_bayesian_update(self):
        updated = EpistemicMath.bayesian_update(0.5, 0.8, 0.4)
        self.assertEqual(updated, 1.0)

if __name__ == '__main__':
    unittest.main()
