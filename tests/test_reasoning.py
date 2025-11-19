import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.epistemic_reasoning import GameModel
import src.config

# Mock Config
src.config.asset_config = {
    "actions": {
        "exploit": 1.5, "lateral_move": 2.0, "exfiltrate": 5.0,
        "encrypt": 10.0, "isolate": 0.0, "patch": 0.0, "monitor": 0.0
    }
}

class TestReasoning(unittest.TestCase):
    def test_defender_strategy(self):
        gm = GameModel()
        state = {"vulnerable": 1.0}
        strategy = gm.calculate_defender_strategy(state)
        self.assertIsNotNone(strategy)

if __name__ == '__main__':
    unittest.main()
