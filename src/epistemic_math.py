import math
from typing import Dict, List

class EpistemicMath:
    @staticmethod
    def brier_score(probability: float, outcome: bool) -> float:
        truth_value = 1.0 if outcome else 0.0
        return (truth_value - probability) ** 2

    @staticmethod
    def truth_proximity(belief_state: Dict[str, float], true_state: Dict[str, bool]) -> float:
        total_score = 0.0
        count = 0
        for prop, prob in belief_state.items():
            if prop in true_state:
                total_score += EpistemicMath.brier_score(prob, true_state[prop])
                count += 1
        if count == 0: return 0.0
        avg_error = total_score / count
        return 1.0 - avg_error

    @staticmethod
    def bayesian_update(prior: float, likelihood: float, marginal: float) -> float:
        if marginal == 0: return prior
        return (likelihood * prior) / marginal

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
