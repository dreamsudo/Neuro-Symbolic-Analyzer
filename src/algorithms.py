from typing import List, Dict
from src.config import asset_config

class Algorithms:
    @staticmethod
    def consistency_checker(propositions: Dict[str, float]) -> List[str]:
        inconsistent = []
        for prop, val in propositions.items():
            neg_prop = f"NOT_{prop}"
            if neg_prop in propositions:
                if val + propositions[neg_prop] > 1.2:
                    inconsistent.append(prop)
        return inconsistent

    @staticmethod
    def utility_scoring(action: str, state: Dict[str, float]) -> float:
        base_score = asset_config['actions'].get(action, 1.0)
        if action == "exploit" and state.get("vulnerable", 0.0) > 0.5:
            base_score *= 2.0
        return base_score
