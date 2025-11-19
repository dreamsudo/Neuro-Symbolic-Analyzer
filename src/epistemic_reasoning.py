import copy
from typing import List, Dict
from src.ontology import OntologyGraph
from src.config import settings
from src.logging_utils import get_logger
from src.epistemic_math import EpistemicMath
from src.algorithms import Algorithms

logger = get_logger(__name__)

class World:
    def __init__(self, id: int, facts: Dict[str, float]):
        self.id = id
        self.facts = facts
    def __repr__(self):
        active = [f"{k}:{v:.2f}" for k, v in self.facts.items() if v > 0.5]
        return f"World {self.id}: {active}"

class GameModel:
    def __init__(self):
        self.attacker_actions = ["exploit", "lateral_move", "exfiltrate"]
        self.defender_actions = ["isolate", "patch", "monitor"]
        
    def get_action_probabilities(self, state: Dict[str, float]) -> Dict[str, float]:
        utilities = {act: Algorithms.utility_scoring(act, state) for act in self.attacker_actions}
        exp_sum = sum(EpistemicMath.sigmoid(u) for u in utilities.values())
        probs = {k: EpistemicMath.sigmoid(v) / exp_sum for k, v in utilities.items()}
        return probs

    def calculate_defender_strategy(self, state: Dict[str, float]) -> str:
        best_action = "monitor"
        min_attacker_utility = float('inf')
        
        for def_act in self.defender_actions:
            simulated_state = state.copy()
            if def_act == "isolate": simulated_state["network_access"] = 0.1
            
            att_utils = {act: Algorithms.utility_scoring(act, simulated_state) for act in self.attacker_actions}
            max_util = max(att_utils.values())
            
            if max_util < min_attacker_utility:
                min_attacker_utility = max_util
                best_action = def_act
        return best_action

class EpistemicEngine:
    def __init__(self, ontology: OntologyGraph):
        self.ontology = ontology
        self.game_model = GameModel()
        self.worlds: List[World] = []
        self.trace_log = []

    def generate_permutations(self, initial_facts: Dict[str, float]):
        logger.info("Generating Epistemological Permutations (Fuzzy)...")
        w0 = World(0, initial_facts)
        self.worlds.append(w0)
        
        current_world = w0
        depth = settings.reasoning.max_simulation_depth
        
        for i in range(1, depth + 1):
            action_probs = self.game_model.get_action_probabilities(current_world.facts)
            
            if settings.reasoning.enable_defender_simulation:
                def_move = self.game_model.calculate_defender_strategy(current_world.facts)
                self.trace_log.append(f"Step {i}: Defender should '{def_move}' to minimize risk.")

            next_facts = copy.deepcopy(current_world.facts)
            graph = self.ontology.graph
            
            for node in graph.nodes:
                if next_facts.get(node, 0.0) > settings.reasoning.fuzzy_threshold:
                    for succ in graph.successors(node):
                        edge_data = graph.get_edge_data(node, succ)
                        if edge_data.get("relation") == "believes_leads_to":
                            prior = edge_data.get("probability", 0.5)
                            updated_prob = EpistemicMath.bayesian_update(prior, 0.9, 0.7)
                            
                            parent_conf = next_facts[node]
                            new_conf = min(parent_conf * updated_prob, 1.0)
                            
                            if new_conf > settings.reasoning.fuzzy_threshold:
                                next_facts[succ] = new_conf
                                self.trace_log.append(f"Step {i}: {node} -> {succ} (Conf: {new_conf:.2f})")

            w_next = World(i, next_facts)
            self.worlds.append(w_next)
            current_world = w_next
        
        logger.info(f"Generated {len(self.worlds)} possible worlds.")

    def check_satisfaction(self, world_id: int, formula: Dict) -> bool:
        w = next((x for x in self.worlds if x.id == world_id), None)
        if not w: return False
        ftype = formula.get('type')
        if ftype == 'atom': 
            return w.facts.get(formula['value'], 0.0) > settings.reasoning.fuzzy_threshold
        elif ftype == 'Ba': return True 
        return False
