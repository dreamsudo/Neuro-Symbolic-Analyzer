import sys
from src.epistemic_reasoning import EpistemicEngine

class CLI:
    def display_header(self):
        print("="*50)
        print("   NEUROSYMBOLIC AI - PSYPHER LABS")
        print("   Cybersecurity Threat Analysis (Enterprise v7.0)")
        print("="*50)

    def display_worlds(self, engine: EpistemicEngine):
        print("\n[Epistemological Permutations Generated]")
        for w in engine.worlds:
            print(w)
        
        print("\n[Reasoning Trace]")
        for trace in engine.trace_log:
            print(f" > {trace}")
            
    def display_alert(self, message):
        print(f"\n[ALERT] {message}")
