import sys
import os
import sqlite3

# Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.system_env import SystemEnvironment
from src.data_handling import DataHandler
from src.semantic_analysis import SemanticAnalyzer
from src.ontology import OntologyGraph
from src.model import ThreatClassifier
from src.epistemic_reasoning import EpistemicEngine
from src.visualization import GraphVisualizer
from src.siem_connector import SIEMConnector
from src.graph_db import GraphDatabase
from src.ui import CLI
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    SystemEnvironment.initialize()
    cli = CLI()
    cli.display_header()

    data_handler = DataHandler()
    mitre_data = data_handler.load_mitre_data()
    reports = data_handler.load_raw_reports()
    
    siem = SIEMConnector()
    reports.extend(siem.poll_alerts())

    if not reports:
        logger.warning("No reports found. Exiting.")
        return

    ontology = OntologyGraph()
    semantic = SemanticAnalyzer()
    classifier = ThreatClassifier()
    graph_db = GraphDatabase()

    for technique in mitre_data:
        ontology.add_concept(technique['id'], "Technique", metadata={"name": technique['name']})

    initial_facts = {} 
    
    for report in reports:
        logger.info("Processing report...")
        risk = classifier.predict(report)
        logger.info(f"Risk Assessment ({risk['method']}): {risk['threat_confidence']:.2f}")

        for technique in mitre_data:
            tid = technique['id']
            if tid in report:
                if semantic.is_current_fact(report, tid):
                    initial_facts[tid] = 1.0 
                else:
                    logger.info(f"Mentioned: {technique['name']} ({tid})")

        relations = semantic.analyze_epistemic_context(report)
        for rel in relations:
            ontology.add_relation(rel['source'], rel['target'], rel['relation'], rel['probability'])

    engine = EpistemicEngine(ontology)
    engine.generate_permutations(initial_facts)
    cli.display_worlds(engine)
    
    viz = GraphVisualizer(engine.worlds)
    viz.generate_html()

    if engine.worlds:
        current_threats = set(k for k,v in engine.worlds[0].facts.items() if v > 0.5)
        future_threats = set(k for k,v in engine.worlds[-1].facts.items() if v > 0.5)
        new_threats = future_threats - current_threats
        
        if new_threats:
            for threat in new_threats:
                cli.display_alert(f"System BELIEVES '{threat}' is imminent.")
        else:
            print("\n[INFO] No new imminent critical threats predicted.")

if __name__ == "__main__":
    main()
