import networkx as nx
import sqlite3
from typing import Dict, List
from src.config import settings
from src.logging_utils import get_logger
from src.graph_db import GraphDatabase

logger = get_logger(__name__)

class OntologyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.db = GraphDatabase() if settings.database.use_sqlite_graph else None

    def add_concept(self, name: str, category: str, metadata: Dict = None):
        if metadata is None: metadata = {}
        self.graph.add_node(name, category=category, **metadata)
        if self.db:
            self.db.add_node(name, name, category, metadata.get("description", ""))

    def add_relation(self, source: str, target: str, relation_type: str, probability: float = 1.0, metadata: Dict = None):
        if metadata is None: metadata = {}
        
        if self.db:
            weight = self.db.get_feedback_weight(source)
            probability *= weight
            self.db.add_edge(source, target, relation_type, probability)
        
        self.graph.add_edge(source, target, relation=relation_type, probability=probability, **metadata)

    def get_successors(self, node: str) -> List[str]:
        if node in self.graph:
            return list(self.graph.successors(node))
        elif self.db:
            return [x[0] for x in self.db.get_successors(node)]
        return []

    def get_edge_data(self, u, v):
        if self.graph.has_edge(u, v):
            return self.graph.get_edge_data(u, v)
        return None
