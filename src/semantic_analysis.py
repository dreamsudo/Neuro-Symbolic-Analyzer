import re
from typing import List, Dict, Tuple
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class SemanticAnalyzer:
    def __init__(self):
        self.use_ai = settings.ai_engine.enabled
        self.nlp = None
        if self.use_ai:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if self.nlp:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        return self._regex_fallback(text)

    def _regex_fallback(self, text: str) -> List[Tuple[str, str]]:
        ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
        return [(ip, "IP_ADDRESS") for ip in ips]

    def analyze_epistemic_context(self, text: str) -> List[Dict]:
        results = []
        clean_text = text.replace('\n', ' ').lower()
        threat_matches = [(m.group(0).upper(), m.start()) for m in re.finditer(r't\d{4}', clean_text)]
        keywords = ["leads to", "suggests", "implies", "results in", "if", "then", "suspect", "believe"]
        
        for i in range(len(threat_matches) - 1):
            source_code, source_idx = threat_matches[i]
            target_code, target_idx = threat_matches[i+1]
            between_text = clean_text[source_idx:target_idx]
            
            if len(between_text) < 200:
                for kw in keywords:
                    if kw in between_text:
                        results.append({
                            "source": source_code,
                            "target": target_code,
                            "relation": "believes_leads_to",
                            "probability": 0.9,
                            "type": "future"
                        })
                        break
        return results

    def is_current_fact(self, text: str, technique_id: str) -> bool:
        idx = text.find(technique_id)
        if idx == -1: return False
        start = max(0, idx - 50)
        end = min(len(text), idx + 50)
        window = text[start:end].lower()
        indicators = ["detected", "found", "active", "indicate", "seen", "execution", "campaign"]
        return any(x in window for x in indicators)
