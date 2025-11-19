import os
import torch
from typing import Dict
from src.config import settings
from src.logging_utils import get_logger
from src.llm_interface import LocalLLM
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logger = get_logger(__name__)

class ThreatClassifier:
    def __init__(self):
        self.config = settings.ai_engine
        self.model_path = settings.paths.models
        self.model = None
        self.tokenizer = None
        self.llm = None
        
        if self.config.enabled:
            if self.config.model_type == "llm":
                self.llm = LocalLLM()
            else:
                self._load_bert()
        else:
            logger.info("AI Engine is DISABLED. Using heuristic mode.")

    def _load_bert(self):
        model_name = self.config.model_name
        local_model_dir = os.path.join(self.model_path, model_name)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
            else:
                logger.info(f"Downloading {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.tokenizer.save_pretrained(local_model_dir)
                self.model.save_pretrained(local_model_dir)
        except Exception as e:
            logger.error(f"BERT Load Failed: {e}")

    def predict(self, text: str) -> Dict[str, float]:
        if not self.config.enabled:
            return self._heuristic_predict(text)
        
        if self.llm:
            extraction = self.llm.reason(text)
            if "T" in extraction:
                return {"threat_confidence": 0.95, "method": "llm", "extraction": extraction}
            else:
                return {"threat_confidence": 0.2, "method": "llm", "extraction": ""}

        if self.model:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    confidence = probs[0][1].item() 
                return {"threat_confidence": confidence, "method": "neural"}
            except Exception:
                pass
        
        return self._heuristic_predict(text)

    def _heuristic_predict(self, text: str) -> Dict[str, float]:
        keywords = ["attack", "malware", "exploit", "unauthorized", "dumping"]
        score = 0.0
        text_lower = text.lower()
        for k in keywords:
            if k in text_lower: score += 0.2
        return {"threat_confidence": min(score, 1.0), "method": "heuristic"}
