import os
from src.logging_utils import get_logger
from src.config import settings

logger = get_logger(__name__)

class LocalLLM:
    def __init__(self):
        self.model_path = settings.ai_engine.llm_path
        self.llm = None
        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"LLM Model not found at {self.model_path}")
            return

        try:
            from llama_cpp import Llama
            logger.info(f"Loading LLM from {self.model_path}...")
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
            logger.info("Local LLM loaded successfully.")
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")

    def reason(self, text: str) -> str:
        if not self.llm:
            return "LLM_NOT_LOADED"
        
        prompt = f"""
        You are a cybersecurity analyst. Extract MITRE ATT&CK Technique IDs (like T1059, T1078) from the text below.
        Text: "{text}"
        Return ONLY the IDs found, separated by commas.
        IDs:
        """
        try:
            output = self.llm(prompt, max_tokens=50, stop=["\n", "Text:"], echo=False)
            result = output['choices'][0]['text'].strip()
            logger.info(f"LLM Extraction Result: {result}")
            return result
        except Exception as e:
            logger.error(f"LLM Inference Error: {e}")
            return ""
