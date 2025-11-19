import time
import random
from typing import List
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class SIEMConnector:
    def __init__(self):
        self.config = settings.siem

    def poll_alerts(self) -> List[str]:
        if not self.config.enabled:
            return []
        
        logger.info(f"Polling SIEM source: {self.config.source}...")
        time.sleep(0.5)
        
        if random.random() > 0.8:
            logger.info("New alerts found in SIEM.")
            return ["Alert: Suspicious PowerShell execution detected on Host-001 (T1059)."]
        
        return []
