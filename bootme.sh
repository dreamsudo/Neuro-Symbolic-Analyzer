#!/bin/sh

# ============================================================
# Neurosymbolic AI - Production Master Bootstrap (v7.1)
# Author: Psypher Labs
# Description: Includes new Visualization Utility Script.
# ============================================================

set -e

# --- Safety Check ---
if [ "$(basename "$PWD")" = "directory" ]; then
    echo "[!] ERROR: You are running this script INSIDE the 'directory' folder."
    echo "    Please run 'cd ..' to go up one level, then run this script again."
    exit 1
fi

PROJECT_DIR="directory"

# --- 1. Cleanup and Setup ---
if [ -d "$PROJECT_DIR" ]; then
    echo "[!] Directory '$PROJECT_DIR' already exists. Backing up..."
    mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%s)"
fi

echo "[-] Bootstrapping Neurosymbolic AI v7.1 in '$PROJECT_DIR'..."

# Create Directory Structure
mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/knowledge_base"
mkdir -p "$PROJECT_DIR/data/models"
mkdir -p "$PROJECT_DIR/data/db"
mkdir -p "$PROJECT_DIR/docs"
mkdir -p "$PROJECT_DIR/scripts"
mkdir -p "$PROJECT_DIR/src"
mkdir -p "$PROJECT_DIR/tests"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/reports"

echo "[-] Directories created."

# --- 2. Download Models (TinyLlama) ---
echo "[-] Downloading Local LLM (TinyLlama)..."
if [ ! -f "$PROJECT_DIR/data/models/tinyllama-1.1b-chat.Q4_K_M.gguf" ]; then
    wget -q --show-progress -O "$PROJECT_DIR/data/models/tinyllama-1.1b-chat.Q4_K_M.gguf" \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

# --- 3. Generate Configuration Files ---

# config/settings.json
cat << 'EOF' > "$PROJECT_DIR/config/settings.json"
{
  "app_name": "Neurosymbolic AI",
  "version": "7.1.0",
  "paths": {
    "raw_data": "data/raw",
    "knowledge_base": "data/knowledge_base",
    "models": "data/models",
    "logs": "logs",
    "reports": "reports",
    "database": "data/db/neurosymbolic.db"
  },
  "ai_engine": {
    "enabled": true,
    "provider": "local",
    "model_type": "llm", 
    "model_name": "tinyllama",
    "llm_path": "data/models/tinyllama-1.1b-chat.Q4_K_M.gguf",
    "confidence_threshold": 0.75
  },
  "database": {
    "download_enabled": true,
    "mitre_url": "https://github.com/mitre/cti/archive/refs/heads/master.zip",
    "local_folder_name": "cti-master",
    "use_cache": true,
    "use_sqlite_graph": true
  },
  "reasoning": {
    "max_simulation_depth": 5,
    "fuzzy_threshold": 0.6,
    "enable_defender_simulation": true
  },
  "siem": {
    "enabled": false,
    "poll_interval": 60,
    "source": "mock_api"
  }
}
EOF

# config/assets.json
cat << 'EOF' > "$PROJECT_DIR/config/assets.json"
{
  "assets": {
    "Workstation": 10,
    "Server": 50,
    "DomainController": 100,
    "Database": 80,
    "Printer": 5
  },
  "actions": {
    "exploit": 1.5,
    "lateral_move": 2.0,
    "exfiltrate": 5.0,
    "encrypt": 10.0,
    "phishing": 1.0
  }
}
EOF

# config/logging.yaml
cat << 'EOF' > "$PROJECT_DIR/config/logging.yaml"
version: 1
disable_existing_loggers: False
formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: standard
    filename: "logs/neurosymbolic.log"
    mode: "a"
root:
  level: INFO
  handlers: [console, file]
EOF

# --- 4. Generate Data Files ---

# data/raw/complex_threat_report.txt
cat << 'EOF' > "$PROJECT_DIR/data/raw/complex_threat_report.txt"
INCIDENT REPORT: IR-2024-ALPHA
DATE: 2023-11-15
SEVERITY: CRITICAL
STATUS: ACTIVE

EXECUTIVE SUMMARY:
SOC analysts have detected a sophisticated intrusion attempt targeting the HR subnet.

OBSERVATIONS (CURRENT FACTS):
1. Network logs detected a successful Phishing (T1566) campaign targeting employee emails.
2. EDR telemetry found active execution of PowerShell (T1059) scripts on host HR-WKSTN-04.
3. We have also seen indications that Masquerading (T1036) is active, as the malware is disguising itself as 'svchost.exe'.

ANALYSIS & PROJECTIONS (EPISTEMIC REASONING):
Based on the behavior, we suspect the attacker has already achieved Persistence via Scheduled Task (T1053). 

If the PowerShell execution continues unchecked, historical data suggests this leads to Impair Defenses (T1562) to disable our antivirus.

CRITICAL PATH PREDICTION:
If defenses are impaired, we believe this leads to Data Encrypted for Impact (T1486) (Ransomware deployment).
However, if they pivot to the network, this leads to Remote Services (T1021) for lateral movement.
EOF

# data/raw/structured_log.json
cat << 'EOF' > "$PROJECT_DIR/data/raw/structured_log.json"
[
  {
    "timestamp": "2023-11-15T10:00:00Z",
    "event_id": 4624,
    "host": "HR-WKSTN-04",
    "message": "Successful logon detected. Potential Valid Accounts (T1078) usage."
  },
  {
    "timestamp": "2023-11-15T10:05:00Z",
    "event_id": 1,
    "host": "HR-WKSTN-04",
    "message": "Process creation: powershell.exe -enc <payload>. Detected Command and Scripting Interpreter (T1059)."
  }
]
EOF

touch "$PROJECT_DIR/data/models/.keep"

# --- 5. Generate Source Code ---

# src/__init__.py
cat << 'EOF' > "$PROJECT_DIR/src/__init__.py"
EOF

# src/config.py
cat << 'EOF' > "$PROJECT_DIR/src/config.py"
import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel

class PathsConfig(BaseModel):
    raw_data: str
    knowledge_base: str
    models: str
    logs: str
    reports: str
    database: str

class AIEngineConfig(BaseModel):
    enabled: bool
    provider: str
    model_type: str
    model_name: str
    llm_path: Optional[str]
    confidence_threshold: float

class DatabaseConfig(BaseModel):
    download_enabled: bool
    mitre_url: str
    local_folder_name: str
    use_cache: bool
    use_sqlite_graph: bool

class ReasoningConfig(BaseModel):
    max_simulation_depth: int
    fuzzy_threshold: float
    enable_defender_simulation: bool

class SIEMConfig(BaseModel):
    enabled: bool
    poll_interval: int
    source: str

class AppConfig(BaseModel):
    app_name: str
    version: str
    paths: PathsConfig
    ai_engine: AIEngineConfig
    database: DatabaseConfig
    reasoning: ReasoningConfig
    siem: SIEMConfig

class ConfigLoader:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self._config = self._load_config()
        self.assets = self._load_assets()

    def _load_config(self) -> AppConfig:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        return AppConfig(**data)

    def _load_assets(self) -> Dict:
        path = "config/assets.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {"assets": {}, "actions": {}}

    @property
    def settings(self) -> AppConfig:
        return self._config

try:
    config_loader = ConfigLoader()
    settings = config_loader.settings
    asset_config = config_loader.assets
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    settings = None
    asset_config = {}
EOF

# src/logging_utils.py
cat << 'EOF' > "$PROJECT_DIR/src/logging_utils.py"
import logging
import logging.config
import yaml
import os

def setup_logging(config_path="config/logging.yaml", default_level=logging.INFO):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error in Logging Configuration: {e}. Using default configs.")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print("Failed to load configuration file. Using default configs")
    
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

def get_logger(name):
    return logging.getLogger(name)
EOF

# src/system_env.py
cat << 'EOF' > "$PROJECT_DIR/src/system_env.py"
import os
import sys
import sqlite3
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class SystemEnvironment:
    @staticmethod
    def initialize():
        logger.info(f"Initializing {settings.app_name} Environment...")
        dirs = [
            settings.paths.raw_data,
            settings.paths.knowledge_base,
            settings.paths.models,
            settings.paths.logs,
            settings.paths.reports,
            os.path.dirname(settings.paths.database)
        ]
        for d in dirs:
            if not os.path.exists(d):
                try:
                    os.makedirs(d)
                    logger.info(f"Created directory: {d}")
                except OSError as e:
                    logger.error(f"Failed to create directory {d}: {e}")
                    sys.exit(1)
        
        SystemEnvironment._init_db()
        logger.info("Environment initialized successfully.")

    @staticmethod
    def _init_db():
        try:
            conn = sqlite3.connect(settings.paths.database)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_id TEXT,
                    is_correct BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
EOF

# src/graph_db.py
cat << 'EOF' > "$PROJECT_DIR/src/graph_db.py"
import sqlite3
import os
from typing import List, Dict, Tuple
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class GraphDatabase:
    def __init__(self):
        self.db_path = settings.paths.database
        self._init_db()

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                relation TEXT,
                probability REAL,
                FOREIGN KEY(source) REFERENCES nodes(id),
                FOREIGN KEY(target) REFERENCES nodes(id),
                PRIMARY KEY (source, target, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                threat_id TEXT,
                is_correct BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_node(self, id: str, name: str, category: str, description: str = ""):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT OR IGNORE INTO nodes (id, name, category, description) VALUES (?, ?, ?, ?)",
                         (id, name, category, description))
            conn.commit()
        finally:
            conn.close()

    def add_edge(self, source: str, target: str, relation: str, probability: float):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT OR REPLACE INTO edges (source, target, relation, probability) VALUES (?, ?, ?, ?)",
                         (source, target, relation, probability))
            conn.commit()
        finally:
            conn.close()

    def get_successors(self, node_id: str) -> List[Tuple[str, Dict]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT target, relation, probability FROM edges WHERE source=?", (node_id,))
        rows = cursor.fetchall()
        conn.close()
        return [(r[0], {"relation": r[1], "probability": r[2]}) for r in rows]

    def save_feedback(self, threat_id: str, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO feedback (threat_id, is_correct) VALUES (?, ?)", (threat_id, is_correct))
        conn.commit()
        conn.close()
        logger.info(f"Feedback saved for {threat_id}: Correct={is_correct}")

    def get_feedback_weight(self, threat_id: str) -> float:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT is_correct FROM feedback WHERE threat_id=?", (threat_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows: return 1.0
        correct_count = sum(1 for r in rows if r[0])
        return correct_count / len(rows)
EOF

# src/data_handling.py
cat << 'EOF' > "$PROJECT_DIR/src/data_handling.py"
import json
import os
import urllib.request
import zipfile
import io
import pickle
import concurrent.futures
from typing import List, Dict
from src.config import settings
from src.logging_utils import get_logger
from src.graph_db import GraphDatabase

logger = get_logger(__name__)

def parse_subdir(args):
    root_path, subdir = args
    full_path = os.path.join(root_path, subdir)
    techniques = []
    if not os.path.exists(full_path):
        return techniques
    
    for filename in os.listdir(full_path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(full_path, filename), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    objects = content.get("objects", [])
                    for obj in objects:
                        if obj.get("type") == "attack-pattern":
                            t_code = None
                            for ref in obj.get("external_references", []):
                                if ref.get("source_name") in ["mitre-attack", "mitre-pre-attack", "mitre-mobile-attack", "capec"]:
                                    t_code = ref.get("external_id")
                                    break
                            if t_code:
                                techniques.append({
                                    "id": t_code,
                                    "name": obj.get("name"),
                                    "description": obj.get("description", "No description"),
                                    "platforms": obj.get("x_mitre_platforms", [])
                                })
            except Exception:
                continue
    return techniques

class DataHandler:
    def __init__(self):
        self.raw_path = settings.paths.raw_data
        self.kb_path = settings.paths.knowledge_base
        self.db_config = settings.database
        self.graph_db = GraphDatabase() if self.db_config.use_sqlite_graph else None

    def load_mitre_data(self) -> List[Dict]:
        cache_path = os.path.join(self.kb_path, "mitre_cache.pkl")
        if self.db_config.use_cache and os.path.exists(cache_path):
            logger.info("Loading MITRE data from local cache (Fast)...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        extracted_folder = os.path.join(self.kb_path, self.db_config.local_folder_name)
        if not os.path.exists(extracted_folder):
            if self.db_config.download_enabled:
                self._download_and_extract_mitre()
            else:
                return self._get_fallback_data()

        data = self._parse_cti_repository_parallel(extracted_folder)
        
        if self.db_config.use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        
        if self.graph_db:
            logger.info("Populating SQLite Graph Database...")
            for item in data:
                self.graph_db.add_node(item['id'], item['name'], "Technique", item['description'])
        
        return data

    def _download_and_extract_mitre(self):
        url = self.db_config.mitre_url
        logger.info(f"Downloading MITRE CTI from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                zip_content = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(self.kb_path)
        except Exception as e:
            logger.error(f"Download failed: {e}")

    def _parse_cti_repository_parallel(self, root_path: str) -> List[Dict]:
        logger.info(f"Scanning CTI repository (Parallel) at {root_path}...")
        target_subdirs = [
            "enterprise-attack/attack-pattern",
            "mobile-attack/attack-pattern",
            "ics-attack/attack-pattern",
            "pre-attack/attack-pattern",
            "capec/2.1/attack-pattern"
        ]
        
        all_techniques = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(parse_subdir, [(root_path, sub) for sub in target_subdirs])
            for res in results:
                all_techniques.extend(res)
        
        logger.info(f"Loaded {len(all_techniques)} techniques.")
        return all_techniques if all_techniques else self._get_fallback_data()

    def _get_fallback_data(self):
        return [{"id": "T1078", "name": "Valid Accounts"}, {"id": "T1059", "name": "Command Scripting"}, {"id": "T1003", "name": "Credential Dumping"}]

    def load_raw_reports(self) -> List[str]:
        reports = []
        if not os.path.exists(self.raw_path): return reports
        
        for f in os.listdir(self.raw_path):
            path = os.path.join(self.raw_path, f)
            if f.endswith(".txt"):
                with open(path, 'r') as file: reports.append(file.read())
            elif f.endswith(".json"):
                try:
                    with open(path, 'r') as file:
                        data = json.load(file)
                        if isinstance(data, list):
                            text_blob = " ".join([f"{entry.get('message', '')} (EventID: {entry.get('event_id', '')})" for entry in data])
                            reports.append(text_blob)
                except Exception as e:
                    logger.error(f"Failed to parse JSON log {f}: {e}")
        return reports
EOF

# src/siem_connector.py
cat << 'EOF' > "$PROJECT_DIR/src/siem_connector.py"
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
EOF

# src/epistemic_math.py
cat << 'EOF' > "$PROJECT_DIR/src/epistemic_math.py"
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
EOF

# src/algorithms.py
cat << 'EOF' > "$PROJECT_DIR/src/algorithms.py"
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
EOF

# src/ontology.py
cat << 'EOF' > "$PROJECT_DIR/src/ontology.py"
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
EOF

# src/llm_interface.py
cat << 'EOF' > "$PROJECT_DIR/src/llm_interface.py"
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
EOF

# src/model.py
cat << 'EOF' > "$PROJECT_DIR/src/model.py"
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
EOF

# src/semantic_analysis.py
cat << 'EOF' > "$PROJECT_DIR/src/semantic_analysis.py"
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
EOF

# src/visualization.py
cat << 'EOF' > "$PROJECT_DIR/src/visualization.py"
import os
import json
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class GraphVisualizer:
    def __init__(self, worlds):
        self.worlds = worlds
        self.output_dir = settings.paths.reports

    def generate_html(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        nodes = []
        edges = []
        
        for i, world in enumerate(self.worlds):
            world_node_id = f"World_{i}"
            active_threats = [f"{k} ({v:.2f})" for k, v in world.facts.items() if v > 0.5]
            label = f"World {i}\n" + "\n".join(active_threats)
            
            nodes.append({"id": world_node_id, "label": label, "color": "#97c2fc", "shape": "box"})
            if i > 0:
                edges.append({"from": f"World_{i-1}", "to": world_node_id, "arrows": "to", "label": "Transition"})

        data = {"nodes": nodes, "edges": edges}
        json_data = json.dumps(data)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neurosymbolic Attack Graph</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">#mynetwork {{ width: 100%; height: 600px; border: 1px solid lightgray; }}</style>
        </head>
        <body>
            <h2>Epistemological Permutations (Fuzzy Logic)</h2>
            <div id="mynetwork"></div>
            <script type="text/javascript">
                var data = {json_data};
                var container = document.getElementById('mynetwork');
                var options = {{ layout: {{ hierarchical: {{ direction: "LR", sortMethod: "directed" }} }}, physics: false }};
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, "attack_graph.html")
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"Graph visualization generated at: {output_path}")
EOF

# src/epistemic_reasoning.py
cat << 'EOF' > "$PROJECT_DIR/src/epistemic_reasoning.py"
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
EOF

# src/ui.py
cat << 'EOF' > "$PROJECT_DIR/src/ui.py"
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
EOF

# src/main.py
cat << 'EOF' > "$PROJECT_DIR/src/main.py"
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
EOF

# --- 6. Generate Tests (Mocked) ---

cat << 'EOF' > "$PROJECT_DIR/tests/test_math.py"
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.epistemic_math import EpistemicMath

class TestEpistemicMath(unittest.TestCase):
    def test_brier_score(self):
        score = EpistemicMath.brier_score(0.8, True)
        self.assertAlmostEqual(score, 0.04)

    def test_bayesian_update(self):
        updated = EpistemicMath.bayesian_update(0.5, 0.8, 0.4)
        self.assertEqual(updated, 1.0)

if __name__ == '__main__':
    unittest.main()
EOF

cat << 'EOF' > "$PROJECT_DIR/tests/test_reasoning.py"
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.epistemic_reasoning import GameModel
import src.config

# Mock Config
src.config.asset_config = {
    "actions": {
        "exploit": 1.5, "lateral_move": 2.0, "exfiltrate": 5.0,
        "encrypt": 10.0, "isolate": 0.0, "patch": 0.0, "monitor": 0.0
    }
}

class TestReasoning(unittest.TestCase):
    def test_defender_strategy(self):
        gm = GameModel()
        state = {"vulnerable": 1.0}
        strategy = gm.calculate_defender_strategy(state)
        self.assertIsNotNone(strategy)

if __name__ == '__main__':
    unittest.main()
EOF

# --- 7. Generate Scripts ---

# scripts/preflight.py
cat << 'EOF' > "$PROJECT_DIR/scripts/preflight.py"
import sys
import os
import importlib.util
import subprocess

REQUIRED_PACKAGES = [
    "networkx",
    "pydantic",
    "torch",
    "transformers",
    "spacy",
    "pyyaml",
    "pytest",
    "mypy",
    "llama-cpp-python"
]

def check_dependencies():
    print("[*] Checking dependencies...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    
    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print("[*] Attempting to install missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("[*] Installation successful.")
        except subprocess.CalledProcessError:
            print("[!] Failed to install packages automatically.")
            return False
    else:
        print("[*] All dependencies installed.")
    return True

def check_spacy_model():
    print("[*] Checking spaCy model 'en_core_web_sm'...")
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("[*] spaCy model found.")
    except OSError:
        print("[!] spaCy model not found. Downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("[*] Download successful.")
        except Exception as e:
            print(f"[!] Failed to download spaCy model: {e}")

def main():
    print("=== Neurosymbolic AI Preflight Check ===")
    if not check_dependencies():
        sys.exit(1)
    check_spacy_model()
    print("\n[SUCCESS] Preflight complete. System is ready.")

if __name__ == "__main__":
    main()
EOF

# scripts/visualize_graph.py (NEW)
cat << 'EOF' > "$PROJECT_DIR/scripts/visualize_graph.py"
import argparse
import sqlite3
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import settings

def serve_reports():
    report_dir = settings.paths.reports
    if not os.path.exists(report_dir):
        print(f"[!] Report directory {report_dir} does not exist.")
        return
    os.chdir(report_dir)
    port = 8000
    server_address = ("", port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    url = f"http://localhost:{port}/attack_graph.html"
    print(f"[-] Serving reports at {url}")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[-] Server stopped.")

def inspect_node(node_id, depth=1):
    db_path = settings.paths.database
    if not os.path.exists(db_path):
        print("[!] Database not found.")
        return
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        print("[!] Please install pyvis: pip install pyvis")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    G = nx.DiGraph()
    queue = [(node_id, 0)]
    visited = set()
    while queue:
        current, current_depth = queue.pop(0)
        if current in visited or current_depth > depth: continue
        visited.add(current)
        cursor.execute("SELECT name, description FROM nodes WHERE id=?", (current,))
        row = cursor.fetchone()
        label = f"{current}\n{row[0]}" if row else current
        G.add_node(current, label=label, color="#ff9999" if current == node_id else "#97c2fc")
        cursor.execute("SELECT target, relation, probability FROM edges WHERE source=?", (current,))
        edges = cursor.fetchall()
        for target, relation, prob in edges:
            G.add_edge(current, target, label=f"{relation}\n({prob:.2f})")
            if target not in visited: queue.append((target, current_depth + 1))
    conn.close()
    if len(G.nodes) == 0:
        print(f"[!] Node {node_id} not found.")
        return
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    output_file = f"inspection_{node_id}.html"
    net.save_graph(output_file)
    target_path = os.path.join(settings.paths.reports, output_file)
    if os.path.abspath(output_file) != os.path.abspath(target_path):
        os.rename(output_file, target_path)
    print(f"[-] Graph saved to {target_path}")
    webbrowser.open("file://" + os.path.abspath(target_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("id")
    inspect_parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    if args.command == "serve": serve_reports()
    elif args.command == "inspect": inspect_node(args.id, args.depth)
EOF

echo "[-] Scripts generated."

# --- 8. Generate Documentation ---

# docs/user_manual.md
cat << 'EOF' > "$PROJECT_DIR/docs/user_manual.md"
# Neurosymbolic AI - User Manual
**Author:** Psypher Labs  
**Version:** 7.0.0 (Enterprise)

## 1. Introduction
Neurosymbolic AI is a hybrid cybersecurity threat analysis tool. It combines Neural Networks, Epistemic Logic, and Game Theory.

## 2. New Features (v7.0)
*   **Caching:** MITRE data is pickled for fast startup.
*   **Parallelism:** Data ingestion uses multi-core processing.
*   **Visualization:** Generates `reports/attack_graph.html`.
*   **Feedback Loop:** Stores user feedback in `data/db/neurosymbolic.db`.
*   **Game Theory:** Includes Defender Minimax strategy.
*   **Sliding Window NLP:** Robust context extraction for complex reports.
*   **LLM Integration:** Uses TinyLlama for advanced reasoning.

## 3. Usage
1. **Bootstrap**: Run `./bootstrap_prod.sh`.
2. **Preflight**: Run `python scripts/preflight.py`.
3. **Run**: `python -m src.main`.
4. **View Report**: Open `reports/attack_graph.html`.
EOF

# README.md
cat << 'EOF' > "$PROJECT_DIR/README.md"
# Neurosymbolic AI (Enterprise Edition)

**By Psypher Labs**

A production-ready cybersecurity threat analysis tool using a hybrid neuro-symbolic approach.

## Overview
Neurosymbolic AI analyzes real cybersecurity data to generate epistemological permutations (representations of possible future states) via epistemic logic, game theory, and probability models.

## Quick Start

1. **Setup**:
   ```bash
   ./bootstrap_prod.sh
   python scripts/preflight.py
