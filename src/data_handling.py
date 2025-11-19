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
