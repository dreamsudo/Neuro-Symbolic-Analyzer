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
