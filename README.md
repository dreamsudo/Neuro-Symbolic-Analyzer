Here is your clean, bullet-listed version of the Neurosymbolic AI Enterprise Manual formatted for a README.md on GitHub.

Neurosymbolic AI – Enterprise Manual

Version: 7.0.0
Author: Psypher Labs
Architecture: Hybrid Neuro-Symbolic (Deep Learning + Epistemic Logic)



System Overview

	•	Bridges unstructured data (logs, reports) and structured reasoning (logic, math)
	•	Core Capabilities:
	•	Perception: Transformer models (BERT/LLM) extract threat entities from text
	•	Knowledge: MITRE ATT&CK mapping using NetworkX + SQLite
	•	Reasoning: Predicts future attack paths using Kripke Semantics and Game Theory
	•	Prediction: Uses Bayesian updates + Fuzzy Logic for threat probability



Installation & Setup

	•	Prerequisites:
	•	OS: Ubuntu/Debian Linux or macOS
	•	Python: 3.8+
	•	RAM: 4GB minimum (8GB+ recommended for LLM)
	•	Tools: gcc/g++ (for llama-cpp-python)
	•	Step 1: Bootstrap

./bootstrap_prod.sh


	•	Step 2: Preflight Check

cd directory
python3 scripts/preflight.py



Configuration Reference

settings.json

	•	Paths
	•	raw_data: Directory for logs
	•	database: Path to SQLite DB
	•	AI Engine
	•	enabled: true (AI mode) or false (regex fallback)
	•	provider: "local" or "external"
	•	model_type: "bert" or "llm"
	•	model_name: e.g. distilbert-base-uncased
	•	llm_path: path to GGUF model file
	•	Database
	•	download_enabled: Fetch MITRE CTI from GitHub
	•	use_cache: Load .pkl cache if available
	•	use_sqlite_graph: Save nodes/edges to SQLite
	•	Reasoning
	•	max_simulation_depth: How far to simulate (e.g., 5)
	•	fuzzy_threshold: Threat confidence threshold
	•	enable_defender_simulation: Use minimax logic
	•	SIEM
	•	enabled: Enable alert polling

assets.json

	•	Assets:
	•	"DomainController": 100
	•	"Printer": 5
	•	Actions:
	•	"encrypt": 10.0
	•	"phishing": 1.0



Data Management

	•	Input Files:
	•	.txt: Incident narratives
	•	.json: Structured logs (event logs)
	•	Caching:
	•	Cached to: data/knowledge_base/mitre_cache.pkl
	•	Delete .pkl or set "use_cache": false to refresh
	•	Graph DB (SQLite):
	•	Path: data/db/neurosymbolic.db
	•	Tables:
	•	nodes: Threat IDs and descriptions
	•	edges: Relationships and probabilities
	•	feedback: User corrections (true/false positives)



Operational Guide

	•	Run Analysis

python3 -m src.main


	•	Output Types:
	•	Risk Score: BERT, LLM, or heuristic confidence
	•	Worlds:
	•	World 0: Current
	•	World 1+: Predicted futures
	•	Reasoning Trace: Fact → Inference → Prediction
	•	Alerts: Triggered only for future threats



Tools & Utilities
	•	Graph Visualizer
	•	Serve HTML:

python3 scripts/visualize_graph.py serve


	•	Inspect a threat node:

python3 scripts/visualize_graph.py inspect T1059 --depth 2


	•	Preflight Checker
	•	Validates Python deps, model files, paths:

python3 scripts/preflight.py





Algorithms & Logic

	•	Sliding Window NLP (semantic_analysis.py)
	•	Detects threat chains using a 200-char window
	•	Example keywords: "leads to", "if/then", "results in"
	•	Epistemic Math (epistemic_math.py)
	•	Bayesian Update: P(H|E) = P(E|H) * P(H) / P(E)
	•	Brier Score: Accuracy evaluator
	•	Clamping: Probabilities capped between 0.0 and 1.0
	•	Game Theory (epistemic_reasoning.py)
	•	Minimax simulation
	•	Chooses optimal Defender action based on Attacker utility



Troubleshooting

	•	Data changes ignored:
	•	Cause: Old .pkl cache
	•	Fix: Delete mitre_cache.pkl or disable cache in settings.json
	•	ModuleNotFoundError: src:
	•	Fix: Run from repo root or use patched main.py
	•	LLM_NOT_LOADED:
	•	Cause: Missing model or dependency
	•	Fix: Run preflight.py, ensure model and llama-cpp-python exist
	•	1.16 Confidence Score:
	•	Cause: Math overflow
	•	Fix: Use version 7.0 (includes clamping)
	•	Blank graph output:
	•	Cause: No input data
	•	Fix: Add .txt to data/raw/, enable auto-download in settings.json
