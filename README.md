Neurosymbolic AI - Enterprise Manual
Version: 7.0
Architecture: Hybrid Neuro-Symbolic (Deep Learning + Epistemic Logic)
Author: Psypher Labs

1. System Overview

Neurosymbolic AI is a threat analysis platform that bridges the gap between unstructured data (e.g., logs, reports) and structured reasoning (e.g., logic, math).

Core Capabilities
	•	Perception: Uses Transformer models (e.g., BERT or LLMs) to extract threat entities from text.
	•	Knowledge: Maps entities to MITRE ATT&CK using a hybrid graph (NetworkX + SQLite).
	•	Reasoning: Simulates future attack states using Kripke Semantics and Game Theory.
	•	Prediction: Uses Bayesian updates and Fuzzy Logic to compute future threat probabilities.


2. Installation & Setup

Prerequisites
	•	OS: Ubuntu/Debian Linux or macOS
	•	Python: 3.8+
	•	System: Minimum 4GB RAM (8GB+ recommended for LLM mode)
	•	Compilers: gcc / g++ (required for llama-cpp-python)

Step 1: Bootstrap

./bootstrap_prod.sh

Step 2: Preflight Check

cd directory
python3 scripts/preflight.py


3. Configuration Reference

A. settings.json

Located in directory/config/.

Section	Key	Type	Description
Paths	raw_data	String	Folder with input logs
	database	String	Path to SQLite DB
AI Engine	enabled	Bool	Enable AI
	provider	String	"local" or "external"
	model_type	String	"bert" or "llm"
	model_name	String	HuggingFace model ID
	llm_path	String	Path to GGUF model
Database	download_enabled	Bool	Download MITRE CTI
	use_cache	Bool	Use .pkl cache
	use_sqlite_graph	Bool	Persist graph to DB
Reasoning	max_simulation_depth	Int	Future steps to simulate
	fuzzy_threshold	Float	Confidence cutoff
	enable_defender_simulation	Bool	Run minimax defender sim
SIEM	enabled	Bool	Poll mock alert API

B. assets.json

Defines target values and action impact:

"DomainController": 100,
"Printer": 5,
"encrypt": 10.0,
"phishing": 1.0



4. Data Management

Input Formats
	•	.txt – Human reports:
T1059 was detected on Host A...
	•	.json – Machine logs:
[{"message": "...", "event_id": 4624}]

Caching Mechanism
	•	Cache file: data/knowledge_base/mitre_cache.pkl
	•	To refresh data:
Delete .pkl file or set "use_cache": false in settings.json

SQLite Graph Database
	•	Path: data/db/neurosymbolic.db
	•	Tables: nodes, edges, feedback

5. Operational Guide

Run the Analyzer

python3 -m src.main

Interpreting Output
	•	Risk Assessment: Confidence from BERT, LLM, or heuristic
	•	Worlds:
	•	World 0 = current state
	•	World 1+ = predicted future threats
	•	Trace: Fact → Inference → Prediction
	•	Alerts: Trigger if threat appears in a future world but not present


6. Tools & Utilities

A. Graph Visualizer

Serve reports:

python3 scripts/visualize_graph.py serve

Inspect threats:

python3 scripts/visualize_graph.py inspect T1059 --depth 2

B. Preflight Checker

python3 scripts/preflight.py

Checks:
	•	Python dependencies
	•	GGUF model presence
	•	File access and structure

7. Algorithms & Logic

1. Sliding Window NLP (src/semantic_analysis.py)
	•	Scans 200-character windows for chains like:
Threat A → keyword → Threat B
	•	Keywords: leads to, implies, if/then, etc.

2. Epistemic Math (src/epistemic_math.py)
	•	Bayesian Update:
P(H|E) = P(E|H) * P(H) / P(E)
	•	Brier Score: Accuracy of probabilistic predictions
	•	Clamping: Keeps probabilities in range [0.0, 1.0]

3. Game Theory (src/epistemic_reasoning.py)
	•	Minimax Algorithm
	•	Simulates defender actions and attacker utilities
	•	Chooses defender moves to minimize attacker advantage


8. Troubleshooting

Symptom	Cause	Solution
Output doesn’t update	Stale cache	Delete .pkl or set "use_cache": false
ModuleNotFoundError: src	Wrong path	Run from root or use patched main.py
LLM_NOT_LOADED	Model missing	Run preflight.py to download
1.16 Confidence	Overflow	Ensure v7.0 is used (includes clamping)
Empty graph HTML	No data	Place .txt in data/raw/ and enable auto-download
