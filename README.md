# Neurosymbolic AI – Enterprise Manual

**Version**: 7.0.0  
**Author**: Psypher Labs  
**Architecture**: Hybrid Neuro-Symbolic (Deep Learning + Epistemic Logic)

---

## System Overview

Neurosymbolic AI is a threat analysis platform that bridges unstructured data (logs, reports) and structured reasoning (logic, math).

### Core Capabilities

- **Perception**: Uses Transformer models (e.g., BERT or LLMs) to extract threat entities from unstructured text.
- **Knowledge**: Maps those entities to the MITRE ATT&CK framework using a hybrid NetworkX + SQLite graph.
- **Reasoning**: Simulates future attack states using Kripke Semantics (Possible Worlds) and Game Theory.
- **Prediction**: Calculates threat probabilities using Bayesian updates and Fuzzy Logic.

---

## Installation & Setup

### Prerequisites

- OS: Ubuntu/Debian Linux or macOS
- Python: 3.8+
- Memory: 4GB RAM minimum (8GB+ recommended for LLM mode)
- Compilers: `gcc`, `g++` (required for `llama-cpp-python`)

### Step 1: Bootstrap

./bootstrap_prod.sh

### Step 2: Preflight Check


cd directory
python3 scripts/preflight.py


Configuration Reference

A. settings.json

Located at: directory/config/settings.json

Core Keys

	•	raw_data (string): Directory for log input files
	•	database (string): Path to the SQLite file
	•	enabled (bool): Enable AI (true) or fallback to regex (false)
	•	provider (string): "local" or "external" model execution
	•	model_type (string): "bert" or "llm"
	•	model_name (string): HuggingFace model ID (e.g., distilbert-base-uncased)
	•	llm_path (string): Path to local .gguf model file
	•	download_enabled (bool): Automatically pull MITRE CTI from GitHub
	•	use_cache (bool): Use pickled .pkl MITRE cache
	•	use_sqlite_graph (bool): Save knowledge graph to SQLite
	•	max_simulation_depth (int): Number of future states to simulate
	•	fuzzy_threshold (float): Confidence score cutoff (0.0–1.0)
	•	enable_defender_simulation (bool): Enable Minimax-based defense simulation
	•	enabled under SIEM: Enable polling of mock alerts

B. assets.json

Defines Game Theory weights:

"DomainController": 100,
"Printer": 5,
"encrypt": 10.0,
"phishing": 1.0


Data Management

Input Files
	•  .txt: Human-written incident narratives
Example:

T1059 was detected on Host A. This suggests T1003.
	•	.json: Structured machine logs
Example:

[{"message": "...", "event_id": 4624}]

Caching Behavior

	•	Cache file: data/knowledge_base/mitre_cache.pkl
	•	To refresh:
	•	Delete .pkl, or
	•	Set "use_cache": false in settings.json

Graph Database (SQLite)

	•	File: data/db/neurosymbolic.db
	•	Tables:
	•	nodes: Threat identifiers and descriptions
	•	edges: Relationships and their weights
	•	feedback: Stores user-provided truth labels


Operational Guide

Run Main Analyzer

python3 -m src.main

Output Meaning

	•	Risk Scores:
	•	neural: BERT classifier
	•	llm: Extractive confidence
	•	heuristic: Keyword density
	•	Possible Worlds:
	•	World 0 = Current
	•	World N = Predicted
	•	Reasoning Trace:
	•	Fact → Inference → Prediction
	•	Defender strategy: isolate / monitor / patch
	•	Alerts:
	•	Raised when threats appear in future worlds but not World 0


Tools & Utilities

Graph Visualizer

Start local web server:

python3 scripts/visualize_graph.py serve

Inspect threat node:

python3 scripts/visualize_graph.py inspect T1059 --depth 2

Preflight Environment Checker

python3 scripts/preflight.py

Validates:

	•	Python packages (e.g., torch, networkx, llama-cpp-python)
	•	Model file presence
	•	Directory write permissions


Algorithms & Logic

1. Sliding Window NLP

File: src/semantic_analysis.py

	•	Treats text as a character stream
	•	Searches for:

[Threat A] → [Keyword] → [Threat B]


	•	Trigger keywords: leads to, implies, if/then, results in

2. Epistemic Math

File: src/epistemic_math.py

	•	Bayesian Update:

P(H|E) = [P(E|H) * P(H)] / P(E)


	•	Brier Score: Measures probability accuracy
	•	Clamping: Keeps outputs within [0.0, 1.0]

3. Game Theory Engine

File: src/epistemic_reasoning.py

	•	Minimax Algorithm:
	•	Simulates defender actions: isolate, monitor, patch
	•	Chooses action that minimizes attacker’s utility
