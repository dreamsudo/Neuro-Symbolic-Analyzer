Neurosymbolic AI - Enterprise Manual
Version: 7.0.0
Architecture: Hybrid Neuro-Symbolic (Deep Learning + Epistemic Logic)
Author: Psypher Labs
1. System Overview
Neurosymbolic AI is a threat analysis platform that bridges the gap between unstructured data (logs, reports) and structured reasoning (logic, math).
Core Capabilities
Perception: Reads text using Transformer models (BERT) or Large Language Models (LLM) to extract threat entities.
Knowledge: Maps entities to the MITRE ATT&CK framework using a hybrid Graph Database (NetworkX + SQLite).
Reasoning: Simulates future attack states using Kripke Semantics (Possible Worlds) and Game Theory (Minimax).
Prediction: Calculates the probability of future attacks using Bayesian updates and Fuzzy Logic.
2. Installation & Setup
Prerequisites
OS: Linux (Ubuntu/Debian) or macOS.
Python: 3.8+.
System: 4GB RAM minimum (8GB+ recommended for LLM mode).
Compilers: gcc / g++ (Required for llama-cpp-python).
Step 1: Bootstrap
Run the master installer. This generates the directory structure, downloads models, and writes source code.
code
Bash
./bootstrap_prod.sh
Step 2: Preflight
Install dependencies and verify the environment.
code
Bash
cd directory
python3 scripts/preflight.py
3. Configuration Reference
The system is controlled by two JSON files in directory/config/.
A. Main Settings (settings.json)
Section	Key	Value Type	Description
Paths	raw_data	String	Directory for input logs (.txt, .json).
database	String	Path to the SQLite file (.db).
AI Engine	enabled	Bool	true: Use AI. false: Use Regex (Heuristic).
provider	String	"local" (runs on VM) or "external" (API).
model_type	String	"bert" (Fast classification) or "llm" (Deep reasoning).
model_name	String	HuggingFace ID (e.g., distilbert-base-uncased).
llm_path	String	Path to GGUF file (e.g., data/models/tinyllama...).
Database	download_enabled	Bool	true: Auto-download MITRE CTI from GitHub.
use_cache	Bool	true: Load from .pkl (Fast). false: Re-parse JSONs (Slow). Disable this to force data refresh.
use_sqlite_graph	Bool	true: Persist graph nodes to SQLite.
Reasoning	max_simulation_depth	Int	How many future steps to predict (e.g., 5).
fuzzy_threshold	Float	Confidence cutoff (0.0-1.0) to consider a threat "Real".
enable_defender_simulation	Bool	true: Calculate optimal Defender moves (Minimax).
SIEM	enabled	Bool	true: Poll mock API for alerts.
B. Asset & Utility Scoring (assets.json)
Defines the "Value" of targets for Game Theory calculations.
Assets: Higher value = Higher probability attacker targets it.
"DomainController": 100
"Printer": 5
Actions: Cost/Reward multiplier for specific attack types.
"encrypt": 10.0 (High impact)
"phishing": 1.0 (Low cost)
4. Data Management
Input Formats
Place files in data/raw/.
Narrative Text (.txt): Human-written incident reports.
Example: "T1059 was detected on Host A. This suggests T1003."
Structured Logs (.json): Machine logs.
Format: [{"message": "...", "event_id": 4624}]
Caching Mechanism (Crucial)
The system pickles the MITRE Ontology to data/knowledge_base/mitre_cache.pkl after the first run.
Issue: If you change raw data or MITRE files, the system might ignore them and load the old cache.
Fix: Delete the .pkl file or set "use_cache": false in settings to force a rebuild.
Graph Database (SQLite)
Located at data/db/neurosymbolic.db.
Nodes Table: Stores Threat IDs and Descriptions.
Edges Table: Stores relationships and probabilities.
Feedback Table: Stores user corrections (True/False Positives).
5. Operational Guide
Running the Analysis
code
Bash
python3 -m src.main
Interpreting Output
Risk Assessment:
(neural): BERT confidence score.
(llm): LLM extraction confidence.
(heuristic): Keyword density score.
Epistemological Permutations:
World 0: Current Facts (What is happening now).
World 1+: Predicted Futures (What will happen).
Format: ['ThreatID:Confidence'] (e.g., T1059:1.00).
Reasoning Trace:
Shows the logic chain: Fact -> Inference -> Prediction.
Shows Defender Strategy: Defender should 'isolate'.
Alerts:
[ALERT] System BELIEVES 'Txxxx' is imminent.
Only triggers for threats that appear in Future Worlds but not World 0.
6. Tools & Utilities
A. Graph Visualizer (scripts/visualize_graph.py)
A CLI tool to inspect the database and view reports.
Serve Reports (Web View):
Starts a local web server to view the interactive HTML graph.
code
Bash
python3 scripts/visualize_graph.py serve
Inspect Specific Threat:
Queries SQLite to generate a graph for a single node and its neighbors.
code
Bash
python3 scripts/visualize_graph.py inspect T1059 --depth 2
B. Preflight Checker (scripts/preflight.py)
Verifies:
Python Libraries (torch, networkx, llama-cpp-python).
Model existence (.gguf files).
Directory permissions.
7. Algorithms & Logic (Deep Dive)
1. Sliding Window NLP (src/semantic_analysis.py)
Instead of splitting by sentences (which fails on bullet points), the system treats text as a stream.
Logic: It scans for [Threat A] ... [Causal Keyword] ... [Threat B] within a 200-character window.
Keywords: "leads to", "implies", "results in", "if/then".
2. Epistemic Math (src/epistemic_math.py)
Bayesian Update: 
P
(
H
∣
E
)
=
P
(
E
∣
H
)
⋅
P
(
H
)
P
(
E
)
P(H∣E)= 
P(E)
P(E∣H)⋅P(H)
​
 
. Used to update the probability of a future threat based on the certainty of the current threat.
Brier Score: Measures the accuracy of probabilistic predictions against ground truth (used in testing).
Clamping: Ensures probabilities never exceed 1.0 or drop below 0.0.
3. Adversarial Game Theory (src/epistemic_reasoning.py)
Algorithm: Minimax (Simplified).
Process:
The system looks at the current state.
It simulates available Defender Actions (isolate, patch, monitor).
For each action, it calculates the Attacker's maximum possible Utility.
It chooses the Defender Action that minimizes the Attacker's maximum Utility.
8. Troubleshooting
Symptom	Cause	Solution
Output doesn't change after editing data	Stale Cache	Run rm directory/data/knowledge_base/mitre_cache.pkl.
"ModuleNotFoundError: src"	Path Issue	Run from the root folder, or use the patched main.py which auto-fixes paths.
"LLM_NOT_LOADED"	Missing Model/Lib	Run scripts/preflight.py to download the model. Ensure llama-cpp-python is installed.
"1.16 Confidence"	Math Overflow	Ensure you are running the v7.0 code (Math Clamping patch applied).
Empty Graph HTML	No Data	Ensure data/raw has text files and settings.json has download_enabled: true.

