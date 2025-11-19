# 🧠 Neurosymbolic AI (Enterprise Edition)

> **"Where Deep Learning meets Formal Logic."**

**Version:** 7.1.0  
**Author:** Psypher Labs  
**Architecture:** Hybrid Neuro-Symbolic (Deep Learning + Epistemic Logic)

---

## 📋 Table of Contents

1.  [System Overview](#1-system-overview)
2.  [Installation & Setup](#2-installation--setup)
3.  [Configuration Reference](#3-configuration-reference)
4.  [Operational Workflow](#4-operational-workflow)
5.  [Tools & Utilities](#5-tools--utilities)
6.  [Troubleshooting](#6-troubleshooting)

---

## 1. System Overview

Neurosymbolic AI is a next-generation threat analysis platform designed to bridge the gap between unstructured data (human-written reports, logs) and structured reasoning (logic, math).

### Core Capabilities
*   **Perception (Neuro):** Uses Transformer models (BERT) or Large Language Models (LLM) to read text and extract threat entities.
*   **Knowledge (Ontology):** Maps entities to the MITRE ATT&CK framework using a hybrid Graph Database (NetworkX + SQLite).
*   **Reasoning (Symbolic):** Simulates future attack states using Kripke Semantics (Possible Worlds) and Game Theory (Minimax).
*   **Prediction:** Calculates the probability of future attacks using Bayesian updates and Fuzzy Logic.

---

## 2. Installation & Setup

### Prerequisites
*   **OS:** Linux (Ubuntu/Debian) or macOS.
*   **Python:** 3.8 or higher.
*   **Hardware:** 4GB RAM minimum (8GB+ recommended for LLM mode).

### Step 1: Bootstrap
Run the master installer script. This generates the directory structure, downloads necessary models, and writes the source code.

\`\`\`bash
chmod +x bootstrap_prod.sh
./bootstrap_prod.sh
\`\`\`

### Step 2: Preflight Check
Install dependencies and verify the environment integrity.

\`\`\`bash
cd directory
python3 scripts/preflight.py
\`\`\`

---

## 3. Configuration Reference

The system is highly configurable via JSON files located in \`directory/config/\`.

### A. Main Settings (\`settings.json\`)

| Section | Key | Default | Description |
| :--- | :--- | :--- | :--- |
| **AI Engine** | \`enabled\` | \`true\` | Use AI models (True) or Regex Heuristics (False). |
| | \`model_type\` | \`"llm"\` | \`"bert"\` (Fast) or \`"llm"\` (Deep Reasoning). |
| | \`confidence_threshold\` | \`0.75\` | Minimum score to accept a prediction. |
| **Database** | \`download_enabled\` | \`true\` | Auto-download MITRE CTI from GitHub. |
| | \`use_cache\` | \`true\` | Load from \`.pkl\` (Fast). Set \`false\` to force refresh. |
| | \`use_sqlite_graph\` | \`true\` | Persist graph nodes to SQLite. |
| **Reasoning** | \`max_simulation_depth\` | \`5\` | Steps into the future to predict. |
| | \`fuzzy_threshold\` | \`0.6\` | Minimum probability (0.0-1.0) for a "plausible" world. |
| | \`enable_defender_simulation\` | \`true\` | Calculate optimal Defender moves (Minimax). |

### B. Asset Scoring (\`assets.json\`)

Defines the "Game Board" for the Game Theory engine.

| Asset Type | Value | Description |
| :--- | :--- | :--- |
| \`DomainController\` | **100** | High Value Target. High probability of attack. |
| \`Database\` | **80** | Critical Data Store. |
| \`Server\` | **50** | Standard Infrastructure. |
| \`Workstation\` | **10** | End-user device. |
| \`Printer\` | **5** | Low Value Target. |

---

## 4. Operational Workflow

### Step 1: Ingest Data
Place your input files in the \`directory/data/raw/\` folder.
*   **Narrative Text (\`.txt\`):** Human-written incident reports.
*   **Structured Logs (\`.json\`):** Machine logs.

### Step 2: Run Analysis
Execute the main module.

\`\`\`bash
python3 -m src.main
\`\`\`

### Step 3: Interpret Output
The console will display:
1.  **Risk Assessment:** A confidence score (0.0 - 1.0).
2.  **Active Threats:** Threats detected as currently present.
3.  **Epistemological Permutations:** A list of "Possible Worlds" showing how the threat might evolve.
4.  **Reasoning Trace:** A step-by-step log of the AI's logic.
5.  **Alerts:** Warnings if the system *believes* a critical attack path is active.

### Step 4: Visualize
Open the generated report in your browser to see the interactive attack graph.
*   **File:** \`directory/reports/attack_graph.html\`

---

## 5. Tools & Utilities

### Graph Visualizer (\`scripts/visualize_graph.py\`)

| Command | Usage | Description |
| :--- | :--- | :--- |
| **Serve** | \`python3 scripts/visualize_graph.py serve\` | Starts a local web server to view reports. |
| **Inspect** | \`python3 scripts/visualize_graph.py inspect T1059\` | Generates a graph for a specific threat ID. |

---

## 6. Troubleshooting

| Symptom | Solution |
| :--- | :--- |
| **Output doesn't change** | Run \`rm directory/data/knowledge_base/mitre_cache.pkl\` to force a refresh. |
| **"ModuleNotFoundError"** | Run from the root folder, or use the patched \`main.py\`. |
| **"LLM_NOT_LOADED"** | Run \`scripts/preflight.py\` to download the model. |
| **Empty Graph HTML** | Ensure \`data/raw\` has text files and \`download_enabled\` is true. |

---
**License:** Enterprise / Proprietary  
**Copyright:** © 2023 Psypher Labs
