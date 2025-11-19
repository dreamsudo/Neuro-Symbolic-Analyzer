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
