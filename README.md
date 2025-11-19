# 📄 Neurosymbolic AI: Technical White Paper
**Version:** 7.1.0 (Enterprise)  

**Author:** Psypher Labs

---

## 1. Executive Summary
Neurosymbolic AI represents a paradigm shift in automated threat analysis. While traditional systems rely on rigid signatures (SIEMs) or opaque statistical models (Deep Learning), this system bridges the gap. It fuses **Neural Networks (NLP)** for unstructured data perception with **Formal Epistemic Logic** and **Game Theory** for structured reasoning. 

The result is a system that reads human-written threat reports, understands context via sliding-window analysis, and mathematically simulates future attack vectors with explainable precision.

## 2. System Architecture

The architecture is defined by a unidirectional data pipeline transforming unstructured text into probabilistic future worlds.

### 2.1. The Perception Layer (Neuro)
*   **Objective:** Convert unstructured text logs into structured facts.
*   **Hybrid Engine:**
    *   **BERT (DistilBERT):** Used for high-speed classification and entity extraction.
    *   **LLM (TinyLlama/Mistral):** Used for "Chain of Thought" reasoning to extract complex causal relationships (e.g., *"If X happens, it implies Y"*).
*   **Sliding Window NLP:** A custom algorithm scans text streams for causal keywords (`leads to`, `suggests`) within a 200-character window to link threat actors to techniques dynamically, overcoming sentence boundary limitations.

### 2.2. The Knowledge Layer (Ontology)
*   **Objective:** Map extracted facts to a universal standard.
*   **MITRE CTI Integration:** The system ingests the full STIX 2.0 repository to build a graph of 500+ known techniques.
*   **Hybrid Graph Storage:** 
    *   **NetworkX:** In-memory traversal for rapid simulation.
    *   **SQLite:** Persistent storage for nodes, edges, and feedback history.
*   **Caching:** `pickle` serialization ensures sub-second startup times after the initial parse.

### 2.3. The Reasoning Layer (Symbolic)
*   **Objective:** Simulate future states based on current facts.
*   **Epistemic Logic (Kripke Semantics):** Models "Possible Worlds" ($W$). It evaluates formulas like $B_a(\phi)$ (The system *Believes* $\phi$) to distinguish between confirmed breaches and predicted pivots.
*   **Bayesian Belief Updates:** Updates the probability of a future event based on new evidence:
    $$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$
*   **Adversarial Game Theory (Minimax):** Models the interaction between an **Attacker** (maximizing utility) and a **Defender** (minimizing risk). It calculates the optimal next move for both agents to predict the "Critical Path."

## 3. Algorithmic Deep Dive

### 3.1. Fuzzy Consistency Checker
Instead of binary True/False, the system uses fuzzy confidence scores ($0.0 - 1.0$). If the system detects $P(A) + P(\neg A) > 1.2$, it flags a **Consistency Violation**, preventing contradictory intelligence from polluting the reasoning engine.

### 3.2. Dynamic Utility Scoring
The Game Theory engine uses a configurable asset value system.
*   **Asset Value:** Exploiting a `DomainController` (Value: 100) is prioritized over a `Printer` (Value: 5).
*   **Action Cost:** `Encryption` (Cost: 10.0) is treated as a high-impact, end-game move, whereas `Phishing` (Cost: 1.0) is a low-cost entry move.

### 3.3. Active Learning (Feedback Loop)
When an analyst marks an alert as a "False Positive", the system records this in the SQLite database. Future probability calculations query this history and apply a penalty weight to that specific threat path.

---
