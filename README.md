# SafeRAG: Semantic Claim-Level Verification for Reliable RAG Systems

SafeRAG is a **post-generation verification and safety control system** designed to detect, measure, and mitigate hallucinations in Retrieval-Augmented Generation (RAG) pipelines.

Unlike standard RAG systems that implicitly trust model outputs, SafeRAG **treats hallucination as a system-level failure** and explicitly verifies generated content **at the claim level** using **semantic entailment, evidence aggregation, and decision gating**.

The system is **fully executable**, **deterministic**, and **auditable**, making it suitable for **safety-critical domains** such as healthcare, finance, and policy.

---

## Key Idea

> Hallucinations are not just a model problem — they are a **missing verification problem**.

SafeRAG introduces an explicit verification layer **after generation** that:

1. decomposes model outputs into atomic claims
2. retrieves supporting evidence
3. verifies each claim semantically
4. enforces safety decisions (ACCEPT / REJECT / REFUSE)

---

## Core Capabilities

* **Claim-level verification** (not answer-level scoring)
* **Semantic entailment** using sentence embeddings
* **Multi-evidence aggregation**
* **Explicit safety decisions**
* **Quantitative reliability metrics**
* **PyTorch-only execution (no TensorFlow dependency)**

---

## High-Level Architecture

```
Generated Answer
      ↓
Claim Extraction
      ↓
Evidence Retrieval (BM25)
      ↓
Semantic Claim–Evidence Verification
      ↓
Aggregation & Metrics
      ↓
Decision Gating (ACCEPT / REJECT / REFUSE)
```

SafeRAG acts as a **control layer**, not a generator.

---

## End-to-End Workflow

### 1. Model Output (Input to SafeRAG)

SafeRAG operates on **already-generated text** from any LLM or RAG system.

Example:

```
The first line treatment for type 2 diabetes is metformin.
Combining ACE inhibitors and ARBs is a recommended strategy.
```

---

### 2. Claim Extraction

The output is decomposed into **atomic factual claims**:

* “The first line treatment for type 2 diabetes is metformin”
* “Combining ACE inhibitors and ARBs is a recommended strategy”

This avoids hiding hallucinations inside long responses.

---

### 3. Evidence Retrieval

For each claim:

* Top-k evidence passages are retrieved using **BM25**
* Each passage is tracked with an ID and relevance score

---

### 4. Semantic Verification

Each claim–evidence pair is evaluated using **sentence embeddings**:

* semantic similarity score
* negation and polarity checks
* claim classified as:

  * **SUPPORTED**
  * **CONTRADICTED**
  * **INSUFFICIENT_EVIDENCE**

This handles paraphrases and avoids brittle lexical matching.

---

### 5. Multi-Evidence Aggregation

Verification decisions are aggregated across multiple evidence sources to reduce single-document bias.

Final claim label is chosen conservatively:

* any contradiction → CONTRADICTED
* otherwise supported evidence → SUPPORTED
* else → INSUFFICIENT_EVIDENCE

---

### 6. Metrics & Safety Decision

SafeRAG computes system-level reliability metrics and enforces a decision:

* **ACCEPT** → output is reliable
* **REJECT** → contradiction detected
* **REFUSE** → insufficient evidence

This mirrors real safety policies in clinical and enterprise AI systems.

---

## Example Output

```
Claim: The first line treatment for type 2 diabetes is metformin
Final Label: SUPPORTED
Semantic Score: 0.975

Claim: Combining ACE inhibitors and ARBs is a recommended strategy
Final Label: CONTRADICTED
Semantic Score: 0.71

METRICS:
{
  "support_rate": 0.5,
  "contradiction_rate": 0.5,
  "insufficient_rate": 0.0,
  "hallucination_rate": 0.5,
  "avg_semantic_score": 0.84
}

SYSTEM DECISION: REJECT
```

---

## Reliability Metrics

SafeRAG reports:

* **Support Rate**
* **Contradiction Rate**
* **Insufficient Evidence Rate**
* **Hallucination Rate**
* **Average Semantic Alignment Score**

These metrics allow **quantitative evaluation**, not subjective judgment.

---

## Repository Structure

```
saferag/
├── main.py            # End-to-end execution & decision gating
├── claims.py          # Atomic claim extraction
├── retriever.py       # Evidence retrieval (BM25)
├── semantic.py        # Semantic entailment scoring
├── verifier.py        # Claim classification logic
├── metrics.py         # Reliability metrics
├── data/
│   ├── documents.txt  # Evidence corpus
│   └── generations.txt# Model outputs to verify
├── requirements.txt
└── README.md
```

---

## Installation

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Force PyTorch-only execution:

```bash
export TRANSFORMERS_NO_TF=1
```

---

## Run SafeRAG

```bash
python main.py
```

SafeRAG executes fully offline and deterministically.

---

## Design Principles

* **Verification over trust**
* **Claims over answers**
* **Explicit safety decisions**
* **Auditable system behavior**
* **Model-agnostic integration**

SafeRAG does not replace RAG — it **controls it**.

---

## Limitations

* Uses embedding similarity, not full logical inference
* Performance depends on evidence corpus quality
* Designed for post-generation verification (not generation itself)

These are intentional trade-offs for transparency and reliability.

---

## Relevance Beyond Healthcare

Although demonstrated on clinical examples, SafeRAG generalizes to:

* financial compliance systems
* legal document analysis
* policy verification
* autonomous decision pipelines
* safety-critical AI control systems

The architecture is domain-agnostic.

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer — LLMs, RAG, Reinforcement Learning, AI Safety
Focus: **Reliable, auditable, system-level AI design**

---

## Final Note

SafeRAG is intentionally **simple, explicit, and controllable**.

Its strength is not scale —
its strength is **correctness under uncertainty**.

That is what reliable AI systems are built on.

---

