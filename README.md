# SafeRAG — Safety-Gated, Claim-Level Verification for RAG Systems

SafeRAG is a **post-generation verification and safety-gating system** for Retrieval-Augmented Generation (RAG) and LLM pipelines.

It prevents hallucinations by **decomposing model outputs into atomic factual claims**, verifying each claim against retrieved evidence, and enforcing a **policy-driven safety decision** *before* any output reaches the user.

SafeRAG treats hallucination not as a stylistic flaw, but as a **system-level reliability failure**.

---

## Why SafeRAG Exists

> **Hallucinations are not just a model problem — they are a missing verification problem.**

Most RAG systems retrieve documents *before* generation and implicitly trust the model to use them correctly.

SafeRAG introduces a **separate, explicit verification layer after generation** that asks a different question:

> *Are the claims the model produced actually supported by evidence?*

If not, the output is **blocked, refused, or rejected**.

---

## What SafeRAG Is (and Is Not)

### SafeRAG **is**:

* Deterministic
* Auditable
* Policy-controlled
* Model-agnostic
* Post-generation
* Domain-agnostic
* Evaluation-driven

### SafeRAG **is not**:

* A text generator
* A prompt-engineering trick
* A heuristic filter
* A black-box safety wrapper

SafeRAG is a **control plane**, not a generation system.

---

## High-Level Workflow

```
LLM / RAG Output
        ↓
Claim Extraction
        ↓
Evidence Retrieval (BM25)
        ↓
Claim–Evidence Verification
        ↓
Conservative Aggregation
        ↓
Dominant Claim Clustering
        ↓
Safety Decision
 (ACCEPT / REFUSE / REJECT)
```

---

## Core Pipeline

### 1. Input: Generated Text

SafeRAG operates on text produced by **any LLM or RAG system**.

Example input:

```
Metformin is the first line treatment for type 2 diabetes.
ACE inhibitors and ARBs should always be combined.
```

---

### 2. Claim Extraction

The output is decomposed into **atomic, factual claims**.

Example:

* “Metformin is the first line treatment for type 2 diabetes”
* “ACE inhibitors and ARBs should always be combined”

This prevents hallucinations from being hidden inside longer answers.

---

### 3. Evidence Retrieval

For each claim:

* Top-k passages are retrieved using **BM25**
* Evidence is deterministic and inspectable
* No embeddings are required for retrieval

---

### 4. Claim Truth Classification

Each claim–evidence pair is classified into one of four **explicit truth states**:

| Label              | Meaning                                       |
| ------------------ | --------------------------------------------- |
| **VERIFIED**       | Claim is supported by evidence                |
| **REFUTED**        | Claim contradicts evidence                    |
| **UNSUPPORTED**    | Evidence is insufficient                      |
| **RISKY_ABSOLUTE** | Overconfident or absolute claim without proof |

Classification uses:

* Semantic similarity (embedding-optional)
* Negation and polarity detection
* Phrase-level grounding for high-confidence cases

No policy decisions happen at this stage.

---

### 5. Conservative Claim Aggregation

If multiple evidence passages exist for a claim:

* Any **REFUTED** verdict dominates
* Otherwise **VERIFIED** dominates
* Otherwise uncertainty is preserved

This ensures **safety-first behavior**.

---

### 6. Dominant Claim Clustering

Claims are grouped by **lexical similarity** to identify the **dominant intent** of the output.

This prevents:

* One strong claim masking multiple weak or unsafe claims
* Mixed outputs being incorrectly accepted

System decisions are based on the **dominant claim cluster**, not isolated claims.

---

### 7. System-Level Safety Decision

Final decision is enforced **after verification**:

| Decision   | Meaning                          |
| ---------- | -------------------------------- |
| **ACCEPT** | All dominant claims are verified |
| **REFUSE** | Claims are uncertain or risky    |
| **REJECT** | Contradictions detected          |

SafeRAG is **fail-safe by default**.

---

## Quantitative Reliability Metrics

For every request, SafeRAG computes:

* Support rate
* Contradiction rate
* Hallucination rate
* Decision distribution

Metrics are deterministic and reproducible.

---

## Evaluation & Benchmarking

SafeRAG includes a **fully reproducible evaluation harness**.

### Datasets

* Clinical domain
* Finance domain

Each dataset contains:

* Supported claims
* Unsupported claims
* Explicit contradictions
* Overconfident absolutes

---

### Run Evaluation

```bash
python eval/run_eval.py
```

---

### Final Results (Actual System Output)

```
=== CLINICAL DATASET ===
Baseline hallucination rate: 0.5
SafeRAG pass-through hallucination rate: 0.0

Decision distribution:
  ACCEPT: 3
  REFUSE: 4
  REJECT: 1

=== FINANCE DATASET ===
Baseline hallucination rate: 0.333
SafeRAG pass-through hallucination rate: 0.0

Decision distribution:
  ACCEPT: 3
  REFUSE: 4
  REJECT: 1
```

**Zero hallucinations reached the user after gating.**

---

## API Usage (Demo Ready)

SafeRAG exposes a **FastAPI service** for integration and demonstration.

### Start API

```bash
python run_api.py
```

### Interactive Docs

```
http://127.0.0.1:8000/docs
```

### Endpoint

```
POST /verify
```

The API returns:

* Extracted claims
* Claim truth labels
* Metrics
* Final decision

---

## Auditability

Every request generates a **structured audit log**:

```
logs/saferag_audit.jsonl
```

Each entry contains:

* Input text
* Claim breakdown
* Evidence usage
* Metrics
* Final decision
* Timestamp

Audit logging is:

* Deterministic
* Non-blocking
* Fail-safe

---

## Automated Safety Tests

SafeRAG includes **system-level tests** validating:

* Supported claims → ACCEPT
* Contradictions → REJECT
* Mixed claims → REFUSE / REJECT
* Gibberish → REFUSE
* Long inputs bounded
* Deterministic execution
* Audit logging

Run:

```bash
pytest -v
```

All tests must pass before deployment.

---

## Repository Structure

```
SafeRAG/
├── app/        # API, schemas, audit logging
├── core/       # Claims, retrieval, verification, policy
├── eval/       # Evaluation harness & datasets
├── tests/      # Automated safety tests
├── data/       # Evidence corpus
├── policies/   # Decision policies
├── run_api.py
├── saferag_bootstrap.py
└── README.md
```

---
Architecture Overview (Conceptual)
```
 ┌─────────────────────────────┐
 │     LLM / RAG Generator     │
 │ (Any model, any framework)  │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │   Generated Natural Text    │
 │ (Untrusted model output)    │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │     Claim Extraction        │
 │  (Atomic factual claims)   │
 │  core/claims.py             │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │   Evidence Retrieval        │
 │   (BM25 / lexical search)  │
 │   core/retriever.py         │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │ Claim–Evidence Verification │
 │  Truth Classification       │
 │ VERIFIED / REFUTED / etc.   │
 │ core/verifier.py            │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │ Conservative Aggregation    │
 │ + Dominant Claim Clustering │
 │ app/service.py              │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │ Policy-Based Safety Gating  │
 │ ACCEPT / REFUSE / REJECT    │
 │ policies/default.yaml       │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │  Auditable Output & Metrics │
 │ logs/saferag_audit.jsonl    │
 └─────────────────────────────┘


## Design Principles

* Claims over answers
* Verification over trust
* Fail-safe by default
* Explicit uncertainty
* Deterministic behavior
* Audit-first design

SafeRAG does not replace RAG — **it controls it**.

---

## Limitations (Intentional)

* No full logical inference engine
* Depends on evidence corpus quality
* Post-generation only (by design)

These trade-offs favor **transparency and control over opacity**.

---

## Applications

SafeRAG is suitable for:

* Clinical decision support
* Financial compliance
* Legal document validation
* Policy analysis
* Enterprise AI safety layers
* Autonomous decision pipelines

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer — LLMs, RAG, AI Safety
Focus: **Reliable, auditable, system-level AI systems**

---

## Final Note

SafeRAG is intentionally **simple, explicit, and bounded**.

Its strength is not scale —
its strength is **correct behavior under uncertainty**.

That is what production-grade AI systems require.

---

