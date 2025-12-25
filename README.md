---

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

If not, the output is **safely contained** — blocked, refused, or rejected — rather than silently passed to the user.

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

Example:

```
Metformin is the first line treatment for type 2 diabetes.
ACE inhibitors and ARBs should always be combined.
```

---

### 2. Claim Extraction

The output is decomposed into **atomic, factual claims**:

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
| **VERIFIED**       | Supported by evidence                         |
| **REFUTED**        | Contradicts evidence                          |
| **UNSUPPORTED**    | Insufficient evidence                         |
| **RISKY_ABSOLUTE** | Overconfident or absolute claim without proof |

Classification uses:

* Semantic similarity (embedding-optional)
* Negation and polarity detection
* Phrase-level grounding for high-confidence cases

**No policy decisions happen here.**

---

### 5. Conservative Claim Aggregation

If multiple evidence passages exist:

* Any **REFUTED** verdict dominates
* Otherwise **VERIFIED** dominates
* Otherwise uncertainty is preserved

This enforces **safety-first behavior**.

---

### 6. Dominant Claim Clustering

Claims are grouped by **lexical similarity** to identify the **dominant intent** of the output.

This prevents:

* One strong claim masking multiple unsafe claims
* Mixed outputs being incorrectly accepted

System decisions are based on the **dominant claim cluster**, not isolated claims.

---

### 7. System-Level Safety Decision

Final decision is enforced **after verification**:

| Decision   | Meaning                                         |
| ---------- | ----------------------------------------------- |
| **ACCEPT** | Dominant claims are verified                    |
| **REFUSE** | Output is uncertain or risky (safe containment) |
| **REJECT** | Explicit contradiction detected                 |

**REFUSE is not failure** — it is an intentional, safe outcome when reliability cannot be established.

---

## Quantitative Reliability Metrics

For every request, SafeRAG computes:

* Support rate
* Contradiction rate
* Decision distribution
* Pass-through hallucination rate

> **Key distinction:**
> SafeRAG measures not just hallucinations *detected*, but hallucinations **reaching the user**.

---

## Evaluation & Benchmarking

SafeRAG includes a **fully reproducible evaluation harness**.

### Datasets

* Clinical domain
* Finance domain

Each dataset includes:

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

**Hallucinations still exist — but none reach the user after gating.**

---

## API Usage (Demo Ready)

SafeRAG exposes a **FastAPI service**.

```bash
python run_api.py
```

Docs:

```
http://127.0.0.1:8000/docs
```

Endpoint:

```
POST /verify
```

---

## Auditability

Every request generates a structured audit log:

```
logs/saferag_audit.jsonl
```

Includes:

* Claims
* Labels
* Metrics
* Final decision
* Timestamp

Audit logging is deterministic, non-blocking, and fail-safe.

---

## Automated Safety Tests

SafeRAG includes system-level tests validating:

* Supported claims → ACCEPT
* Contradictions → REJECT
* Mixed claims → REFUSE / REJECT
* Gibberish → REFUSE
* Determinism
* Audit logging

```bash
pytest -v
```

---

## Architecture Overview (Conceptual)

*(keep your ASCII diagram exactly as is — it’s excellent)*

---

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
* Dependent on evidence corpus quality
* Post-generation only (by design)

These trade-offs favor **transparency and control over opacity**.

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer — LLMs, RAG, AI Safety

---

## Final Note

SafeRAG is intentionally **simple, explicit, and bounded**.

Its strength is not scale —
its strength is **correct behavior under uncertainty**.

That is what production-grade AI systems require.

---


