# SafeRAG — Claim-Level Hallucination Detection & Verification Engine

SafeRAG is a **production-ready, post-generation verification and safety control system** for Retrieval-Augmented Generation (RAG) pipelines.
It detects, measures, and prevents hallucinations by **verifying model outputs at the semantic claim level**, rather than trusting the generated answer as a whole.

Unlike standard RAG systems, SafeRAG treats hallucination as a **system-level reliability failure** and enforces **explicit safety decisions** before any output reaches the user.

SafeRAG is:

* **Deterministic**
* **Auditable**
* **Policy-controlled**
* **Quantitatively evaluated**
* **Deployable via API**

It is suitable for **safety-critical domains** such as healthcare, finance, policy, and enterprise AI systems.

---

## Core Insight

> **Hallucinations are not just a model problem — they are a missing verification problem.**

SafeRAG introduces an explicit verification layer **after generation** that independently checks whether the model’s claims are actually supported by evidence.

---

## What SafeRAG Does (At a Glance)

SafeRAG performs the following steps on any LLM output:

1. **Extracts atomic factual claims**
2. **Retrieves supporting evidence**
3. **Verifies claims semantically**
4. **Aggregates evidence conservatively**
5. **Applies policy-driven safety gating**
6. **Logs auditable verification results**

SafeRAG **does not generate text**.
It **controls whether generated text is allowed to pass**.

---

## System Architecture

```
LLM / RAG Output
        ↓
Claim Extraction
        ↓
Evidence Retrieval (BM25)
        ↓
Semantic Claim–Evidence Verification
        ↓
Conservative Aggregation
        ↓
Reliability Metrics
        ↓
Safety Decision
 (ACCEPT / REJECT / REFUSE)
```

SafeRAG acts as a **control plane**, not a generator.

---

## Verification Workflow

### 1. Input (Already-Generated Text)

SafeRAG operates on outputs from any LLM or RAG system.

Example:

```
Metformin is the first line treatment for type 2 diabetes.
ACE inhibitors and ARBs should always be combined.
```

---

### 2. Claim Extraction

The text is decomposed into **atomic claims**:

* “Metformin is the first line treatment for type 2 diabetes”
* “ACE inhibitors and ARBs should always be combined”

This prevents hallucinations from being hidden inside long responses.

---

### 3. Evidence Retrieval

For each claim:

* Top-k evidence passages are retrieved using **BM25**
* Each passage is tracked with a document ID and relevance score

---

### 4. Semantic Claim Verification

Each claim–evidence pair is evaluated using **sentence embeddings**:

* Semantic similarity scoring
* Negation and polarity detection
* Claim classification:

  * **SUPPORTED**
  * **CONTRADICTED**
  * **INSUFFICIENT_EVIDENCE**

This avoids brittle keyword matching and handles paraphrases.

---

### 5. Conservative Aggregation

Multiple evidence results are aggregated conservatively:

* Any contradiction → **CONTRADICTED**
* Otherwise any support → **SUPPORTED**
* Else → **INSUFFICIENT_EVIDENCE**

---

### 6. Safety Decision

System-level decision enforced via policy:

* **ACCEPT** → Output is reliable
* **REJECT** → Contradiction detected
* **REFUSE** → Insufficient evidence or uncertainty

This mirrors real-world safety policies in regulated systems.

---

## Reliability Metrics

SafeRAG computes quantitative reliability metrics for every request:

* Support rate
* Contradiction rate
* Insufficient evidence rate
* Hallucination rate
* Average semantic alignment score

These metrics enable **measured evaluation**, not subjective judgment.

---

## Example Output

```json
{
  "decision": "REFUSE",
  "claims": [
    {
      "claim": "ACE inhibitors and ARBs should always be combined",
      "label": "INSUFFICIENT_EVIDENCE",
      "score": 0.485,
      "evidence_ids": ["0", "1", "2"]
    }
  ],
  "metrics": {
    "support_rate": 0.0,
    "contradiction_rate": 0.0,
    "insufficient_rate": 1.0,
    "hallucination_rate": 1.0,
    "avg_semantic_score": 0.485
  }
}
```

---

## Evaluation & Benchmarking

SafeRAG includes a **lightweight, reproducible evaluation harness** to quantify hallucination reduction.

### Datasets

* Clinical examples
* Finance examples

### Evaluation Measures

* Baseline hallucination rate (no gating)
* Hallucinations reaching the user **after SafeRAG gating**

### Run Evaluation

```bash
python eval/run_eval.py
```

Example result:

```
Baseline hallucination rate: 0.667
SafeRAG pass-through hallucination rate: 0.0
```

This demonstrates **complete elimination of hallucinations reaching the user** in evaluated cases.

---

## API Usage (Demo-Ready)

SafeRAG exposes a **FastAPI service** for integration and demonstration.

### Start the API

```bash
python run_api.py
```

### Open Interactive Docs

```
http://127.0.0.1:8000/docs
```

### Endpoint

```
POST /verify
```

The UI allows you to:

* Paste generated text
* Run verification
* Inspect claims, metrics, and decisions

This serves as the **live demo surface** for SafeRAG.

---

## Auditability

Every verification request generates an **immutable audit log**:

```
logs/saferag_audit.jsonl
```

Each entry includes:

* Input text
* Extracted claims
* Evidence usage
* Metrics
* Final decision
* Timestamp

Audit logging is **non-blocking** and **fail-safe**.

---

## Automated Safety Tests

SafeRAG includes **automated system-level tests** that verify:

* Supported claims → ACCEPT
* Contradictions → REJECT
* Gibberish / low-signal input → REFUSE
* Long inputs are bounded
* Deterministic execution
* Audit logs are written

Run tests:

```bash
pytest -v
```

All tests must pass before deployment.

---

## Repository Structure

```
saferag/
├── app/            # API, schemas, audit logging
├── core/           # Claim extraction, retrieval, verification
├── eval/           # Evaluation harness & datasets
├── tests/          # Automated safety tests
├── data/           # Evidence corpus
├── policies/       # Verification policies
├── run_api.py      # API entrypoint
├── saferag_bootstrap.py
├── requirements.txt
└── README.md
```

---

## Design Principles

* **Verification over trust**
* **Claims over answers**
* **Fail-safe by default**
* **Policy-driven decisions**
* **Auditable execution**
* **Model-agnostic integration**

SafeRAG does not replace RAG — it **controls it**.

---

## Limitations

* Uses semantic similarity rather than full logical inference
* Dependent on evidence corpus quality
* Post-generation verification only (intentional)

These trade-offs favor **transparency, control, and reliability**.

---

## Applications

SafeRAG is applicable to:

* Clinical decision support
* Financial compliance checks
* Legal document validation
* Policy analysis
* Enterprise AI safety layers
* Autonomous decision pipelines

The architecture is **domain-agnostic**.

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer — LLMs, RAG, AI Safety
Focus: **Reliable, auditable, system-level AI systems**

---

## Final Note

SafeRAG is intentionally **simple, explicit, and bounded**.

Its strength is not scale —
its strength is **correctness under uncertainty**.

That is what production-grade AI systems are built on.


