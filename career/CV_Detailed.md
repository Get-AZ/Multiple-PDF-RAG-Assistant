# Detailed CV project entry

## Multiple PDF RAG Assistant
Getamesay Zegeye | Independent portfolio project

Developed a local multi-document question-answering application with a shared Python core serving a Streamlit UI and an authenticated HTTP API.

- Implemented page-local 1,500-character chunks with 200-character overlap, stable evidence identities and filename/page metadata.
- Used task-specific Gemini document/query embeddings, validated normalized float32 vectors and exact FAISS IndexFlatIP search over the full uploaded collection.
- Added bounded completed-turn history and standalone follow-up question rewriting.
- Built collection creation, query and deletion endpoints with bearer authentication, upload/body bounds, collection isolation and sanitized provider errors.
- Added an opt-in document-vector cache bound to content, extraction policy, model/configuration and implementation identity, with checksum validation and atomic writes.
- Added citation diagnostics that check filename/page references against retrieved headers while explicitly leaving semantic support and claim coverage unverified.
- Expanded offline validation to 201 tests, covering extraction, retrieval, provenance, saved evaluation bindings, UI state, API behavior and cache integrity.

## Measured development work

On 12 answerable questions within a frozen 15-case development set, original retrieval achieved Recall@6 0.8125, MRR@6 0.6972 and nDCG@6 0.6836. Single-reviewer answer assessment found 15/15 correctness, 13/15 completeness and 11/15 citation support; 9/15 passed all five criteria.

Rejected a citation-prompt candidate because completeness fell from 13/15 to 12/15 despite better citation support. Also rejected a reranker that regressed retrieval metrics.

In one local two-PDF cache benchmark, four cold document-embedding requests fell to zero on each of three warm runs. Cold processing took 4.2626 seconds; warm median processing took 0.8397 seconds. This is a local document-processing observation, not an answer-latency or general speedup claim.

## Scope

Local single-user operation with in-memory indexes and optional persistent document vectors. Google APIs perform embedding and generation. Results concern familiar development documents, not held-out generalization or production deployment.

Skills: Python, Starlette, Streamlit, pypdf, NumPy, FAISS, Google Gen AI SDK, automated testing, provenance, evaluation and API integration.
