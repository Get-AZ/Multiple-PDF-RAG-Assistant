# Technical report: Multiple PDF RAG Assistant v1.0.0

Getamesay Zegeye | Independent portfolio project | Updated 12 November 2026

## Objective and scope

The application answers questions across text-based PDFs and exposes the retrieved evidence for inspection. Version 1.0.0 provides a Streamlit interface and a bearer-authenticated Starlette API backed by the same Python pipeline. It is a local single-user application, not a hosted production service.

The implementation separates retrieval, answer generation and citation reference diagnostics. Retrieving a passage does not establish that the model used it, and a matching filename/page citation does not prove that its claim is supported.

## Architecture

| Component | Responsibility |
|---|---|
| `streamlit_app.py` | Upload controls, session state, chat and evidence display |
| `service_api.py` / `run_service.py` | HTTP validation, bearer authentication and localhost startup |
| `rag_service.py` | In-memory collection creation, query, deletion and operation locking |
| `rag_core.py` | Shared embedding, retrieval, follow-up rewriting and generation |
| `rag_provenance.py` | Page-local chunks, hashes, offsets and stable evidence identities |
| `embedding_cache.py` | Optional integrity-checked document-vector persistence |
| `citation_validation.py` | Deterministic reference matching against retrieved headers |

The UI calls the core directly; it does not require the HTTP server or its service token. UI sessions and API collections have separate indexes and lifecycles. The historical `app.py` remains unchanged for evaluation source binding; it is not the recommended current UI entry point.

## Extraction and retrieval

pypdf extracts existing page text; OCR is not implemented. Chunks use 1,500-character windows with 200-character overlap and never cross page boundaries. Empty pages retain their original PDF page numbering. Overlap helps continuity but can produce redundant passages and split sentences.

Document identities bind the PDF bytes. Chunk identities bind document identity, extraction/chunk policy and text location; renaming a file does not change its content identity. Display filenames remain current, including document-number labels for duplicate filenames.

The core uses `gemini-embedding-001`, with RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions. Document calls contain up to 20 chunks. Embeddings are converted to contiguous float32 arrays, checked for shape, finite values and nonzero norms, then L2-normalized. Exact FAISS IndexFlatIP search returns up to six passages across the whole collection. It does not guarantee representation from every PDF.

## Follow-ups and generation

The shared core uses `gemini-3.5-flash-lite`. The latest three completed user/assistant pairs are bounded to 1,500 user and 3,000 assistant characters per pair. History is provided to resolve references, not as factual evidence.

With history, a separate request rewrites the current question into a standalone search question, with a 256-token output cap. The answer request receives the retrieved text, history and original/rewritten question, with a 1,000-token output cap. The configured client timeout is 60 seconds with one attempt.

The original answer prompt asks for filename/page citations, supported comparisons and brief identification of missing details. These instructions do not guarantee complete answers or correct citations. Provider model IDs and returned metadata improve auditability but do not independently pin provider weights.

## State, API boundaries and data handling

Successful processing builds a replacement index before publishing it. Failed UI processing preserves the previous index; changed upload selections disable questions until restored or successfully reprocessed. Clearing chat retains the index.

The HTTP API exposes unauthenticated GET /health and authenticated collection creation, query and deletion. Health checks local process/configuration state without calling Google. The service token is separate from the Google key; every token holder can access every collection.

| Boundary | Limit |
|---|---|
| Upload | 1–4 PDFs, 6 MiB each |
| Total HTTP body | 12 MiB including multipart overhead |
| Resident collections | 4 |
| Chunks per collection | 2,000 |
| Question | 1–4,000 characters |
| HTTP history | Up to 3 completed alternating pairs, 6,000 characters per message before core caps |
| Collection operations | One at a time; overlapping operations return 409 |

The server runs on 127.0.0.1 with one worker. Collections disappear after deletion or server restart. The API does not store chat history; callers supply it. Provider failures return sanitized errors. Authentication and body bounds do not constitute multi-user authorization or production hardening.

Chunk text is sent to Google for uncached document embedding. Questions, retrieved passages and bounded history are sent for their respective embedding/rewrite/generation operations. Private PDFs, credentials and saved evaluation artifacts are excluded from the repository.

## Optional embedding cache

The cache is disabled by default and enabled with RAG_EMBEDDING_CACHE_ENABLED=1. It persists document vectors only; extraction and FAISS construction still run. Questions, rewrites, answers and indexes are not cached.

A cache key binds the ordered collection's content/provenance, extraction policy, model/task/dimension, normalization, implementation identity, relevant package versions and operator revision. Records use integrity validation, non-pickle loading and atomic replacement. Invalid records trigger recomputation; storage failures can skip persistence while using computed vectors.

A cache hit can survive a filename change because labels are supplied by fresh extraction. Collection reorder, content/configuration changes and implementation changes can invalidate reuse. Deleting a collection or clearing chat does not remove cache records. Vectors are private derived document data. An operator revision change is needed when a provider embedding change is not reflected in its model ID.

### Local observation

One two-PDF benchmark used one cold run and three sequential warm runs:

| Observation | Result |
|---|---:|
| Cold document-embedding requests | 4 |
| Warm document-embedding requests | 0, 0, 0 |
| Cold processing | 4.2626 seconds |
| Warm median processing | 0.8397 seconds |

Exact vectors, provenance and fixed self-vector probe rankings matched. These are not natural-language retrieval queries. The approximately 5.08× cold/warm-median ratio is a single local document-processing observation, not an answer-latency, dollar-savings or general performance claim. Extraction/font warnings in the run do not establish extraction completeness.

## Citation reference diagnostics

The API returns citation_check; the UI displays the same kind of diagnostic alongside the unchanged answer. The parser checks supported parenthetical filename/PDF-page formats against supplied retrieval headers, including page lists/ranges and semicolon-separated references.

- matched: parsed references match retrieved headers.
- needs_review: a possible citation has a missing filename, unknown source/page, invalid range or another parsing/limit issue.
- not_checked: no parseable reference was assessed.

Semantic support and claim coverage remain explicitly unchecked in every status. A wrong factual statement with a valid source/page can still receive matched. A brief missing-information response without citations is not automatically failed. The bounded parser is not exhaustive, adds no Google requests, and neither repairs answers nor assigns human-review scores.

## Reproducible development evaluation

The frozen corpus contains 74 chunks from familiar YOLO and Transformer papers. Fifteen approved cases include 12 answerable and three missing-information questions, with two completed follow-up histories. Queries search both documents together. Gold answers and evidence labels are excluded from model inputs.

Source, corpus, prompt, response and review bindings detect incompatible changes. Frozen evidence labels remain fixed even when equivalent answering content is retrieved from another chunk.

### Original retrieval

Scored on the 12 answerable cases at k=6:

| Recall@6 | MRR@6 | nDCG@6 |
|---:|---:|---:|
| 0.8125 | 0.6972 | 0.6836 |

### Human answer review

| Criterion | Original | Citation instruction candidate |
|---|---:|---:|
| Correctness | 15/15 | 15/15 |
| Completeness | 13/15 | 12/15 |
| Groundedness | 15/15 | 15/15 |
| Citation support | 11/15 | 12/15 |
| Appropriate abstention | 15/15 | 15/15 |
| All five | 9/15 | 10/15 |

The original baseline appropriately abstained on all three missing-information cases. Correctness assesses stated facts separately from completeness and citation support.

The candidate retained the same saved six passages/order and generation settings, adding citation instructions. It was not promoted because completeness regressed. A separate top-30-to-six cross-encoder reranker also regressed retrieval: Recall@6 0.5625, MRR@6 0.5972 and nDCG@6 0.4992. The current core retains original retrieval and answer instructions.

These are single-reviewer development results on familiar documents. They do not establish held-out generalization, inter-rater agreement, statistical significance or production reliability. Historical evaluation scores are not automatically assigned to new service outputs.

## Validation

The expanded implementation passed 201 offline tests, including Windows/Python 3.14 PR checks and local validation. Tests use simulated provider responses and include real PDF extraction and FAISS behavior. They cover provenance, saved evaluation bindings, prompt parity, history, UI state, API boundaries, collection isolation, cache integrity and citation diagnostics.

Separate live smoke checks exercised health, unauthorized access, two-document upload, direct answers, follow-up rewriting, missing-information handling and deletion (204 followed by 404). They demonstrate functionality, not a new aggregate answer-quality benchmark.

## Remaining work

A next evaluation should use unfamiliar PDFs and independently reviewed questions, frozen before tuning. Compare changes under fixed input/context budgets and preserve regression checks. Potential future work includes OCR, token-aware chunking and measured hybrid retrieval. Shared deployment would additionally require user-specific authorization, durable collection management, operational monitoring and appropriate transport/security controls.

See [README](../README.md), [cache documentation](EMBEDDING_CACHE.md) and [citation checks](CITATION_REFERENCE_CHECKS.md) for operational details.
