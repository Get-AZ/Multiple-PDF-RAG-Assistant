# v1.0.0 release candidate

Status: prepared for review; no tag or GitHub release has been published by this document.
Proposed first release: v1.0.0. Repository releases and tag refs were empty when checked on 12 November 2026.

## Release summary

A local multi-PDF RAG assistant with a shared Python core, Streamlit UI and bearer-authenticated Starlette API. Includes reproducible development evaluation, optional document-vector caching and citation reference diagnostics.

## Included capabilities

- Page-local extraction and stable evidence provenance.
- Gemini embeddings, normalized FAISS top-six retrieval and bounded follow-up history.
- Collection creation, query and deletion with upload/body limits and sanitized errors.
- Opt-in document embedding cache with integrity checks; indexes and chat remain in memory.
- Filename/page checks against supplied retrieval headers, with semantic support and claim coverage explicitly unchecked.
- Frozen development evaluation runners, saved-response validation and human-review scoring.

The preserved historical app and rejected experimental prompts/rerankers remain separate from the current shared-core runtime.

## Validation and observations

The expanded suite passed 201 offline tests. Provider responses are simulated in CI; live API/UI smoke checks are separate.

Original retrieval on 12 answerable development cases: Recall@6 0.8125, MRR@6 0.6972, nDCG@6 0.6836. The full 15-case single-reviewer evaluation yielded 15/15 correctness, 13/15 completeness, 15/15 groundedness, 11/15 citation support and 15/15 appropriate abstention; 9/15 passed all five criteria.

The citation candidate was not promoted: citation support rose to 12/15 while completeness fell to 12/15. A cross-encoder reranker also regressed retrieval metrics and was not promoted.

One local two-PDF cache benchmark used four cold document-embedding requests and zero on each of three warm runs. Cold processing: 4.2626 seconds; warm median: 0.8397 seconds. This does not establish general speedup, answer latency, dollar savings or retrieval quality.

## Limits

Local single-user operation; no public deployment, multi-user accounts or durable collection/chat storage. Google APIs are required for embedding and generation. Cached vectors are persistent local derived data when enabled. Citation reference matching does not prove factual support. Familiar development cases are not a held-out benchmark.

Private PDFs, keys, caches, saved responses and review data are excluded from the release source.

## Publish checklist

1. Merge the documentation PR after its CI check succeeds.
2. Run the full offline suite on the intended main commit and confirm the working tree contains only intended changes.
3. Verify private data and credentials remain excluded.
4. Create tag v1.0.0 on that verified main commit, then publish a GitHub release using the summary, capabilities, validation and limits above.
5. Keep raw local benchmark/review artifacts private; link public documentation for the reported results.

Do not describe this release as production-ready or the citation diagnostics as semantic verification.
