# 30 interview questions and answers

Multiple PDF RAG Assistant v1.0.0 | Getamesay Zegeye

These answers describe implemented behavior and measured development work. Adapt first-person wording to your own explanation; do not present local development observations as production results.

## 1. What did you build?

I built a local multi-PDF RAG assistant with a shared Python pipeline, a Streamlit UI and a bearer-authenticated Starlette API. Users can inspect retrieved text, filenames, pages and evidence IDs.

## 2. How is the architecture separated?

rag_core.py handles the model pipeline; rag_provenance.py handles evidence identities; rag_service.py manages collections; service_api.py validates HTTP requests. The UI calls the core directly and does not require the API server.

## 3. What does RAG mean in this project?

The system embeds a question, retrieves relevant PDF passages, then supplies those passages to Gemini for generation. Retrieval and generation are separate failure points; a fluent answer is not proof that the evidence supports it.

## 4. How do you extract and chunk PDFs?

pypdf extracts existing text page by page. I use 1,500-character windows with 200-character overlap without crossing page boundaries. Scanned documents need OCR first; overlap can still split sentences or add redundancy.

## 5. Why preserve stable evidence identities?

Document byte hashes and policy/location-bound chunk IDs let me trace saved results to exact source content. Renaming a PDF does not change its content identity, while changing its bytes or extraction policy can change the relevant identities.

## 6. Which embedding model and task types do you use?

gemini-embedding-001 through the Google Gen AI SDK. Document batches use RETRIEVAL_DOCUMENT and question embeddings use RETRIEVAL_QUERY. Document batches contain up to 20 chunks.

## 7. How do you validate embeddings?

I convert responses to contiguous float32 arrays, verify the response shape and row count, reject nonfinite or zero-norm vectors, then L2-normalize them. Invalid vectors must not silently enter the index.

## 8. Why FAISS IndexFlatIP?

For a small collection, exact inner-product search is straightforward to inspect. On unit-length vectors its ranking corresponds to cosine similarity. It is not approximate nearest-neighbor search, and its memory cost grows with the collection.

## 9. Does global top-six retrieval guarantee cross-document coverage?

No. All six passages can come from one PDF. I inspect coverage and frozen evidence ranks before changing retrieval; possible future comparisons include hybrid or per-document candidates under the same context budget.

## 10. How do follow-up questions work?

The latest three completed user/assistant pairs are bounded and sent to a separate standalone-question rewrite request. The rewritten question drives retrieval. History resolves references but is explicitly not source evidence.

## 11. Which model and request bounds generate answers?

The shared core uses gemini-3.5-flash-lite, a 60-second client timeout and one attempt. Rewrite output is capped at 256 tokens and answer output at 1,000. A model ID does not independently pin provider weights.

## 12. How do you avoid stale UI indexes?

The UI compares the selected filenames and content fingerprints with the indexed selection. Changed uploads disable questions. New processing replaces the old index only after success; clearing chat retains the index.

## 13. What endpoints does the service expose?

GET /health, POST /collections, POST /collections/{id}/query and DELETE /collections/{id}. Health makes no Google request. Creation embeds documents; queries embed questions and generate answers, with an extra rewrite for follow-ups.

## 14. How is authentication implemented?

Non-health endpoints require a bearer service token distinct from the Google key. The configured token must be at least 32 ASCII characters. Every token holder can access all collections, so this is local single-user access rather than tenant authorization.

## 15. What resource limits are enforced?

The API accepts 1–4 PDFs, up to 6 MiB each and a 12 MiB total body including multipart overhead. It allows four resident collections and 2,000 chunks per collection. Questions are capped at 4,000 characters; concurrent collection operations return 409.

## 16. How does collection and chat persistence work?

API collections and UI indexes live in memory and are separate. Restarting the server loses API collections. The API stores no chat history; clients send up to three completed pairs. Optional document-vector persistence does not persist collections or chat.

## 17. How do you handle API failures?

The boundary returns sanitized errors instead of provider details that might include private content. Examples include 401 for auth, 404 for missing collections, 409 for busy/capacity, 413 for size and 502 for other processing failures. Some calls may consume quota before failure.

## 18. What exactly is cached?

Only the complete ordered collection's document-vector matrix, when explicitly enabled. Extraction and FAISS construction still run. Query embeddings, rewrites, answers and FAISS indexes are not cached.

## 19. How do you prevent invalid cache reuse?

The cache binds content/provenance, extraction policy, model/task/dimension, normalization, implementation/package identity and an operator revision. It validates checksums and arrays, loads without pickle and writes atomically. Provider weight drift under the same ID needs an operator revision change.

## 20. What did the cache benchmark show?

In one local two-PDF run, four cold document-embedding requests became zero in each of three warm runs. Cold processing took 4.2626 seconds and the warm median was 0.8397 seconds. That observation does not establish general speedup, answer latency or retrieval quality.

## 21. What do the citation diagnostics verify?

They check supported filename/PDF-page references against supplied retrieved headers. matched means those references matched, not that the claim is true. Semantic support and claim coverage remain unchecked; the original answer is never repaired or blocked.

## 22. How do you interpret not_checked citations?

It means no parseable reference was assessed. A brief missing-information response may reasonably have no citation, but uncited factual content may still need review. The parser does not assign automatic human-quality scores.

## 23. How did you structure the evaluation?

I froze 15 reviewed development cases on familiar YOLO and Transformer papers: 12 answerable, three missing-information and two completed follow-up histories. Gold answers and evidence labels stay out of model inputs, and saved contexts/responses are bound to their sources.

## 24. What retrieval results did you measure?

On the 12 answerable development cases, original FAISS retrieval achieved Recall@6 0.8125, MRR@6 0.6972 and nDCG@6 0.6836. These measure frozen labeled evidence retrieval, not final answer quality or held-out generalization.

## 25. How do correctness and completeness differ?

An answer can state only correct facts yet omit a requested part. In baseline human review, correctness was 15/15, completeness 13/15, groundedness 15/15, citation support 11/15 and appropriate abstention 15/15. Only 9/15 passed all five criteria.

## 26. Describe an experiment you decided not to ship.

A citation-instruction candidate improved citation support from 11/15 to 12/15, but completeness fell from 13/15 to 12/15. I retained the baseline because the change introduced a regression despite improving another metric.

## 27. What happened with reranking?

A saved-query experiment reranked 30 candidates down to six with a cross-encoder. Recall@6 fell to 0.5625, MRR@6 to 0.5972 and nDCG@6 to 0.4992, so I did not promote it. Input truncation was a diagnostic concern, not a proven explanation for every failure.

## 28. How do you diagnose an apparently missing answer?

I inspect extraction and exact retrieved passages before blaming generation. The fact may exist elsewhere in the PDF. In one case equivalent answering evidence appeared outside the frozen labeled chunk; I preserved the label-based score rather than retroactively changing the test.

## 29. What do the 201 tests establish?

They check implementation behavior offline, using simulated Google responses plus real PDF extraction and FAISS operations. They cover provenance, bindings, history, state, API validation, caches and reference checks. Separate live smoke checks establish basic functionality, not an aggregate quality score.

## 30. What would you do next?

I would freeze an unfamiliar-document evaluation before more tuning and add independent review where possible. Production sharing would also need user-specific authorization, durable collection management and operational controls. The current small single-reviewer development set cannot establish production reliability.
