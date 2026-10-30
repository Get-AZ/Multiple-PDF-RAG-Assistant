# Technical report

## Objective
Enable source-aware question answering across several text-based PDFs in a local Streamlit app. Expose the passages behind each answer so users can inspect grounding and retrieval failures.

## Engineering decisions
Page-local character chunks preserve traceable citations while remaining easy to inspect. Overlap improves continuity but can retrieve redundant passages and split sentences. Gemini embedding task types align document and question representations. Normalized exact FAISS search is simple for small collections; index memory grows with chunk count and embedding dimension.

The upload selection is a tuple of filenames and SHA-256 content hashes. A changed selection disables question entry until successful reprocessing. The hash is a consistency check, not a persistent cache. Processing builds new data before replacing the previous session index. A partial API failure leaves the old index intact, but consumes any requests already completed.

## Generation
The prompt permits comparison across different passages, requests file/page citations, and asks for partial answers when only some requested facts are supported. Source text is framed as evidence rather than instructions. Prompt rules do not guarantee faithful synthesis or valid citations. A retrieval miss can cause a false missing-information answer even when the full PDF contains the fact.

## Development observations
Earlier tests on YOLO and Transformer papers included direct questions, cross-passage comparisons and deliberately absent information. Reported answers improved for some comparisons as retrieval and prompts changed, but task comparisons and network-depth questions still showed incomplete evidence. These historical outputs are diagnostic examples, not a measured release score. The Single PDF assistant's 12-question result does not apply to this project.

## Limitations and next work
No OCR, disk persistence, API quota backoff, local embedding fallback, per-document coverage, reranking, calibrated refusal threshold, multi-user service controls or automated citation verification is implemented. Reprocessing can exhaust embedding quota. Next experiments should fix the source/model configuration, record exact contexts and evaluate unseen questions before tuning. Consider persistent indexes, token-aware chunks, hybrid candidates and a bounded context budget after measuring failures.
