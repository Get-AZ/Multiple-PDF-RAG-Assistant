# Multiple PDF RAG Assistant

## Problem and implementation

A useful PDF assistant must expose which documents support its answers and distinguish missing retrieval coverage from information absent in the source.

I built a shared Python pipeline for a Streamlit interface and a bearer-authenticated Starlette API. It extracts page-local chunks, preserves stable evidence IDs, embeds documents and questions using Gemini, retrieves six passages with normalized FAISS search, and generates answers with bounded follow-up history. Users can inspect retrieved passages and their source pages.

The API adds bounded uploads, isolated in-memory collections and deletion. An opt-in integrity-checked cache reuses document vectors. Citation diagnostics flag filename/page mismatches against retrieved headers; they do not establish that a claim is supported.

## Evidence and decisions

A frozen development set contains 15 human-reviewed cases on YOLO and Transformer papers: 12 answerable, three missing-information and two completed follow-up histories. Reference answers and evidence labels are excluded from model inputs.

| Measure | Original baseline |
|---|---:|
| Recall@6, answerable cases | 0.8125 |
| MRR@6, answerable cases | 0.6972 |
| nDCG@6, answerable cases | 0.6836 |
| Correctness | 15/15 |
| Completeness | 13/15 |
| Citation support | 11/15 |
| All five answer criteria | 9/15 |

A citation-prompt candidate improved citation support to 12/15 but reduced completeness to 12/15, so I retained the original prompt. A separate cross-encoder reranker also regressed retrieval metrics and was rejected.

One local cache benchmark reduced document-embedding requests from four on a cold run to zero on each of three warm runs. Processing took 4.2626 seconds cold versus a 0.8397-second warm median. Exact cached vectors, provenance and fixed self-vector probe rankings matched. These probes do not measure natural-language retrieval quality.

## Validation and limits

The implementation passed 201 offline tests with simulated provider responses; live API/UI smoke checks were performed separately. The development set is small, familiar and reviewed by one person. Correctness alone does not imply complete or well-cited answers.

This is a local single-user portfolio application, not a hosted production service. Indexes and chat remain in memory; optional cached vectors persist locally. Google APIs are required for model calls. The portfolio presents the project and does not host inference.

[Source and setup](https://github.com/Get-AZ/Multiple-PDF-RAG-Assistant)
