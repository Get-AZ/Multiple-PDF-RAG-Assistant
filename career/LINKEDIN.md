# LinkedIn draft

I extended my Multiple PDF RAG Assistant into a shared Python service with a Streamlit interface and an authenticated Starlette API.

The project now includes page-aware evidence IDs, FAISS retrieval, bounded follow-up history, optional document embedding caching and citation reference diagnostics.

The most useful lesson came from evaluation: a citation-prompt change improved citation support on my development cases but reduced completeness. I kept the original prompt. A reranking experiment also regressed retrieval results and was not promoted.

The expanded implementation passed 201 offline tests. Live smoke checks were separate from the frozen 15-case, single-reviewer development evaluation.

This remains a local single-user portfolio application. Its citation diagnostics check filenames and pages, not whether every claim is supported.

Code, setup and measured results:
https://github.com/Get-AZ/Multiple-PDF-RAG-Assistant

#RAG #Python #GenerativeAI #InformationRetrieval #SoftwareTesting

## Short project description

Local multi-PDF RAG assistant with a shared Streamlit/Starlette core, Gemini embeddings, FAISS retrieval, stable evidence provenance, reproducible development evaluation, optional document-vector caching and citation reference diagnostics.

## Suggested profile skills

Python; Information Retrieval; API Development; Automated Testing; Generative AI.

This is draft wording for manual review and posting.
