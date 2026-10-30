# Multiple PDF RAG Assistant

A Streamlit application by **Getamesay Zegeye** for asking questions across uploaded PDFs and inspecting the retrieved evidence.

## Features

- Multiple PDF uploads, extracted page by page with pypdf.
- 1,500-character chunks, 200-character overlap; chunks stay within a page.
- Gemini document/query embeddings with distinct retrieval task types.
- L2-normalized vectors and exact FAISS IndexFlatIP search over a shared document collection.
- Up to six retrieved chunks per question, with file names and page numbers.
- Gemini Flash-Lite answers, visible evidence, chat display and clear-chat control.
- SHA-256 file-selection checks prevent questions using an index for a changed selection.
- The prior index is replaced only after successful processing.

## Run locally

Use Python 3.12 as a starting environment; dependencies are unpinned.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, set GOOGLE_API_KEY, then run:

```bash
python -m streamlit run app.py
```

Upload text-based PDFs, click **Process PDFs**, then ask a question. Both embeddings and generation use Google's API and need available quota. No API key or research PDF is included.

## Implementation boundaries

The FAISS index is in session memory; it is not saved to disk. Restarting the session requires processing again. Chat history is displayed but is not passed to Gemini, so each question is answered independently. Retrieval is semantic top-6 across all PDFs; there is no hybrid retrieval, reranking, per-document coverage guarantee or relevance threshold. Scanned PDFs require OCR outside this app.

The source is the attached app.py, preserved without functional changes. Helper tests validate chunking, indexing and selection behavior without API calls; live Gemini inference has not been rerun for this release. Earlier multi-PDF answers showed both successful synthesis and missing-evidence failures. No final numerical accuracy score is claimed for this exact source.

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Setup and troubleshooting](docs/SETUP_AND_TROUBLESHOOTING.md)
- [Evaluation plan](evaluation/EVALUATION.md)
- [Short CV entry](career/CV_Short.md)
- [Detailed CV entry](career/CV_Detailed.md)
- [LinkedIn text](career/LINKEDIN.md)
- [Portfolio case study](career/PORTFOLIO_CASE_STUDY.md)
- [30 interview questions and answers](interview/30_Interview_Questions_and_Answers.md)

Run offline checks: `python -m unittest discover -s tests -v`.

Project: https://github.com/Get-AZ/Multiple-PDF-RAG-Assistant
