# Task 2 validation

## Requirement coverage

| Requirement | Implementation | Evidence |
|---|---|---|
| Multiple PDF input | Multi-file uploader; per-page extraction | Actual pypdf extraction on three sample types and two supplied research papers |
| Chat history | Streamlit session messages; bounded recent turns | AppTest retains two exchanges, including follow-up rewrite |
| Shared vector search | Normalized vectors in FAISS IndexFlatIP | Actual FAISS search chooses matching sample file with simulated embeddings |
| PDF attribution | File/page labels and stored passages per answer | Pipeline checks and three retained source expanders after rerun |
| Separate UI | Upload sidebar, Chat tab, Retrieved context tab | AppTest verifies tabs, controls and source display |
| Diverse documents | Sample paper/manual/report fixtures plus supplied papers | Extraction and offline pipeline checks; live-answer matrix below remains pending |

## Automated results

13 tests passed: four helper checks plus nine pipeline/UI tests. Tests use real pypdf, FAISS and Streamlit AppTest. Google embedding/generation responses are simulated with deterministic vectors and fixed output. This does not measure real semantic retrieval, hallucinations, model reference resolution or answer quality.

The sample documents are generated in memory with reportlab: a research abstract about attention, an operating manual about router rebooting and an annual report about revenue. Duplicate filenames and a blank PDF are also tested. No test fixture PDFs or private documents are published.

Source-paper extraction: supplied pdf1.pdf produced 36 chunks on 10 nonempty pages; pdf2.pdf produced 38 chunks on 15 nonempty pages. Both were loaded into one extraction collection. These are extraction counts, not answer scores.

Validated environment: Python 3.12, Streamlit 1.66.1, faiss-cpu 1.15.1 and google-genai 2.34.0. Requirements remain unpinned. Reproduce with:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## Live API acceptance matrix — not yet run for this source

Use a real research paper, operating manual and annual report you are allowed to process. Record their hashes, actual source pages, questions, contexts, answers, elapsed times and model configuration before tuning.

| Test | Question to adapt to your documents | Pass condition |
|---|---|---|
| Paper fact | What method does the paper propose? | Correct fact with paper/page citation |
| Manual instruction | What steps restart the device? | Ordered supported steps with manual/page citations |
| Report metric | What revenue is reported for the named year? | Correct value, unit, period and report/page citation |
| Follow-up | After the restart answer: How long should I wait? | Search query resolves device/reference correctly; answer cites fresh evidence |
| Cross-document | Compare the paper's method with the manual's implementation | Both sources retrieved or explicitly identify unsupported comparison |
| Missing answer | What exact electricity dollar cost was used? | Brief abstention if absent; no invented value |
| UI history | Switch tabs and inspect an earlier answer | Original answer and its original passages remain |
| Changed selection | Replace one upload, then attempt to ask | Chat disabled until successful processing |

Model/network latency, cloud quota behavior, citation correctness and unseen-document accuracy require this live run. No user credentials were used by automated checks.
