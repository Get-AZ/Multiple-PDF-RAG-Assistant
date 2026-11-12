# Streamlit migration to the shared RAG core

## Run the current interface

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-service.txt reportlab
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Set GOOGLE_API_KEY in the local .env before processing PDFs. UTF-8 files with a BOM are supported. No service token or running HTTP server is required: this UI calls the Python core directly. Processing and chatting make Google requests; importing the UI or running offline tests does not.

## What changed

streamlit_app.py contains the UI, upload selection checks, source rendering and a progress callback adapter. It delegates index creation and answering to rag_core.py. The existing HTTP service uses that same module. Both now select gemini-3.5-flash-lite and gemini-embedding-001.

The original answer/rewrite prompts, page-local extraction, stable IDs, normalized embeddings, FAISS retrieval, top-six selection and bounded completed-turn history remain in the core. No rejected reranker or citation-instruction candidate is enabled. The UI adapter does not implement another prompt or retrieval pipeline.

UI session memory and API collection memory remain separate. This is shared implementation, not shared indexes or an HTTP-backed Streamlit client. The core's 2,000-chunk collection limit also applies to the UI; the HTTP upload/body/authentication limits remain API-boundary behavior.

Processed UI indexes carry pipeline_id=shared_core_v1. An old/unmarked session index cannot be used for questions until reprocessed. Existing success behavior is preserved: evidence survives reruns, follow-ups receive completed history, clearing chat keeps the index, failed reprocessing preserves the old index, changed uploads disable stale-index chat, and failed answers remove their unfinished user turn.

## Preserved evaluation baseline

app.py is unchanged. Its normalized UTF-8 SHA256 is:

```text
613eec2f024d32c27432fa3e610d450af2c329e53bd84acfc7530281eaf29da8
```

The existing evaluation scripts, rag_core.py, rag_provenance.py, datasets, prompts, answers, approved reviews and score bindings are unchanged. Historical runners continue validating against app.py. Do not change frozen manifests or hashes to make new source pass an older baseline check.

The prior 15-case development scores describe the saved evaluated runs. Sharing their original functions and model ID does not constitute a new live-model answer-quality evaluation of this UI. Provider responses may vary.

## Offline verification

Eight additional tests cover:
- Historical source digest and the shared module/model selection.
- Progress delegation and cleanup on success and failure.
- Missing-key controls with a network-call trap.
- Real PDF extraction and FAISS with simulated Google responses through Streamlit AppTest: direct questions, follow-ups, evidence reruns and clear-chat behavior.
- Failed reprocessing, changed-upload blocking and unfinished-turn cleanup.
- Refusal to reuse an index without the current pipeline marker.

The expanded 158-test suite passed during preparation. GitHub Actions runs the same unittest discovery on Windows/Python 3.14 for this PR. This adds behavior coverage; it does not measure live answer quality, latency or production scalability.

## Optional live UI smoke check

After the offline suite and PR check pass, launch streamlit_app.py and open the local URL printed by Streamlit. Upload the two public research PDFs and process them. This makes document-embedding requests.

Ask the FPS question, follow up about convolutional depth, then ask for the missing English-to-Amharic BLEU score. Inspect the rewritten question, filenames, PDF pages and evidence. A direct question makes query-embedding and generation requests; a follow-up adds a rewrite request.

Check evidence after a rerun and clear chat. Confirm that changing the upload selection disables chat until processing succeeds. No such live UI run was made during package preparation, and the earlier API smoke checks are recorded separately.
