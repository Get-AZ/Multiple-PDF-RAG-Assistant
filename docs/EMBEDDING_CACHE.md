# Opt-in document embedding cache and measurement

The shared core can reuse a saved matrix of document embeddings when the same ordered collection is processed again. It is off by default. The Streamlit UI and HTTP service both use this path through rag_core.build_index.

This caches document vectors only. It does not cache query embeddings, follow-up rewrites, generated answers, chat history or FAISS indexes. PDF extraction and FAISS index construction still run. No reranker or new citation instructions are enabled.

## Configuration

In the PowerShell terminal that will launch Streamlit or the API:

```powershell
$env:RAG_EMBEDDING_CACHE_ENABLED = '1'
$env:RAG_EMBEDDING_CACHE_REVISION = 'v1'

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Alternatively, put these two settings in your local .env and restart the process. To opt out, set RAG_EMBEDDING_CACHE_ENABLED=0. Existing files are not deleted when disabled.

Default location: private_data/embedding_cache_v1/ under the project, now excluded by the repository .gitignore. RAG_EMBEDDING_CACHE_DIR optionally selects another directory; keep custom paths private and excluded from Git too. No files are created merely by importing the module or checking configuration.

Vectors persist after clearing chat, deleting an API collection or restarting a process. They are derived document data and should be treated as private, even though the cache does not store raw passage text or display filenames. Collection IDs are not cache isolation boundaries; this remains a single-user local service.

To remove the default cache, stop processes using it, confirm the directory is the cache folder, then delete only private_data/embedding_cache_v1/. Do not delete the rest of private_data, which can contain frozen evaluations and approved reviews.

## What counts as the same collection

The key binds:
- Ordered chunk IDs, exact text hashes and document byte hashes.
- Extraction-policy hashes, PDF page and character offsets.
- Embedding model ID, RETRIEVAL_DOCUMENT task, expected dimension (3072), float32 dtype and L2 normalization.
- Operator revision, normalized shared-core source hash, Google SDK version and NumPy version.

Renaming files can reuse vectors because display names are not evidence identities. Freshly extracted chunks always supply current filenames and metadata. Changing PDF bytes, chunking/extraction policy, collection order, model, implementation, dependency namespace or revision produces a miss. This first version uses collection-matrix granularity: adding, removing or reordering PDFs recomputes the whole collection rather than partially reusing individual documents.

A provider can update embedding weights without changing its model ID. The cache cannot discover that remotely without an external version signal. Change RAG_EMBEDDING_CACHE_REVISION or remove old records when the provider embedding version changes or you intentionally want fresh vectors. Matching dimensions alone do not prove that query and document embeddings belong to the same model space.

## Record integrity and failure behavior

Records are NumPy NPZ files with a JSON identity and matrix checksum. Loading disables pickle and validates the metadata, shape, dtype, finite values, L2 norms and checksum before reuse. Oversized/invalid records are ignored and recomputed. Writes use a temporary file and atomic replacement; no saved Python objects are deserialized.

A normal full hit requires no document-embedding client/request. A miss requests document embeddings in batches of 20, using the original embed_texts function. Storage failures or the cache budget do not discard successfully computed vectors: processing continues uncached. Provider failures propagate and are not silently retried by the cache.

The default write budget is 256 MiB, with a 64 MiB per-record bound. When the budget would be exceeded, the newly computed matrix is used without saving; there is no automatic eviction. The byte-budget check is best effort across processes. Concurrent cold processing in separate processes may duplicate requests; atomic writes prevent partial records, not cross-process request deduplication. Local cache files are not an untrusted-user input interface.

## Processing observations

The UI shows the last collection-processing request count, reused chunk count and elapsed seconds. API collection creation adds a processing object containing:

```json
{
  "document_embedding_requests": 0,
  "embedded_chunks": 0,
  "cache_status": "hit",
  "chunk_count": 74,
  "reused_chunks": 74,
  "processing_seconds": 0.0
}
```

This is an illustrative warm response, not a measured timing. Actual cache_status values include disabled, hit, miss:stored, invalid:stored and miss/invalid statuses followed by a reason storage was skipped. The elapsed time covers extraction, vector loading/generation and index construction in build_index; it is not end-to-end HTTP upload or answer latency.

## Controlled cold/warm benchmark

The benchmark creates its own new cache under a new output folder. It does not reuse, clear or overwrite the app's normal cache or historical evaluation outputs. Default behavior is a dry run without extraction, cache writes or API requests.

In VS Code PowerShell, set the actual research PDF paths:

```powershell
$yolo = Join-Path $env:USERPROFILE 'Downloads\RAG Evaluation and Regression Harness Complete Data\YOLO object-detection paper.pdf'
$transformer = Join-Path $env:USERPROFILE 'Downloads\RAG Evaluation and Regression Harness Complete Data\rag-evaluation-harness\private_data\pdfs\transformer.pdf'

.\.venv\Scripts\python.exe .\benchmark_embedding_cache.py --pdf "$yolo" "$transformer"
```

After the dry run succeeds, intentionally execute the measured run:

```powershell
.\.venv\Scripts\python.exe .\benchmark_embedding_cache.py --pdf "$yolo" "$transformer" --execute
```

The cold run makes document-embedding requests. Three warm repeats should make zero document-embedding requests. For a 74-chunk collection with batches of 20, the expected cold count is four; this is an expectation until measured on your environment. There are no query embedding, rewrite or answer-generation calls in this benchmark.

The report is saved to private_data/embedding_cache_benchmark_v1/benchmark_report.json. An existing output is refused; choose a new --outdir after an earlier or failed run. The report records PDF/source hashes, package versions, counts and timings. Cached vectors, chunks/provenance and fixed self-vector probe rankings must match exactly between cold and warm runs.

The probes are document vectors used as fixed index queries for serialization/integrity checks. They are not natural-language questions, gold relevance labels or a new Recall/MRR/nDCG evaluation. Cold/warm timing includes provider and OS-cache effects, and one cold run does not establish a general speedup. Report observed seconds and counts; do not claim dollar savings or improved answer quality.

## Verification

18 new offline tests cover cold/warm request counts, rename reuse with fresh source labels, batching, uncached query embeddings, model/revision/SDK/implementation/policy invalidation, byte/order changes, text-hash mismatch, corrupt/checksum/object-array rejection, invalid vectors, write-budget/I/O fallback, provider failure propagation, opt-out, service observations and benchmark report integrity.

The expanded 176-test suite passed during preparation using simulated provider responses. No real Google requests or real-PDF timing benchmark were performed during preparation. Windows CI and the user's measured run remain separate steps. Historical app.py, evaluation helpers, saved prompts/reviews/scores and the six original core prompt/retrieval functions remain unchanged; build_index gains the opt-in reuse/instrumentation path.

Reference for NPZ loading without object deserialization: https://numpy.org/doc/stable/reference/generated/numpy.load.html

## Reported local measurement — 2026-11-12

The user ran the controlled benchmark on the YOLO and Transformer PDFs and supplied its completed terminal summary:

| Observation | Cold run | Warm runs |
|---|---:|---:|
| Document-embedding requests | 4 | 0, 0, 0 |
| Processing time | 4.2626 s | 0.8397 s median across three runs |

The observed cold/warm-median ratio was approximately **5.08x**, or **80.30% less processing time** for the warm median in this run. All three warm runs avoided document-embedding requests. This does not measure total application request savings, answer latency or dollar savings.

The runner reported identical cached vectors, extracted provenance and fixed self-vector probe rankings. It generated no answers and saved benchmark_report.json under the local private_data/embedding_cache_benchmark_v1/ directory. The raw report and vectors remain local; this public summary records the user-reported console observations.

pypdf emitted warnings about missing fontTools for CFF Type1 font encodings. The run still completed its consistency checks; these checks do not establish that PDF text extraction was complete or free of encoding errors.

This is one cold run followed by three sequential warm runs in one local environment. Provider latency, repeated extraction and OS caching affect the comparison. It is evidence of request avoidance and an observed local timing difference, not a general speedup guarantee or a new retrieval/answer-quality score. The benchmark used its own cache and did not enable caching by default in the UI or API.
