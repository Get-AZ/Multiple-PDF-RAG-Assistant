# Detailed CV Project Entry

## Multiple PDF RAG Assistant
Getamesay Zegeye | Independent AI portfolio project

### Objective
Developed a local application that searches several uploaded PDFs and generates answers supported by retrieved passages with file and page references.

### Implementation
- Built a Streamlit interface with multi-file upload, embedding progress, processing controls, chat display, error reporting and source inspection.
- Extracted text per page and created 1,500-character windows with 200-character overlap, retaining document filenames and page metadata.
- Batched Gemini document embeddings in groups of up to 20 and used a separate retrieval-query task for question encoding.
- Normalized float32 vectors and built an exact FAISS IndexFlatIP index for semantic search across the collection.
- Selected up to six passages and integrated Gemini Flash-Lite with a grounded-answer prompt, request timeout and output limit.
- Fingerprinted selected files using SHA-256 and disabled questions when uploads no longer matched the indexed collection.
- Replaced the active index only after processing succeeded and separated chat clearing from document processing.

### Evaluation and problem solving
Investigated direct-answer, cross-document and missing-information questions using YOLO and Transformer research papers. Distinguished absent evidence in retrieved context from absence in the document. Documented quota exhaustion during cloud embedding and API timeout behavior. Prepared offline helper checks and an exact-source evaluation plan; no final aggregate answer-accuracy score is claimed.

### Scope
Local educational application. Indexes are held in session memory, and displayed chat history is not sent to the model. Planned improvements include persistent indexing, quota-aware retries and stronger retrieval coverage.

### Technical skills
Python, Streamlit, pypdf, NumPy, FAISS, Google Gen AI SDK, embeddings, information retrieval, prompt design, state management and evaluation.
