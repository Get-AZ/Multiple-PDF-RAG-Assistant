# Release provenance

Task 2 extends the supplied multi-PDF source with bounded conversation history, standalone query rewriting, persistent per-answer evidence within the session, separated UI areas and expanded tests. The original uploaded source remains in repository history.

Current app.py SHA-256: 613a22fa8c569127d201ab614b834c8e31c623ee27c77b26b1643ee012ff77ed.

The real .env was not read or published. API calls in automated tests use simulated responses. Tests exercise actual PDF extraction, FAISS search and Streamlit UI state, but do not establish real embedding relevance or generated-answer accuracy. Earlier career Word/PDF documents describe the initial release; this README and Task 2 validation describe current behavior.
