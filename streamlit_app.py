"""Streamlit interface using the same RAG core as the local service.
app.py remains the source-bound historical evaluation baseline.
"""
import hashlib
import os
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import rag_core as core

PIPELINE_ID = "shared_core_v1"

def upload_selection(files):
    return tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files)


def build_index(files, stats=None):
    """Adapt the core's progress callback to Streamlit without duplicating RAG."""
    progress = st.progress(0, text="Creating document embeddings...")
    try:
        def update(completed, total):
            progress.progress(completed / total,
                              text=f"Embedded {completed} of {total} chunks")
        return core.build_index(files, progress=update, stats=stats)
    finally:
        progress.empty()


def render_sources(message):
    sources = message.get("passages", [])
    if not sources:
        st.info("No retrieved passages are stored for this answer.")
        return
    st.caption("Retrieved PDFs (retrieval does not prove that every passage was used): "
               + ", ".join(sorted({p["source"] for p in sources})))
    if message.get("retrieval_question"):
        st.caption("Search question: " + message["retrieval_question"])
    for n, passage in enumerate(sources, start=1):
        with st.expander(f"Passage {n} · {passage['source']} · page {passage['page']}"):
            st.write(passage["text"])


def main():
    load_dotenv(Path(__file__).with_name(".env"), encoding="utf-8-sig")
    st.set_page_config(page_title="MultiPDF Chat", page_icon="📚", layout="wide")
    st.title("📚 Multiple PDF RAG Assistant")
    st.caption("Upload documents, chat across them, and inspect the retrieved evidence.")
    st.caption(f"Shared RAG core · {core.CHAT_MODEL} · {core.EMBEDDING_MODEL}")
    st.session_state.setdefault("pdf_data", None)
    st.session_state.setdefault("messages", [])
    with st.sidebar:
        st.header("1. Upload and process")
        files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        selection = upload_selection(files or [])
        has_key = bool(os.getenv("GOOGLE_API_KEY"))
        if not has_key:
            st.warning("Set GOOGLE_API_KEY in .env and restart to process PDFs.")
        if st.button("Process PDFs", disabled=not files or not has_key):
            try:
                with st.spinner("Reading PDFs and creating a shared index..."):
                    processing_stats = {}
                    index, chunks = build_index(files, stats=processing_stats)
                # Keep the existing index and chat intact until processing succeeds.
                st.session_state.pdf_data = {"provenance_schema": 1, "pipeline_id": PIPELINE_ID, "processing": processing_stats, "selection": selection, "index": index,
                                             "chunks": chunks}
                st.session_state.messages = []
                st.success(f"Ready: {len(chunks)} chunks indexed.")
                represented = {c["upload_position"] for c in chunks}
                for n, f in enumerate(files, 1):
                    if n not in represented:
                        st.warning(f"No readable text in {f.name}; it was not indexed.")
            except Exception as error:
                st.error(f"Processing failed: {error}")
        if st.button("Clear chat"):
            st.session_state.messages = []
        st.caption("Indexes and chat stay in this session only. Optional document embeddings can persist in a local cache when enabled. This UI uses the shared core directly; the HTTP server is not required. Both embeddings and "
                   "answers use Google APIs. Follow-ups use an additional rewrite request.")

    data = st.session_state.pdf_data
    ready = (data is not None and data.get("provenance_schema") == 1
             and data.get("pipeline_id") == PIPELINE_ID
             and data["selection"] == selection and has_key)
    if ready:
        counts = Counter(c["source"] for c in data["chunks"])
        with st.sidebar:
            st.subheader("Indexed documents")
            for name, count in counts.items():
                st.write(f"{name}: {count} chunks")
            processing = data.get("processing", {})
            if processing:
                st.caption(f"Reused document embeddings: {processing['reused_chunks']}; "
                           f"document embedding requests: {processing['document_embedding_requests']}; "
                           f"processing: {processing['processing_seconds']:.2f}s")
    else:
        st.info("Select your PDFs and click Process PDFs before asking questions.")
    chat_tab, evidence_tab = st.tabs(["2. Chat", "3. Retrieved context"])
    with chat_tab:
        # A changed selection hides the old collection's chat until it is restored.
        if ready:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        st.caption("Retrieved PDFs: " + ", ".join(sorted(
                            {p["source"] for p in message.get("passages", [])})))
        question = st.chat_input("Ask about your PDFs or ask a follow-up", disabled=not ready, max_chars=4000)
        if question:
            history = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": question})
            try:
                with st.spinner("Retrieving evidence and generating an answer..."):
                    answer, passages, rewritten = core.answer_question(
                        question, data["index"], data["chunks"], history)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "passages": passages,
                    "retrieval_question": rewritten,
                })
                st.rerun()
            except Exception as error:
                # Failed requests are removed so retries do not produce duplicate turns.
                st.session_state.messages.pop()
                st.error(f"Answer request failed: {error}")
    with evidence_tab:
        answers = [m for m in st.session_state.messages if m["role"] == "assistant"] if ready else []
        if answers:
            chosen = st.selectbox("Choose an answer to inspect", range(len(answers)),
                                  index=len(answers) - 1,
                                  format_func=lambda n: f"Answer {n + 1}")
            st.markdown(answers[chosen]["content"])
            render_sources(answers[chosen])
        else:
            st.info("Retrieved passages will appear here after a successful answer.")


if __name__ == "__main__":
    main()
