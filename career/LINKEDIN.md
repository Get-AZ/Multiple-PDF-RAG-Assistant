# LinkedIn post

I built a Multiple PDF RAG Assistant with Python, Streamlit, Gemini embeddings, FAISS and Gemini Flash-Lite.

Users can upload several PDFs, ask a question across the collection and inspect the passages behind the answer, including filenames and page numbers.

The project helped me work through practical RAG challenges: preserving source metadata, preventing stale-document queries, handling embedding quota errors and identifying when retrieval misses a fact that exists in the original PDF.

The current app uses session-memory indexing and semantic top-six retrieval. My next steps are persistent indexes and a documented evaluation of cross-document evidence coverage.

Code, technical documentation and 30 interview questions and answers:
https://github.com/Get-AZ/Multiple-PDF-RAG-Assistant

#GenerativeAI #RAG #Python #Streamlit #FAISS

## Project description
A multi-document question-answering application with page-aware chunking, Gemini retrieval embeddings, normalized FAISS semantic search and Gemini Flash-Lite generation. Includes file/page citation prompts, visible retrieved passages and file-selection consistency checks.

## Suggested profile skills
Python; Generative AI; Information Retrieval; Streamlit; Natural Language Processing.
