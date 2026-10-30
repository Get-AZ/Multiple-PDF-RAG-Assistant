# Multiple PDF RAG Assistant

## Problem
Answering a comparison question may require evidence spread across multiple research PDFs. A useful assistant must preserve the source of each fact and expose missing retrieval coverage.

## Solution
A Streamlit interface extracts page text, creates overlapping character chunks, requests Gemini embeddings and searches a normalized FAISS index shared by all uploads. Gemini Flash-Lite receives the question and selected passages, with instructions to cite filenames/pages and identify unsupported details.

## Engineering contribution
Designed a clear processing workflow, embedding batches, source metadata, visible evidence and selection fingerprinting. Successful indexing replaces prior state only at completion. Errors are displayed in the UI.

## What testing revealed
Earlier multi-PDF examples included both successful comparisons and false abstentions when relevant evidence was missing. This motivated inspecting the exact retrieved context before changing prompts. No release-level aggregate accuracy is established.

## Current scope
Local Python application with an in-memory index, semantic top-six search and independent questions. The portfolio page presents the project; it does not host model inference.
