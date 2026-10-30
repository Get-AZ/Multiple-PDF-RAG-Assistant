# Setup and troubleshooting

Install requirements inside an activated virtual environment, create .env from the example, and launch with python -m streamlit run app.py. The app reads .env beside app.py. Keep the real key private; .env is excluded from this release.

- Missing API key: set GOOGLE_API_KEY and restart the app.
- 429 RESOURCE_EXHAUSTED: check the quota metric and retry delay in Google's response. A daily embedding quota is not fixed by repeatedly processing the same files. Wait for reset or adjust the account's available quota. This app does not persist completed embeddings across sessions.
- 504 or timeout: retry later and inspect request errors. A shorter answer does not guarantee a successful backend request.
- No readable text: use a text-based PDF or run OCR separately.
- Missing fact: inspect Retrieved passages first. Distinguish missing retrieved evidence from absent document content.
- Changed uploads: click Process PDFs again. The question box deliberately stays disabled until selection matches the indexed files.
- Follow-up questions: repeat the relevant names/details because prior chat messages are not sent to the model.
- Dependency mismatch: record python --version and python -m pip freeze after a working install; the supplied requirements do not pin a tested environment.
