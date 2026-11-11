"""Page-local PDF chunks with content and extraction provenance.

Identity is not a filename, index row, collection ID, or access-control boundary.
Repeated identical uploads intentionally have the same content IDs.
"""
import hashlib
import io
import json
from collections import Counter
from importlib.metadata import version

from pypdf import PdfReader

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extraction_policy():
    return {
        "identity_schema": 1,
        "extractor": "pypdf.page.extract_text.strip.v1",
        "pypdf_version": version("pypdf"),
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "page_local": True,
    }


def extract_pdf_chunks(files):
    """Preserve original text windows, labels and order; add stable IDs."""
    files = list(files)
    names = Counter(f.name for f in files)
    policy = extraction_policy()
    policy_id = fingerprint(policy)
    chunks = []
    for number, uploaded_file in enumerate(files, start=1):
        raw = uploaded_file.getvalue()
        document_sha256 = hashlib.sha256(raw).hexdigest()
        source = uploaded_file.name
        if names[source] > 1:
            source = f"{source} (document {number})"
        reader = PdfReader(io.BytesIO(raw))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                piece = text[start:start + CHUNK_SIZE]
                end = start + len(piece)
                text_sha256 = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                identity = {
                    "document_sha256": document_sha256,
                    "extraction_policy_sha256": policy_id,
                    "page": page_number,
                    "start_char": start,
                    "end_char": end,
                    "text_sha256": text_sha256,
                }
                chunks.append({
                    "text": piece,
                    "source": source,
                    "page": page_number,
                    "document_id": "sha256:" + document_sha256,
                    "document_sha256": document_sha256,
                    "chunk_id": "chunk:" + fingerprint(identity),
                    "text_sha256": text_sha256,
                    "extraction_policy_sha256": policy_id,
                    "extraction_policy": dict(policy),
                    "start_char": start,
                    "end_char": end,
                    "upload_position": number,
                })
                if start + CHUNK_SIZE >= len(text):
                    break
    return chunks
