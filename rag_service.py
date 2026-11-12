"""Bounded, process-local collection lifecycle around the reusable RAG core."""
from dataclasses import dataclass
from secrets import token_urlsafe
import threading
import rag_core


class BusyError(Exception):
    pass


class CapacityError(Exception):
    pass


class CollectionNotFound(Exception):
    pass


@dataclass(frozen=True)
class PDFUpload:
    name: str
    data: bytes

    def getvalue(self):
        return self.data


class RagService:
    """One token holder, one process; no implicit cross-request chat memory."""
    def __init__(self, core=rag_core, max_collections=4):
        self.core = core
        self.max_collections = max_collections
        self._collections = {}
        self._lock = threading.Lock()

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            raise BusyError('Another collection operation is in progress.')

    def create(self, uploads):
        self._enter()
        try:
            if len(self._collections) >= self.max_collections:
                raise CapacityError('Delete a collection before creating another.')
            processing = {}
            index, chunks = self.core.build_index(uploads, stats=processing)
            cid = token_urlsafe(24)
            # Publish only after all extraction, embeddings and indexing succeed.
            self._collections[cid] = (index, chunks)
            represented = {c['upload_position'] for c in chunks}
            return {'collection_id': cid, 'chunk_count': len(chunks), 'processing': processing,
                    'documents': [{'filename': f.name, 'upload_position': n,
                                   'indexed': n in represented} for n, f in enumerate(uploads, 1)]}
        finally:
            self._lock.release()

    def query(self, cid, question, messages):
        self._enter()
        try:
            if cid not in self._collections:
                raise CollectionNotFound(cid)
            index, chunks = self._collections[cid]
            answer, passages, rewritten = self.core.answer_question(question, index, chunks, messages, k=6)
            return {'collection_id': cid, 'answer': answer, 'retrieval_question': rewritten,
                    'retrieved_passages': passages, 'chat_model': self.core.CHAT_MODEL,
                    'embedding_model': self.core.EMBEDDING_MODEL,
                    'evidence_note': 'Retrieved evidence is not proof that each passage was cited or used.'}
        finally:
            self._lock.release()

    def delete(self, cid):
        self._enter()
        try:
            if cid not in self._collections:
                raise CollectionNotFound(cid)
            del self._collections[cid]
        finally:
            self._lock.release()
