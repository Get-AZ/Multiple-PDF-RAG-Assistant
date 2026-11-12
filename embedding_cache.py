"""Opt-in local document-embedding matrix cache. No pickle or raw text storage."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA = 1
MAX_RECORD_BYTES = 64 * 1024 * 1024
DEFAULT_BUDGET = 256 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def collection_identity(chunks, model, revision, dimension, implementation, sdk_version):
    """Bind ordered content identities and exact text hashes, never display names."""
    if not chunks or not isinstance(revision, str) or not revision.strip() or len(revision) > 100:
        raise ValueError('Cache requires chunks and a nonempty revision of at most 100 characters.')
    records = []
    for c in chunks:
        digest = sha(c['text'].encode('utf-8'))
        if digest != c['text_sha256']:
            raise ValueError('Chunk text no longer matches its provenance hash.')
        records.append({key: c[key] for key in ('chunk_id','document_sha256','extraction_policy_sha256',
                                               'text_sha256','page','start_char','end_char')})
    return {'schema_version': SCHEMA, 'model': model, 'task_type': 'RETRIEVAL_DOCUMENT',
            'revision': revision, 'dimension': dimension, 'dtype': 'float32', 'normalization': 'L2',
            'implementation_sha256': implementation, 'google_genai_version': sdk_version,
            'numpy_version': np.__version__, 'ordered_chunks': records}


@dataclass(frozen=True)
class CacheResult:
    vectors: np.ndarray
    status: str


class EmbeddingCache:
    def __init__(self, directory, dimension=3072, revision='v1', budget_bytes=DEFAULT_BUDGET):
        if type(dimension) is not int or dimension <= 0 or type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('Invalid cache dimension or byte budget.')
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 100:
            raise ValueError('Invalid cache revision.')
        self.directory = Path(directory).resolve()
        self.dimension = dimension
        self.revision = revision
        self.budget_bytes = budget_bytes

    def path_for(self, identity):
        if identity['dimension'] != self.dimension or identity['revision'] != self.revision:
            raise ValueError('Cache identity configuration differs.')
        return self.directory / (fingerprint(identity) + '.npz')

    def valid_vectors(self, vectors, rows):
        return (isinstance(vectors, np.ndarray) and vectors.dtype == np.dtype('float32')
                and vectors.shape == (rows, self.dimension) and np.isfinite(vectors).all()
                and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-4, atol=1e-5))

    def load(self, identity):
        path = self.path_for(identity)
        try:
            if not path.exists():
                return None, 'miss'
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
                return None, 'invalid'
            with zipfile.ZipFile(path) as archive:
                records = archive.infolist()
                if len(records) != 2 or {r.filename for r in records} != {'metadata.npy','vectors.npy'}:
                    return None, 'invalid'
                if sum(r.file_size for r in records) > MAX_RECORD_BYTES:
                    return None, 'invalid'
            with np.load(path, allow_pickle=False) as record:
                metadata = json.loads(record['metadata'].item())
                vectors = np.array(record['vectors'], copy=True, order='C')
            if metadata['identity'] != identity or not self.valid_vectors(vectors, len(identity['ordered_chunks'])):
                return None, 'invalid'
            if metadata['vectors_sha256'] != sha(vectors.tobytes(order='C')):
                return None, 'invalid'
            return vectors, 'hit'
        except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):
            return None, 'invalid'

    def save(self, identity, vectors):
        if not self.valid_vectors(vectors, len(identity['ordered_chunks'])):
            # Preserve the uncached core's dimension behavior, but never persist an
            # unexpected provider/mock dimension in this configured namespace.
            return 'not_stored_invalid_dimension_or_vectors'
        path = self.path_for(identity)
        temporary = None
        try:
            if path.is_symlink():
                return 'not_stored_symlink'
            self.directory.mkdir(parents=True, exist_ok=True)
            metadata = {'identity': identity, 'vectors_sha256': sha(vectors.tobytes(order='C'))}
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='.embedding-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)), vectors=vectors)
                stream.flush()
                os.fsync(stream.fileno())
            size = temporary.stat().st_size
            existing = sum(p.stat().st_size for p in self.directory.glob('*.npz') if p.is_file() and p != path)
            if size > MAX_RECORD_BYTES or existing + size > self.budget_bytes:
                return 'not_stored_budget'
            os.replace(temporary, path)
            return 'stored'
        except OSError:
            return 'not_stored_io_error'
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_or_compute(self, identity, produce):
        vectors, status = self.load(identity)
        if vectors is not None:
            return CacheResult(vectors, 'hit')
        # Provider errors must propagate; do not retry generation after a cache error.
        vectors = produce()
        saved = self.save(identity, vectors)
        return CacheResult(vectors, status + ':' + saved)


def configured_cache():
    """Default off. Configuration never creates files or calls Google."""
    enabled = os.getenv('RAG_EMBEDDING_CACHE_ENABLED', '').strip()
    if enabled not in ('', '0', '1'):
        raise ValueError('RAG_EMBEDDING_CACHE_ENABLED must be 0 or 1.')
    if enabled != '1':
        return None
    default = Path(__file__).resolve().parent / 'private_data' / 'embedding_cache_v1'
    return EmbeddingCache(os.getenv('RAG_EMBEDDING_CACHE_DIR') or default,
                          revision=os.getenv('RAG_EMBEDDING_CACHE_REVISION', 'v1'))
