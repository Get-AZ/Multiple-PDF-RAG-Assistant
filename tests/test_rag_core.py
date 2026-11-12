"""Real PDF/FAISS integration with simulated Google responses; no API requests."""
import ast
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from reportlab.pdfgen import canvas
import rag_core as core
from rag_service import BusyError, CapacityError, CollectionNotFound, PDFUpload, RagService


def pdf(name, text):
    stream = io.BytesIO(); writer = canvas.Canvas(stream)
    writer.drawString(30, 750, text); writer.showPage(); writer.save()
    return PDFUpload(name, stream.getvalue())


def vector(text):
    return [1.0 if word in text.lower() else 0.05 for word in ['attention', 'router', 'revenue']]


class FakeModels:
    def __init__(self): self.calls = []
    def embed_content(self, **kwargs):
        self.calls.append(('embed', kwargs))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector(t)) for t in kwargs['contents']])
    def generate_content(self, **kwargs):
        self.calls.append(('generate', kwargs))
        return SimpleNamespace(text='What does the router manual say?' if kwargs['contents'].startswith('Rewrite')
                               else 'Reboot after 30 seconds (manual.pdf, page 1).')


class FakeClient:
    def __init__(self): self.models = FakeModels()
    def __enter__(self): return self
    def __exit__(self, *args): pass


class CoreTests(unittest.TestCase):
    def documents(self):
        return [pdf('research.pdf', 'Attention relates sequence positions.'),
                pdf('manual.pdf', 'Reboot the router after 30 seconds.'),
                pdf('report.pdf', 'Revenue was 125 million dollars.')]

    def test_original_prompt_and_retrieval_functions_preserved(self):
        root = Path(__file__).resolve().parents[1]
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n, ast.FunctionDef)}
        original, extracted = functions(root/'app.py'), functions(root/'rag_core.py')
        for name in ['make_client', 'extract_chunks', 'embed_texts', 'recent_history', 'search_passages', 'answer_question']:
            self.assertEqual(original[name], extracted[name], name)
        self.assertEqual(core.CHAT_MODEL, 'gemini-3.5-flash-lite')

    def test_multiple_pdfs_real_index_and_provenance(self):
        client = FakeClient(); progress = []
        with patch.object(core, 'make_client', return_value=client):
            index, chunks = core.build_index(self.documents(), lambda done,total: progress.append((done,total)))
        self.assertEqual(index.ntotal, 3); self.assertEqual(progress, [(3,3)])
        for q, source in [('attention','research.pdf'), ('router','manual.pdf'), ('revenue','report.pdf')]:
            result = core.search_passages(client, q, index, chunks, k=1)
            self.assertEqual(result[0]['source'], source)
            self.assertTrue(result[0]['chunk_id'].startswith('chunk:'))
            self.assertEqual(result[0]['page'], 1)
        np.testing.assert_allclose(np.linalg.norm(core.embed_texts(client, ['router'], 'RETRIEVAL_QUERY'), axis=1), [1], atol=1e-6)

    def test_followup_and_generation_receive_history(self):
        client = FakeClient()
        with patch.object(core, 'make_client', return_value=client):
            index, chunks = core.build_index(self.documents())
            answer, passages, rewritten = core.answer_question('How long?', index, chunks,
                [{'role':'user','content':'Tell me about the router.'}, {'role':'assistant','content':'It can be rebooted.'}])
        self.assertEqual(rewritten, 'What does the router manual say?')
        self.assertEqual(passages[0]['source'], 'manual.pdf')
        generated = [v for k,v in client.models.calls if k == 'generate']
        self.assertEqual(len(generated), 2)
        self.assertIn('Tell me about the router.', generated[0]['contents'])
        self.assertIn('Conversation (not evidence):', generated[1]['contents'])
        self.assertIn('manual.pdf, page 1', generated[1]['contents'])
        self.assertIn('Original question:\nHow long?', generated[1]['contents'])
        self.assertEqual(answer, 'Reboot after 30 seconds (manual.pdf, page 1).')

    def test_invalid_vectors_rejected(self):
        for values in [[0,0,0], [float('nan'),1,1]]:
            client = FakeClient()
            client.models.embed_content = lambda **kw: SimpleNamespace(embeddings=[SimpleNamespace(values=values)])
            with self.assertRaises(ValueError): core.embed_texts(client, ['x'], 'RETRIEVAL_QUERY')

    def test_unreadable_pdf_and_chunk_limit_before_client(self):
        for docs, maximum in [([pdf('blank.pdf','')], 2000), (self.documents(), 1)]:
            with patch.object(core, 'make_client', side_effect=AssertionError('Network attempted')):
                with self.assertRaises(ValueError): core.build_index(docs, max_chunks=maximum)

    def test_empty_answer_rejected(self):
        client = FakeClient()
        with patch.object(core, 'make_client', return_value=client):
            index, chunks = core.build_index(self.documents())
            client.models.generate_content = lambda **kw: SimpleNamespace(text='')
            with self.assertRaises(ValueError): core.answer_question('router?', index, chunks)

    def test_service_lifecycle_and_failure_preserves_existing_collection(self):
        service = RagService(max_collections=2); client = FakeClient()
        with patch.object(core, 'make_client', return_value=client):
            created = service.create(self.documents()); cid = created['collection_id']
            with patch.object(core, 'build_index', side_effect=ValueError('failure')):
                with self.assertRaises(ValueError): service.create(self.documents())
            self.assertIn('answer', service.query(cid, 'router?', []))
            service.delete(cid)
            with self.assertRaises(CollectionNotFound): service.query(cid, 'router?', [])

    def test_capacity_and_busy_rejected_before_model_calls(self):
        service = RagService(max_collections=0)
        with patch.object(core, 'make_client', side_effect=AssertionError('Network attempted')):
            with self.assertRaises(CapacityError): service.create(self.documents())
        service._lock.acquire()
        try:
            with self.assertRaises(BusyError): service.query('x', 'x', [])
        finally: service._lock.release()

    def test_collections_are_isolated(self):
        service = RagService(); client = FakeClient()
        with patch.object(core, 'make_client', return_value=client):
            a = service.create([self.documents()[0]])['collection_id']
            b = service.create([self.documents()[1]])['collection_id']
            self.assertNotEqual(a,b)
            self.assertEqual(service.query(a, 'router?', [])['retrieved_passages'][0]['source'], 'research.pdf')
            self.assertEqual(service.query(b, 'router?', [])['retrieved_passages'][0]['source'], 'manual.pdf')
