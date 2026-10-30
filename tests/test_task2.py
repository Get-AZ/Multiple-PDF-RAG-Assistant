"""Offline integration tests: real PDF extraction + FAISS, simulated Gemini."""
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import faiss
import numpy as np
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('multi_rag_test_app', ROOT / 'app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def pdf(name, pages):
    stream = io.BytesIO()
    writer = canvas.Canvas(stream)
    for text in pages:
        writer.drawString(45, 760, text)
        writer.showPage()
    writer.save()
    return SimpleNamespace(name=name, getvalue=lambda: stream.getvalue())


def documents():
    return [pdf('research.pdf', ['Research abstract: attention relates sequence positions.']),
            pdf('manual.pdf', ['Operating manual: reboot the router after 30 seconds.']),
            pdf('report.pdf', ['Annual report: revenue was 125 million dollars.'])]


def embedding(text):
    terms = ['attention', 'router', 'revenue']
    return [1.0 if t in text.lower() else 0.05 for t in terms]


class Models:
    def __init__(self):
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(('embed', kwargs))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=embedding(t))
                                           for t in kwargs['contents']])

    def generate_content(self, **kwargs):
        self.calls.append(('generate', kwargs))
        if kwargs['contents'].startswith('Rewrite'):
            return SimpleNamespace(text='How long should I wait before rebooting the router?')
        return SimpleNamespace(text='Wait 30 seconds (manual.pdf, page 1).')


class Client:
    def __init__(self): self.models = Models()
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Pipeline(unittest.TestCase):
    def test_diverse_pdfs_and_shared_faiss_index(self):
        client = Client()
        with patch.object(app, 'make_client', return_value=client), patch.object(app.st, 'progress'):
            index, chunks = app.build_index(documents())
        self.assertEqual(index.ntotal, 3)
        self.assertEqual({c['source'] for c in chunks}, {'research.pdf', 'manual.pdf', 'report.pdf'})
        for q, expected in [('attention', 'research.pdf'), ('router', 'manual.pdf'), ('revenue', 'report.pdf')]:
            with self.subTest(question=q):
                found = app.search_passages(client, q, index, chunks, k=1)
                self.assertEqual(found[0]['source'], expected)
                self.assertEqual(found[0]['page'], 1)

    def test_follow_up_retrieval_and_generation_receive_history(self):
        chunks = app.extract_chunks(documents())
        client = Client()
        vectors = app.embed_texts(client, [c['text'] for c in chunks], 'RETRIEVAL_DOCUMENT')
        index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
        history = [{'role':'user','content':'Tell me about the router.'},
                   {'role':'assistant','content':'The manual discusses rebooting.'}]
        with patch.object(app, 'make_client', return_value=client):
            answer, passages, query = app.answer_question('How long should I wait?', index, chunks, history, k=1)
        self.assertEqual(passages[0]['source'], 'manual.pdf')
        self.assertIn('router', query)
        prompts = [k['contents'] for t,k in client.models.calls if t=='generate']
        self.assertEqual(len(prompts), 2)
        self.assertIn('Tell me about the router.', prompts[-1])
        self.assertIn('manual.pdf, page 1', prompts[-1])
        self.assertIn('30 seconds', answer)

    def test_overlap_page_boundaries_and_duplicate_names(self):
        chunks = app.extract_chunks([pdf('same.pdf', ['a'*1600, 'second page']), pdf('same.pdf', ['other'])])
        self.assertEqual(chunks[0]['text'][-200:], chunks[1]['text'][:200])
        self.assertEqual(chunks[2]['page'], 2)
        self.assertNotEqual(chunks[0]['source'], chunks[-1]['source'])
        self.assertNotEqual(chunks[0]['document_id'], chunks[-1]['document_id'])

    def test_empty_pdf_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No readable text'):
            app.build_index([pdf('blank.pdf', [''])])

    def test_history_is_bounded_and_excludes_unanswered_turn(self):
        messages = []
        for i in range(5): messages += [{'role':'user','content':f'q{i}'}, {'role':'assistant','content':'a'*5000}]
        messages += [{'role':'user','content':'unanswered'}]
        text = app.recent_history(messages)
        self.assertNotIn('q0', text)
        self.assertNotIn('unanswered', text)
        self.assertIn('q4', text)
        self.assertLess(len(text), 14000)

    def test_invalid_embeddings_rejected(self):
        client = Client()
        with patch.object(client.models, 'embed_content', return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[0.,0.,0.])])):
            with self.assertRaisesRegex(ValueError, 'invalid vectors'):
                app.embed_texts(client, ['text'], 'RETRIEVAL_QUERY')


class Interface(unittest.TestCase):
    def test_missing_key_disables_processing_and_chat(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY':''}):
            at = AppTest.from_file(str(ROOT/'app.py')).run()
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(at.chat_input[0].disabled)
            self.assertTrue(at.button[0].disabled)
            self.assertEqual([t.label for t in at.tabs], ['2. Chat', '3. Retrieved context'])

    def test_chat_evidence_survives_rerun_and_clear(self):
        files = documents()
        client = Client()
        with patch.dict(os.environ, {'GOOGLE_API_KEY':'offline-test-key'}), \
             patch('streamlit.file_uploader', return_value=files), \
             patch('google.genai.Client', return_value=client):
            at = AppTest.from_file(str(ROOT/'app.py')).run()
            at.button[0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertFalse(at.chat_input[0].disabled)
            at.chat_input[0].set_value('How do I reboot the router?').run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state['messages']), 2)
            self.assertEqual(len(at.session_state['messages'][1]['passages']), 3)
            at.run()
            self.assertEqual(len(at.expander), 3)
            at.chat_input[0].set_value('How long should I wait?').run()
            self.assertEqual(len(at.session_state['messages']), 4)
            self.assertIn('router', at.session_state['messages'][-1]['retrieval_question'])
            at.button[1].click().run()
            self.assertEqual(at.session_state['messages'], [])
            self.assertIsNotNone(at.session_state['pdf_data'])

    def test_failed_reprocessing_preserves_index_and_changed_upload_blocks_chat(self):
        files = documents()
        with patch.dict(os.environ, {'GOOGLE_API_KEY':'offline-test-key'}), \
             patch('streamlit.file_uploader', return_value=files) as uploader, \
             patch('google.genai.Client', return_value=Client()) as factory:
            at = AppTest.from_file(str(ROOT/'app.py')).run()
            at.button[0].click().run()
            old = at.session_state['pdf_data']['selection']
            uploader.return_value = [pdf('new.pdf', ['new text'])]
            factory.side_effect = RuntimeError('simulated quota failure')
            at.button[0].click().run()
            self.assertTrue(at.chat_input[0].disabled)
            self.assertEqual(at.session_state['pdf_data']['selection'], old)
            self.assertIn('simulated quota failure', at.error[0].value)


if __name__ == '__main__': unittest.main()
