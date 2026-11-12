"""Shared-core Streamlit integration; real PDFs/FAISS, simulated provider only."""
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import rag_core as core
from test_task2 import Client, documents, pdf

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'streamlit_app.py'
spec = importlib.util.spec_from_file_location('shared_ui_under_test', ENTRY)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class SharedStreamlitTests(unittest.TestCase):
    def test_historical_source_and_model_selection(self):
        digest = hashlib.sha256((ROOT/'app.py').read_text(encoding='utf-8-sig').encode()).hexdigest()
        self.assertEqual(digest, '613eec2f024d32c27432fa3e610d450af2c329e53bd84acfc7530281eaf29da8')
        self.assertIs(ui.core, core)
        self.assertEqual(core.CHAT_MODEL, 'gemini-3.5-flash-lite')

    def test_progress_adapter_delegates_and_cleans_up(self):
        with patch.object(core,'build_index', return_value=('index',['chunk'])) as build, patch.object(ui.st,'progress') as progress:
            self.assertEqual(ui.build_index(['file']), ('index',['chunk']))
            build.assert_called_once()
            callback = build.call_args.kwargs['progress']
            callback(1,2)
            progress.return_value.progress.assert_called_once_with(0.5,text='Embedded 1 of 2 chunks')
            progress.return_value.empty.assert_called_once()

    def test_progress_cleaned_after_failure(self):
        with patch.object(core,'build_index', side_effect=ValueError('simulated failure')), patch.object(ui.st,'progress') as progress:
            with self.assertRaisesRegex(ValueError,'simulated failure'): ui.build_index([])
            progress.return_value.empty.assert_called_once()

    def test_missing_key_disables_processing_and_chat(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY':''}), patch.object(core,'make_client',side_effect=AssertionError('Network attempted')):
            at=AppTest.from_file(str(ENTRY)).run()
            self.assertEqual(len(at.exception),0)
            self.assertTrue(at.button[0].disabled); self.assertTrue(at.chat_input[0].disabled)

    def test_shared_pipeline_chat_followup_evidence_and_clear(self):
        files=documents(); client=Client()
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}), patch('streamlit.file_uploader',return_value=files), \
             patch.object(core,'make_client',return_value=client), patch.object(core,'build_index',wraps=core.build_index) as build, \
             patch.object(core,'answer_question',wraps=core.answer_question) as answer:
            at=AppTest.from_file(str(ENTRY)).run()
            at.button[0].click().run()
            self.assertEqual(len(at.exception),0); build.assert_called_once()
            self.assertEqual(at.session_state['pdf_data']['pipeline_id'],'shared_core_v1')
            at.chat_input[0].set_value('How do I reboot the router?').run()
            self.assertEqual(len(at.exception),0); self.assertEqual(answer.call_count,1)
            saved=at.session_state['messages'][1]
            self.assertEqual(len(saved['passages']),3)
            self.assertTrue(all(p['chunk_id'].startswith('chunk:') for p in saved['passages']))
            at.run(); self.assertEqual(len(at.expander),3)
            at.chat_input[0].set_value('How long should I wait?').run()
            self.assertEqual(len(at.exception),0); self.assertEqual(answer.call_count,2)
            self.assertEqual(len(at.session_state['messages']),4)
            self.assertIn('router',at.session_state['messages'][-1]['retrieval_question'])
            generated=[kw for kind,kw in client.models.calls if kind=='generate']
            self.assertEqual(len(generated),3)
            self.assertTrue(all(kw['model']==core.CHAT_MODEL for kw in generated))
            at.button[1].click().run()
            self.assertEqual(at.session_state['messages'],[])
            self.assertIsNotNone(at.session_state['pdf_data'])

    def test_failed_reprocessing_keeps_previous_index_and_blocks_changed_selection(self):
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}), patch('streamlit.file_uploader',return_value=documents()) as uploader, \
             patch.object(core,'make_client',return_value=Client()) as factory:
            at=AppTest.from_file(str(ENTRY)).run(); at.button[0].click().run()
            old=at.session_state['pdf_data']['selection']
            uploader.return_value=[pdf('new.pdf',['new text'])]
            factory.side_effect=RuntimeError('simulated failure')
            at.button[0].click().run()
            self.assertEqual(at.session_state['pdf_data']['selection'],old)
            self.assertTrue(at.chat_input[0].disabled)
            self.assertEqual(len(at.exception),0)
            self.assertIn('simulated failure',at.error[0].value)

    def test_failed_answer_removes_unfinished_turn(self):
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}), patch('streamlit.file_uploader',return_value=documents()), \
             patch.object(core,'make_client',return_value=Client()), patch.object(core,'answer_question',side_effect=RuntimeError('simulated failure')):
            at=AppTest.from_file(str(ENTRY)).run(); at.button[0].click().run()
            at.chat_input[0].set_value('router?').run()
            self.assertEqual(at.session_state['messages'],[])
            self.assertIsNotNone(at.session_state['pdf_data'])
            self.assertEqual(len(at.exception),0)

    def test_unmarked_old_index_is_not_used(self):
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}), patch('streamlit.file_uploader',return_value=documents()), \
             patch.object(core,'make_client',return_value=Client()):
            at=AppTest.from_file(str(ENTRY)).run(); at.button[0].click().run()
            data=dict(at.session_state['pdf_data']); data.pop('pipeline_id')
            at.session_state['pdf_data']=data
            at.run()
            self.assertTrue(at.chat_input[0].disabled)
            self.assertEqual(len(at.exception),0)

    def test_missing_filename_warning_with_unchanged_answer(self):
        from types import SimpleNamespace
        client=Client();answer='Wait 30 seconds (Passage 1, p. 1).'
        client.models.generate_content=lambda **kwargs:SimpleNamespace(text=answer)
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}),patch('streamlit.file_uploader',return_value=documents()),patch.object(core,'make_client',return_value=client):
            at=AppTest.from_file(str(ENTRY)).run();at.button[0].click().run()
            at.chat_input[0].set_value('router?').run()
            self.assertEqual(len(at.exception),0)
            message=at.session_state['messages'][-1]
            self.assertEqual(message['content'],answer)
            self.assertEqual(message['citation_check']['status'],'needs_review')
            self.assertTrue(any('Citation references need review' in warning.value for warning in at.warning))

    def test_supported_reference_shows_scope_disclaimer(self):
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}),patch('streamlit.file_uploader',return_value=documents()),patch.object(core,'make_client',return_value=Client()):
            at=AppTest.from_file(str(ENTRY)).run();at.button[0].click().run()
            at.chat_input[0].set_value('router?').run()
            self.assertEqual(at.session_state['messages'][-1]['citation_check']['status'],'matched')
            self.assertTrue(any('Claim support and citation coverage still need review' in caption.value for caption in at.caption))

    def test_uncited_absence_response_is_not_automatic_failure(self):
        from types import SimpleNamespace
        client=Client();answer='The supplied passages do not report this detail.'
        client.models.generate_content=lambda **kwargs:SimpleNamespace(text=answer)
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}),patch('streamlit.file_uploader',return_value=documents()),patch.object(core,'make_client',return_value=client):
            at=AppTest.from_file(str(ENTRY)).run();at.button[0].click().run()
            at.chat_input[0].set_value('unknown result?').run()
            self.assertEqual(at.session_state['messages'][-1]['content'],answer)
            self.assertEqual(at.session_state['messages'][-1]['citation_check']['status'],'not_checked')
            self.assertTrue(any('brief missing-information response' in item.value for item in at.info))

    def test_previous_messages_are_checked_on_rerender_without_new_provider_calls(self):
        client=Client()
        with patch.dict(os.environ,{'GOOGLE_API_KEY':'offline-test-key'}),patch('streamlit.file_uploader',return_value=documents()),patch.object(core,'make_client',return_value=client):
            at=AppTest.from_file(str(ENTRY)).run();at.button[0].click().run()
            at.chat_input[0].set_value('router?').run()
            messages=list(at.session_state['messages']);messages[-1]=dict(messages[-1]);messages[-1].pop('citation_check')
            at.session_state['messages']=messages;before=len(client.models.calls)
            at.run()
            self.assertEqual(len(at.exception),0);self.assertEqual(len(client.models.calls),before)
            self.assertTrue(any('Filename/page references match retrieved headers' in caption.value for caption in at.caption))


if __name__ == '__main__': unittest.main()
