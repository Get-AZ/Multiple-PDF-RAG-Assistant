"""Reference-format/context checks are separate from semantic support and scores."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from citation_validation import check_citations
import rag_core as core
from rag_service import RagService
from test_rag_core import FakeClient, pdf


class CitationValidationTests(unittest.TestCase):
    def context(self):
        return [{'source':'YOLO object-detection paper.pdf','page':4,'text':'CNN evidence'},
                {'source':'YOLO object-detection paper.pdf','page':6,'text':'FPS table'},
                {'source':'Attention Is All You Need..pdf','page':2,'text':'self-attention'},
                {'source':'Attention Is All You Need..pdf','page':6,'text':'positions'}]

    def test_exact_filename_page_matches(self):
        result=check_citations('YOLO runs at 45 FPS (YOLO object-detection paper.pdf, page 6).',self.context())
        self.assertEqual(result['status'],'matched')
        self.assertEqual(result['references'][0]['pages'],[6])
        self.assertFalse(result['semantic_support_checked']);self.assertFalse(result['claim_coverage_checked'])

    def test_markdown_filename_and_table_suffix(self):
        for name in ['*YOLO object-detection paper.pdf*','**YOLO object-detection paper.pdf**','`YOLO object-detection paper.pdf`']:
            result=check_citations(f'45 FPS ({name}, page 6, Table 1).',self.context())
            self.assertEqual(result['status'],'matched')

    def test_passage_number_is_not_filename(self):
        result=check_citations('45 FPS (Passage 2, p. 6).',self.context())
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['issues'][0]['code'],'missing_filename')

    def test_page_only_reference_needs_review(self):
        result=check_citations('45 FPS (p. 6).',self.context())
        self.assertEqual(result['status'],'needs_review')

    def test_page_not_supplied_is_flagged(self):
        result=check_citations('CNN (YOLO object-detection paper.pdf, page 3).',self.context())
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['issues'][0]['code'],'pages_not_in_context')
        self.assertEqual(result['issues'][0]['pages'],[3])

    def test_page_present_in_other_document_does_not_pass(self):
        result=check_citations('Claim (Attention Is All You Need..pdf, page 4).',self.context())
        self.assertEqual(result['references'][0]['missing_pages'],[4])

    def test_unknown_filename(self):
        result=check_citations('Claim (transformer.pdf, page 2).',self.context())
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['issues'][0]['code'],'source_not_in_context')

    def test_filename_case_is_exact(self):
        result=check_citations('Claim (attention is all you need..pdf, page 2).',self.context())
        self.assertEqual(result['issues'][0]['code'],'source_not_in_context')

    def test_all_pages_in_multi_page_citation_checked(self):
        result=check_citations('Attention (Attention Is All You Need..pdf, pages 2, 4, 6).',self.context())
        self.assertEqual(result['references'][0]['pages'],[2,4,6])
        self.assertEqual(result['references'][0]['missing_pages'],[4])

    def test_range_expansion_checks_missing_middle(self):
        result=check_citations('Attention (Attention Is All You Need..pdf, pages 2–6).',self.context())
        self.assertEqual(result['references'][0]['missing_pages'],[3,4,5])

    def test_semicolon_separated_sources(self):
        result=check_citations('Comparison (YOLO object-detection paper.pdf, page 4; Attention Is All You Need..pdf, page 2).',self.context())
        self.assertEqual(result['status'],'matched');self.assertEqual(len(result['references']),2)

    def test_nested_duplicate_name_label(self):
        result=check_citations('Claim (paper.pdf (document 2), page 1).',[{'source':'paper.pdf (document 2)','page':1}])
        self.assertEqual(result['status'],'matched')

    def test_missing_information_response_is_not_automatically_failed(self):
        result=check_citations('The supplied context does not report that result.',self.context())
        self.assertEqual(result['status'],'not_checked');self.assertEqual(result['issues'],[])

    def test_uncited_factual_answer_is_not_automatically_passed(self):
        result=check_citations('YOLO runs at 45 FPS.',self.context())
        self.assertEqual(result['status'],'not_checked')
        self.assertFalse(result['claim_coverage_checked'])

    def test_valid_reference_does_not_prove_claim_truth(self):
        result=check_citations('YOLO runs at 999 FPS (YOLO object-detection paper.pdf, page 6).',self.context())
        self.assertEqual(result['status'],'matched')
        self.assertFalse(result['semantic_support_checked'])

    def test_invalid_page_and_huge_range_are_bounded(self):
        for pages in ['0','6-2','1-999999999999']:
            result=check_citations(f'Claim (YOLO object-detection paper.pdf, pages {pages}).',self.context())
            self.assertEqual(result['issues'][0]['code'],'invalid_pages')

    def test_unparsed_pdf_reference_needs_review(self):
        result=check_citations('Claim (YOLO object-detection paper.pdf, page six).',self.context())
        self.assertEqual(result['issues'][0]['code'],'unparsed_reference')

    def test_non_citation_parentheses_ignored(self):
        result=check_citations('The model uses attention (N = 6).',self.context())
        self.assertEqual(result['status'],'not_checked')

    def test_service_adds_diagnostics_without_rewriting_answer(self):
        client=FakeClient();answer='Reboot (manual.pdf, page 99).'
        with patch.object(core,'make_client',return_value=client):
            service=RagService();cid=service.create([pdf('manual.pdf','Reboot the router after 30 seconds.')])['collection_id']
            client.models.generate_content=lambda **kwargs:SimpleNamespace(text=answer)
            result=service.query(cid,'router?',[])
        self.assertEqual(result['answer'],answer)
        self.assertEqual(result['citation_check']['issues'][0]['code'],'pages_not_in_context')

    def test_inputs_are_not_mutated(self):
        from copy import deepcopy
        context=self.context();saved=deepcopy(context);answer='CNN (Passage 1, p. 4).'
        check_citations(answer,context)
        self.assertEqual(context,saved);self.assertEqual(answer,'CNN (Passage 1, p. 4).')

    def test_http_response_retains_answer_and_reference_report(self):
        from starlette.testclient import TestClient
        from service_api import create_app
        token='offline-reference-check-'+'x'*40
        provider=FakeClient();text='Reboot (Passage 1, p. 1).'
        with patch.object(core,'make_client',return_value=provider):
            service=RagService();cid=service.create([pdf('manual.pdf','Reboot the router after 30 seconds.')])['collection_id']
            provider.models.calls.clear()
            provider.models.generate_content=lambda **kwargs:SimpleNamespace(text=text)
            with TestClient(create_app(service,token)) as client:
                response=client.post(f'/collections/{cid}/query',headers={'Authorization':'Bearer '+token},json={'question':'router?'})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['answer'],text)
            self.assertEqual(response.json()['citation_check']['issues'][0]['code'],'missing_filename')
            self.assertFalse(response.json()['citation_check']['semantic_support_checked'])
