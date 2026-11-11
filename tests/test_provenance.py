"""Offline extraction/provenance tests; no API calls or model downloads."""
import hashlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reportlab.pdfgen import canvas
import rag_provenance as p


def pdf(name, pages):
    data = io.BytesIO()
    writer = canvas.Canvas(data, invariant=1)
    for text in pages:
        writer.drawString(45, 760, text)
        writer.showPage()
    writer.save()
    return upload(name, data.getvalue())


def upload(name, raw):
    return SimpleNamespace(name=name, getvalue=lambda: raw)


def ids(chunks):
    return {(c['document_id'], c['chunk_id']) for c in chunks}


class Provenance(unittest.TestCase):
    def test_repeated_extraction_is_stable(self):
        f = pdf('a.pdf', ['Attention relates positions.'])
        self.assertEqual(p.extract_pdf_chunks([f]), p.extract_pdf_chunks([f]))

    def test_rename_does_not_change_ids(self):
        f = pdf('a.pdf', ['Attention relates positions.'])
        renamed = upload('renamed.pdf', f.getvalue())
        a, b = p.extract_pdf_chunks([f]), p.extract_pdf_chunks([renamed])
        self.assertEqual(ids(a), ids(b))
        self.assertNotEqual(a[0]['source'], b[0]['source'])

    def test_upload_order_does_not_change_ids(self):
        a, b = pdf('a.pdf', ['Alpha evidence']), pdf('b.pdf', ['Beta evidence'])
        first = p.extract_pdf_chunks([a, b]); second = p.extract_pdf_chunks([b, a])
        self.assertEqual(ids(first), ids(second))
        self.assertEqual(first[0]['upload_position'], 1)
        self.assertEqual(second[-1]['upload_position'], 2)

    def test_same_filename_different_bytes_are_distinct(self):
        a, b = pdf('same.pdf', ['Alpha']), pdf('same.pdf', ['Beta'])
        chunks = p.extract_pdf_chunks([a, b])
        self.assertNotEqual(chunks[0]['document_id'], chunks[1]['document_id'])
        self.assertNotEqual(chunks[0]['source'], chunks[1]['source'])

    def test_byte_change_with_same_extracted_text_changes_ids(self):
        f = pdf('a.pdf', ['Same text'])
        changed = upload('a.pdf', f.getvalue() + b'\n% changed source bytes\n')
        a, b = p.extract_pdf_chunks([f]), p.extract_pdf_chunks([changed])
        self.assertEqual(a[0]['text'], b[0]['text'])
        self.assertNotEqual(ids(a), ids(b))

    def test_overlap_and_character_offsets_are_preserved(self):
        f = pdf('long.pdf', ['a' * 1600])
        chunks = p.extract_pdf_chunks([f])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]['text'][-200:], chunks[1]['text'][:200])
        self.assertEqual((chunks[0]['start_char'], chunks[0]['end_char']), (0, 1500))
        self.assertEqual((chunks[1]['start_char'], chunks[1]['end_char']), (1300, 1600))
        self.assertNotEqual(chunks[0]['chunk_id'], chunks[1]['chunk_id'])

    def test_identical_text_on_different_pages_has_distinct_ids(self):
        chunks = p.extract_pdf_chunks([pdf('a.pdf', ['Repeat', 'Repeat'])])
        self.assertEqual([c['page'] for c in chunks], [1, 2])
        self.assertNotEqual(chunks[0]['chunk_id'], chunks[1]['chunk_id'])

    def test_blank_page_does_not_renumber_evidence(self):
        chunks = p.extract_pdf_chunks([pdf('a.pdf', ['', 'Actual evidence'])])
        self.assertEqual(chunks[0]['page'], 2)
        self.assertEqual(p.extract_pdf_chunks([pdf('blank.pdf', [''])]), [])

    def test_policy_version_changes_chunk_not_document_identity(self):
        f = pdf('a.pdf', ['Evidence'])
        original = p.extraction_policy()
        a = p.extract_pdf_chunks([f])
        with patch.object(p, 'extraction_policy', return_value={**original, 'pypdf_version': 'different'}):
            b = p.extract_pdf_chunks([f])
        self.assertEqual(a[0]['document_id'], b[0]['document_id'])
        self.assertNotEqual(a[0]['chunk_id'], b[0]['chunk_id'])

    def test_repeated_identical_uploads_share_content_identity(self):
        f = pdf('a.pdf', ['Evidence'])
        chunks = p.extract_pdf_chunks([f, upload('copy.pdf', f.getvalue())])
        self.assertEqual(chunks[0]['chunk_id'], chunks[1]['chunk_id'])
        self.assertNotEqual(chunks[0]['upload_position'], chunks[1]['upload_position'])

    def test_hashes_identify_actual_bytes_and_text(self):
        f = pdf('a.pdf', ['Evidence'])
        c = p.extract_pdf_chunks([f])[0]
        self.assertEqual(c['document_sha256'], hashlib.sha256(f.getvalue()).hexdigest())
        self.assertEqual(c['text_sha256'], hashlib.sha256(c['text'].encode()).hexdigest())
        self.assertEqual(c['extraction_policy_sha256'], p.fingerprint(c['extraction_policy']))

    def test_json_round_trip_and_empty_input(self):
        import json
        chunks = p.extract_pdf_chunks([pdf('a.pdf', ['Evidence'])])
        self.assertEqual(json.loads(json.dumps(chunks)), chunks)
        self.assertEqual(p.extract_pdf_chunks([]), [])


if __name__ == '__main__':
    unittest.main()
