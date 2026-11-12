"""Offline corpus export validation with generated PDF fixtures."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reportlab.pdfgen import canvas
import export_evaluation_corpus as exporter
from rag_provenance import extract_pdf_chunks
from types import SimpleNamespace


def write_pdf(path, text):
    buffer = io.BytesIO(); pdf = canvas.Canvas(buffer, invariant=1)
    pdf.drawString(45, 760, text); pdf.showPage();pdf.save()
    path.write_bytes(buffer.getvalue());return path


class CorpusExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.y = write_pdf(self.root/'yolo.pdf', 'YOLO evidence')
        self.t = write_pdf(self.root/'transformer.pdf', 'Transformer evidence')
        self.out = self.root/'output'

    def test_exact_app_chunks_and_file_hashes(self):
        m = exporter.export_corpus(self.y, self.t, self.out)
        corpus = json.loads((self.out/'corpus.json').read_text())
        uploads = [SimpleNamespace(name=p.name, getvalue=lambda p=p:p.read_bytes()) for p in [self.y,self.t]]
        self.assertEqual(corpus, [dict(c,id=c['chunk_id']) for c in extract_pdf_chunks(uploads)])
        for name, sha in m['files_sha256'].items():
            self.assertEqual(hashlib.sha256((self.out/name).read_bytes()).hexdigest(),sha)
        self.assertEqual(m['approved_cases'],0)

    def test_drafts_are_pending_and_scope_valid(self):
        m = exporter.export_corpus(self.y, self.t, self.out)
        with (self.out/'development_review.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),15)
        docs = {x['document_id'] for x in m['documents']}
        self.assertTrue(all(set(json.loads(r['document_ids'])) <= docs for r in rows))
        self.assertTrue(all(r['review_status']=='pending' and not r['answerable'] and not r['reference_answer'] for r in rows))

    def test_existing_output_refused_unchanged(self):
        self.out.mkdir();(self.out/'marker').write_text('keep')
        with self.assertRaises(ValueError):exporter.export_corpus(self.y,self.t,self.out)
        self.assertEqual(list(self.out.iterdir()),[self.out/'marker'])

    def test_duplicate_pdf_refused(self):
        duplicate = self.root/'copy.pdf';duplicate.write_bytes(self.y.read_bytes())
        with self.assertRaisesRegex(ValueError,'different PDF'):exporter.export_corpus(self.y,duplicate,self.out)
        self.assertFalse(self.out.exists())

    def test_blank_document_refused(self):
        write_pdf(self.y,'')
        with self.assertRaisesRegex(ValueError,'No readable text'):exporter.export_corpus(self.y,self.t,self.out)
        self.assertFalse(self.out.exists())

    def test_missing_source_refused(self):
        with self.assertRaises(OSError):exporter.export_corpus(self.root/'missing.pdf',self.t,self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_pdf_refused_before_outputs(self):
        self.y.write_bytes(b'not a pdf')
        with self.assertRaises(Exception):exporter.export_corpus(self.y,self.t,self.out)
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
