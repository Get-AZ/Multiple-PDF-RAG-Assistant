"""Export the real app's chunks for review. No embeddings or generation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader
from rag_provenance import extract_pdf_chunks, extraction_policy


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def review_rows(document_ids):
    yolo, transformer = document_ids
    drafts = [
        ('MP-DEV-001', 'direct', 'What FPS are reported for YOLO and Fast YOLO?', [yolo], []),
        ('MP-DEV-002', 'direct', 'How many convolutional layers do YOLO and Fast YOLO use?', [yolo], []),
        ('MP-DEV-003', 'direct', 'What does each YOLO bounding box predict?', [yolo], []),
        ('MP-DEV-004', 'direct', 'How many layers are in the Transformer encoder?', [transformer], []),
        ('MP-DEV-005', 'direct', 'What two sublayers does a Transformer encoder layer contain?', [transformer], []),
        ('MP-DEV-006', 'direct', 'What GPUs, training steps and duration were used for the base Transformer?', [transformer], []),
        ('MP-DEV-007', 'cross_document', 'Compare the task, input and output of YOLO and the Transformer.', document_ids, []),
        ('MP-DEV-008', 'cross_document', 'How do YOLO and the Transformer represent position?', document_ids, []),
        ('MP-DEV-009', 'cross_document', 'Compare the role of convolution in YOLO with the main mechanism of the Transformer.', document_ids, []),
        ('MP-DEV-010', 'multiple_passage', 'Compare YOLO and Fast YOLO in speed, PASCAL VOC 2007 mAP and convolutional depth.', [yolo], []),
        ('MP-DEV-011', 'follow_up', 'How does the fast version compare?', [yolo], [{'role':'user','content':'Compare the reported frame rates of YOLO and Fast YOLO.'}]),
        ('MP-DEV-012', 'follow_up', 'What hardware did the base model use?', [transformer], [{'role':'user','content':'Tell me about the Transformer base model.'}]),
        ('MP-DEV-013', 'missing_information', 'What was the exact electricity cost in dollars to train the base Transformer?', [transformer], []),
        ('MP-DEV-014', 'missing_information', 'What FPS did original YOLO achieve on an NVIDIA RTX 4090?', [yolo], []),
        ('MP-DEV-015', 'missing_information', 'What English-to-Amharic BLEU score does the Transformer paper report?', [transformer], []),
    ]
    rows = []
    for cid, category, question, scope, history in drafts:
        rows.append(dict(id=cid, split='dev', category=category, question=question,
                         document_ids=json.dumps(scope), history_scenario=json.dumps(history),
                         relevant_ids='[]', pdf_pages='', exact_evidence_quote='',
                         reference_answer='', answerable='', review_status='pending',
                         reviewer='', reviewed_at='', notes='Draft development case; verify against these PDF editions. Follow-ups need a reviewed complete conversation before execution.'))
    return rows


def export_corpus(yolo, transformer, outdir):
    paths = [Path(yolo), Path(transformer)]
    outdir = Path(outdir)
    if outdir.exists():
        raise ValueError('Output already exists; preserve it and choose a new --outdir.')
    sources, uploads = [], []
    for path in paths:
        raw = path.read_bytes()
        reader = PdfReader(io.BytesIO(raw))
        sha = digest(raw)
        sources.append({'filename':path.name, 'document_id':'sha256:' + sha,
                        'document_sha256':sha, 'pdf_page_count':len(reader.pages)})
        uploads.append(SimpleNamespace(name=path.name, getvalue=lambda b=raw: b))
    if sources[0]['document_id'] == sources[1]['document_id']:
        raise ValueError('Choose two different PDF byte contents, not duplicate copies.')
    chunks = extract_pdf_chunks(uploads)
    for source in sources:
        selected = [c for c in chunks if c['document_id'] == source['document_id']]
        if not selected:
            raise ValueError('No readable text for ' + source['filename'])
        source['chunk_count'] = len(selected)
        source['nonempty_page_count'] = len({c['page'] for c in selected})
    corpus = [dict(c, id=c['chunk_id']) for c in chunks]
    if len({c['id'] for c in corpus}) != len(corpus):
        raise ValueError('Duplicate chunk IDs; inspect source documents before review.')
    rows = review_rows([s['document_id'] for s in sources])
    files = {'corpus.json':json_text(corpus)}
    preview = ['# Development corpus evidence browser', '',
               'PDF page numbers are one-based. Match text against the actual PDF before approval.', '']
    for c in corpus:
        preview += ['## ' + c['id'], f"Source: {c['source']} | PDF page {c['page']}",
                    f"Characters {c['start_char']}:{c['end_char']} in stripped extracted page text", '', c['text'], '']
    files['passages.md'] = '\n'.join(preview)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    files['development_review.csv'] = buffer.getvalue()
    policy = extraction_policy()
    manifest = {'schema_version':1, 'scope':'Actual multi-PDF app extraction only; development, not held-out.',
                'documents':sources, 'extraction_policy':policy,
                'source_files':{name:digest((Path(__file__).resolve().parent / name).read_bytes())
                                for name in ['app.py','rag_provenance.py']},
                'exporter_sha256':digest(Path(__file__).read_bytes()),
                'files_sha256':{name:digest(text.encode('utf-8')) for name,text in files.items()},
                'draft_cases':len(rows), 'approved_cases':0, 'api_requests':0,
                'status':'Pending human evidence review. No retrieval or answer quality measured.'}
    # All source parsing and payload preparation complete before creating outputs.
    outdir.mkdir(parents=True)
    for name, text in files.items():
        (outdir / name).write_bytes(text.encode('utf-8'))
    (outdir / 'manifest.json').write_bytes(json_text(manifest).encode('utf-8'))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yolo', required=True)
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--outdir', default='private_data/multi_pdf_development_v1')
    args = parser.parse_args()
    manifest = export_corpus(args.yolo, args.transformer, args.outdir)
    for d in manifest['documents']:
        print(f"{d['filename']}: {d['pdf_page_count']} pages; {d['chunk_count']} chunks; {d['document_sha256']}")
    print('EXPORTED: actual app chunks and 15 pending development review cases.')
    print('NO API REQUESTS. No embeddings, retrieval, answers or scores produced.')
    print('Output:', Path(args.outdir).resolve())


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit('STOP: ' + str(error))
