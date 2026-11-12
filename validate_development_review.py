"""Validate human-reviewed multi-PDF development evidence and freeze cases.

Standard-library only. No model imports or API requests. Never approves rows.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re

EXPECTED_CATEGORIES = {
    **{f'MP-DEV-{i:03d}':'direct' for i in range(1, 7)},
    **{f'MP-DEV-{i:03d}':'cross_document' for i in range(7, 10)},
    'MP-DEV-010':'multiple_passage', 'MP-DEV-011':'follow_up',
    'MP-DEV-012':'follow_up',
    **{f'MP-DEV-{i:03d}':'missing_information' for i in range(13, 16)},
}
REQUIRED_FIELDS = set('id split category question document_ids history_scenario relevant_ids pdf_pages exact_evidence_quote reference_answer answerable review_status reviewer reviewed_at notes'.split())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact_hash(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode())


def dataset_fingerprint(value):
    # Match the established harness.evaluate.fingerprint serialization.
    return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(raw):
    def reject(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(raw, parse_constant=reject)


def id_list(text, field, cid, allow_empty=False):
    values = load_json(text)
    require(isinstance(values,list) and all(isinstance(v,str) and v for v in values), cid+': invalid '+field)
    require(len(values)==len(set(values)),cid+': duplicate '+field)
    require(allow_empty or bool(values),cid+': empty '+field)
    return values


def validate(folder, project):
    folder, project = Path(folder), Path(project)
    raw = {name:(folder/name).read_bytes() for name in ['manifest.json','corpus.json','passages.md','development_review.csv']}
    manifest = load_json(raw['manifest.json']); corpus = load_json(raw['corpus.json'])
    require(manifest.get('schema_version')==1,'Unknown export manifest schema.')
    for name in ['corpus.json','passages.md']:
        require(sha(raw[name])==manifest['files_sha256'][name], 'Export changed: '+name)
    # The human-reviewed CSV is EXPECTED to differ from its initial pending export.
    # Preserve its original hash and current reviewed hash rather than replacing either.
    for name in ['app.py','rag_provenance.py']:
        require(sha((project/name).read_bytes())==manifest['source_files'][name],
                'Source changed since export: '+name+'. Inspect before freezing.')
    require(sha((project/'export_evaluation_corpus.py').read_bytes())==manifest['exporter_sha256'],
            'Exporter changed since export. Inspect before freezing.')
    docs=manifest['documents']
    require(isinstance(docs,list) and len(docs)==2,'Expected two development PDFs.')
    docmap={d['document_id']:d for d in docs}
    require(len(docmap)==2,'Duplicate source document identities.')
    for d in docs:
        require(bool(re.fullmatch('[0-9a-f]{64}',d['document_sha256'])),'Invalid PDF hash.')
        require(d['document_id']=='sha256:'+d['document_sha256'],'Document ID/hash mismatch.')
        require(type(d['pdf_page_count']) is int and d['pdf_page_count']>0,'Invalid PDF page count.')
    require(isinstance(corpus,list) and bool(corpus),'Empty corpus.')
    passages={}
    policy=manifest['extraction_policy']; policy_sha=compact_hash(policy)
    for c in corpus:
        pid=c['id']; docid=c['document_id']
        require(isinstance(pid,str) and pid not in passages,'Duplicate/invalid corpus ID.')
        require(docid in docmap,'Corpus document outside manifest.')
        require(c['document_sha256']==docmap[docid]['document_sha256'],'Passage PDF hash mismatch.')
        require(type(c['page']) is int and 1<=c['page']<=docmap[docid]['pdf_page_count'],'Invalid passage page.')
        require(isinstance(c['text'],str) and c['text'].strip(),'Empty passage text.')
        require(type(c['start_char']) is int and type(c['end_char']) is int and c['start_char']>=0
                and c['end_char']-c['start_char']==len(c['text']),'Invalid character offsets.')
        require(c['text_sha256']==sha(c['text'].encode()),'Passage text hash mismatch.')
        require(c['extraction_policy']==policy and c['extraction_policy_sha256']==policy_sha,'Extraction policy mismatch.')
        identity={k:c[k] for k in ['document_sha256','extraction_policy_sha256','page','start_char','end_char','text_sha256']}
        require(pid==c['chunk_id']=='chunk:'+compact_hash(identity),'Chunk identity mismatch.')
        passages[pid]=c
    for d in docs:
        items=[c for c in corpus if c['document_id']==d['document_id']]
        require(len(items)==d['chunk_count'] and len({c['page'] for c in items})==d['nonempty_page_count'],'Document chunk/page counts differ.')
    reader=csv.DictReader(io.StringIO(raw['development_review.csv'].decode('utf-8-sig'),newline=''))
    require(reader.fieldnames is not None and REQUIRED_FIELDS<=set(reader.fieldnames),'Missing review columns.')
    rows=list(reader)
    require(len(rows)==15 and {r['id'] for r in rows}==set(EXPECTED_CATEGORIES),'Expected exactly 15 unique MP-DEV cases.')
    cases=[];questions=set();collection_ids=sorted(docmap)
    for row in rows:
        cid=row['id'];category=row['category']
        require(None not in row and all(row.get(k) is not None for k in REQUIRED_FIELDS),cid+': malformed CSV fields.')
        require(row['split']=='dev' and category==EXPECTED_CATEGORIES[cid],cid+': split/category changed.')
        require(row['review_status']=='approved',cid+': review not approved.')
        for field in ['question','reference_answer','reviewer','reviewed_at','notes','pdf_pages']:
            require(bool(row[field].strip()),cid+': missing '+field)
        date=datetime.fromisoformat(row['reviewed_at'])
        require(date.tzinfo is not None,cid+': review date must include timezone.')
        question=' '.join(row['question'].lower().split())
        require(question not in questions,cid+': duplicate question.');questions.add(question)
        require(row['answerable'] in ['true','false'],cid+': answerable must be true or false.')
        answerable=row['answerable']=='true'
        require(answerable==(category!='missing_information'),cid+': unexpected answerability; inspect reviewed question.')
        evidence_docs=id_list(row['document_ids'],'document_ids',cid)
        require(set(evidence_docs)<=set(docmap),cid+': document outside corpus.')
        refs=id_list(row['relevant_ids'],'relevant_ids',cid,allow_empty=True)
        require(bool(refs)==answerable,cid+': answerability/evidence mismatch.')
        require(all(pid in passages for pid in refs),cid+': unknown evidence ID.')
        require(all(passages[pid]['document_id'] in evidence_docs for pid in refs),cid+': evidence outside reviewed scope.')
        quote=row['exact_evidence_quote']
        if answerable:
            require(bool(quote.strip()),cid+': missing evidence quote.')
            # Multi-source quotes may have source headers; each text segment must
            # occur verbatim in at least one of the approved reference passages.
            without_headers=re.sub(r'^\[[^\n]*\]\s*\n','',quote.replace('\r\n','\n'),flags=re.M)
            segments=[x.strip() for x in re.split(r'\n\s*\n',without_headers) if x.strip()]
            require(bool(segments),cid+': no quote text.')
            require(all(any(segment in passages[pid]['text'].replace('\r\n','\n') for pid in refs) for segment in segments),cid+': quote does not match referenced source text.')
            if category=='cross_document':
                require(len({passages[pid]['document_id'] for pid in refs})==2,cid+': both PDFs need evidence.')
        else:
            require(not quote.strip(),cid+': missing-information case must not pretend to quote an absent answer.')
        history=load_json(row['history_scenario'])
        require(isinstance(history,list),cid+': history must be a list.')
        if category=='follow_up':
            require(bool(history) and len(history)%2==0 and len(history)<=6,cid+': complete bounded conversation required.')
            for i,message in enumerate(history):
                require(isinstance(message,dict) and set(message)=={'role','content'},cid+': invalid history message.')
                require(message['role']==('user' if i%2==0 else 'assistant'),cid+': history roles must alternate.')
                limit=1500 if message['role']=='user' else 3000
                require(isinstance(message['content'],str) and 0<len(message['content'])<=limit,cid+': history would be empty or truncated.')
        else:
            require(not history,cid+': unexpected history on a non-follow-up case.')
        cases.append({
            'id':cid,'split':'dev','category':category,'question':row['question'],
            # Every app query searches the full uploaded collection. Reviewed
            # evidence scope is metadata, NOT an oracle filter for retrieval.
            'document_ids':collection_ids,'evidence_document_ids':evidence_docs,
            'relevant_ids':refs,'reference_answer':row['reference_answer'],
            'answerable':answerable,'history':history,
            'review_status':row['review_status'],'reviewer':row['reviewer'],
            'reviewed_at':row['reviewed_at'],'notes':row['notes'],
            'pdf_pages':row['pdf_pages'],'exact_evidence_quote':quote,
        })
    cases.sort(key=lambda c:c['id'])
    require(sum(c['answerable'] for c in cases)==12,'Expected 12 answerable cases.')
    return corpus,cases,raw,manifest


def freeze(folder, outdir, project):
    outdir=Path(outdir)
    require(not outdir.exists(),'Approved output already exists. Preserve it and choose a new --outdir.')
    corpus,cases,raw,source=validate(folder,project)
    cases_bytes=(json.dumps(cases,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    report={
        'schema_version':1,'scope':'Human-reviewed development data; no retrieval or answer scores.',
        'case_count':len(cases),'answerable_count':12,'missing_information_count':3,
        'follow_up_count':2,'document_count':2,'passage_count':len(corpus),
        'categories':dict(Counter(c['category'] for c in cases)),
        'dataset_sha256':dataset_fingerprint({'corpus':corpus,'cases':cases}),
        'initial_pending_review_sha256':source['files_sha256']['development_review.csv'],
        'reviewed_csv_sha256':sha(raw['development_review.csv']),
        'source_export_manifest_sha256':sha(raw['manifest.json']),
        'corpus_file_sha256':sha(raw['corpus.json']),'cases_file_sha256':sha(cases_bytes),
        'validator_sha256':sha(Path(__file__).read_bytes()),
        'retrieval_scope':'Both PDFs for every case; gold evidence scope is not a retrieval filter.',
        'history_scope':'Human-approved controlled fixtures, not recorded live model responses.',
        'api_requests':0,'limitations':'Development-only; no independent held-out claims. Human review correctness is not proven by validation.'}
    outdir.mkdir(parents=True)
    outputs={'corpus.json':raw['corpus.json'],'cases.json':cases_bytes,
             'reviewed_development.csv':raw['development_review.csv'],
             'source_export_manifest.json':raw['manifest.json'],
             'validation.json':(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode()}
    for name,data in outputs.items():(outdir/name).write_bytes(data)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='private_data/multi_pdf_development_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_approved_dev_v1')
    args=parser.parse_args()
    r=freeze(args.data,args.outdir,Path(__file__).resolve().parent)
    print('VERIFIED: corpus hashes, stable IDs, review fields and evidence scope.')
    print('VERIFIED: 15 approved development cases: 12 answerable, 3 missing-information.')
    print('VERIFIED: 2 completed follow-up histories; full collection search scope.')
    print('FROZEN:',Path(args.outdir).resolve())
    print('Dataset SHA256:',r['dataset_sha256'])
    print('NO API REQUESTS. Retrieval and answer quality have not been scored.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,OSError,csv.Error) as error:
        raise SystemExit('INVALID REVIEW: '+str(error))
