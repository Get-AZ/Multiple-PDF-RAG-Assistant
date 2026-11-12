"""Synthetic validation fixtures, not human-reviewed research-paper cases."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_development_review as v


class ReviewValidation(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.project=Path(temp.name);self.data=self.project/'export';self.data.mkdir()
        self.out=self.project/'approved'
        for name in ['app.py','rag_provenance.py','export_evaluation_corpus.py']:
            (self.project/name).write_text('# synthetic fixture source\n')
        self.policy={'identity_schema':1,'extractor':'fixture','pypdf_version':'fixture'}
        docs=[];self.corpus=[]
        for n in range(2):
            raw=f'fixture document {n}'.encode();sha=v.sha(raw)
            docid='sha256:'+sha;text=f'Synthetic evidence from document {n}.'
            c={'text':text,'source':f'{n}.pdf','document_id':docid,'document_sha256':sha,
               'page':1,'start_char':0,'end_char':len(text),'text_sha256':v.sha(text.encode()),
               'extraction_policy':self.policy,'extraction_policy_sha256':v.compact_hash(self.policy),
               'upload_position':n+1}
            identity={k:c[k] for k in ['document_sha256','extraction_policy_sha256','page','start_char','end_char','text_sha256']}
            c['id']=c['chunk_id']='chunk:'+v.compact_hash(identity);self.corpus.append(c)
            docs.append({'filename':c['source'],'document_id':docid,'document_sha256':sha,'pdf_page_count':1,'chunk_count':1,'nonempty_page_count':1})
        self.rows=[]
        for cid,category in v.EXPECTED_CATEGORIES.items():
            answerable=category!='missing_information';refs=self.corpus if category=='cross_document' else self.corpus[:1]
            self.rows.append({'id':cid,'split':'dev','category':category,'question':'Fixture question '+cid,
                'document_ids':json.dumps([c['document_id'] for c in refs]),
                'history_scenario':json.dumps([{'role':'user','content':'Fixture preceding question'},{'role':'assistant','content':'Fixture preceding answer'}] if category=='follow_up' else []),
                'relevant_ids':json.dumps([c['id'] for c in refs] if answerable else []),
                'pdf_pages':'Fixture page 1','exact_evidence_quote':'\n\n'.join('[Fixture source]\n'+c['text'] for c in refs) if answerable else '',
                'reference_answer':'Synthetic fixture reference','answerable':'true' if answerable else 'false',
                'review_status':'approved','reviewer':'UNIT TEST FIXTURE ONLY','reviewed_at':'2026-11-11T12:00:00+00:00','notes':'Synthetic test, not a real human review.'})
        (self.data/'corpus.json').write_text(json.dumps(self.corpus))
        (self.data/'passages.md').write_text('Synthetic evidence browser')
        self.write_rows()
        m={'schema_version':1,'documents':docs,'extraction_policy':self.policy,
           'files_sha256':{name:v.sha((self.data/name).read_bytes()) for name in ['corpus.json','passages.md','development_review.csv']},
           'source_files':{name:v.sha((self.project/name).read_bytes()) for name in ['app.py','rag_provenance.py']},
           'exporter_sha256':v.sha((self.project/'export_evaluation_corpus.py').read_bytes())}
        (self.data/'manifest.json').write_text(json.dumps(m))

    def write_rows(self):
        with (self.data/'development_review.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)

    def rejected(self):
        self.write_rows()
        with self.assertRaises(ValueError):v.freeze(self.data,self.out,self.project)
        self.assertFalse(self.out.exists())

    def test_valid_freeze_keeps_full_collection_and_originals(self):
        original=(self.data/'corpus.json').read_bytes()
        # Human CSV changes after the initial pending export are allowed and bound.
        self.rows[0]['notes']='Updated synthetic rationale';self.write_rows()
        r=v.freeze(self.data,self.out,self.project)
        self.assertEqual((r['case_count'],r['answerable_count'],r['missing_information_count'],r['follow_up_count']),(15,12,3,2))
        self.assertEqual((self.out/'corpus.json').read_bytes(),original)
        cases=json.loads((self.out/'cases.json').read_bytes())
        self.assertTrue(all(len(c['document_ids'])==2 for c in cases))
        self.assertEqual(len(cases[0]['evidence_document_ids']),1)
        self.assertNotEqual(r['initial_pending_review_sha256'],r['reviewed_csv_sha256'])
        self.assertEqual(r['dataset_sha256'],v.dataset_fingerprint({'corpus':self.corpus,'cases':cases}))

    def test_pending_review_rejected(self):
        self.rows[0]['review_status']='pending';self.rejected()

    def test_missing_rationale_rejected(self):
        self.rows[0]['notes']='';self.rejected()

    def test_unknown_reference_rejected(self):
        self.rows[0]['relevant_ids']='["chunk:unknown"]';self.rejected()

    def test_quote_not_in_source_rejected(self):
        self.rows[0]['exact_evidence_quote']='Invented evidence';self.rejected()

    def test_evidence_outside_scope_rejected(self):
        self.rows[0]['document_ids']=json.dumps([self.corpus[1]['document_id']]);self.rejected()

    def test_cross_document_needs_both_sources(self):
        self.rows[6]['relevant_ids']=json.dumps([self.corpus[0]['id']])
        self.rows[6]['exact_evidence_quote']=self.corpus[0]['text'];self.rejected()

    def test_duplicate_case_rejected(self):
        self.rows[1]['id']=self.rows[0]['id'];self.rejected()

    def test_follow_up_needs_history(self):
        self.rows[10]['history_scenario']='[]';self.rejected()

    def test_follow_up_needs_completed_turn(self):
        self.rows[10]['history_scenario']=json.dumps([{'role':'user','content':'Unanswered'}]);self.rejected()

    def test_missing_information_must_have_no_gold(self):
        self.rows[12]['relevant_ids']=json.dumps([self.corpus[0]['id']]);self.rejected()

    def test_changed_corpus_hash_rejected(self):
        (self.data/'corpus.json').write_text('[]')
        self.rejected()

    def test_changed_app_source_rejected(self):
        (self.project/'app.py').write_text('# changed')
        self.rejected()

    def test_existing_output_refused_without_overwrite(self):
        self.out.mkdir();marker=self.out/'keep';marker.write_text('keep')
        with self.assertRaises(ValueError):v.freeze(self.data,self.out,self.project)
        self.assertEqual(list(self.out.iterdir()),[marker])

    def test_invalid_boolean_rejected(self):
        self.rows[0]['answerable']='yes';self.rejected()

    def test_non_finite_json_rejected(self):
        self.rows[0]['document_ids']='[NaN]';self.rejected()


if __name__=='__main__':unittest.main()
