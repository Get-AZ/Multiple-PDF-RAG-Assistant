import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_saved_reranking as r


class SavedRerankingTests(unittest.TestCase):
    def test_known_metrics(self):
        value=r.metric(['a','b'],['x','a','b'])
        self.assertEqual(value['recall'],1)
        self.assertEqual(value['mrr'],.5)
        self.assertAlmostEqual(value['ndcg'],(1/math.log2(3)+.5)/(1+1/math.log2(3)))

    def test_missing_information_not_scored(self):
        self.assertIsNone(r.metric([],['a']))

    def test_duplicate_retrieval_rejected(self):
        with self.assertRaises(ValueError):r.metric(['a'],['a','a'])

    def test_stable_score_ties(self):
        passages={'a':{'text':'one'},'b':{'text':'two'}}
        self.assertEqual(r.stable_rerank('q',['b','a'],passages,lambda pairs:[1,1]),[('a',1),('b',1)])

    def test_invalid_scores(self):
        for scores in [[float('nan')],[float('inf')],[]]:
            with self.assertRaises(ValueError):r.stable_rerank('q',['a'],{'a':{'text':'text'}},lambda pairs:scores)

    def test_unknown_and_duplicate_candidates(self):
        for ids in [['z'],['a','a']]:
            with self.assertRaises(ValueError):r.stable_rerank('q',ids,{'a':{'text':'text'}},lambda pairs:[1]*len(pairs))

    def test_only_query_and_passage_text_enter_scorer(self):
        seen=[]
        passage={'text':'public evidence','reference_answer':'SECRET','relevant_ids':['GOLD']}
        r.stable_rerank('user query',['a'],{'a':passage},lambda pairs:seen.extend(pairs) or [1])
        self.assertEqual(seen,[['user query','public evidence']])

    def test_empty_candidates(self):
        self.assertEqual(r.stable_rerank('q',[],{},lambda _:self.fail('Called scorer')),[])

    def test_aggregate_regression_detected(self):
        self.assertFalse(r.comparison({'recall':1,'mrr':1},{'recall':1,'mrr':.9})['aggregate_no_regression'])

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):r.read(p)

    def fixture(self,root):
        project=Path(root);data=project/'data';base=project/'base';data.mkdir();base.mkdir()
        for name in ['app.py','rag_provenance.py','run_app_retrieval_baseline.py']:(project/name).write_text('# test fixture\n')
        corpus=[{'id':str(i),'text':'fixture passage '+str(i),'document_id':'doc'+str(i%2)} for i in range(74)]
        cases=[{'id':f'test{i}','question':'question '+str(i),'history':[],'relevant_ids':['0'] if i<12 else [],
            'answerable':i<12,'split':'dev','review_status':'approved','document_ids':['doc0','doc1']} for i in range(15)]
        for name,value in [('corpus.json',corpus),('cases.json',cases),('source_export_manifest.json',{'source_files':{n:r.sha((project/n).read_bytes()) for n in ['app.py','rag_provenance.py']}})]:r.save(data/name,value)
        digest=r.fingerprint({'corpus':corpus,'cases':cases})
        validation={'dataset_sha256':digest}
        for name,key in [('corpus.json','corpus_file_sha256'),('cases.json','cases_file_sha256'),('source_export_manifest.json','source_export_manifest_sha256')]:validation[key]=r.sha((data/name).read_bytes())
        r.save(data/'validation.json',validation)
        binding={'dataset_sha256':digest,'k':6,'document_batch_size':20,'embedding_model':'gemini-embedding-001','chat_model_for_rewrite':'gemini-3.5-flash-lite',
            'app_sha256':r.sha((project/'app.py').read_bytes()),'provenance_sha256':r.sha((project/'rag_provenance.py').read_bytes()),'runner_sha256':r.sha((project/'run_app_retrieval_baseline.py').read_bytes())}
        r.save(base/'run_manifest.json',binding)
        def cached(name,request,value):r.save(base/(name+'.json'),{'key':r.fingerprint({'run':binding,'request':request}),'value':value,'value_sha256':r.fingerprint(value)})
        for start in range(0,74,20):
            texts=[p['text'] for p in corpus[start:start+20]]
            cached('documents-'+str(start),{'texts':texts,'model':binding['embedding_model'],'task_type':'RETRIEVAL_DOCUMENT'},[[1.,0.]]*len(texts))
        rows=[]
        for c in cases:
            row={'id':c['id'],'retrieval_question':c['question'],'retrieved_ids':[str(i) for i in range(6)],'answer_generated':False}
            cached(c['id']+'-retrieval',{'question':c['question'],'history':[]},row)
            cached(c['id']+'-query',{'model':binding['embedding_model'],'texts':[c['question']],'task_type':'RETRIEVAL_QUERY'},[[1.,0.]])
            rows.append({**row,'metrics':r.metric(c['relevant_ids'],row['retrieved_ids'])})
        r.save(base/'retrieval_report.json',{'binding':binding,'cases':rows,'metrics':r.aggregate(rows)})
        return data,base,project,digest,r.sha((project/'app.py').read_text().encode())

    def check_fixture(self,tmp,mutate=None):
        data,base,project,digest,apphash=self.fixture(tmp)
        if mutate:mutate(data,base,project)
        with patch.object(r,'DATASET',digest),patch.object(r,'APP_SOURCE',apphash):return r.preflight(data,base,project)

    def test_complete_preflight_and_no_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.check_fixture(tmp)
            self.assertEqual(len(result[0]),74)
            self.assertEqual(len(result[3]),15)
            self.assertFalse((Path(tmp)/'experiment_manifest.json').exists())

    def test_changed_query_cache_rejected(self):
        def mutate(data,base,project):
            p=base/'test0-query.json';item=r.read(p);item['value']=[[0.,1.]];r.save(p,item)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):self.check_fixture(tmp,mutate)

    def test_changed_baseline_metric_rejected(self):
        def mutate(data,base,project):
            p=base/'retrieval_report.json';item=r.read(p);item['metrics']['recall']=0;r.save(p,item)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):self.check_fixture(tmp,mutate)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):self.check_fixture(tmp,lambda d,b,p:(p/'app.py').write_text('changed'))


if __name__=='__main__':unittest.main()
