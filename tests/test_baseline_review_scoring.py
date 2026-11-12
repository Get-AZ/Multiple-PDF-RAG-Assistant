from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import score_baseline_reviews as s


class ReviewScoringTests(unittest.TestCase):
    def fixture(self):
        cases=[{'id':'test','category':'direct','answerable':True}]
        prompts=[{'id':'test','prompt':'source context'}]
        answer={'answer':'supported answer','finish_reason':'FinishReason.STOP'}
        binding={'model':'test'}
        answers={'binding':binding,'cases':[{'id':'test',**answer}]}
        review={'binding_sha256':s.f.fingerprint(binding),'reviews':[{
            'id':'test','answer_sha256':s.f.fingerprint(answer),'prompt_sha256':s.f.fingerprint(prompts[0]),
            'scores':{k:True for k in s.CRITERIA},'review_status':'approved','reviewer':'TEST FIXTURE ONLY',
            'reviewed_at':'2026-11-12T00:00:00+00:00','rationale':'Synthetic testing only.'}]}
        return [cases,prompts,answers,binding,review]

    def test_complete_review_scores(self):
        rows=s.validate_reviews(*self.fixture())
        self.assertEqual(s.summarize(rows)['all_criteria']['rate'],1)

    def test_citation_failure_separate_from_correctness(self):
        values=self.fixture();values[-1]['reviews'][0]['scores']['citation_support']=False
        report=s.summarize(s.validate_reviews(*values))
        self.assertEqual(report['criteria']['correctness']['passed'],1)
        self.assertEqual(report['all_criteria']['passed'],0)

    def test_partial_abstention_separate_from_completeness(self):
        values=self.fixture();values[-1]['reviews'][0]['scores']['completeness']=False
        report=s.summarize(s.validate_reviews(*values))
        self.assertEqual(report['criteria']['appropriate_abstention']['rate'],1)
        self.assertEqual(report['criteria']['completeness']['rate'],0)

    def test_changed_answer_rejected(self):
        values=self.fixture();values[2]['cases'][0]['answer']='changed'
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_changed_prompt_rejected(self):
        values=self.fixture();values[1][0]['prompt']='changed'
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_changed_binding_rejected(self):
        values=self.fixture();values[3]['model']='other'
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_pending_review_rejected(self):
        values=self.fixture();values[-1]['reviews'][0]['review_status']='pending'
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_incomplete_review_rejected(self):
        for field in ['reviewer','reviewed_at','rationale']:
            values=self.fixture();values[-1]['reviews'][0][field]=' '
            with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_timezone_required(self):
        values=self.fixture();values[-1]['reviews'][0]['reviewed_at']='2026-11-12'
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_boolean_scores_required(self):
        for invalid in [None,1,0,'true',float('nan')]:
            values=self.fixture();values[-1]['reviews'][0]['scores']['correctness']=invalid
            with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_unknown_or_missing_criterion_rejected(self):
        values=self.fixture();values[-1]['reviews'][0]['scores']['other']=True
        with self.assertRaises(ValueError):s.validate_reviews(*values)
        values=self.fixture();del values[-1]['reviews'][0]['scores']['groundedness']
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_duplicate_unknown_missing_review_rejected(self):
        for mode in ['duplicate','unknown','missing']:
            values=self.fixture();rows=values[-1]['reviews']
            if mode=='duplicate':rows.append(deepcopy(rows[0]))
            elif mode=='unknown':rows[0]['id']='unknown'
            else:rows.clear()
            with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_duplicate_answers_rejected(self):
        values=self.fixture();values[2]['cases'].append(deepcopy(values[2]['cases'][0]))
        with self.assertRaises(ValueError):s.validate_reviews(*values)

    def test_missing_information_can_be_scored(self):
        values=self.fixture();values[0][0].update(category='missing_information',answerable=False)
        rows=s.validate_reviews(*values)
        self.assertFalse(rows[0]['answerable'])
        self.assertEqual(s.summarize(rows)['case_count'],1)

    def test_checkpoint_reconciliation(self):
        # Full loader with synthetic trusted preflight/prompt results; verify
        # original individual saved responses match aggregate exports/reviews.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);baseline=root/'baseline';folder=root/'answers';baseline.mkdir();folder.mkdir()
            for name in ['app.py','run_baseline_answers.py','run_saved_reranking.py','run_app_retrieval_baseline.py']:
                (root/name).write_text('# fixture only\n')
            s.f.save(baseline/'retrieval_report.json',{'metrics':{'recall':.5}})
            binding={'dataset_sha256':s.f.DATASET,'model':s.runner.CHAT_MODEL,
                'baseline_report_sha256':s.f.sha((baseline/'retrieval_report.json').read_bytes()),
                'max_output_tokens':1000,'request_interval_seconds':15}
            for name,key in [('app.py','app_sha256'),('run_baseline_answers.py','script_sha256'),('run_saved_reranking.py','validation_helper_sha256'),('run_app_retrieval_baseline.py','store_helper_sha256')]:binding[key]=s.f.sha((root/name).read_bytes())
            cases=[];prompts=[];answers=[];reviews=[]
            for i in range(15):
                cid=str(i);cases.append({'id':cid,'category':'direct' if i<12 else 'missing_information','answerable':i<12})
                p={'id':cid,'model':s.runner.CHAT_MODEL,'prompt':'context '+cid,'config':{'max_output_tokens':1000}}
                prompts.append(p);answer={'answer':'response '+cid,'finish_reason':'STOP'};answers.append({'id':cid,**answer})
                reviews.append({'id':cid,'answer_sha256':s.f.fingerprint(answer),'prompt_sha256':s.f.fingerprint(p),
                    'scores':{k:True for k in s.CRITERIA},'review_status':'approved','reviewer':'TEST FIXTURE ONLY',
                    'reviewed_at':'2026-11-12T00:00:00+00:00','rationale':'Synthetic only.'})
            binding['prompt_list_sha256']=s.f.fingerprint(prompts)
            s.f.save(folder/'run_manifest.json',binding);s.f.save(folder/'prompts.json',prompts)
            s.f.save(folder/'answers.json',{'binding':binding,'cases':answers})
            s.f.save(folder/'answer_reviews.json',{'binding_sha256':s.f.fingerprint(binding),'reviews':reviews})
            for p,a in zip(prompts,answers):
                request={'model':p['model'],'prompt':p['prompt'],'config':p['config']}
                answer={k:v for k,v in a.items() if k!='id'}
                s.f.save(folder/(p['id']+'-answer.json'),{'key':s.f.fingerprint({'run':binding,'request':request}),
                    'value':answer,'value_sha256':s.f.fingerprint(answer)})
            with patch.object(s.f,'preflight',return_value=([],cases,[],{},{},{'metrics':{'recall':.5}})),patch.object(s.runner,'prepare_prompts',return_value=prompts):
                result=s.load_and_validate(root,baseline,folder,root)
                self.assertEqual(len(result[0]),15)
                checkpoint=folder/'0-answer.json';item=s.f.read(checkpoint);item['value']['answer']='changed'
                # Even a recomputed checkpoint value hash cannot hide disagreement
                # with the previously exported/reviewed aggregate response.
                item['value_sha256']=s.f.fingerprint(item['value']);s.f.save(checkpoint,item)
                with self.assertRaises(ValueError):s.load_and_validate(root,baseline,folder,root)


if __name__=='__main__':unittest.main()
