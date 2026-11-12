from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_citation_experiment as c


class CitationExperimentTests(unittest.TestCase):
    def prompts(self):
        return [{'id':f'case{i}','model':c.runner.CHAT_MODEL,'config':{'max_output_tokens':1000},
            'retrieved_ids':[str(n) for n in range(6)],
            'prompt':'Original question '+str(i)+'\nConversation: fixture\nReference passages:\n[Passage 1; doc.pdf, page 2]\nEvidence.'} for i in range(15)]

    def test_exact_prefix_only(self):
        old=self.prompts();new=c.citation_prompts(old)
        c.verify_prompt_only_change(old,new)
        self.assertEqual(new[0]['prompt'],c.CITATION_RULES+old[0]['prompt'])

    def test_original_inputs_unchanged(self):
        old=self.prompts();snapshot=deepcopy(old);c.citation_prompts(old)
        self.assertEqual(old,snapshot)

    def test_context_order_and_config_preserved(self):
        old=self.prompts();new=c.citation_prompts(old)
        for a,b in zip(old,new):
            self.assertEqual(a['retrieved_ids'],b['retrieved_ids'])
            self.assertEqual(a['model'],b['model'])
            self.assertEqual(a['config'],b['config'])

    def test_changed_candidate_context_rejected(self):
        old=self.prompts();new=c.citation_prompts(old);new[0]['prompt']+='altered context'
        with self.assertRaises(ValueError):c.verify_prompt_only_change(old,new)

    def test_changed_candidate_model_rejected(self):
        old=self.prompts();new=c.citation_prompts(old);new[0]['model']='other-model'
        with self.assertRaises(ValueError):c.verify_prompt_only_change(old,new)

    def test_changed_candidate_passage_order_rejected(self):
        old=self.prompts();new=c.citation_prompts(old)
        new[0]['retrieved_ids']=list(reversed(new[0]['retrieved_ids']))
        with self.assertRaises(ValueError):c.verify_prompt_only_change(old,new)

    def test_wrong_original_model_rejected(self):
        old=self.prompts();old[0]['model']='other-model'
        with self.assertRaises(ValueError):c.citation_prompts(old)

    def test_wrong_token_limit_rejected(self):
        old=self.prompts();old[0]['config']={'max_output_tokens':1500}
        with self.assertRaises(ValueError):c.citation_prompts(old)

    def test_duplicate_ids_rejected(self):
        old=self.prompts();old[1]['id']=old[0]['id']
        with self.assertRaises(ValueError):c.citation_prompts(old)

    def test_bad_context_count_rejected(self):
        for ids in [['a'],['a']*6]:
            old=self.prompts();old[0]['retrieved_ids']=ids
            with self.assertRaises(ValueError):c.citation_prompts(old)

    def test_output_cannot_overlap_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            protected=Path(tmp)/'baseline'
            for candidate in [protected,protected/'child',Path(tmp)]:
                with self.assertRaises(ValueError):c.safe_output(candidate,[protected])
            self.assertEqual(c.safe_output(Path(tmp)/'candidate',[protected]),Path(tmp)/'candidate')

    def test_version_audit_detects_changed_and_unknown(self):
        old={'cases':[{'id':'a','model_version':'v1'},{'id':'b','model_version':None}]}
        result=c.version_audit(old,{'a':{'model_version':'v2'},'b':{'model_version':'v2'}})
        self.assertFalse(result[0]['same_returned_model_version'])
        self.assertIsNone(result[1]['same_returned_model_version'])

    def test_no_gold_fields_in_added_instructions(self):
        for word in ['reference_answer','relevant_ids','review_status','MP-DEV-']:
            self.assertNotIn(word,c.CITATION_RULES)

    def test_fresh_reviews_not_copied_from_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompts=c.citation_prompts(self.prompts())
            corpus=[{'id':str(i),'source':'doc.pdf','page':i+1,'text':'evidence '+str(i)} for i in range(6)]
            cases=[{'id':p['id'],'question':'Question','reference_answer':'Human reference','relevant_ids':['0'],'answerable':True} for p in prompts]
            answers={p['id']:{'answer':'Candidate','finish_reason':'STOP'} for p in prompts}
            c.runner.export_review(tmp,corpus,cases,prompts,answers,{'experiment':'fixture'})
            rows=c.f.read(Path(tmp)/'answer_reviews.json')['reviews']
            self.assertEqual(len(rows),15)
            self.assertTrue(all(r['review_status']=='pending' and all(v is None for v in r['scores'].values()) for r in rows))

    def test_prepare_dry_without_api_and_baseline_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';retrieval=root/'retrieval';answers=root/'answers';scores=root/'scores';output=root/'candidate'
            for folder in [data,retrieval,answers,scores]:folder.mkdir()
            c.f.save(answers/'prompts.json',self.prompts())
            c.f.save(data/'corpus.json',[{'id':str(i),'text':'fixture'} for i in range(6)])
            c.f.save(data/'cases.json',[{'id':p['id']} for p in self.prompts()])
            c.f.save(scores/'human_review_scores.json',{'test':'fixture'})
            c.f.save(retrieval/'retrieval_report.json',{'test':'fixture'})
            for name in ['app.py','score_baseline_reviews.py','run_baseline_answers.py','run_saved_reranking.py','run_app_retrieval_baseline.py']:
                (root/name).write_text('# fixture only\n')
            before={str(p):p.read_bytes() for d in [data,retrieval,answers,scores] for p in d.iterdir()}
            with patch.object(c,'verified_baseline',return_value=({'model':'fixture'},{},{})),patch.object(c.importlib.metadata,'version',return_value='fixture'):
                store,prompts,*_=c.prepare(data,retrieval,answers,scores,output,root)
                self.assertEqual(len(prompts),15)
                self.assertEqual(len(c.f.read(output/'prompts.json')),15)
                # An identical repeat reuses artifacts rather than mixing config.
                c.prepare(data,retrieval,answers,scores,output,root)
            self.assertEqual(before,{p:Path(p).read_bytes() for p in before})


if __name__=='__main__':unittest.main()
