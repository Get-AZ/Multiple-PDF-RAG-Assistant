from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_baseline_answers as r


class BaselineAnswerTests(unittest.TestCase):
    def app_path(self):
        p=Path(__file__).resolve().parents[1]/'app.py'
        return p if p.exists() else Path(__file__).resolve().parents[3]/'upload'/'app(2).py'

    def passages(self):
        return [{'id':str(i),'source':'paper.pdf','page':i+1,'text':'Source text '+str(i)} for i in range(6)]

    def test_actual_app_prompt_no_history(self):
        p=r.capture_prompt(self.app_path(),'Question?',[],self.passages(),'Question?')
        self.assertEqual(p['model'],r.CHAT_MODEL)
        self.assertEqual(p['config'],{'max_output_tokens':1000})
        self.assertIn('[Passage 6; paper.pdf, page 6]',p['prompt'])
        self.assertIn('Original question:\nQuestion?',p['prompt'])

    def test_saved_followup_rewrite_reused(self):
        history=[{'role':'user','content':'Which model?'},{'role':'assistant','content':'A model.'}]
        p=r.capture_prompt(self.app_path(),'What hardware?',history,self.passages(),'What hardware did the base model use?')
        self.assertIn('Standalone question:\nWhat hardware did the base model use?',p['prompt'])
        self.assertIn('User: Which model?',p['prompt'])

    def test_gold_fields_never_enter_prompt(self):
        row={'id':'case','question':'Question?','history':[], 'reference_answer':'SECRET_ANSWER','relevant_ids':['SECRET_ID']}
        corpus=self.passages()
        prompts=r.prepare_prompts(self.app_path().parent,corpus,[row],{'case':{'retrieved_ids':[str(i) for i in range(6)],'retrieval_question':'Question?'}}) if self.app_path().name=='app.py' else None
        if prompts is None:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/'app.py').write_bytes(self.app_path().read_bytes())
                prompts=r.prepare_prompts(tmp,corpus,[row],{'case':{'retrieved_ids':[str(i) for i in range(6)],'retrieval_question':'Question?'}})
        self.assertNotIn('SECRET',prompts[0]['prompt'])

    def test_wrong_saved_standalone_rejected(self):
        with self.assertRaises(ValueError):r.capture_prompt(self.app_path(),'Question?',[],self.passages(),'Changed query')

    def test_missing_context_rejected(self):
        with self.assertRaises(ValueError):r.capture_prompt(self.app_path(),'Q',[],self.passages()[:5],'Q')

    def test_pacing(self):
        self.assertEqual(r.pacing_delay(100,105),10)
        self.assertEqual(r.pacing_delay(100,120),0)
        self.assertEqual(r.pacing_delay(0,100),0)
        with self.assertRaises(ValueError):r.pacing_delay(200,100)
        with self.assertRaises(ValueError):r.pacing_delay(100,105,interval=1)

    def test_empty_answer_not_successful(self):
        with self.assertRaises(ValueError):r.response_value(SimpleNamespace(text=' '))

    def test_finish_reason_preserved_for_review(self):
        value=r.response_value(SimpleNamespace(text='partial answer',candidates=[SimpleNamespace(finish_reason='MAX_TOKENS')],model_version='test-version'))
        self.assertEqual(value['finish_reason'],'MAX_TOKENS')
        self.assertEqual(value['model_version'],'test-version')

    def fixtures(self):
        corpus=self.passages()
        case={'id':'case','question':'Question?','reference_answer':'Reference','relevant_ids':['0'],'answerable':True}
        prompts=[{'id':'case','retrieved_ids':[str(i) for i in range(6)],'prompt':'No gold answer'}]
        answers={'case':{'answer':'Source answer','finish_reason':'STOP'}}
        return corpus,[case],prompts,answers,{'model':'test'}

    def test_review_pending_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            r.export_review(tmp,*self.fixtures())
            review=r.frozen.read(Path(tmp)/'answer_reviews.json')['reviews'][0]
            self.assertEqual(review['review_status'],'pending')
            self.assertTrue(all(v is None for v in review['scores'].values()))
            self.assertIn('Source answer',(Path(tmp)/'review_packet.md').read_text())

    def test_existing_human_review_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixtures();r.export_review(tmp,*f)
            p=Path(tmp)/'answer_reviews.json';review=r.frozen.read(p)
            review['reviews'][0]['rationale']='Human work';r.frozen.save(p,review)
            before=p.read_bytes();r.export_review(tmp,*f)
            self.assertEqual(p.read_bytes(),before)

    def test_changed_reviewed_answer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=list(self.fixtures());r.export_review(tmp,*f)
            f[3]={'case':{'answer':'changed','finish_reason':'STOP'}}
            with self.assertRaises(ValueError):r.export_review(tmp,*f)

    def test_resumption_does_not_repeat_answer_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=r.Store(tmp,{'model':'a'})
            calls=[]
            def api():calls.append(1);return {'answer':'saved'}
            one=store.cached('answer',{'prompt':'q'},api)
            two=store.cached('answer',{'prompt':'q'},api)
            self.assertEqual(one,two)
            self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
