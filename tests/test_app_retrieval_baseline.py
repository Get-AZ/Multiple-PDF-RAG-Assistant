import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import run_app_retrieval_baseline as r


class BaselineTests(unittest.TestCase):
    def test_metrics_known_values(self):
        m=r.metrics(['a','b'],['x','a','b'])
        self.assertEqual(m['recall'],1)
        self.assertEqual(m['mrr'],0.5)
        self.assertAlmostEqual(m['ndcg'],(1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3)))

    def test_missing_information_excluded(self):
        self.assertIsNone(r.metrics([],['x']))

    def test_duplicate_rank_rejected(self):
        with self.assertRaises(ValueError):r.metrics(['a'],['a','a'])

    def test_progress_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=r.Store(tmp,{'model':'m'})
            self.assertEqual(store.cached('a',{'q':'q'},lambda:[1]),[1])
            self.assertEqual(store.cached('a',{'q':'q'},lambda:self.fail('Repeated request')),[1])

    def test_changed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r.Store(tmp,{'model':'a'})
            with self.assertRaises(ValueError):r.Store(tmp,{'model':'b'})

    def test_changed_request_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=r.Store(tmp,{})
            store.cached('x',{'q':'a'},lambda:'v')
            with self.assertRaises(ValueError):store.cached('x',{'q':'b'},lambda:'v')

    def test_tampered_progress_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=r.Store(tmp,{})
            store.cached('x',{},lambda:'v')
            p=Path(tmp)/'x.json';item=json.loads(p.read_text());item['value']='changed';p.write_text(json.dumps(item))
            with self.assertRaises(ValueError):store.cached('x',{},lambda:'v')

    def test_unbound_folder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'other').write_text('do not replace')
            with self.assertRaises(ValueError):r.Store(tmp,{})

    def test_no_final_answer_request(self):
        proxy=r.RetrievalOnlyModels(None,False,lambda _:self.fail('No rewrite expected'))
        self.assertIn('NOT GENERATED',proxy.generate_content(contents='Reference passages: x').text)
        with self.assertRaises(ValueError):proxy.generate_content(contents='Reference passages: x')

    def test_only_rewrite_is_forwarded(self):
        seen=[]
        proxy=r.RetrievalOnlyModels(None,True,lambda args:seen.append(args) or 'standalone')
        self.assertEqual(proxy.generate_content(contents='Rewrite the current question as one standalone search question. test').text,'standalone')
        proxy.generate_content(contents='Reference passages: example')
        self.assertEqual(len(seen),1)

    def test_unexpected_prompt_stopped_before_api(self):
        proxy=r.RetrievalOnlyModels(None,True,lambda _:self.fail('API must not be called'))
        with self.assertRaises(ValueError):proxy.generate_content(contents='Answer this instead')

    def test_vector_validation(self):
        r.check_vectors(np.array([[1.,0.]]),1,np,2)
        for value,dimension in [(np.array([[float('nan'),0.]]),2),(np.array([[0.,0.]]),2),(np.array([[1.,0.]]),3)]:
            with self.assertRaises(ValueError):r.check_vectors(value,1,np,dimension)

    def test_actual_app_functions_without_reference_leakage(self):
        # Use the inspected app's actual function AST, with simulated API/index.
        app_path=Path(__file__).resolve().parents[1]/'app.py'
        if not app_path.exists():
            app_path=Path(__file__).resolve().parents[3]/'upload'/'app(2).py'
        tree=ast.parse(app_path.read_text(encoding='utf-8-sig'))
        names={'recent_history','search_passages','answer_question'}
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        env={'types':SimpleNamespace(GenerateContentConfig=lambda **kw:kw),'CHAT_MODEL':r.CHAT_MODEL}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<app functions>','exec'),env)
        app=SimpleNamespace(**{name:env[name] for name in names})
        # Functions must resolve patched app attributes just as a real module does.
        class Models:
            def __init__(self):self.prompts=[]
            def generate_content(self,**kwargs):
                self.prompts.append(kwargs['contents'])
                return SimpleNamespace(text='standalone user question')
        models=Models();client=SimpleNamespace(models=models)
        class Index:
            ntotal=6;d=2
            def search(self,vector,k):return np.ones((1,k)),np.array([list(range(k))])
        corpus=[{'id':str(i),'text':'document text','source':'a.pdf','page':i+1} for i in range(6)]
        embedded=[]
        def embed(client,texts,task_type):
            embedded.append((texts,task_type))
            return np.array([[1.,0.]],dtype=np.float32)
        app.embed_texts=embed
        app.make_client=lambda:None
        # Bridge globals to patched attributes.
        env['embed_texts']=lambda *args:app.embed_texts(*args)
        env['make_client']=lambda:app.make_client()
        case={'id':'test','question':'What hardware?','history':[{'role':'user','content':'Base model?'},{'role':'assistant','content':'Previously discussed model.'}],
              'reference_answer':'SECRET_REFERENCE','relevant_ids':['SECRET_GOLD']}
        with tempfile.TemporaryDirectory() as tmp:
            store=r.Store(tmp,{})
            row=r.run_case(app,client,Index(),corpus,case,store,np)
            self.assertEqual(row['retrieved_ids'],[str(i) for i in range(6)])
            self.assertFalse(row['answer_generated'])
            self.assertEqual(len(models.prompts),1)
            self.assertNotIn('SECRET',models.prompts[0])
            self.assertEqual(embedded[0][0],['standalone user question'])
            r.run_case(app,client,Index(),corpus,case,store,np)
            self.assertEqual(len(models.prompts),1)
            self.assertEqual(len(embedded),1)


if __name__=='__main__':unittest.main()
