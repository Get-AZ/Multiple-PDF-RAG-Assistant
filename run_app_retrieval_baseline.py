"""Resumable development retrieval baseline using the actual app functions.

Default: offline preflight only. --execute makes embedding and follow-up rewrite
requests. Answer generation is intercepted and never sent to Gemini.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace
from unittest.mock import patch

EXPECTED_DATASET = 'a19ba7d8e9aa5abe61d777fbfffc3f1ea9417f4542c84abd9a12080f5731178e'
EXPECTED_APP_AST = '6bf8f1a022d9c3df96fd01bb4a8d8a92761dd1071944f4ff740c51ba010ec5fa'
CHAT_MODEL = 'gemini-3.5-flash-lite'
EMBEDDING_MODEL = 'gemini-embedding-001'
PACKAGES = ['google-genai', 'numpy', 'faiss-cpu', 'pypdf', 'streamlit', 'python-dotenv']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def read(path):
    def reject(value):
        raise ValueError('Non-finite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def metrics(relevant, retrieved, k=6):
    require(type(k) is int and k > 0, 'Invalid k.')
    require(len(relevant)==len(set(relevant)), 'Duplicate evidence IDs.')
    require(len(retrieved)==len(set(retrieved)) and len(retrieved)<=k, 'Duplicate/excess retrieval IDs.')
    if not relevant:
        return None
    gold=set(relevant)
    hits=[i+1 for i,pid in enumerate(retrieved) if pid in gold]
    dcg=sum(1/math.log2(rank+1) for rank in hits)
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
    return {'recall':len(hits)/len(gold), 'mrr':1/hits[0] if hits else 0.0, 'ndcg':dcg/ideal}


def load_frozen(folder, project):
    folder,project=Path(folder),Path(project)
    validation=read(folder/'validation.json')
    corpus=read(folder/'corpus.json');cases=read(folder/'cases.json')
    for filename,key in [('corpus.json','corpus_file_sha256'),('cases.json','cases_file_sha256'),
                         ('source_export_manifest.json','source_export_manifest_sha256')]:
        require(sha((folder/filename).read_bytes())==validation[key], 'Frozen file changed: '+filename)
    digest=fingerprint({'corpus':corpus,'cases':cases})
    require(digest==validation['dataset_sha256']==EXPECTED_DATASET, 'Unexpected frozen dataset. Do not edit hashes to bypass.')
    source=read(folder/'source_export_manifest.json')
    for name in ['app.py','rag_provenance.py']:
        require(sha((project/name).read_bytes())==source['source_files'][name], 'Source differs from export: '+name)
    app_tree=ast.dump(ast.parse((project/'app.py').read_text(encoding='utf-8-sig')), include_attributes=False)
    require(
        sha((project/'app.py').read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode())
        == '613eec2f024d32c27432fa3e610d450af2c329e53bd84acfc7530281eaf29da8',
        'App source differs from inspected upload.'
    )
    require(len(corpus)==74 and len(cases)==15, 'Unexpected corpus/case count.')
    require(len({p['id'] for p in corpus})==len(corpus), 'Duplicate corpus IDs.')
    docs={p['document_id'] for p in corpus}
    require(len(docs)==2, 'Expected two PDFs.')
    ids={p['id'] for p in corpus}
    for c in cases:
        require(set(c['document_ids'])==docs, 'Retrieval must search both PDFs.')
        require(c['review_status']=='approved' and c['split']=='dev', 'Unapproved/non-development case.')
        require(set(c['relevant_ids'])<=ids, 'Unknown gold evidence.')
        require(bool(c['relevant_ids'])==c['answerable'], 'Evidence/answerability mismatch.')
    require(len({c['id'] for c in cases})==15 and sum(c['answerable'] for c in cases)==12, 'Case counts differ.')
    require(sum(bool(c['history']) for c in cases)==2, 'Expected two follow-up fixtures.')
    return corpus,cases,digest


class Store:
    """Atomic, content-bound artifacts; no pickle or FAISS deserialization."""
    def __init__(self, folder, binding):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.binding=binding
        self.path=self.folder/'run_manifest.json'
        if self.path.exists():
            require(read(self.path)==binding, 'Run configuration changed; choose a fresh --outdir.')
        else:
            require(not any(self.folder.iterdir()), 'Output folder contains unbound files.')
            save(self.path,binding)

    def cached(self, name, request, produce):
        path=self.folder/(name+'.json')
        key=fingerprint({'run':self.binding,'request':request})
        if path.exists():
            item=read(path)
            require(item['key']==key and fingerprint(item['value'])==item['value_sha256'], 'Cache mismatch: '+name)
            return item['value']
        value=produce()
        save(path,{'key':key,'value':value,'value_sha256':fingerprint(value)})
        return value


class RetrievalOnlyModels:
    """Forward only the app's rewrite call; intercept its final answer call."""
    def __init__(self, real_models, has_history, rewrite):
        self.real_models=real_models
        self.has_history=has_history
        self.rewrite=rewrite
        self.calls=0

    def generate_content(self, **kwargs):
        self.calls+=1
        if self.has_history and self.calls==1:
            require(kwargs.get('contents','').startswith('Rewrite the current question as one standalone search question.'), 'Unexpected rewrite prompt; no API request sent.')
            return SimpleNamespace(text=self.rewrite(kwargs))
        expected=2 if self.has_history else 1
        require(self.calls==expected, 'Unexpected generation call; stopped before API request.')
        require('Reference passages:' in kwargs.get('contents',''), 'Unexpected answer prompt.')
        return SimpleNamespace(text='[RETRIEVAL ONLY: ANSWER NOT GENERATED]')


def run_case(app, client, index, corpus, case, store, np):
    # Only these user-facing fields enter the app call. Gold labels are scored later.
    question=case['question'];history=case['history'];cid=case['id']
    def rewrite(kwargs):
        def produce():
            response=client.models.generate_content(**kwargs)
            require(bool(response.text and response.text.strip()), 'Empty rewrite.')
            return response.text
        return store.cached(cid+'-rewrite', {'model':CHAT_MODEL,'prompt':kwargs['contents'],'max_output_tokens':256},produce)
    models=RetrievalOnlyModels(client.models, bool(app.recent_history(history)), rewrite)
    @contextmanager
    def proxy_client():
        yield SimpleNamespace(models=models)
    original_embed=app.embed_texts
    def query_embed(unused_client, texts, task_type):
        require(task_type=='RETRIEVAL_QUERY' and len(texts)==1, 'Unexpected embedding request.')
        values=store.cached(cid+'-query', {'model':EMBEDDING_MODEL,'texts':texts,'task_type':task_type},
            lambda:original_embed(client,texts,task_type).tolist())
        array=np.ascontiguousarray(values,dtype=np.float32)
        check_vectors(array,1,np,index.d)
        return array
    with patch.object(app,'make_client',proxy_client),patch.object(app,'embed_texts',query_embed):
        _,passages,rewritten=app.answer_question(question,index,corpus,messages=history,k=6)
    retrieved=[p['id'] for p in passages]
    require(len(retrieved)==min(6,len(corpus)) and len(set(retrieved))==len(retrieved), 'Invalid retrieval result.')
    require(set(retrieved)<={p['id'] for p in corpus}, 'Unknown retrieved ID.')
    return {'id':cid,'retrieval_question':rewritten,'retrieved_ids':retrieved,
            'retrieved_sources':[{'id':p['id'],'source':p['source'],'page':p['page']} for p in passages],
            'answer_generated':False}


def check_vectors(array, count, np, dimension=None):
    require(array.ndim==2 and array.shape[0]==count and array.shape[1]>0, 'Invalid embedding shape.')
    require(dimension is None or array.shape[1]==dimension, 'Embedding dimension mismatch.')
    require(np.isfinite(array).all(), 'Non-finite embedding.')
    require(np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4), 'Embeddings are not normalized.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--data',default='private_data/multi_pdf_approved_dev_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_faiss_baseline_v1')
    args=parser.parse_args()
    project=Path(__file__).resolve().parent
    corpus,cases,digest=load_frozen(args.data,project)
    versions={name:importlib.metadata.version(name) for name in PACKAGES}
    binding={'schema_version':1,'dataset_sha256':digest,'app_sha256':sha((project/'app.py').read_bytes()),
        'provenance_sha256':sha((project/'rag_provenance.py').read_bytes()),
        'runner_sha256':sha(Path(__file__).read_bytes()),'python':platform.python_version(),
        'packages':versions,'embedding_model':EMBEDDING_MODEL,'chat_model_for_rewrite':CHAT_MODEL,
        'k':6,'document_batch_size':20,'pipeline':'actual app normalized FAISS IndexFlatIP; no final generation'}
    print('VERIFIED: frozen dataset, source hashes and full two-document search scope.')
    print('Cases: 15; answerable: 12; missing-information: 3; follow-ups: 2.')
    print('Models:',EMBEDDING_MODEL,'/',CHAT_MODEL,'(rewrite only)')
    if not args.execute:
        print('DRY RUN: no API requests or output files. Use --execute for the retrieval baseline.')
        return
    import faiss
    import numpy as np
    import app
    from dotenv import load_dotenv
    load_dotenv(project/'.env',encoding='utf-8-sig',override=True)
    # Explicit per-process override; no source edits or frozen-file changes.
    app.CHAT_MODEL=CHAT_MODEL
    require(app.EMBEDDING_MODEL==EMBEDDING_MODEL,'Unexpected app embedding model.')
    store=Store(args.outdir,binding)
    with app.make_client() as client:
        batches=[]
        for start in range(0,len(corpus),20):
            texts=[p['text'] for p in corpus[start:start+20]]
            values=store.cached('documents-'+str(start),{'texts':texts,'model':EMBEDDING_MODEL,'task_type':'RETRIEVAL_DOCUMENT'},
                lambda:app.embed_texts(client,texts,'RETRIEVAL_DOCUMENT').tolist())
            array=np.ascontiguousarray(values,dtype=np.float32)
            check_vectors(array,len(texts),np,batches[0].shape[1] if batches else None)
            batches.append(array)
            print('Document embeddings ready:',min(start+20,len(corpus)),'/',len(corpus),flush=True)
        vectors=np.vstack(batches)
        index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
        results=[]
        for case in cases:
            start=time.perf_counter()
            row=store.cached(case['id']+'-retrieval',{'question':case['question'],'history':case['history']},
                lambda:run_case(app,client,index,corpus,case,store,np))
            require(row['id']==case['id'] and not row['answer_generated'],'Invalid saved case.')
            row=dict(row)
            row['metrics']=metrics(case['relevant_ids'],row['retrieved_ids'])
            row['category']=case['category'];row['answerable']=case['answerable']
            results.append(row)
            print(case['id']+': ready (new or reused); current processing seconds: '+str(round(time.perf_counter()-start,3)),flush=True)
    scored=[r['metrics'] for r in results if r['metrics'] is not None]
    aggregate={key:sum(r[key] for r in scored)/len(scored) for key in ['recall','mrr','ndcg']}
    report={'binding':binding,'metrics':aggregate,'cases':results,
        'scored_answerable_cases':len(scored),'excluded_missing_information_cases':3,
        'limitations':['Development-only labeled-chunk retrieval at k=6; not answer accuracy.',
            'Missing-information cases are not retrieval successes; abstention remains untested.',
            'Saved progress is for reproducibility, not a measured product cache.',
            'Provider model aliases are recorded; immutable provider weight revisions are unavailable.',
            'No service latency percentile or cost claim; interrupted/reused work is not benchmark latency.']}
    save(store.folder/'retrieval_report.json',report)
    print('DEVELOPMENT RETRIEVAL @6:',json.dumps(aggregate))
    print('Report:',(store.folder/'retrieval_report.json').resolve())
    print('Final answers were NOT generated. No comparison gate or held-out claim assigned.')


if __name__=='__main__':
    main()
