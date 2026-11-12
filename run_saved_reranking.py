"""Development-only top-30 -> cross-encoder -> top-six experiment.
Default is an offline artifact check. --execute downloads/loads a pinned local
reranker. This module never imports the Google SDK or calls Gemini.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import time

DATASET = 'a19ba7d8e9aa5abe61d777fbfffc3f1ea9417f4542c84abd9a12080f5731178e'
APP_SOURCE = '613eec2f024d32c27432fa3e610d450af2c329e53bd84acfc7530281eaf29da8'
MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())


def read(path):
    def reject(value):raise ValueError('Non-finite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def metric(gold,ids,k=6):
    require(len(ids)==len(set(ids)) and len(ids)<=k,'Duplicate/excess retrieved IDs.')
    require(len(gold)==len(set(gold)),'Duplicate gold IDs.')
    if not gold:return None
    hits=[i+1 for i,pid in enumerate(ids) if pid in set(gold)]
    ideal=sum(1/math.log2(i+2) for i in range(min(len(gold),k)))
    return {'recall':len(hits)/len(gold),'mrr':1/hits[0] if hits else 0.,
            'ndcg':sum(1/math.log2(i+1) for i in hits)/ideal}


def aggregate(rows):
    values=[r['metrics'] for r in rows if r['metrics'] is not None]
    require(bool(values),'No scored cases.')
    return {key:sum(m[key] for m in values)/len(values) for key in ['recall','mrr','ndcg']}


def checked_cache(folder,name,binding,request):
    item=read(Path(folder)/(name+'.json'))
    require(item['key']==fingerprint({'run':binding,'request':request}),'Saved request binding differs: '+name)
    require(item['value_sha256']==fingerprint(item['value']),'Saved value hash differs: '+name)
    return item['value']


def close_metrics(actual,expected):
    require(set(actual)==set(expected),'Metric fields differ.')
    require(all(math.isfinite(actual[k]) and abs(actual[k]-expected[k])<1e-10 for k in expected),'Recorded metrics do not reproduce.')


def preflight(data,baseline,project):
    data,baseline,project=Path(data),Path(baseline),Path(project)
    corpus=read(data/'corpus.json');cases=read(data/'cases.json');validation=read(data/'validation.json')
    require(fingerprint({'corpus':corpus,'cases':cases})==validation['dataset_sha256']==DATASET,'Frozen dataset differs.')
    for name,key in [('corpus.json','corpus_file_sha256'),('cases.json','cases_file_sha256'),
                     ('source_export_manifest.json','source_export_manifest_sha256')]:
        require(sha((data/name).read_bytes())==validation[key],'Frozen file changed: '+name)
    source=read(data/'source_export_manifest.json')
    for name in ['app.py','rag_provenance.py']:
        require(sha((project/name).read_bytes())==source['source_files'][name],'Source changed: '+name)
    require(sha((project/'app.py').read_text(encoding='utf-8-sig').encode())==APP_SOURCE,'App differs from inspected source.')
    binding=read(baseline/'run_manifest.json');report=read(baseline/'retrieval_report.json')
    require(report['binding']==binding and binding['dataset_sha256']==DATASET,'Baseline report binding differs.')
    require(binding['k']==6 and binding['document_batch_size']==20,'Unexpected baseline settings.')
    require(binding['embedding_model']=='gemini-embedding-001' and binding['chat_model_for_rewrite']=='gemini-3.5-flash-lite','Unexpected baseline models.')
    require(binding['app_sha256']==sha((project/'app.py').read_bytes()),'Baseline app differs.')
    require(binding['provenance_sha256']==sha((project/'rag_provenance.py').read_bytes()),'Baseline provenance differs.')
    require(binding['runner_sha256']==sha((project/'run_app_retrieval_baseline.py').read_bytes()),'Baseline runner changed after run.')
    require(len(corpus)==74 and len(cases)==15 and len({p['id'] for p in corpus})==74,'Unexpected corpus/cases.')
    require(len({c['id'] for c in cases})==15,'Duplicate cases.')
    docs={p['document_id'] for p in corpus};require(len(docs)==2,'Expected two documents.')
    require(sum(c['answerable'] for c in cases)==12,'Expected 12 answerable cases.')
    saved_rows={r['id']:r for r in report['cases']}
    require(len(report['cases'])==15 and set(saved_rows)=={c['id'] for c in cases},'Baseline case coverage differs.')
    vectors=[];queries={};retrievals={}
    for start in range(0,len(corpus),20):
        texts=[p['text'] for p in corpus[start:start+20]]
        values=checked_cache(baseline,'documents-'+str(start),binding,
            {'texts':texts,'model':binding['embedding_model'],'task_type':'RETRIEVAL_DOCUMENT'})
        require(len(values)==len(texts),'Document batch count differs.')
        vectors.extend(values)
    ids={p['id'] for p in corpus}
    for c in cases:
        cid=c['id']
        require(c['split']=='dev' and c['review_status']=='approved' and set(c['document_ids'])==docs,'Invalid review/search scope.')
        require(bool(c['relevant_ids'])==c['answerable'] and set(c['relevant_ids'])<=ids,'Invalid gold evidence.')
        row=checked_cache(baseline,cid+'-retrieval',binding,{'question':c['question'],'history':c['history']})
        require(row['id']==cid and row['answer_generated'] is False,'Invalid saved retrieval.')
        require(len(row['retrieved_ids'])==6 and set(row['retrieved_ids'])<=ids,'Invalid baseline top six.')
        require(saved_rows[cid]['retrieved_ids']==row['retrieved_ids'] and saved_rows[cid]['retrieval_question']==row['retrieval_question'],'Report differs from saved retrieval.')
        expected=metric(c['relevant_ids'],row['retrieved_ids'])
        if expected is None:require(saved_rows[cid]['metrics'] is None,'Missing case was retrieval-scored.')
        else:close_metrics(saved_rows[cid]['metrics'],expected)
        queries[cid]=checked_cache(baseline,cid+'-query',binding,
            {'model':binding['embedding_model'],'texts':[row['retrieval_question']],'task_type':'RETRIEVAL_QUERY'})
        require(isinstance(row['retrieval_question'],str) and row['retrieval_question'].strip(),'Empty saved query.')
        retrievals[cid]=row
    close_metrics(report['metrics'],aggregate(report['cases']))
    return corpus,cases,vectors,queries,retrievals,report


def stable_rerank(query,candidates,passages,predict):
    require(len(candidates)==len(set(candidates)),'Duplicate candidate ID.')
    require(all(pid in passages for pid in candidates),'Unknown candidate.')
    if not candidates:return []
    # No review fields, gold answers, labels or evidence-document filters.
    pairs=[[query,passages[pid]['text']] for pid in candidates]
    scores=[float(s) for s in predict(pairs)]
    require(len(scores)==len(candidates) and all(math.isfinite(s) for s in scores),'Invalid reranker scores.')
    return sorted(zip(candidates,scores),key=lambda item:(-item[1],item[0]))


def comparison(old,new):
    regressions=[key for key in old if new[key]<old[key]-1e-10]
    return {'aggregate_no_regression':not regressions,'regressed_metrics':regressions,
            'note':'Development comparison only; not a required CI promotion gate.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='private_data/multi_pdf_approved_dev_v1')
    parser.add_argument('--baseline',default='private_data/multi_pdf_faiss_baseline_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_reranking_v1')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();project=Path(__file__).resolve().parent
    corpus,cases,vectors,queries,retrievals,baseline=preflight(args.data,args.baseline,project)
    print('VERIFIED: frozen data, baseline report and 19 saved embedding artifacts.')
    print('EXPERIMENT: 30 candidates -> cross-encoder -> 6 passages; same saved questions.')
    if not args.execute:
        print('DRY RUN: no model imports/downloads, Gemini calls or file changes.');return
    import faiss
    import numpy as np
    from huggingface_hub import HfApi
    from sentence_transformers import CrossEncoder
    document_vectors=np.ascontiguousarray(vectors,dtype=np.float32)
    require(document_vectors.ndim==2 and document_vectors.shape==(74,3072),'Unexpected document embedding shape.')
    require(np.isfinite(document_vectors).all() and np.allclose(np.linalg.norm(document_vectors,axis=1),1,atol=1e-4),'Invalid document vectors.')
    index=faiss.IndexFlatIP(document_vectors.shape[1]);index.add(document_vectors)
    pools={}
    # Reproduce all original rankings BEFORE loading/running the reranker.
    for c in cases:
        query=np.ascontiguousarray(queries[c['id']],dtype=np.float32)
        require(query.shape==(1,index.d) and np.isfinite(query).all() and np.allclose(np.linalg.norm(query,axis=1),1,atol=1e-4),'Invalid query vector.')
        _,positions=index.search(query,30)
        ids=[corpus[int(i)]['id'] for i in positions[0]]
        require(ids[:6]==retrievals[c['id']]['retrieved_ids'],'Baseline FAISS ranking does not reproduce: '+c['id'])
        pools[c['id']]=ids
    print('REPLAY PASS: all 15 original top-six rankings match.')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'experiment_manifest.json'
    packages={p:importlib.metadata.version(p) for p in ['numpy','faiss-cpu','sentence-transformers','torch','transformers','huggingface-hub']}
    binding={'dataset_sha256':DATASET,'baseline_report_sha256':sha((Path(args.baseline)/'retrieval_report.json').read_bytes()),
        'baseline_manifest_sha256':sha((Path(args.baseline)/'run_manifest.json').read_bytes()),
        'script_sha256':sha(Path(__file__).read_bytes()),'model':MODEL,'pool_size':30,'k':6,
        'device':'cpu','max_seq_length':512,'packages':packages,'python':platform.python_version(),
        'candidate_pools_sha256':fingerprint(pools)}
    if manifest_path.exists():
        old=read(manifest_path);revision=old['model_revision']
        require({k:v for k,v in old.items() if k!='model_revision'}==binding,'Experiment configuration changed; use a new output folder.')
    else:
        require(not any(out.iterdir()),'Output folder contains unbound files.')
        revision=HfApi().model_info(MODEL).sha
        require(bool(re.fullmatch('[0-9a-f]{40}',revision)),'Invalid Hugging Face revision.')
        save(manifest_path,{**binding,'model_revision':revision})
    print('Loading pinned reranker on CPU:',MODEL,revision,flush=True)
    model=CrossEncoder(MODEL,revision=revision,device='cpu',trust_remote_code=False,max_length=512)
    require(model.max_seq_length==512,'Unexpected model input limit.')
    passages={p['id']:p for p in corpus};rows=[];audits={}
    for c in cases:
        cid=c['id'];query=retrievals[cid]['retrieval_question'];candidates=pools[cid]
        path=out/(cid+'.json');key=fingerprint({'binding':{**binding,'model_revision':revision},'query':query,'candidates':candidates})
        if path.exists():
            saved=read(path);require(saved['key']==key and saved['value_sha256']==fingerprint(saved['value']),'Case checkpoint differs.')
            audit=saved['value']
        else:
            lengths={pid:len(model.tokenizer(query,passages[pid]['text'],truncation=False,verbose=False)['input_ids']) for pid in candidates}
            started=time.perf_counter()
            ranked=stable_rerank(query,candidates,passages,lambda pairs:model.predict(pairs,batch_size=16,show_progress_bar=False))
            audit={'candidate_ids':candidates,'ranking':[{'id':pid,'score':score} for pid,score in ranked],
                   'over_limit_ids':[pid for pid in candidates if lengths[pid]>512],
                   'scoring_seconds':time.perf_counter()-started}
            save(path,{'key':key,'value':audit,'value_sha256':fingerprint(audit)})
        ranked_ids=[p['id'] for p in audit['ranking']]
        require(len(ranked_ids)==30 and set(ranked_ids)==set(candidates),'Saved ranking candidate coverage differs.')
        require(all(math.isfinite(r['score']) for r in audit['ranking']),'Invalid saved score.')
        top=ranked_ids[:6]
        rows.append({'id':cid,'category':c['category'],'answerable':c['answerable'],
            'retrieval_question':query,'retrieved_ids':top,'metrics':metric(c['relevant_ids'],top),
            'candidate_pool_recall':metric(c['relevant_ids'],candidates,k=30),
            'baseline_metrics':metric(c['relevant_ids'],retrievals[cid]['retrieved_ids'])})
        audits[cid]=audit
        print(cid+': reranked (new or reused)',flush=True)
    result=aggregate(rows);gate=comparison(baseline['metrics'],result)
    report={'experiment':{**binding,'model_revision':revision},'baseline_metrics':baseline['metrics'],'metrics':result,
        'comparison':gate,'cases':rows,'audit':audits,'scored_cases':12,'unscored_missing_information_cases':3,
        'limitations':['Development-only retrieval; no answer/abstention scores or held-out claims.',
            'Top-30 pool chosen after examining development failures; requires independent held-out validation.',
            'No corpus filters, gold scope restrictions or label-driven ranking.',
            'Pairs longer than 512 tokens are truncated by the model and audited.',
            'Batch scoring seconds are not service latency; progress reuse is not product cache evaluation.']}
    save(out/'reranking_report.json',report)
    print('\nMethod       Recall@6  MRR@6    nDCG@6')
    for name,values in [('Baseline',baseline['metrics']),('Reranked',result)]:
        print(f"{name:12} {values['recall']:.4f}    {values['mrr']:.4f}   {values['ndcg']:.4f}")
    print('Aggregate comparison:', 'PASS' if gate['aggregate_no_regression'] else 'REGRESSION')
    print('Per-case recall (baseline -> reranked):')
    for row in rows:
        print(row['id'], 'unscored' if row['metrics'] is None else f"{row['baseline_metrics']['recall']:.4f} -> {row['metrics']['recall']:.4f}")
    print('Report:',(out/'reranking_report.json').resolve())
    print('NO GEMINI REQUESTS. Baseline, app and frozen data unchanged.')


if __name__=='__main__':main()
