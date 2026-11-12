"""Generate and manually review answers from the original saved FAISS baseline.
Default prepares prompts without API calls. --execute sends at most 15 new
answer requests, reusing saved follow-up rewrites and six-passage retrievals.
"""
import argparse
import ast
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import run_saved_reranking as frozen
from run_app_retrieval_baseline import Store

CHAT_MODEL='gemini-3.5-flash-lite'
CRITERIA=['correctness','completeness','groundedness','citation_support','appropriate_abstention']


def capture_prompt(app_path,question,history,passages,standalone):
    """Execute only inspected app function AST with simulated retrieval/client.

    No app imports or Google calls. Follow-up rewriting is replayed from the
    saved baseline. The actual app's final generation prompt/config is captured.
    """
    tree=ast.parse(Path(app_path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'recent_history','answer_question'}]
    frozen.require(len(nodes)==2,'Expected app answer/history functions.')
    env={'CHAT_MODEL':CHAT_MODEL,'types':SimpleNamespace(GenerateContentConfig=lambda **kwargs:kwargs)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<inspected app functions>','exec'),env)
    calls=[]
    has_history=bool(env['recent_history'](history))
    class Models:
        def generate_content(self,**kwargs):
            calls.append(kwargs)
            if has_history and len(calls)==1:
                frozen.require(kwargs['contents'].startswith('Rewrite the current question'),'Unexpected rewrite prompt.')
                return SimpleNamespace(text=standalone)
            frozen.require(len(calls)==(2 if has_history else 1),'Unexpected generation call.')
            frozen.require('Reference passages:' in kwargs['contents'],'Unexpected final prompt.')
            return SimpleNamespace(text='[PROMPT CAPTURE ONLY]')
    @contextmanager
    def client():yield SimpleNamespace(models=Models())
    def search(unused_client,q,index,chunks,k):
        frozen.require(q==standalone and k==6 and len(passages)==6,'Saved retrieval differs from app request.')
        return passages
    env.update(make_client=client,search_passages=search)
    env['answer_question'](question,None,None,messages=history,k=6)
    final=calls[-1]
    frozen.require(final['config']=={'max_output_tokens':1000},'App answer config changed.')
    return {'model':CHAT_MODEL,'prompt':final['contents'],'config':final['config']}


def prepare_prompts(project,corpus,cases,retrievals):
    passages={p['id']:p for p in corpus}
    prompts=[]
    for case in cases:
        row=retrievals[case['id']]
        selected=[passages[pid] for pid in row['retrieved_ids']]
        payload=capture_prompt(Path(project)/'app.py',case['question'],case['history'],selected,row['retrieval_question'])
        prompts.append({'id':case['id'],'retrieved_ids':row['retrieved_ids'],**payload})
    return prompts


def pacing_delay(last_start,now,interval=15):
    frozen.require(interval>=15,'Request interval must be at least 15 seconds.')
    frozen.require(last_start<=now+5,'Saved request timestamp is ahead of current clock.')
    return max(0,interval-(now-last_start)) if last_start else 0


def response_value(response):
    text=response.text
    frozen.require(isinstance(text,str) and bool(text.strip()),'No answer text returned; no successful answer saved.')
    candidates=getattr(response,'candidates',None) or []
    reason=getattr(candidates[0],'finish_reason',None) if candidates else None
    usage=getattr(response,'usage_metadata',None)
    return {'answer':text.strip(),'model_version':getattr(response,'model_version',None),
        'finish_reason':str(reason) if reason is not None else None,
        'usage':{name:getattr(usage,name,None) for name in ['prompt_token_count','candidates_token_count','total_token_count']},
        'saved_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def export_review(folder,corpus,cases,prompts,answers,binding):
    """Human reference fields appear only in review artifacts, never prompts."""
    folder=Path(folder);passages={p['id']:p for p in corpus};by_prompt={p['id']:p for p in prompts}
    frozen.require(set(answers)=={c['id'] for c in cases},'Answers incomplete; human review not prepared yet.')
    records=[];markdown=['# Original FAISS baseline — development answer review',
        'No answer-quality score has been assigned. Review each answer against the source PDF and supplied context.',
        'Score each criterion true/false; keep pending until human review. Explain each decision in rationale.',
        'Correctness: factual accuracy against the PDF. Completeness: all requested parts addressed.',
        'Groundedness: factual claims supported by the six supplied passages, not model knowledge or chat history.',
        'Citation support: correct filename/page citations supporting factual claims.',
        'Appropriate abstention: missing details acknowledged, no fabricated answer, and supported parts still answered.',
        'A true fact absent from supplied context may pass correctness but fail groundedness. Record retrieval limitations separately.',
        'For missing-information questions, a brief correct refusal needs no fabricated citation; citation_support may pass if no unsupported claims/citations are made.',
        'Finish reasons and token usage are diagnostics; non-STOP outputs may be incomplete and need review.',
        'The two histories are controlled human-reviewed fixtures, not a continuous live conversation.']
    for c in cases:
        cid=c['id'];answer=answers[cid];p=by_prompt[cid]
        answer_hash=frozen.fingerprint(answer)
        records.append({'id':cid,'answer_sha256':answer_hash,'prompt_sha256':frozen.fingerprint(p),
            'scores':{key:None for key in CRITERIA},'review_status':'pending','reviewer':'','reviewed_at':'','rationale':''})
        markdown.extend(['',f'## {cid}',f"Question: {c['question']}",f"Expected answerability: {c['answerable']}",
            f"Finish reason: {answer.get('finish_reason')}",'','### Generated answer',answer['answer'],
            '', '### Human-reviewed reference answer',c['reference_answer'],
            '', '### Supplied retrieval context'])
        for i,pid in enumerate(p['retrieved_ids'],1):
            item=passages[pid]
            markdown.extend(['',f"Passage {i} | {item['source']} | PDF page {item['page']} | {pid}",item['text']])
        markdown.extend(['','### Labeled evidence IDs', '\n'.join(c['relevant_ids']) or '(Missing-information case: none)',
            '', '### Human review', 'Correctness: ___ | Completeness: ___ | Groundedness: ___ | Citation support: ___ | Appropriate abstention: ___',
            'Rationale: ___'])
    template=folder/'answer_reviews.json'
    # Never overwrite someone's completed or in-progress review.
    if template.exists():
        existing=frozen.read(template)
        frozen.require(existing['binding_sha256']==frozen.fingerprint(binding),'Existing review has different binding.')
        frozen.require(len(existing['reviews'])==len(records) and {r['id'] for r in existing['reviews']}=={r['id'] for r in records},'Existing review case set differs.')
        expected={r['id']:r for r in records}
        for row in existing['reviews']:
            frozen.require(all(row[k]==expected[row['id']][k] for k in ['answer_sha256','prompt_sha256']),'Reviewed answer/prompt changed.')
    frozen.save(folder/'answers.json',{'binding':binding,'cases':[{'id':c['id'],**answers[c['id']]} for c in cases]})
    (folder/'review_packet.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    if not template.exists():
        frozen.save(template,{'binding_sha256':frozen.fingerprint(binding),'reviews':records})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='private_data/multi_pdf_approved_dev_v1')
    parser.add_argument('--baseline',default='private_data/multi_pdf_faiss_baseline_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_baseline_answers_v1')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();project=Path(__file__).resolve().parent
    corpus,cases,_,_,retrievals,_=frozen.preflight(args.data,args.baseline,project)
    prompts=prepare_prompts(project,corpus,cases,retrievals)
    binding={'schema_version':1,'dataset_sha256':frozen.DATASET,'model':CHAT_MODEL,
        'prompt_list_sha256':frozen.fingerprint(prompts),
        'baseline_report_sha256':frozen.sha((Path(args.baseline)/'retrieval_report.json').read_bytes()),
        'app_sha256':frozen.sha((project/'app.py').read_bytes()),
        'script_sha256':frozen.sha(Path(__file__).read_bytes()),
        'validation_helper_sha256':frozen.sha((project/'run_saved_reranking.py').read_bytes()),
        'store_helper_sha256':frozen.sha((project/'run_app_retrieval_baseline.py').read_bytes()),
        'packages':{p:importlib.metadata.version(p) for p in ['google-genai','python-dotenv']},
        'python':platform.python_version(),'request_interval_seconds':15,'max_output_tokens':1000,
        'scope':'Original FAISS baseline development answers; saved rewrites; no new retrieval.'}
    store=Store(args.outdir,binding)
    prompt_path=store.folder/'prompts.json'
    if prompt_path.exists():frozen.require(frozen.read(prompt_path)==prompts,'Saved prompts changed.')
    else:frozen.save(prompt_path,prompts)
    print('VERIFIED: 15 exact app answer prompts from original FAISS baseline.')
    print('Reference answers and evidence labels are absent from model inputs.')
    print('Prompts:',prompt_path.resolve())
    if not args.execute:
        print('DRY RUN: prompts saved; no API requests, new embeddings or rewrites.');return
    from dotenv import load_dotenv
    from google.genai import types
    import app
    load_dotenv(project/'.env',encoding='utf-8-sig',override=True)
    answers={};state_path=store.folder/'request_state.json'
    with app.make_client() as client:
        for item in prompts:
            cid=item['id'];request={'model':item['model'],'prompt':item['prompt'],'config':item['config']}
            def produce():
                last=frozen.read(state_path)['last_start'] if state_path.exists() else 0
                delay=pacing_delay(last,time.time())
                if delay:
                    print(f'{cid}: pacing for {delay:.1f}s...',flush=True);time.sleep(delay)
                frozen.save(state_path,{'last_start':time.time()})
                print(cid+': requesting answer...',flush=True)
                response=client.models.generate_content(model=item['model'],contents=item['prompt'],
                    config=types.GenerateContentConfig(**item['config']))
                return response_value(response)
            answers[cid]=store.cached(cid+'-answer',request,produce)
            print(cid+': ready (saved or reused)',flush=True)
    export_review(store.folder,corpus,cases,prompts,answers,binding)
    print('COMPLETE: 15 answers, review_packet.md and pending answer_reviews.json saved.')
    print('Output:',store.folder.resolve())
    print('No answer-quality score assigned. App, baseline and frozen data unchanged.')


if __name__=='__main__':main()
