"""Controlled citation-instruction candidate using original saved FAISS context.
Default: validate and save candidate prompts without API requests.
--execute: generate 15 candidate answers; no embeddings, rewrites or retrieval.
"""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import time

import score_baseline_reviews as scoring

runner=scoring.runner
f=scoring.f
CITATION_RULES='''Citation requirements for this answer:
- Cite each factual answer claim using the exact source filename and PDF page printed in the reference-passage headers.
- Use this format: (<exact source filename>, page <PDF page number>).
- A passage number alone is not a source citation. Do not replace the filename with a passage number.
- Cite only filename/page pairs present in the supplied reference headers, and only when the cited passage supports the claim.
- If a claim needs several sources, cite the supporting filename/page pairs separately. Do not add unsupported pages or page ranges.
- A brief statement that the requested information is absent from the supplied context needs no invented citation. Any additional factual explanation still needs its supporting filename/page citation.

'''


def citation_prompts(original):
    f.require(isinstance(original,list) and bool(original),'Missing baseline prompts.')
    result=[];seen=set()
    for item in original:
        f.require(item['id'] not in seen,'Duplicate baseline prompt ID.');seen.add(item['id'])
        f.require(item['model']==runner.CHAT_MODEL and item['config']=={'max_output_tokens':1000},'Baseline generation configuration differs.')
        f.require(isinstance(item['prompt'],str) and item['prompt'].strip(),'Empty original prompt.')
        f.require(len(item['retrieved_ids'])==6 and len(set(item['retrieved_ids']))==6,'Expected six distinct original passages.')
        result.append({**item,'prompt':CITATION_RULES+item['prompt'],
            'baseline_prompt_sha256':f.fingerprint(item)})
    return result


def verify_prompt_only_change(original,candidate):
    f.require(len(original)==len(candidate),'Candidate prompt count changed.')
    for old,new in zip(original,candidate):
        f.require(new['prompt']==CITATION_RULES+old['prompt'],'Unexpected changes beyond citation prefix.')
        f.require(new['baseline_prompt_sha256']==f.fingerprint(old),'Original prompt hash differs.')
        reconstructed={k:v for k,v in new.items() if k!='baseline_prompt_sha256'}
        reconstructed['prompt']=old['prompt']
        f.require(reconstructed==old,'Question/context/model/configuration/IDs changed.')


def verified_baseline(data,retrieval,answers,scores,project):
    answers,scores,project=Path(answers),Path(scores),Path(project)
    rows,binding,baseline=scoring.load_and_validate(data,retrieval,answers,project)
    report=f.read(scores/'human_review_scores.json')
    f.require(report['dataset_sha256']==f.DATASET,'Scored dataset differs.')
    f.require(report['answer_run_binding_sha256']==f.fingerprint(binding),'Scored answer run differs.')
    f.require(report['scorer_sha256']==f.sha((project/'score_baseline_reviews.py').read_bytes()),'Original scoring script changed.')
    for name,digest in report['input_files_sha256'].items():
        f.require(name in {'run_manifest.json','prompts.json','answers.json','answer_reviews.json'},'Unexpected baseline score input path.')
        f.require(f.sha((answers/name).read_bytes())==digest,'Scored baseline input changed: '+name)
    f.require(set(report['input_files_sha256'])=={'run_manifest.json','prompts.json','answers.json','answer_reviews.json'},'Score report lacks input hashes.')
    f.require(report['cases']==rows and report['overall']==scoring.summarize(rows),'Baseline score summary differs from approved reviews.')
    return binding,baseline,report


def safe_output(output,protected):
    output=Path(output).resolve()
    for path in protected:
        path=Path(path).resolve()
        f.require(not (output==path or output in path.parents or path in output.parents),
                  'Candidate output overlaps a protected baseline/data folder.')
    return output


def prepare(data,retrieval,answers,scores,output,project):
    project=Path(project)
    output=safe_output(output,[data,retrieval,answers,scores])
    original_binding,baseline,report=verified_baseline(data,retrieval,answers,scores,project)
    original=f.read(Path(answers)/'prompts.json')
    prompts=citation_prompts(original);verify_prompt_only_change(original,prompts)
    corpus=f.read(Path(data)/'corpus.json');cases=f.read(Path(data)/'cases.json')
    f.require(len(prompts)==len(cases)==15,'Expected 15 candidate cases.')
    binding={'schema_version':1,'experiment':'citation_instructions_only_v1','dataset_sha256':f.DATASET,
        'model':runner.CHAT_MODEL,'max_output_tokens':1000,'request_interval_seconds':15,
        'citation_rules_sha256':f.sha(CITATION_RULES.encode()),'prompt_list_sha256':f.fingerprint(prompts),
        'original_answer_binding_sha256':f.fingerprint(original_binding),
        'original_answer_scores_sha256':f.sha((Path(scores)/'human_review_scores.json').read_bytes()),
        'original_prompts_sha256':f.sha((Path(answers)/'prompts.json').read_bytes()),
        'original_retrieval_report_sha256':f.sha((Path(retrieval)/'retrieval_report.json').read_bytes()),
        'script_sha256':f.sha(Path(__file__).read_bytes()),
        'source_helpers_sha256':{name:f.sha((project/name).read_bytes()) for name in
            ['app.py','score_baseline_reviews.py','run_baseline_answers.py','run_saved_reranking.py','run_app_retrieval_baseline.py']},
        'packages':{name:importlib.metadata.version(name) for name in ['google-genai','python-dotenv']},
        'python':platform.python_version(),'unchanged':['original questions','history','standalone questions',
            'six passage texts and order','model ID','max_output_tokens','provider decoding defaults'],
        'limitations':['One candidate generation per familiar development case; no causal/significance claim.',
            'Provider model ID may resolve to updated weights; returned model_version is recorded.',
            'Citation formatting alone does not prove semantic citation support. Fresh human review required.']}
    store=runner.Store(output,binding)
    p=output/'prompts.json'
    if p.exists():f.require(f.read(p)==prompts,'Saved candidate prompts changed.')
    else:f.save(p,prompts)
    note=output/'experiment_notes.md'
    if not note.exists():
        note.write_text('# Citation-instruction candidate v1\n\n'
            'Only a fixed citation instruction prefix is added to the exact original model inputs. '
            'All retrieval, questions, history and generation settings remain fixed. '
            'Outputs require fresh manual review; baseline decisions are never copied.\n\n'
            'This is a single-run development experiment, not held-out validation or a significance test. '
            'The provider model alias is fixed but immutable provider weights are unavailable.\n\n'
            '## Added instructions\n\n'+CITATION_RULES,encoding='utf-8')
    return store,prompts,corpus,cases,report


def version_audit(original_answers,candidate_answers):
    old={a['id']:a.get('model_version') for a in original_answers['cases']}
    return [{'id':cid,'baseline_model_version':old[cid],'candidate_model_version':answer.get('model_version'),
             'same_returned_model_version':old[cid]==answer.get('model_version') if old[cid] and answer.get('model_version') else None}
            for cid,answer in candidate_answers.items()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='private_data/multi_pdf_approved_dev_v1')
    parser.add_argument('--retrieval',default='private_data/multi_pdf_faiss_baseline_v1')
    parser.add_argument('--baseline-answers',default='private_data/multi_pdf_baseline_answers_v1')
    parser.add_argument('--baseline-scores',default='private_data/multi_pdf_baseline_answer_scores_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_citation_candidate_v1')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();project=Path(__file__).resolve().parent
    store,prompts,corpus,cases,baseline=prepare(args.data,args.retrieval,args.baseline_answers,
        args.baseline_scores,args.outdir,project)
    print('VERIFIED: all 15 approved baseline reviews, answers, prompts and source bindings.')
    print('VERIFIED: only citation instructions added; original six passages/order/model/config retained.')
    print('BASELINE: citation support',baseline['overall']['criteria']['citation_support']['passed'],'/ 15;',
        'all-five success',baseline['overall']['all_criteria']['passed'],'/ 15.')
    print('Candidate prompts:',(store.folder/'prompts.json').resolve())
    if not args.execute:
        print('DRY RUN: candidate prompts saved; no API requests, embeddings, retrieval or rewrites.');return
    from dotenv import load_dotenv
    from google.genai import types
    import app
    load_dotenv(project/'.env',encoding='utf-8-sig',override=True)
    answers={};state=store.folder/'request_state.json'
    with app.make_client() as client:
        for item in prompts:
            cid=item['id'];request={'model':item['model'],'prompt':item['prompt'],'config':item['config']}
            def produce():
                last=f.read(state)['last_start'] if state.exists() else 0
                delay=runner.pacing_delay(last,time.time())
                if delay:
                    print(f'{cid}: pacing for {delay:.1f}s...',flush=True);time.sleep(delay)
                f.save(state,{'last_start':time.time()})
                print(cid+': requesting candidate answer...',flush=True)
                response=client.models.generate_content(model=item['model'],contents=item['prompt'],
                    config=types.GenerateContentConfig(**item['config']))
                return runner.response_value(response)
            answers[cid]=store.cached(cid+'-answer',request,produce)
            print(cid+': candidate ready (saved or reused)',flush=True)
    runner.export_review(store.folder,corpus,cases,prompts,answers,store.binding)
    packet=store.folder/'review_packet.md'
    text=packet.read_text(encoding='utf-8')
    text=text.replace('# Original FAISS baseline — development answer review',
        '# Citation-instruction candidate v1 — development answer review',1)
    packet.write_text(text,encoding='utf-8')
    versions=version_audit(f.read(Path(args.baseline_answers)/'answers.json'),answers)
    f.save(store.folder/'model_version_audit.json',versions)
    changed=[r['id'] for r in versions if r['same_returned_model_version'] is False]
    unknown=[r['id'] for r in versions if r['same_returned_model_version'] is None]
    print('COMPLETE: 15 candidate answers; fresh pending answer_reviews.json and review_packet.md saved.')
    print('Returned model-version differences:',changed or 'None')
    print('Missing model-version metadata:',unknown or 'None')
    print('Output:',store.folder.resolve())
    print('No candidate answer scores assigned. Original app, baseline and frozen data unchanged.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,OSError) as error:raise SystemExit('CITATION EXPERIMENT STOPPED: '+str(error))
