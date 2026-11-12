"""Validate all 15 saved human reviews and score the development answer baseline.
No API/model calls, no automatic human approvals, and no input-file changes.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import run_baseline_answers as runner

f=runner.frozen
CRITERIA=runner.CRITERIA


def unique_map(rows,expected,label):
    f.require(isinstance(rows,list) and len(rows)==len(expected),label+': incomplete/extra rows.')
    f.require(all(isinstance(row,dict) and isinstance(row.get('id'),str) for row in rows),label+': malformed row.')
    mapping={row['id']:row for row in rows}
    f.require(len(mapping)==len(rows) and set(mapping)==set(expected),label+': duplicate, missing or unknown IDs.')
    return mapping


def validate_reviews(cases,prompts,answers,binding,review):
    ids=[c['id'] for c in cases]
    f.require(len(ids)==len(set(ids)),'Duplicate case IDs.')
    prompt_map=unique_map(prompts,ids,'Prompts')
    answer_map=unique_map(answers['cases'],ids,'Answers')
    f.require(answers['binding']==binding,'Aggregate answer binding differs.')
    f.require(review['binding_sha256']==f.fingerprint(binding),'Review binding differs.')
    reviews=unique_map(review['reviews'],ids,'Reviews')
    verified=[]
    for case in cases:
        cid=case['id'];row=reviews[cid];prompt=prompt_map[cid]
        answer={k:v for k,v in answer_map[cid].items() if k!='id'}
        f.require(isinstance(answer.get('answer'),str) and answer['answer'].strip(),cid+': empty answer.')
        f.require(row['answer_sha256']==f.fingerprint(answer),cid+': reviewed answer hash differs.')
        f.require(row['prompt_sha256']==f.fingerprint(prompt),cid+': reviewed prompt hash differs.')
        f.require(row['review_status']=='approved',cid+': review still pending or not approved.')
        for field in ['reviewer','reviewed_at','rationale']:
            f.require(isinstance(row.get(field),str) and row[field].strip(),cid+': missing '+field)
        date=datetime.fromisoformat(row['reviewed_at'])
        f.require(date.tzinfo is not None,cid+': review timestamp lacks timezone.')
        scores=row['scores']
        f.require(isinstance(scores,dict) and set(scores)==set(CRITERIA),cid+': incorrect criterion fields.')
        f.require(all(type(value) is bool for value in scores.values()),cid+': every score must be true/false; no null/numeric/string scores.')
        f.require(type(case['answerable']) is bool,'Invalid answerability.')
        verified.append({'id':cid,'category':case['category'],'answerable':case['answerable'],
            'scores':scores,'all_criteria_pass':all(scores.values()),
            'failed_criteria':[name for name in CRITERIA if not scores[name]],
            'reviewer':row['reviewer'],'reviewed_at':row['reviewed_at'],'rationale':row['rationale'],
            'answer_sha256':row['answer_sha256'],'prompt_sha256':row['prompt_sha256'],
            'finish_reason':answer.get('finish_reason')})
    return verified


def summarize(rows):
    f.require(bool(rows),'No reviewed cases.')
    count=len(rows)
    return {'case_count':count,
        'criteria':{name:{'passed':sum(row['scores'][name] for row in rows),'total':count,
                          'rate':sum(row['scores'][name] for row in rows)/count} for name in CRITERIA},
        'all_criteria':{'passed':sum(row['all_criteria_pass'] for row in rows),'total':count,
                        'rate':sum(row['all_criteria_pass'] for row in rows)/count}}


def load_and_validate(data,baseline,folder,project):
    folder,project,baseline=Path(folder),Path(project),Path(baseline)
    corpus,cases,_,_,retrievals,baseline_report=f.preflight(data,baseline,project)
    expected_prompts=runner.prepare_prompts(project,corpus,cases,retrievals)
    binding=f.read(folder/'run_manifest.json');prompts=f.read(folder/'prompts.json')
    f.require(prompts==expected_prompts,'Prompts differ from exact original baseline app prompts.')
    f.require(binding['dataset_sha256']==f.DATASET and binding['model']==runner.CHAT_MODEL,'Answer dataset/model differs.')
    f.require(binding['prompt_list_sha256']==f.fingerprint(prompts),'Prompt-list hash differs.')
    f.require(binding['baseline_report_sha256']==f.sha((baseline/'retrieval_report.json').read_bytes()),'Baseline report changed.')
    for name,key in [('app.py','app_sha256'),('run_baseline_answers.py','script_sha256'),
                     ('run_saved_reranking.py','validation_helper_sha256'),('run_app_retrieval_baseline.py','store_helper_sha256')]:
        f.require(binding[key]==f.sha((project/name).read_bytes()),'Bound source changed: '+name)
    f.require(binding['max_output_tokens']==1000 and binding['request_interval_seconds']==15,'Generation settings differ.')
    answers=f.read(folder/'answers.json');review=f.read(folder/'answer_reviews.json')
    verified=validate_reviews(cases,prompts,answers,binding,review)
    answer_map={a['id']:{k:v for k,v in a.items() if k!='id'} for a in answers['cases']}
    # Reconcile aggregate answers against each original atomic API checkpoint.
    for item in prompts:
        payload=f.checked_cache(folder,item['id']+'-answer',binding,
            {'model':item['model'],'prompt':item['prompt'],'config':item['config']})
        f.require(payload==answer_map[item['id']],'Aggregate differs from original saved response: '+item['id'])
    f.require(len(verified)==15 and sum(row['answerable'] for row in verified)==12,'Expected 12 answerable and 3 missing-information cases.')
    return verified,binding,baseline_report


def report_markdown(report):
    summary=report['overall']
    lines=['# Original FAISS baseline — human-reviewed development answers',
        'This report summarizes 15 human-reviewed development cases on two familiar research PDFs. It is not held-out or general accuracy.',
        '| Criterion | Passed | Rate |','|---|---:|---:|']
    for name,value in summary['criteria'].items():
        lines.append(f"| {name} | {value['passed']}/{value['total']} | {value['rate']:.2%} |")
    strict=summary['all_criteria']
    lines.extend([f"| All five criteria | {strict['passed']}/{strict['total']} | {strict['rate']:.2%} |",'',
        '## Missing-information behavior','',
        f"Appropriate abstention: {report['missing_information']['criteria']['appropriate_abstention']['passed']}/3. This is separate from passing every criterion, including citation formatting.",
        '', '## Case decisions','', '| Case | All five pass | Failed criteria |','|---|---|---|'])
    for row in report['cases']:
        lines.append(f"| {row['id']} | {'Yes' if row['all_criteria_pass'] else 'No'} | {', '.join(row['failed_criteria']) or 'None'} |")
    lines.extend(['','## Interpretation limits','',
        '- Correctness measures accuracy of stated claims; incompleteness is reported separately.',
        '- Citation support uses the agreed strict filename/page requirement, including formatting failures.',
        '- Safe partial abstention may pass groundedness/appropriate abstention while failing completeness.',
        '- Missing-information cases are included in answer review, but excluded from retrieval metrics.',
        '- MP-DEV-001 has equivalent supporting evidence outside its frozen labeled chunk. Its retrieval label/score was not changed.',
        '- This records one person’s reviewed decisions; it does not establish independent inter-rater agreement.',
        '- No baseline promotion, CI enforcement, service latency, caching benefit or held-out claim is assigned.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='private_data/multi_pdf_approved_dev_v1')
    parser.add_argument('--baseline',default='private_data/multi_pdf_faiss_baseline_v1')
    parser.add_argument('--answers',default='private_data/multi_pdf_baseline_answers_v1')
    parser.add_argument('--outdir',default='private_data/multi_pdf_baseline_answer_scores_v1')
    args=parser.parse_args();project=Path(__file__).resolve().parent;out=Path(args.outdir)
    f.require(not out.exists(),'Score output already exists; preserve it and choose a new --outdir.')
    rows,binding,baseline=load_and_validate(args.data,args.baseline,args.answers,project)
    answerable=[row for row in rows if row['answerable']];missing=[row for row in rows if not row['answerable']]
    folder=Path(args.answers)
    report={'schema_version':1,'scope':'Single-reviewer development answer evaluation; no held-out claims.',
        'dataset_sha256':f.DATASET,'answer_run_binding_sha256':f.fingerprint(binding),
        'input_files_sha256':{name:f.sha((folder/name).read_bytes()) for name in ['run_manifest.json','prompts.json','answers.json','answer_reviews.json']},
        'scorer_sha256':f.sha(Path(__file__).read_bytes()),'overall':summarize(rows),
        'answerable':summarize(answerable),'missing_information':summarize(missing),
        'by_category':{category:summarize([row for row in rows if row['category']==category]) for category in sorted({row['category'] for row in rows})},
        'retrieval_metrics_for_context_only':baseline['metrics'],'cases':rows,
        'finish_reason_warnings':[row['id'] for row in rows if row['finish_reason'] not in ['STOP','FinishReason.STOP']],
        'scored_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'limitations':['15 development cases; two familiar research PDFs; one human reviewer.',
            'Criterion passes are not interchangeable with complete/cited end-to-end success.',
            'No API calls; input answers, reviews and frozen retrieval labels unchanged.']}
    # All validation completes before any score files are created.
    out.mkdir(parents=True)
    f.save(out/'human_review_scores.json',report)
    (out/'answer_quality_report.md').write_text(report_markdown(report),encoding='utf-8')
    print('VERIFIED: 15 approved reviews; prompt, answer, source and saved-response hashes match.')
    print('\nCriterion                    Passed      Rate')
    for name,value in report['overall']['criteria'].items():
        print(f"{name:28} {value['passed']:2}/{value['total']}       {value['rate']:.2%}")
    strict=report['overall']['all_criteria']
    print(f"All five criteria            {strict['passed']:2}/{strict['total']}       {strict['rate']:.2%}")
    print('Missing-information appropriate abstention:',report['missing_information']['criteria']['appropriate_abstention']['passed'],'/ 3')
    print('Reports:',out.resolve())
    print('NO API REQUESTS. Development-only; input files unchanged.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,OSError) as error:raise SystemExit('INVALID REVIEW: '+str(error))
