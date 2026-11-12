"""Offline citation review comparison. Default requires the full local project.
--package PATH audits an exported review package with explicitly limited scope.
No model/API calls. Existing reports and input files are never overwritten.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

CRITERIA = ('correctness', 'completeness', 'groundedness', 'citation_support', 'appropriate_abstention')
IDS = [f'MP-DEV-{i:03}' for i in range(1, 16)]
DATASET = 'a19ba7d8e9aa5abe61d777fbfffc3f1ea9417f4542c84abd9a12080f5731178e'
HELPERS = {'app.py', 'score_baseline_reviews.py', 'run_baseline_answers.py',
           'run_saved_reranking.py', 'run_app_retrieval_baseline.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def reject_constant(value):
    raise ValueError('Non-finite JSON value: ' + value)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=reject_constant, object_pairs_hook=no_duplicate_keys)


def rows_by_id(rows, label):
    require(isinstance(rows, list) and len(rows) == 15, label + ': expected 15 rows.')
    require(all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in rows), label + ': malformed rows.')
    result = {r['id']: r for r in rows}
    require(len(result) == 15 and set(result) == set(IDS), label + ': duplicate/missing/unknown IDs.')
    return result


def check_scores(row):
    require(row.get('review_status', 'approved') == 'approved', row['id'] + ': review not approved.')
    for field in ('reviewer', 'reviewed_at', 'rationale'):
        require(isinstance(row.get(field), str) and row[field].strip(), row['id'] + ': missing ' + field)
    require(datetime.fromisoformat(row['reviewed_at']).tzinfo is not None, row['id'] + ': timezone missing.')
    scores = row['scores']
    require(isinstance(scores, dict) and set(scores) == set(CRITERIA), row['id'] + ': criterion fields differ.')
    require(all(type(v) is bool for v in scores.values()), row['id'] + ': scores must be booleans.')


def summarize(rows):
    require(bool(rows), 'Empty score group.')
    total = len(rows)
    return {'case_count': total, 'criteria': {
        name: {'passed': sum(r['scores'][name] for r in rows), 'total': total,
               'rate': sum(r['scores'][name] for r in rows) / total} for name in CRITERIA},
        'all_criteria': {'passed': sum(all(r['scores'].values()) for r in rows), 'total': total,
                         'rate': sum(all(r['scores'].values()) for r in rows) / total}}


def citation_rules(script):
    tree = ast.parse(Path(script).read_text(encoding='utf-8-sig'))
    found = [ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CITATION_RULES' for t in node.targets)]
    require(len(found) == 1 and isinstance(found[0], str) and bool(found[0]), 'Missing unique citation prefix.')
    return found[0]


def compare(old, new):
    before, after = summarize(old), summarize(new)
    changes = []
    old_map = {r['id']: r for r in old}
    for row in new:
        previous = old_map[row['id']]
        changes.append({'id': row['id'], 'improved_criteria': [n for n in CRITERIA if not previous['scores'][n] and row['scores'][n]],
                        'regressed_criteria': [n for n in CRITERIA if previous['scores'][n] and not row['scores'][n]],
                        'baseline_all_five': all(previous['scores'].values()),
                        'candidate_all_five': all(row['scores'].values())})
    regressions = [n for n in CRITERIA if after['criteria'][n]['passed'] < before['criteria'][n]['passed']]
    return {'baseline': before, 'candidate': after, 'case_changes': changes,
            'aggregate_regressed_criteria': regressions,
            'citation_pass_count_increased': after['criteria']['citation_support']['passed'] > before['criteria']['citation_support']['passed'],
            'decision': 'Do not promote: completeness regressed.' if 'completeness' in regressions else
                        ('Do not promote: one or more criteria regressed.' if regressions else 'Review paired outcomes before any promotion.'),
            'gate_scope': 'Development comparison only; no CI gate, deployment or held-out claim.'}


def audit(candidate, baseline_score, generator, scorer):
    candidate = Path(candidate)
    binding = read(candidate / 'run_manifest.json')
    baseline = read(baseline_score)
    require(binding['experiment'] == 'citation_instructions_only_v1', 'Unexpected experiment.')
    require(binding['dataset_sha256'] == baseline['dataset_sha256'] == DATASET, 'Dataset differs.')
    require(binding['model'] == 'gemini-3.5-flash-lite' and binding['max_output_tokens'] == 1000
            and binding['request_interval_seconds'] == 15, 'Generation settings differ.')
    require(binding['original_answer_scores_sha256'] == sha(Path(baseline_score).read_bytes()), 'Baseline score report changed.')
    require(binding['original_answer_binding_sha256'] == baseline['answer_run_binding_sha256'], 'Baseline answer binding differs.')
    require(binding['script_sha256'] == sha(Path(generator).read_bytes()), 'Candidate generator changed.')
    require(set(binding['source_helpers_sha256']) == HELPERS, 'Helper source inventory differs.')
    require(binding['source_helpers_sha256']['score_baseline_reviews.py'] == sha(Path(scorer).read_bytes()) == baseline['scorer_sha256'], 'Baseline scorer changed.')
    rules = citation_rules(generator)
    require(sha(rules.encode()) == binding['citation_rules_sha256'], 'Citation instructions differ.')
    prompts = read(candidate / 'prompts.json')
    require(fingerprint(prompts) == binding['prompt_list_sha256'], 'Prompt list changed.')
    pm = rows_by_id(prompts, 'Prompts')
    answers = read(candidate / 'answers.json')
    require(answers['binding'] == binding, 'Answer manifest differs.')
    am = rows_by_id(answers['cases'], 'Answers')
    review = read(candidate / 'answer_reviews.json')
    require(review['binding_sha256'] == fingerprint(binding), 'Review manifest differs.')
    rm = rows_by_id(review['reviews'], 'Reviews')
    bm = rows_by_id(baseline['cases'], 'Baseline scores')
    versions = rows_by_id(read(candidate / 'model_version_audit.json'), 'Model versions')
    rows = []
    for cid in IDS:
        item, row, old = pm[cid], rm[cid], bm[cid]
        require(type(old['answerable']) is bool, cid + ': invalid answerability.')
        require(old['answerable'] == (int(cid[-3:]) <= 12), cid + ': unexpected answerability.')
        check_scores(old)
        require(old['all_criteria_pass'] == all(old['scores'].values()) and old['failed_criteria'] == [n for n in CRITERIA if not old['scores'][n]], cid + ': inconsistent baseline row.')
        require(isinstance(item['prompt'], str) and item['prompt'].startswith(rules), cid + ': citation prefix differs.')
        original = {k: v for k, v in item.items() if k != 'baseline_prompt_sha256'}
        original['prompt'] = item['prompt'][len(rules):]
        require(fingerprint(original) == item['baseline_prompt_sha256'] == old['prompt_sha256'], cid + ': original prompt differs from reviewed baseline.')
        require(item['model'] == binding['model'] and item['config'] == {'max_output_tokens': 1000}, cid + ': prompt configuration differs.')
        require(len(item['retrieved_ids']) == len(set(item['retrieved_ids'])) == 6, cid + ': expected six distinct passages.')
        answer = {k: v for k, v in am[cid].items() if k != 'id'}
        require(isinstance(answer.get('answer'), str) and bool(answer['answer'].strip()), cid + ': empty answer.')
        checkpoint = read(candidate / (cid + '-answer.json'))
        request = {'model': item['model'], 'prompt': item['prompt'], 'config': item['config']}
        require(checkpoint['key'] == fingerprint({'run': binding, 'request': request}), cid + ': checkpoint request binding differs.')
        require(checkpoint['value'] == answer and checkpoint['value_sha256'] == fingerprint(answer), cid + ': saved response differs.')
        require(row['answer_sha256'] == fingerprint(answer) and row['prompt_sha256'] == fingerprint(item), cid + ': reviewed answer/prompt changed.')
        require('review_status' in row, cid + ': missing approval field.')
        check_scores(row)
        version = versions[cid]
        require(version['candidate_model_version'] == answer.get('model_version'), cid + ': model audit differs.')
        a, b = version['baseline_model_version'], version['candidate_model_version']
        expected = a == b if a and b else None
        require(version['same_returned_model_version'] is expected, cid + ': invalid version audit.')
        rows.append({**row, 'category': old['category'], 'answerable': old['answerable'],
                     'all_criteria_pass': all(row['scores'].values()),
                     'failed_criteria': [n for n in CRITERIA if not row['scores'][n]],
                     'finish_reason': answer.get('finish_reason')})
    old_rows = [bm[cid] for cid in IDS]
    require(baseline['overall'] == summarize(old_rows), 'Baseline overall summary differs.')
    for key, subset in [('answerable', [r for r in old_rows if r['answerable']]),
                        ('missing_information', [r for r in old_rows if not r['answerable']])]:
        require(baseline[key] == summarize(subset), 'Baseline subgroup summary differs.')
    report = {'schema_version': 1, 'validation_scope': 'Exported-package consistency audit; original source/data files not included.',
              'dataset_sha256': DATASET, 'candidate_run_binding_sha256': fingerprint(binding),
              'comparison': compare(old_rows, rows), 'cases': rows,
              'missing_information': summarize([r for r in rows if not r['answerable']]),
              'model_versions': [versions[cid] for cid in IDS],
              'finish_reason_warnings': [r['id'] for r in rows if r['finish_reason'] not in ('STOP', 'FinishReason.STOP')],
              'input_files_sha256': {p.name: sha(p.read_bytes()) for p in sorted(candidate.glob('*.json')) if p.name != 'request_state.json'},
              'baseline_scores_sha256': sha(Path(baseline_score).read_bytes()),
              'comparison_scorer_sha256': sha(Path(__file__).read_bytes()),
              'limitations': ['15 familiar development cases; one human reviewer; single generation per case.',
                  'Hashes bind saved artifacts; they do not independently prove the reviewer judgments.',
                  'No causal, statistical significance, held-out, latency or service performance claim.',
                  'Returned model versions are metadata, not proof of immutable provider weights.'],
              'scored_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    return report


def full_project(project, candidate, baseline_score):
    project = Path(project).resolve()
    sys.path.insert(0, str(project))
    import run_citation_experiment as experiment
    expected_script = project / 'run_citation_experiment.py'
    require(Path(experiment.__file__).resolve() == expected_script, 'Imported generator from unexpected location.')
    # This path revalidates the original frozen corpus, retrieval caches, baseline
    # prompts, original answer checkpoints, reviews and baseline score report.
    original = project / 'private_data/multi_pdf_baseline_answers_v1'
    retrieval = project / 'private_data/multi_pdf_faiss_baseline_v1'
    data = project / 'private_data/multi_pdf_approved_dev_v1'
    original_binding, _, baseline = experiment.verified_baseline(data, retrieval, original, baseline_score.parent, project)
    binding = read(candidate / 'run_manifest.json')
    require(binding['original_answer_binding_sha256'] == fingerprint(original_binding), 'Original binding changed.')
    for name, digest in binding['source_helpers_sha256'].items():
        require(sha((project / name).read_bytes()) == digest, 'Bound source changed: ' + name)
    require(sha((original / 'prompts.json').read_bytes()) == binding['original_prompts_sha256'], 'Original prompt file changed.')
    require(sha((retrieval / 'retrieval_report.json').read_bytes()) == binding['original_retrieval_report_sha256'], 'Retrieval changed.')
    prompts = read(candidate / 'prompts.json')
    experiment.verify_prompt_only_change(read(original / 'prompts.json'), prompts)
    report = audit(candidate, baseline_score, expected_script, project / 'score_baseline_reviews.py')
    original_answers = {r['id']: r for r in read(original / 'answers.json')['cases']}
    for row in report['model_versions']:
        require(row['baseline_model_version'] == original_answers[row['id']].get('model_version'), 'Original model metadata changed.')
    report['validation_scope'] = 'Full local project validation: frozen corpus, source, retrieval, both answer runs and approved reviews.'
    return report


def markdown(report):
    c = report['comparison']
    lines = ['# Citation-instruction development comparison', '', report['validation_scope'], '',
             '| Criterion | Baseline | Candidate | Change in passes |', '|---|---:|---:|---:|']
    for name in CRITERIA:
        a, b = c['baseline']['criteria'][name]['passed'], c['candidate']['criteria'][name]['passed']
        lines.append(f'| {name} | {a}/15 | {b}/15 | {b-a:+} |')
    a, b = c['baseline']['all_criteria']['passed'], c['candidate']['all_criteria']['passed']
    lines += [f'| All five | {a}/15 | {b}/15 | {b-a:+} |', '', c['decision'], '',
              '## Paired changes', '', '| Case | Improved criteria | Regressed criteria |', '|---|---|---|']
    for row in c['case_changes']:
        if row['improved_criteria'] or row['regressed_criteria']:
            lines.append(f"| {row['id']} | {', '.join(row['improved_criteria']) or 'None'} | {', '.join(row['regressed_criteria']) or 'None'} |")
    missing = report['missing_information']['criteria']['appropriate_abstention']
    lines += ['', f"Missing-information appropriate abstention: {missing['passed']}/{missing['total']}.", '',
              'Citation errors on 007, 009 and 014 include passage-number/PDF-page confusion. Completeness fails on 007, 008 and 010; 008 newly omits the image-relative width/height detail.', '',
              'Retain the original app prompt while addressing the observed failures in a separate development experiment. Do not change the frozen labels or saved answers.', '',
              '## Limits', ''] + ['- ' + item for item in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--outdir', type=Path, default=Path('private_data/multi_pdf_citation_comparison_v1'))
    args = parser.parse_args()
    out = args.outdir.resolve()
    require(not out.exists(), 'Output exists; choose a new --outdir.')
    if args.package:
        base = args.package.resolve()
        protected = [base]
        report = audit(base, base / 'human_review_scores.json', base / 'run_citation_experiment.py', base / 'score_baseline_reviews.py')
    else:
        project = args.project.resolve()
        base = project / 'private_data/multi_pdf_citation_candidate_v1'
        protected = [project / 'private_data' / n for n in ('multi_pdf_approved_dev_v1', 'multi_pdf_faiss_baseline_v1',
                     'multi_pdf_baseline_answers_v1', 'multi_pdf_baseline_answer_scores_v1', 'multi_pdf_citation_candidate_v1')]
        report = full_project(project, base, project / 'private_data/multi_pdf_baseline_answer_scores_v1/human_review_scores.json')
    require(all(out != p.resolve() and out not in p.resolve().parents and p.resolve() not in out.parents for p in protected), 'Output overlaps protected input.')
    # Create only after every validation succeeds. Exclusive creation prevents overwrite.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'citation_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'citation_comparison.md').write_text(markdown(report), encoding='utf-8')
    print('VERIFIED:', report['validation_scope'])
    for name in CRITERIA:
        c = report['comparison']
        print(f"{name}: {c['baseline']['criteria'][name]['passed']}/15 -> {c['candidate']['criteria'][name]['passed']}/15")
    print('All five:', c['baseline']['all_criteria']['passed'], '/15 ->', c['candidate']['all_criteria']['passed'], '/15')
    print(c['decision'])
    print('Reports:', out)
    print('NO API REQUESTS. Inputs unchanged. Development-only.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit('COMPARISON STOPPED: ' + str(error))
