import copy
from datetime import datetime, timezone
import unittest
from score_citation_reviews import (CRITERIA, IDS, check_scores, compare, fingerprint,
                                     no_duplicate_keys, reject_constant, rows_by_id, summarize)


def rows():
    return [{'id': cid, 'scores': dict.fromkeys(CRITERIA, True), 'review_status': 'approved',
             'reviewer': 'Test reviewer', 'reviewed_at': datetime.now(timezone.utc).isoformat(),
             'rationale': 'Synthetic test decision.'} for cid in IDS]


class CitationComparisonTests(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        value = rows(); value[-1]['id'] = value[0]['id']
        with self.assertRaises(ValueError): rows_by_id(value, 'test')

    def test_missing_id_rejected(self):
        with self.assertRaises(ValueError): rows_by_id(rows()[:-1], 'test')

    def test_unknown_id_rejected(self):
        value = rows(); value[-1]['id'] = 'UNKNOWN'
        with self.assertRaises(ValueError): rows_by_id(value, 'test')

    def test_pending_review_rejected(self):
        row = rows()[0]; row['review_status'] = 'pending'
        with self.assertRaises(ValueError): check_scores(row)

    def test_missing_rationale_rejected(self):
        row = rows()[0]; row['rationale'] = ''
        with self.assertRaises(ValueError): check_scores(row)

    def test_naive_review_date_rejected(self):
        row = rows()[0]; row['reviewed_at'] = '2026-11-12T00:00:00'
        with self.assertRaises(ValueError): check_scores(row)

    def test_numeric_scores_rejected(self):
        row = rows()[0]; row['scores']['correctness'] = 1
        with self.assertRaises(ValueError): check_scores(row)

    def test_null_scores_rejected(self):
        row = rows()[0]; row['scores']['correctness'] = None
        with self.assertRaises(ValueError): check_scores(row)

    def test_known_rates(self):
        value = rows(); value[0]['scores']['completeness'] = False
        result = summarize(value)
        self.assertEqual(result['all_criteria']['passed'], 14)
        self.assertEqual(result['criteria']['completeness']['rate'], 14/15)

    def test_citation_improvement_does_not_hide_completeness_regression(self):
        old = rows(); old[0]['scores']['citation_support'] = False
        new = copy.deepcopy(old); new[0]['scores']['citation_support'] = True
        new[1]['scores']['completeness'] = False
        result = compare(old, new)
        self.assertTrue(result['citation_pass_count_increased'])
        self.assertEqual(result['aggregate_regressed_criteria'], ['completeness'])
        self.assertIn('Do not promote', result['decision'])
        self.assertEqual(result['case_changes'][1]['regressed_criteria'], ['completeness'])

    def test_hash_binds_prompt_and_answer_changes(self):
        self.assertNotEqual(fingerprint({'prompt': 'a'}), fingerprint({'prompt': 'b'}))
        self.assertNotEqual(fingerprint({'answer': '45'}), fingerprint({'answer': '155'}))

    def test_nonfinite_and_duplicate_json_rejected(self):
        with self.assertRaises(ValueError): reject_constant('NaN')
        with self.assertRaises(ValueError): no_duplicate_keys([('id', 1), ('id', 2)])


if __name__ == '__main__':
    unittest.main()
