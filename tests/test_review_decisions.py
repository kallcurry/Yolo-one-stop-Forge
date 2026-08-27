import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.models.annotation_review import ReviewIssue, default_pose_review_config
from app.models.review_decisions import ReviewDecisionStore


def _issue(rule='suspected_left_right_swap', shape=1):
    return ReviewIssue(
        rule=rule,
        severity='warning',
        message='疑似左右反标',
        group_id=0,
        label='left_knee',
        shape_indices=[shape],
        point_indices=[(shape, 0)],
    )


class ReviewDecisionStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'images' / 'batch' / 'frame.jpg'
        self.annotation = self.root / 'annotations' / 'batch' / 'frame.json'
        self.image.parent.mkdir(parents=True)
        self.annotation.parent.mkdir(parents=True)
        self.image.write_bytes(b'image')
        self.annotation.write_text('{"shapes": []}', encoding='utf-8')
        self.config = default_pose_review_config()

    def test_accept_file_persists_and_filters_current_issues(self):
        store = ReviewDecisionStore(self.root)
        issues = [_issue(shape=1), _issue('duplicate_keypoint', shape=2)]

        self.assertEqual(
            store.accept(self.image, self.annotation, self.config, issues), 2
        )
        reloaded = ReviewDecisionStore(self.root)
        result = reloaded.evaluate(
            self.image, self.annotation, self.config, issues
        )

        self.assertTrue(result.manually_passed)
        self.assertEqual(len(result.accepted_issues), 2)
        self.assertEqual(len(result.active_issues), 0)
        payload = json.loads(reloaded.path.read_text(encoding='utf-8'))
        record = next(iter(payload['decisions'].values()))
        self.assertEqual(record['image'], 'images/batch/frame.jpg')

    def test_accept_one_issue_leaves_other_issue_active(self):
        store = ReviewDecisionStore(self.root)
        first = _issue(shape=1)
        second = _issue('duplicate_keypoint', shape=2)
        store.accept(
            self.image, self.annotation, self.config, [first], scope='issue'
        )

        result = store.evaluate(
            self.image, self.annotation, self.config, [first, second]
        )

        self.assertEqual(result.accepted_issues, (first,))
        self.assertEqual(result.active_issues, (second,))
        self.assertFalse(result.manually_passed)

    def test_annotation_change_invalidates_previous_decision(self):
        store = ReviewDecisionStore(self.root)
        issue = _issue()
        store.accept(self.image, self.annotation, self.config, [issue])
        self.annotation.write_text('{"shapes": [{"label": "x"}]}', encoding='utf-8')

        result = store.evaluate(
            self.image, self.annotation, self.config, [issue]
        )

        self.assertEqual(result.active_issues, (issue,))
        self.assertEqual(result.accepted_issues, ())
        self.assertTrue(result.stale)

    def test_template_change_invalidates_previous_decision(self):
        store = ReviewDecisionStore(self.root)
        issue = _issue()
        store.accept(self.image, self.annotation, self.config, [issue])
        changed_config = replace(
            self.config,
            thresholds={**self.config.thresholds, 'outside_tolerance': 99.0},
        )

        result = store.evaluate(
            self.image, self.annotation, changed_config, [issue]
        )

        self.assertEqual(result.active_issues, (issue,))
        self.assertEqual(result.accepted_issues, ())
        self.assertTrue(result.stale)

    def test_revoke_restores_automatic_issues(self):
        store = ReviewDecisionStore(self.root)
        issue = _issue()
        store.accept(self.image, self.annotation, self.config, [issue])

        self.assertTrue(store.revoke(
            self.image, self.annotation, self.config
        ))
        result = store.evaluate(
            self.image, self.annotation, self.config, [issue]
        )
        self.assertEqual(result.active_issues, (issue,))
        self.assertFalse(result.manually_passed)


if __name__ == '__main__':
    unittest.main()
