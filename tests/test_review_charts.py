import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from app.controllers.app_controller import AppController
from app.models.annotation_review import AnnotationSummary, ReviewIssue
from app.models.review_decisions import ReviewDecisionResult
from app.views.detail_panel import DetailPanel, _ReviewChartsWidget


def _folder_summary():
    return {
        'task_type': 'pose',
        'annotation_files': 10,
        'missing_annotations': 1,
        'invalid_annotations': 0,
        'issue_files': 2,
        'issue_count': 2,
        'person_boxes': 10,
        'target_class_counts': {'person_dress_middle': 8},
        'target_class_file_counts': {'person_dress_middle': 7},
        'keypoint_counts': {'nose': 9},
        'keypoint_file_counts': {'nose': 9},
        'shape_type_counts': {'point': 9, 'rectangle': 8},
        'shape_type_file_counts': {'point': 9, 'rectangle': 7},
        'rule_counts': {'duplicate_keypoint': 2},
        'severity_counts': {'error': 2},
        'metric_files': {
            'overview:images': [
                {'index': idx, 'filename': f'{idx}.jpg', 'count': 1,
                 'status': '图片', 'detail': '当前文件夹中的图片'}
                for idx in range(10)
            ],
            'keypoint:nose': [
                {'index': 3, 'filename': '3.jpg', 'count': 0,
                 'expected': 1, 'status': '缺失 1',
                 'detail': 'nose: 实际 0，期望 1'},
            ],
        },
    }


class ReviewChartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dispose_widget(self, widget):
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()

    def test_bar_click_selects_and_second_click_clears_selection(self):
        charts = _ReviewChartsWidget()
        charts.resize(900, 500)
        charts.show()
        self.addCleanup(self._dispose_widget, charts)
        selected = []
        charts.metric_selected.connect(selected.append)
        charts.set_summary(10, _folder_summary())
        self.app.processEvents()
        for chart in charts._charts:
            chart._animation.setCurrentTime(chart._animation.duration())
            chart.repaint()

        chart = charts.overview_chart
        self.assertEqual(len(chart._bar_hit_rects), len(chart._data))
        click_pos = chart._bar_hit_rects[0].center().toPoint()

        QTest.mouseClick(chart, Qt.LeftButton, pos=click_pos)
        self.assertEqual(selected[-1]['key'], 'overview:images')
        self.assertEqual(chart._selected_key, 'overview:images')

        QTest.mouseClick(chart, Qt.LeftButton, pos=click_pos)
        self.assertIsNone(selected[-1])
        self.assertIsNone(chart._selected_key)

    def test_metric_detail_lists_files_and_reuses_jump_signal(self):
        panel = DetailPanel()
        panel.resize(900, 900)
        panel.show()
        self.addCleanup(self._dispose_widget, panel)
        summary = _folder_summary()
        self.assertIn(
            '待处理图片: **3**',
            panel._build_review_report(10, summary),
        )
        panel.show_review_stats(10, [], summary)
        self.assertIn('3 / 10', panel.lbl_review_stats.text())
        self.assertEqual(panel.lbl_review_stats.property('tone'), 'warning')
        payload = {
            'key': 'keypoint:nose',
            'label': 'nose',
            'value': 9,
            'file_count': 9,
            'expected': 10,
            'kind': 'keypoint',
        }

        panel._on_chart_metric_selected(payload)

        self.assertEqual(panel.review_results_tabs.currentIndex(), 1)
        self.assertEqual(panel.metric_detail_tree.topLevelItemCount(), 1)
        self.assertIn('缺失 1', panel.metric_detail_tree.topLevelItem(0).text(1))
        jumped = []
        panel.review_file_selected.connect(jumped.append)
        panel._on_review_stats_item_activated(
            panel.metric_detail_tree.topLevelItem(0), 0
        )
        self.assertEqual(jumped, [3])

        panel._on_chart_metric_selected(None)
        self.assertEqual(panel.review_results_tabs.currentIndex(), 0)
        self.assertFalse(panel.review_results_tabs.isTabEnabled(1))

    def test_metric_detail_search_filters_and_opens_first_match(self):
        panel = DetailPanel()
        panel.show()
        self.addCleanup(self._dispose_widget, panel)
        summary = _folder_summary()
        panel.show_review_stats(10, [], summary)
        payload = {
            'key': 'overview:images',
            'label': '图片数量',
            'value': 10,
            'file_count': 10,
            'kind': 'images',
        }
        panel._on_chart_metric_selected(payload)

        panel.metric_detail_search.setText('7.jpg 当前文件夹')

        self.assertEqual(panel.lbl_metric_search_count.text(), '匹配 1 / 10')
        self.assertFalse(panel.metric_detail_tree.topLevelItem(7).isHidden())
        opened = []
        panel.review_file_selected.connect(opened.append)
        QTest.keyClick(panel.metric_detail_search, Qt.Key_Return)
        self.assertEqual(opened, [7])

    def test_manual_pass_rows_are_visible_and_can_be_opened(self):
        panel = DetailPanel()
        panel.show()
        self.addCleanup(self._dispose_widget, panel)
        issue = ReviewIssue(
            rule='suspected_left_right_swap', severity='warning',
            message='疑似左右反标', group_id=0, label='left_knee',
            shape_indices=[1], point_indices=[(1, 0)],
        )
        summary = _folder_summary()
        summary.update({
            'missing_annotations': 0,
            'issue_files': 0,
            'issue_count': 0,
            'manual_pass_files': 1,
            'accepted_issue_count': 1,
        })
        rows = [{
            'index': 4,
            'filename': '4.jpg',
            'issues': [],
            'accepted_issues': [issue],
            'status': 'manual',
        }]

        panel.show_review_stats(10, rows, summary)
        self.assertEqual(panel.review_stats_tree.topLevelItemCount(), 1)
        item = panel.review_stats_tree.topLevelItem(0)
        self.assertIn('人工通过', item.text(1))
        opened = []
        panel.review_file_selected.connect(opened.append)
        panel._on_review_stats_item_activated(item, 0)
        self.assertEqual(opened, [4])

    def test_problem_file_search_filters_and_opens_first_match(self):
        panel = DetailPanel()
        panel.show()
        self.addCleanup(self._dispose_widget, panel)
        duplicate = ReviewIssue(
            rule='duplicate_keypoint', severity='error',
            message='nose 出现 2 次', group_id=0, label='nose',
            shape_indices=[1], point_indices=[(1, 0)],
        )
        swapped = ReviewIssue(
            rule='suspected_left_right_swap', severity='warning',
            message='left_knee / right_knee 可能互换', group_id=0,
            label='left_knee', shape_indices=[2, 3],
            point_indices=[(2, 0), (3, 0)],
        )
        rows = [
            {
                'index': 1, 'filename': 'frame_000001.jpg',
                'issues': [duplicate], 'status': 'problem',
            },
            {
                'index': 2, 'filename': 'frame_000002.jpg',
                'issues': [swapped], 'status': 'problem',
            },
        ]
        summary = _folder_summary()
        summary.update({'issue_files': 2, 'issue_count': 2})
        panel.show_review_stats(10, rows, summary)

        panel.review_file_search.setText('000002 左右')

        self.assertTrue(panel.review_stats_tree.topLevelItem(0).isHidden())
        self.assertFalse(panel.review_stats_tree.topLevelItem(1).isHidden())
        self.assertEqual(panel.lbl_review_search_count.text(), '匹配 1 / 2')
        opened = []
        panel.review_file_selected.connect(opened.append)
        QTest.keyClick(panel.review_file_search, Qt.Key_Return)
        self.assertEqual(opened, [2])

        panel.review_file_search.clear()
        self.assertFalse(panel.review_stats_tree.topLevelItem(0).isHidden())
        self.assertEqual(panel.lbl_review_search_count.text(), '共 2 项')

    def test_current_issue_can_be_ignored_and_then_shows_manual_pass(self):
        panel = DetailPanel()
        panel.show()
        self.addCleanup(self._dispose_widget, panel)
        issue = ReviewIssue(
            rule='suspected_left_right_swap', severity='warning',
            message='疑似左右反标', group_id=0, label='left_knee',
            shape_indices=[1], point_indices=[(1, 0)],
        )
        active = ReviewDecisionResult((issue,), (issue,), ())
        panel._load_review('/unused.json', None, active)
        item = panel.review_tree.topLevelItem(0)
        panel.review_tree.setCurrentItem(item)
        ignored = []
        panel.manual_ignore_issue_requested.connect(ignored.append)

        panel.btn_manual_ignore_issue.click()

        self.assertEqual(ignored, [issue])
        accepted = ReviewDecisionResult((issue,), (), (issue,))
        panel._load_review('/unused.json', None, accepted)
        self.assertIn('人工复核通过', panel.lbl_review_summary.text())
        self.assertEqual(panel.lbl_review_summary.property('tone'), 'manual')
        self.assertFalse(panel.btn_manual_accept_current.isEnabled())
        self.assertTrue(panel.btn_manual_restore_current.isEnabled())

    def test_review_workspace_exposes_stable_style_hooks(self):
        panel = DetailPanel()
        panel.show()
        self.addCleanup(self._dispose_widget, panel)

        self.assertEqual(panel.objectName(), 'detailPanel')
        self.assertEqual(panel.review_config_bar.objectName(), 'reviewConfigBar')
        self.assertEqual(panel.review_stats_bar.objectName(), 'reviewStatsBar')
        self.assertEqual(panel.review_stats_tree.objectName(), 'reviewProblemTree')
        self.assertEqual(panel.review_tree.objectName(), 'currentIssueTree')
        self.assertTrue(panel.lbl_review_summary.isHidden())

        panel._set_review_summary('发现问题', 'danger')
        self.assertEqual(panel.lbl_review_summary.property('tone'), 'danger')

    def test_annotation_metrics_index_keypoint_count_anomalies(self):
        folder_summary = {
            'target_class_file_counts': {},
            'keypoint_file_counts': {},
            'shape_type_file_counts': {},
            'metric_files': {},
        }
        annotation_summary = AnnotationSummary(
            valid=True,
            person_boxes=2,
            target_class_counts={'person_dress_middle': 2},
            keypoint_counts={'nose': 1},
            shape_type_counts={'rectangle': 2, 'point': 1},
        )

        AppController._index_annotation_metrics(
            folder_summary, annotation_summary, 4, '4.jpg'
        )

        self.assertEqual(
            folder_summary['target_class_file_counts']['person_dress_middle'], 1
        )
        self.assertEqual(folder_summary['keypoint_file_counts']['nose'], 1)
        record = folder_summary['metric_files']['keypoint:nose'][0]
        self.assertEqual(record['count'], 1)
        self.assertEqual(record['expected'], 2)
        self.assertIn('缺失 1', record['status'])


if __name__ == '__main__':
    unittest.main()
