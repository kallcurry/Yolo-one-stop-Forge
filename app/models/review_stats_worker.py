"""Background review-stats scan (off-UI-thread) for the data management panel.

The scan walks every image of the current folder, runs annotation review and
aggregates per-file metrics.  Doing that synchronously on the UI thread
freezes the window for large folders (thousands of images).  This module runs
the computation in a QThread, reusing the controller's pure helpers (folder
summary factory / metric file aggregation) while keeping every Qt widget out
of the worker; the ready result is handed back to the UI thread.
"""

from __future__ import annotations

from collections import Counter

from PyQt5.QtCore import QThread, pyqtSignal

from app.models.annotation_review import (
    current_pose_review_config,
    review_annotation_file,
)
from app.models.file_system import find_annotation
from app.models.review_decisions import (  # noqa: F401 - type reference
    ReviewDecisionResult,
)


class ReviewStatsWorker(QThread):
    result_ready = pyqtSignal(object)   # (total, rows, folder_summary, metric_rows)
    failed = pyqtSignal(str)

    def __init__(self, controller, images, annotation_dir: str, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._images = list(images)
        self._annotation_dir = str(annotation_dir)

    def run(self):
        try:
            result = compute_review_stats(
                self._controller, self._images, self._annotation_dir,
            )
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001 - report, never crash UI
            self.failed.emit(f'{type(exc).__name__}: {exc}')


def _review_result_core(controller, image_path, annotation_path):
    """Thread-safe review decision (no widget/status-bar access)."""
    issues = tuple(review_annotation_file(annotation_path, image_path))
    store = getattr(controller, '_review_decision_store', None)
    if store is None:
        return ReviewDecisionResult(issues, issues, ())
    try:
        return store.evaluate(
            image_path, annotation_path, current_pose_review_config(), issues,
        )
    except OSError:
        return ReviewDecisionResult(issues, issues, ())


def compute_review_stats(controller, images, annotation_dir: str):
    """Pure computation; no Qt widgets. Returns 3-tuple result payload."""
    total = len(images)
    folder_summary = controller._new_review_folder_summary(total)
    rows = []
    config = current_pose_review_config()
    task_type = config.task_type

    first_ann_dir = None
    if images:
        first_ann_dir = find_annotation(
            images[0], annotation_dir=annotation_dir,
        )
        if first_ann_dir is not None:
            folder_summary['annotation_set_dir'] = str(
                first_ann_dir.parent.parent
                if first_ann_dir.parent.parent.name != annotation_dir
                else first_ann_dir.parent
            )

    for idx, img_path in enumerate(images):
        filename = img_path.name
        controller._add_metric_file(
            folder_summary, 'overview:images', idx, filename,
            status='图片', detail='当前文件夹中的图片',
        )
        ann = find_annotation(img_path, annotation_dir=annotation_dir)
        if ann is None:
            folder_summary['missing_annotations'] += 1
            controller._add_metric_file(
                folder_summary, 'overview:missing', idx, filename,
                status='缺失标注', detail='未找到对应 JSON 标注文件',
            )
            controller._add_metric_file(
                folder_summary, 'quality:issue', idx, filename,
                status='有问题', detail='缺失标注文件',
            )
            continue

        try:
            summary = controller._add_annotation_summary(
                folder_summary, ann,
            )
        except Exception:  # noqa: BLE001 - invalid json → treat as invalid
            folder_summary['invalid_annotations'] += 1
            controller._add_metric_file(
                folder_summary, 'overview:invalid', idx, filename,
                status='JSON 无效', detail='标注文件无法解析',
            )
            continue
        controller._add_metric_file(
            folder_summary, 'overview:annotations', idx, filename,
            status='标注有效' if summary.valid else 'JSON 无效',
            detail=summary.error or '已找到对应 JSON 标注文件',
        )
        if not summary.valid:
            folder_summary['invalid_annotations'] += 1
            controller._add_metric_file(
                folder_summary, 'quality:issue', idx, filename,
                status='有问题', detail=summary.error,
            )
            continue

        controller._index_annotation_metrics(
            folder_summary, summary, idx, filename,
        )
        review_result = _review_result_core(controller, img_path, ann)
        raw_issues = list(review_result.raw_issues)
        issues = list(review_result.active_issues)
        accepted_issues = list(review_result.accepted_issues)

        if not raw_issues:
            folder_summary['ok_files'] += 1
            controller._add_metric_file(
                folder_summary, 'quality:ok', idx, filename,
                status='规则通过',
                detail='当前已执行规则未发现标注问题',
            )
            continue

        folder_summary['raw_issue_files'] += 1
        folder_summary['raw_issue_count'] += len(raw_issues)
        folder_summary['accepted_issue_count'] += len(accepted_issues)
        for issue in raw_issues:
            folder_summary['raw_rule_counts'][issue.rule] = (
                folder_summary['raw_rule_counts'].get(issue.rule, 0) + 1
            )

        if review_result.manually_passed:
            folder_summary['manual_pass_files'] += 1
            rows.append({
                'index': idx, 'filename': filename, 'issues': [],
                'accepted_issues': accepted_issues, 'status': 'manual',
            })
            controller._add_metric_file(
                folder_summary, 'quality:manual', idx, filename,
                count=len(accepted_issues), status='人工通过',
                detail=f'人工确认 {len(accepted_issues)} 个算法问题为误报',
            )
            continue

        rows.append({
            'index': idx,
            'filename': filename,
            'issues': issues,
            'accepted_issues': accepted_issues,
            'status': 'stale' if review_result.stale else 'problem',
        })
        if review_result.stale:
            folder_summary['stale_review_files'] += 1
        controller._add_issue_summary(folder_summary, issues)
        controller._add_metric_file(
            folder_summary, 'quality:issue', idx, filename,
            count=len(issues), status='有问题',
            detail=f'发现 {len(issues)} 个审查问题',
        )
        for rule, count in Counter(issue.rule for issue in issues).items():
            controller._add_metric_file(
                folder_summary, f'rule:{rule}', idx, filename,
                count=count, status=f'{count} 个问题',
                detail='; '.join(
                    issue.message for issue in issues if issue.rule == rule
                ),
            )

    return total, rows, folder_summary
