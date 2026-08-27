import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QWidget

from app.controllers.app_controller import AppController
from app.models.annotation_review import current_pose_review_config
from app.models.dataset_preparation import (
    DatasetPreparationRequest,
    prepare_dataset,
)
from app.views.detail_panel import DetailPanel
from app.views.dir_tree import DirTreePanel
from app.views.file_list_panel import FileListPanel
from app.views.image_viewer import ImageViewer
from app.views.main_window import MainWindow
from app.views.model_management import ModelManagementView
from app.views.training_management import TrainingManagementView


class TrainingManagementTest(unittest.TestCase):
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

    def _build_dataset(self, root: Path, count: int = 5):
        for directory in (
            root / 'images' / 'Collect_A',
            root / 'annotations' / 'Collect_A',
            root / 'labels' / 'Collect_A',
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f'frame_{index:03d}'
            QImage(16, 12, QImage.Format_RGB32).save(
                str(root / 'images' / 'Collect_A' / f'{stem}.png')
            )
            (root / 'annotations' / 'Collect_A' / f'{stem}.json').write_text(
                json.dumps({
                    'imageWidth': 16,
                    'imageHeight': 12,
                    'shapes': [
                        {
                            'label': 'person_dress_finish',
                            'shape_type': 'rectangle',
                            'group_id': 0,
                            'points': [[1, 1], [12, 10]],
                        },
                        *[
                            {
                                'label': f'kp_{point_index:02d}',
                                'shape_type': 'point',
                                'group_id': 0,
                                'points': [[point_index, point_index]],
                            }
                            for point_index in range(23)
                        ],
                    ],
                }),
                encoding='utf-8',
            )
            (root / 'labels' / 'Collect_A' / f'{stem}.txt').write_text(
                ' '.join(
                    ['0', '0.5', '0.5', '0.5', '0.5']
                    + ['0.5', '0.5', '2'] * 23
                ) + '\n',
                encoding='utf-8',
            )

    def _prepare(self, root: Path):
        return prepare_dataset(DatasetPreparationRequest(
            dataset_root=root,
            source_names=('Collect_A',),
            target_name='2026-08-07',
            task_type='pose',
            annotation_dir='annotations',
            label_dir='labels',
            use_copy=True,
            class_names=('person_dress_finish',),
        ))

    def _wait_for_training(self, view, timeout: float = 4.0):
        deadline = time.monotonic() + timeout
        while view.is_training() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.app.processEvents()
        self.assertFalse(view.is_training(), 'training subprocess did not exit')

    @staticmethod
    def _view(temp_dir: str | Path) -> TrainingManagementView:
        root = Path(temp_dir)
        return TrainingManagementView(
            task_registry_path=':memory:',
            training_root=root / 'platform_training',
            models_root=root / 'models',
        )

    def test_training_workspace_lists_raw_and_existing_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)
            view = self._view(temp_dir)
            view.resize(1200, 760)
            view.show()
            self.addCleanup(self._dispose_widget, view)

            self.assertTrue(view.set_dataset_root(root))
            self.app.processEvents()

            self.assertEqual(view.source_tree.topLevelItemCount(), 1)
            source = view.source_tree.topLevelItem(0)
            self.assertEqual(source.text(1), 'Collect_A')
            self.assertEqual(source.text(2), '5')
            self.assertEqual(source.text(3), '5')
            self.assertEqual(source.text(4), '5')
            source_status = view.source_tree.itemWidget(source, 5)
            self.assertEqual(source.text(5), '')
            self.assertEqual(source.data(5, Qt.UserRole), '完整')
            self.assertEqual(source_status.text(), '完整')
            self.assertEqual(source_status.property('tone'), 'success')
            self.assertGreaterEqual(view.existing_batch_tree.topLevelItemCount(), 1)
            existing = view.existing_batch_tree.topLevelItem(0)
            existing_status = view.existing_batch_tree.itemWidget(existing, 4)
            self.assertEqual(existing_status.text(), '就绪')
            self.assertEqual(existing_status.property('tone'), 'success')
            self.assertEqual(
                view._selected_existing_batch_path(),
                prepared.batch_root,
            )
            self.assertTrue(view.btn_use_existing.isEnabled())
            self.assertTrue(view.btn_view_existing.isEnabled())
            self.assertEqual(source.checkState(0), Qt.Checked)
            self.assertIn('Collect_A', view.lbl_batch_source_hint.text())

    def test_raw_batch_row_click_and_selection_commands_toggle_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            source = view.source_tree.topLevelItem(0)

            self.assertEqual(source.checkState(0), Qt.Unchecked)
            view.source_tree.itemClicked.emit(source, 1)
            self.assertEqual(source.checkState(0), Qt.Checked)
            self.assertEqual(view._selected_source_names(), ('Collect_A',))
            self.assertTrue(view.btn_prepare.isEnabled())
            request = view._build_request()
            self.assertTrue(request.skip_incomplete_samples)
            self.assertTrue(request.skip_duplicate_samples)
            self.assertEqual(request.class_names, ())
            self.assertEqual(request.keypoints, ())
            self.assertEqual(request.left_right_pairs, ())

            view.source_tree.itemClicked.emit(source, 4)
            self.assertEqual(source.checkState(0), Qt.Unchecked)
            view.btn_select_all_sources.click()
            self.assertEqual(source.checkState(0), Qt.Checked)
            view.btn_clear_sources.click()
            self.assertEqual(source.checkState(0), Qt.Unchecked)
            self.assertFalse(view.btn_prepare.isEnabled())

    def test_existing_batch_can_enter_review_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)

            requested = []
            view.review_dataset_requested.connect(
                lambda path, task: requested.append((path, task))
            )
            view._use_existing_batch()
            self.assertEqual(view.content_stack.currentIndex(), 1)
            self.assertEqual(view._current_batch, prepared.batch_root)
            self.assertTrue(view.btn_open_review.isEnabled())

            view.btn_open_review.click()
            self.assertEqual(requested, [(str(prepared.batch_root), 'pose')])

    def test_existing_batch_can_open_directly_in_data_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            requested = []
            view.review_dataset_requested.connect(
                lambda path, task: requested.append((path, task))
            )

            view.btn_view_existing.click()

            self.assertEqual(
                requested, [(str(prepared.batch_root), 'pose')]
            )

    def test_existing_batch_can_be_renamed_without_changing_source_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)

            with patch(
                'app.views.training_management.QInputDialog.getText',
                return_value=('renamed-batch', True),
            ):
                view._rename_existing_batch()

            renamed = root / 'training_data' / 'renamed-batch'
            self.assertTrue(renamed.is_dir())
            self.assertFalse(prepared.batch_root.exists())
            self.assertTrue((root / 'images' / 'Collect_A').is_dir())
            manifest = json.loads(
                (renamed / 'preparation_manifest.json').read_text(encoding='utf-8')
            )
            self.assertEqual(manifest['request']['target_name'], 'renamed-batch')

    def test_existing_batch_can_be_deleted_without_deleting_raw_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)

            with patch(
                'app.views.training_management.QMessageBox.warning',
                return_value=QMessageBox.Yes,
            ):
                view._delete_existing_batch()

            self.assertFalse(prepared.batch_root.exists())
            self.assertTrue((root / 'images' / 'Collect_A').is_dir())
            self.assertTrue((root / 'annotations' / 'Collect_A').is_dir())
            self.assertTrue((root / 'labels' / 'Collect_A').is_dir())
            self.assertEqual(view.existing_batch_tree.topLevelItemCount(), 0)

    def test_core_form_updates_template_without_losing_advanced_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            original = view.current_training_config()
            self.assertIn('pose', original.parameters)

            view.epochs_spin.setValue(240)
            view.optimizer_combo.setCurrentText('AdamW')
            updated = view.current_training_config()

            self.assertEqual(updated.parameters['epochs'], 240)
            self.assertEqual(updated.parameters['optimizer'], 'AdamW')
            self.assertNotIn('resume', updated.parameters)
            self.assertEqual(
                updated.parameters['pose'], original.parameters['pose']
            )

    def test_training_subprocess_updates_monitor_and_registers_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            output = Path(temp_dir) / 'training_runs'
            self._build_dataset(root)
            self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            view._use_existing_batch()
            view.output_root_edit.setText(str(output))
            view.project_name_edit.setText('ShengSong')
            view.run_name_edit.setText('pose-test-run')
            view.epochs_spin.setValue(2)
            view._confirm_training_start = lambda _job: True
            fixture = Path(__file__).parent / 'fixtures' / 'fake_training_runner.py'
            view._training_command = lambda job_path: (
                sys.executable, [str(fixture), str(job_path)]
            )
            completed = []
            view.training_completed.connect(
                lambda repository, run: completed.append((repository, run))
            )

            view._start_training()
            self._wait_for_training(view)

            run_dir = output / 'ShengSong' / 'pose-test-run'
            self.assertEqual(view._training_terminal_event, 'completed')
            self.assertEqual(view.training_run_progress.value(), 1000)
            self.assertEqual(view.lbl_monitor_epoch.text(), '2 / 2')
            self.assertIn('train/box_loss', view._metric_items)
            self.assertEqual(
                len(view.training_curve_chart.series['train/box_loss']), 2
            )
            self.assertEqual(
                len(
                    view.training_curve_chart.series[
                        'metrics/mAP50-95(P)'
                    ]
                ),
                2,
            )
            self.assertTrue((run_dir / 'weights' / 'best.pt').is_file())
            self.assertTrue(view.btn_view_training_result.isEnabled())
            self.assertEqual(completed, [(str(output.resolve()), str(run_dir))])
            records = ModelManagementView.scan_model_directory(output)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].name, 'pose-test-run')
            self.assertEqual(records[0].task_type, 'pose')
            tasks = view._task_registry.list_tasks()
            self.assertEqual(len(tasks), 1)
            view._load_task_monitor(tasks[0])
            self.assertEqual(
                view.training_curve_chart.series['train/box_loss'][-1][0], 2
            )
            task_dir = Path(tasks[0].request_path).parent
            self.assertEqual(
                task_dir.parent,
                Path(temp_dir) / 'platform_training' / 'tasks',
            )
            self.assertTrue((task_dir / 'training_request.json').is_file())
            self.assertTrue((task_dir / 'dataset.yaml').is_file())
            self.assertTrue((task_dir / 'run_training.py').is_file())
            self.assertTrue((task_dir / 'training.log').is_file())

            requested_models = []
            view.model_result_requested.connect(
                lambda repository, run: requested_models.append(
                    (repository, run)
                )
            )
            view._request_task_model(tasks[0])
            self.assertEqual(
                requested_models,
                [(str(output.resolve()), str(run_dir))],
            )
            self.assertEqual(
                view.task_tree.contextMenuPolicy(), Qt.CustomContextMenu
            )

    def test_running_training_can_be_stopped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            output = Path(temp_dir) / 'training_runs'
            self._build_dataset(root)
            self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            view._use_existing_batch()
            view.output_root_edit.setText(str(output))
            view.project_name_edit.setText('ShengSong')
            view.run_name_edit.setText('slow-pose-run')
            view._confirm_training_start = lambda _job: True
            view._confirm_training_stop = lambda: True
            fixture = Path(__file__).parent / 'fixtures' / 'fake_training_runner.py'
            view._training_command = lambda job_path: (
                sys.executable, [str(fixture), str(job_path)]
            )

            view._start_training()
            deadline = time.monotonic() + 2.0
            while (
                view._training_process is not None
                and view._training_process.state() != view._training_process.Running
                and time.monotonic() < deadline
            ):
                self.app.processEvents()
                QTest.qWait(10)
            view._stop_training()
            self._wait_for_training(view)

            self.assertEqual(view._training_terminal_event, 'cancelled')
            self.assertEqual(view.lbl_monitor_status.text(), '已停止')

    def test_interrupted_task_can_resume_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            output = Path(temp_dir) / 'training_runs'
            self._build_dataset(root)
            self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            view._use_existing_batch()
            view.output_root_edit.setText(str(output))
            view.project_name_edit.setText('ShengSong')
            view.run_name_edit.setText('resume-pose-run')
            view.epochs_spin.setValue(10)
            record = view._persist_current_task('draft')
            checkpoint = Path(record.run_dir) / 'weights' / 'last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'checkpoint')
            view._task_registry.update_progress(
                record.task_id, epoch=4, epochs=10, progress=40.0
            )
            view._task_registry.set_status(
                record.task_id, 'interrupted',
                error_message='模拟异常中断',
            )
            view.refresh_training_tasks(select_task_id=record.task_id)

            self.assertEqual(view.btn_task_start.text(), '继续训练')
            self.assertTrue(view.btn_task_start.isEnabled())
            self.assertEqual(view.btn_task_retry.text(), '重新训练')
            view._confirm_training_resume = lambda _record, _path: True
            launched = []
            view._launch_task = lambda task: launched.append(task)

            view._start_selected_task()

            resumed = view._task_registry.require(record.task_id)
            self.assertEqual(resumed.status, 'queued')
            self.assertEqual(Path(resumed.model), checkpoint.resolve())
            self.assertTrue(resumed.parameters['resume'])
            self.assertEqual(resumed.current_epoch, 4)
            self.assertEqual(resumed.progress, 40.0)
            self.assertEqual(len(launched), 1)
            self.assertEqual(launched[0].task_id, record.task_id)
            history = Path(resumed.request_path).parent / 'resume_history'
            self.assertEqual(len(list(history.glob('epoch_4_*.json'))), 1)

    def test_stopped_task_without_checkpoint_must_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            self._prepare(root)
            view = self._view(temp_dir)
            self.addCleanup(self._dispose_widget, view)
            view.set_dataset_root(root)
            view._use_existing_batch()
            view.output_root_edit.setText(str(Path(temp_dir) / 'runs'))
            view.run_name_edit.setText('stopped-before-epoch')
            record = view._persist_current_task('draft')
            view._task_registry.set_status(record.task_id, 'cancelled')

            view.refresh_training_tasks(select_task_id=record.task_id)

            self.assertEqual(view.btn_task_start.text(), '无可用断点')
            self.assertFalse(view.btn_task_start.isEnabled())
            self.assertTrue(view.btn_task_retry.isEnabled())

    def test_training_review_uses_shared_data_workspace_and_returns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Dataset'
            self._build_dataset(root)
            prepared = self._prepare(root)

            window = MainWindow()
            tree = DirTreePanel()
            viewer = ImageViewer()
            detail = DetailPanel()
            file_list = FileListPanel()
            model_view = ModelManagementView(load_saved_directory=False)
            training_view = self._view(temp_dir)
            detail.set_file_list(file_list)
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.addWidget(detail)
            window.set_dir_tree(tree)
            window.set_image_viewer(viewer)
            window.set_detail_panel(right_panel)
            window.set_model_manager(model_view)
            window.set_training_manager(training_view)
            controller = AppController(
                window, tree, viewer, detail, file_list,
                model_view, training_view,
            )
            window.show()
            self.addCleanup(self._dispose_widget, window)
            self.app.processEvents()

            window.select_module('train')
            controller.open_training_dataset(str(prepared.batch_root), 'pose')
            self.app.processEvents()

            self.assertEqual(window.current_module(), 'data')
            self.assertEqual(Path(tree.selected_path()), prepared.batch_root)
            self.assertFalse(window.data_source_context_bar.isHidden())
            self.assertIn('训练数据审查', window.lbl_data_source_context.text())
            review_config = current_pose_review_config()
            self.assertEqual(
                review_config.target_classes, ('person_dress_finish',)
            )
            self.assertEqual(len(review_config.keypoints), 23)
            self.assertEqual(review_config.keypoints[0], 'kp_00')

            QTest.mouseClick(window.btn_return_to_model, Qt.LeftButton)
            self.app.processEvents()
            self.assertEqual(window.current_module(), 'train')
            self.assertTrue(window.data_source_context_bar.isHidden())


if __name__ == '__main__':
    unittest.main()
