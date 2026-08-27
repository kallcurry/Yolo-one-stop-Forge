import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from app.controllers.app_controller import AppController
from app.models.model_registry import (
    DatasetSource,
    MetricPoint,
    MetricSeries,
    ModelRecord,
)
from app.views.detail_panel import DetailPanel
from app.views.dir_tree import DirTreePanel
from app.views.file_list_panel import FileListPanel
from app.views.image_viewer import ImageViewer
from app.views.main_window import MainWindow
from app.views.model_management import ModelManagementView


class ModelManagementTest(unittest.TestCase):
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

    def test_scan_ultralytics_repository_builds_one_record_per_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / 'datasets' / 'ShengSong_Datasets'
            train_images = dataset / 'training_data' / '2026-07-29' / 'images'
            train_annotations = dataset / 'training_data' / '2026-07-29' / 'annotations'
            train_labels = dataset / 'training_data' / '2026-07-29' / 'labels'
            val_images = dataset / 'test_data' / 'images'
            val_annotations = dataset / 'test_data' / 'annotations'
            val_labels = dataset / 'test_data' / 'labels'
            for directory in (
                train_images, train_annotations, train_labels,
                val_images, val_annotations, val_labels,
            ):
                directory.mkdir(parents=True)
            (train_images / 'train.jpg').write_bytes(b'image')
            (train_annotations / 'train.json').write_text('{}', encoding='utf-8')
            (train_labels / 'train.txt').write_text('0', encoding='utf-8')
            (val_images / 'val.jpg').write_bytes(b'image')
            (val_annotations / 'val.json').write_text('{}', encoding='utf-8')
            (val_labels / 'val.txt').write_text('0', encoding='utf-8')

            data_yaml = root / 'shengsong-pose.yaml'
            data_yaml.write_text(
                f'train: {train_images}\nval: {val_images}\nnames: [worker]\n',
                encoding='utf-8',
            )
            run = root / 'models' / 'Shengsong' / 'yolov8x-pose-2026-07-29'
            weights = run / 'weights'
            weights.mkdir(parents=True)
            (run / 'args.yaml').write_text(
                'task: pose\n'
                'model: yolov8m-pose.pt\n'
                f'data: {data_yaml}\n'
                'epochs: 100\n'
                'batch: 16\n'
                'imgsz: 640\n'
                'optimizer: SGD\n',
                encoding='utf-8',
            )
            (run / 'results.csv').write_text(
                'epoch,metrics/mAP50-95(B),metrics/mAP50-95(P),'
                'train/pose_loss,val/pose_loss,lr/pg0,time\n'
                '0,0.31,0.42,4.2,4.6,0.001,10\n'
                '1,0.55,0.73,2.8,3.1,0.0005,20\n',
                encoding='utf-8',
            )
            (run / 'results.png').write_bytes(b'plot')
            (weights / 'best.pt').write_bytes(b'best')
            (weights / 'last.pt').write_bytes(b'last')
            (weights / 'best.onnx').write_bytes(b'onnx')

            records = ModelManagementView.scan_model_directory(root / 'models')

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.name, 'yolov8x-pose-2026-07-29')
        self.assertEqual(record.project_name, 'Shengsong')
        self.assertEqual(record.task_type, 'pose')
        self.assertEqual(record.architecture, 'yolov8m-pose')
        self.assertEqual(record.actual_epochs, 2)
        self.assertEqual(record.planned_epochs, 100)
        self.assertIn(('task', 'pose'), record.training_args)
        self.assertIn(('optimizer', 'SGD'), record.training_args)
        self.assertEqual(record.primary_metric_name, 'Pose mAP50-95')
        self.assertAlmostEqual(record.primary_metric_value, 0.73)
        series = {item.key: item for item in record.metric_series}
        self.assertEqual(len(series['metrics/mAP50-95(P)'].points), 2)
        self.assertEqual(series['train/pose_loss'].category, 'loss')
        self.assertFalse(series['train/pose_loss'].higher_is_better)
        self.assertEqual(series['lr/pg0'].category, 'learning_rate')
        self.assertAlmostEqual(series['val/pose_loss'].points[-1].value, 3.1)
        self.assertEqual(len(record.artifacts), 3)
        self.assertEqual(len(record.result_assets), 1)
        self.assertEqual(len(record.data_sources), 2)
        train_source, val_source = record.data_sources
        self.assertEqual(train_source.batch_name, '2026-07-29')
        self.assertEqual(train_source.image_count, 1)
        self.assertEqual(train_source.annotation_count, 1)
        self.assertEqual(val_source.batch_name, '默认测试集')
        self.assertEqual(val_source.role, 'val')

    def test_filter_and_card_click_open_inline_details(self):
        view = ModelManagementView(load_saved_directory=False)
        view.resize(1220, 720)
        view.show()
        self.addCleanup(self._dispose_widget, view)
        self.app.processEvents()
        self.assertGreaterEqual(view._grid_columns, 2)

        view._select_task('pose')
        self.app.processEvents()
        self.assertEqual(len(view._visible_models), 1)
        self.assertEqual(len(view._cards), 1)
        selected = []
        view.model_selected.connect(selected.append)

        card = view._cards[0]
        QTest.mouseClick(card, Qt.LeftButton, pos=card.rect().center())
        self.app.processEvents()

        self.assertEqual(selected[0].task_type, 'pose')
        self.assertIs(view.content_stack.currentWidget(), view.details_page)
        self.assertEqual(view.lbl_detail_title.text(), 'ShengSong-Pose-23')

        view.show_library()
        self.app.processEvents()
        self.assertIs(view.content_stack.currentWidget(), view.library_page)

    def test_detail_tables_are_resizable_and_results_are_selectable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / 'results.png'
            QImage(320, 180, QImage.Format_RGB32).save(str(result_path))

            view = ModelManagementView(load_saved_directory=False)
            view.resize(1200, 760)
            record = replace(
                view._preview_models()[1],
                result_assets=(str(result_path),),
                training_args=(('task', 'pose'), ('imgsz', '640')),
            )
            view.show()
            view.show_model_details(record)
            self.addCleanup(self._dispose_widget, view)
            self.app.processEvents()

            self.assertTrue(view.source_tree.header().stretchLastSection())
            self.assertGreaterEqual(view.source_tree.columnWidth(7), 72)
            self.assertEqual(view.detail_tabs.count(), 5)
            self.assertEqual(view.config_tree.topLevelItemCount(), 2)
            self.assertEqual(view.result_asset_combo.count(), 1)
            self.assertEqual(view.result_asset_combo.currentData(), str(result_path))
            self.assertFalse(view._current_result_pixmap.isNull())
            self.assertGreater(view.detail_body_splitter.handleWidth(), 0)

    def test_same_task_models_can_compare_and_return_from_details(self):
        view = ModelManagementView(load_saved_directory=False)
        view.resize(1280, 800)
        base = view._preview_models()[1]
        curve_a = MetricSeries(
            key='metrics/mAP50-95(P)',
            label='Pose mAP50-95',
            category='evaluation',
            higher_is_better=True,
            points=(MetricPoint(0, 0.42), MetricPoint(1, 0.73)),
        )
        curve_b = replace(
            curve_a,
            points=(MetricPoint(0, 0.39), MetricPoint(1, 0.68)),
        )
        box_curve = replace(
            curve_a,
            key='metrics/mAP50-95(B)',
            label='Box mAP50-95',
        )
        model_a = replace(
            base,
            model_id='pose:a',
            name='pose-a',
            metric_series=(box_curve, curve_a),
        )
        model_b = replace(
            base,
            model_id='pose:b',
            name='pose-b',
            metric_series=(box_curve, curve_b),
        )
        detection = replace(
            view._preview_models()[0], model_id='det:c', name='det-c'
        )
        view._all_models = [model_a, model_b, detection]
        view._update_category_counts()
        view._apply_filters()
        view.show()
        self.addCleanup(self._dispose_widget, view)
        self.app.processEvents()

        view.btn_compare_mode.click()
        cards = {card.record.model_id: card for card in view._cards}
        cards['pose:a'].compare_button.click()
        cards['pose:b'].compare_button.click()
        cards['det:c'].compare_button.click()

        self.assertEqual(
            [record.model_id for record in view._comparison_models],
            ['pose:a', 'pose:b'],
        )
        self.assertIn('相同任务', view.lbl_comparison_state.text())
        self.assertTrue(view.btn_start_comparison.isEnabled())

        view.btn_start_comparison.click()
        self.app.processEvents()
        self.assertIs(
            view.content_stack.currentWidget(), view.comparison_page
        )
        self.assertEqual(view.comparison_page.metric_combo.count(), 2)
        self.assertEqual(
            view.comparison_page.metric_combo.currentData(),
            'metrics/mAP50-95(P)',
        )
        self.assertEqual(view.comparison_page.table.topLevelItemCount(), 10)

        requested = []
        view.comparison_page.model_requested.connect(requested.append)
        chart = view.comparison_page.bar_chart
        chart._animation.setCurrentTime(chart._animation.duration())
        chart.grab()
        self.app.processEvents()
        self.assertGreaterEqual(len(chart._bar_hits), 2)
        QTest.mouseClick(
            chart,
            Qt.LeftButton,
            pos=chart._bar_hits[0][0].center().toPoint(),
        )
        self.assertEqual(requested[-1].model_id, 'pose:a')
        self.assertIs(view.content_stack.currentWidget(), view.details_page)

        view.btn_back.click()
        self.app.processEvents()
        self.assertIs(
            view.content_stack.currentWidget(), view.comparison_page
        )
        self.assertEqual(len(view._comparison_models), 2)

    def test_data_source_context_bar_is_reversible(self):
        window = MainWindow()
        model_view = ModelManagementView(load_saved_directory=False)
        window.set_model_manager(model_view)
        window.show()
        self.addCleanup(self._dispose_widget, window)
        self.app.processEvents()

        model = model_view._all_models[0]
        source = type('Source', (), {
            'role': 'train',
            'dataset_name': 'ShengSong_Datasets',
            'batch_name': '2026-07-29',
            'image_path': '/tmp/images',
        })()
        window.show_data_source_context(model, source)
        self.assertFalse(window.data_source_context_bar.isHidden())
        self.assertIn(model.name, window.lbl_data_source_context.text())
        self.assertIn('2026-07-29', window.lbl_data_source_context.text())
        window.hide_data_source_context()
        self.assertTrue(window.data_source_context_bar.isHidden())

    def test_model_source_opens_batch_and_returns_to_model_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / 'ShengSong_Datasets'
            raw_images = dataset / 'images' / 'Collect_A'
            raw_annotations = dataset / 'annotations' / 'Collect_A'
            batch = dataset / 'training_data' / '2026-07-29'
            for directory in (
                raw_images,
                raw_annotations,
                batch / 'images',
                batch / 'annotations',
                batch / 'labels',
            ):
                directory.mkdir(parents=True)
            image_path = batch / 'images' / 'frame_001.png'
            QImage(8, 8, QImage.Format_RGB32).save(str(image_path))
            (batch / 'annotations' / 'frame_001.json').write_text(
                '{"imageWidth": 8, "imageHeight": 8, "shapes": []}',
                encoding='utf-8',
            )
            (batch / 'labels' / 'frame_001.txt').write_text('', encoding='utf-8')

            source = DatasetSource(
                role='train',
                dataset_name='ShengSong_Datasets',
                batch_name='2026-07-29',
                dataset_root=str(dataset),
                batch_root=str(batch),
                image_path=str(batch / 'images'),
                annotation_path=str(batch / 'annotations'),
                labels_path=str(batch / 'labels'),
                annotation_dir='annotations',
                image_count=1,
                annotation_count=1,
                label_count=1,
                available=True,
            )
            model = ModelRecord(
                model_id='run-1',
                name='pose-run-2026-07-29',
                project_name='Shengsong',
                task_type='pose',
                framework='Ultralytics YOLO',
                file_format='PyTorch',
                path=str(Path(temp_dir) / 'models' / 'run-1'),
                data_sources=(source,),
            )

            window = MainWindow()
            tree = DirTreePanel()
            viewer = ImageViewer()
            detail = DetailPanel()
            file_list = FileListPanel()
            model_view = ModelManagementView(load_saved_directory=False)
            detail.set_file_list(file_list)
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.addWidget(detail)
            window.set_dir_tree(tree)
            window.set_image_viewer(viewer)
            window.set_detail_panel(right_panel)
            window.set_model_manager(model_view)
            controller = AppController(
                window, tree, viewer, detail, file_list, model_view
            )
            window.show()
            self.addCleanup(self._dispose_widget, window)
            self.app.processEvents()

            window.select_module('model')
            controller.open_model_dataset(model, source)
            self.app.processEvents()

            self.assertEqual(window.current_module(), 'data')
            self.assertEqual(Path(tree.selected_path()), batch.resolve())
            self.assertEqual(Path(viewer.current_path()), image_path.resolve())
            self.assertFalse(window.data_source_context_bar.isHidden())

            controller.return_to_model_details()
            self.app.processEvents()
            self.assertEqual(window.current_module(), 'model')
            self.assertEqual(model_view.lbl_detail_title.text(), model.name)
            self.assertIs(
                model_view.content_stack.currentWidget(),
                model_view.details_page,
            )

    def test_main_window_switches_between_isolated_workspaces(self):
        window = MainWindow()
        model_view = ModelManagementView(load_saved_directory=False)
        window.set_model_manager(model_view)
        window.resize(1280, 760)
        window.show()
        self.addCleanup(self._dispose_widget, window)
        self.app.processEvents()
        window.set_action_enabled(True)

        window.module_buttons['model'].click()
        self.app.processEvents()
        self.assertEqual(window.current_module(), 'model')
        self.assertIs(window.workspace_stack.currentWidget(), model_view)
        self.assertTrue(window.btn_task_picker.isHidden())
        self.assertFalse(window.data_workspace.isVisible())
        self.assertFalse(window.action_delete.isEnabled())
        self.assertFalse(window.shortcut_annotation.isEnabled())

        window.module_buttons['data'].click()
        self.app.processEvents()
        self.assertEqual(window.current_module(), 'data')
        self.assertIs(window.workspace_stack.currentWidget(), window.data_workspace)
        self.assertFalse(window.btn_task_picker.isHidden())
        self.assertTrue(window.action_delete.isEnabled())
        self.assertTrue(window.shortcut_annotation.isEnabled())


if __name__ == '__main__':
    unittest.main()
