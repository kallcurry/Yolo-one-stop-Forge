"""Central controller — wires views to models."""

import json
from collections import Counter
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QSettings, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.utils import log
from app.models.file_system import (
    DirFormat,
    annotation_set_dir_for_image,
    detect_format,
    expected_annotation_path,
    find_annotation,
    list_images,
    create_folder as fs_create_folder,
)
from app.models.annotation_sync import synchronize_annotation_folder
from app.models.label_tool import build_xanylabeling_folder_command
from app.models.annotation_review import (
    TASK_PRESETS,
    apply_pose_review_config,
    current_pose_review_config,
    default_task_review_config,
    load_pose_review_config,
    load_and_apply_pose_review_config,
    pose_review_config_to_dict,
    review_config_from_data,
    reorder_keypoints_file,
    review_annotation_file,
    summarize_annotation_file,
)
from app.models.label_tool import (
    build_xanylabeling_command,
    command_to_text,
)
from app.models.annotation_sync import (
    AnnotationSyncError,
    annotation_file_fingerprint,
    synchronize_annotation_replicas,
)
from app.models.review_decisions import (
    ReviewDecisionResult,
    ReviewDecisionStore,
)
from app.models.operations import (
    copy_images,
    delete_images,
    move_images,
    rename_image,
)

POSE_TEMPLATE_BUILTIN_ID = '__builtin_pose_review_template__'
TASK_TEMPLATE_PREFIX = '__builtin_task_review_template__:'
DATA_TASK_ORDER = ('pose', 'detection', 'segmentation', 'obb')
POSE_TEMPLATE_LIST_KEY = 'poseReviewTemplatePaths'
POSE_TEMPLATE_SELECTED_KEY = 'poseReviewSelectedTemplateId'
LAST_DIRECTORY_KEY = 'lastDirectory'
LAST_SELECTED_DIRECTORY_KEY = 'lastSelectedDirectory'
LAST_SELECTED_IMAGE_KEY = 'lastSelectedImage'


class AppController(QObject):
    """Coordinates models and views. Created in main.py after all views exist."""

    def __init__(self, main_window, dir_tree, image_viewer,
                 detail_panel, file_list_panel, model_manager=None,
                 training_manager=None, evaluation_manager=None,
                 inference_manager=None):
        super().__init__()
        self._win = main_window
        self._tree = dir_tree
        self._viewer = image_viewer
        self._detail = detail_panel
        self._file_list = file_list_panel
        self._model_manager = model_manager
        self._training_manager = training_manager
        self._evaluation_manager = evaluation_manager
        self._inference_manager = inference_manager

        self._images: list[Path] = []
        self._current_index: int = -1
        self._sync_annotation = True
        self._label_tool_processes: list[QProcess] = []
        self._pose_template_entries: list[dict] = []
        self._active_pose_template_id = POSE_TEMPLATE_BUILTIN_ID
        self._review_policy_config = None
        self._model_navigation_context = None
        self._training_navigation_context = None
        self._last_open_directory = ''
        self._last_selected_directory = ''
        self._last_selected_image = ''
        self._review_decision_store: ReviewDecisionStore | None = None

        self._connect_own_signals()
        self._load_saved_pose_review_config()

    # --- Public API for main.py ---

    def open_directory_dialog(self):
        """Prompt user to pick a root directory."""
        path = QFileDialog.getExistingDirectory(self._win, '选择数据根目录')
        if path:
            self.open_directory(path)

    def open_directory(self, path: str):
        """Load a directory tree root and remember it."""
        normalized_path = Path(str(path)).expanduser().resolve()
        if not normalized_path.is_dir():
            return
        path = str(normalized_path)
        self._review_decision_store = ReviewDecisionStore(normalized_path)
        self._model_navigation_context = None
        if hasattr(self._win, 'hide_data_source_context'):
            self._win.hide_data_source_context()
        log(f'📁 open_directory: {path}')
        self._last_open_directory = path
        self._save_last_dir(path)
        self._win.status_bar.showMessage(f'已打开: {path}')
        self._tree.load_root(path)
        self._tree.populate_annotation_dirs(
            normalized_path, current=self._active_annotation_dir(),
        )
        if self._training_manager is not None:
            self._training_manager.set_dataset_root(path)
        self._images = []
        self._current_index = -1
        self._batch_annotate = False
        self._win.set_counter_text(0, 0)
        self._win.set_nav_enabled(False)
        self._win.set_action_enabled(False)
        if self._file_list:
            self._file_list.populate([])
        if self._detail:
            self._detail.clear()

    def last_directory(self) -> str | None:
        return self._stored_path(LAST_DIRECTORY_KEY)

    def restore_last_selection(self):
        """Restore the last selected data folder and image after the tree loads."""
        selected_directory = self._stored_path(LAST_SELECTED_DIRECTORY_KEY)
        if not selected_directory or not Path(selected_directory).is_dir():
            return False
        if not self._tree.select_path(selected_directory, emit=True):
            return False

        selected_image = self._stored_path(LAST_SELECTED_IMAGE_KEY)
        if selected_image:
            target = Path(selected_image)
            for index, image_path in enumerate(self._images):
                if self._same_path(image_path, target):
                    self._current_index = index
                    self._load_current()
                    break
        return True

    def save_last_directory(self):
        """Persist the last data root while the application is closing."""
        if self._last_open_directory:
            self._save_last_dir(self._last_open_directory)
        if self._last_selected_directory or self._last_selected_image:
            self._save_selection_state()

    def open_model_dataset(self, model, source):
        """Open a parsed training/validation source in the data workspace."""
        image_path = Path(str(getattr(source, 'image_path', ''))).expanduser()
        if not image_path.is_dir():
            QMessageBox.warning(
                self._win,
                '数据路径不可用',
                f'模型记录中的数据目录不存在或已移动:\n{image_path}',
            )
            return

        dataset_root = Path(
            str(getattr(source, 'dataset_root', '') or image_path)
        ).expanduser()
        if not dataset_root.is_dir():
            dataset_root = image_path

        task_type = str(getattr(model, 'task_type', 'pose') or 'pose')
        if task_type not in DATA_TASK_ORDER:
            task_type = 'pose'
        self._win.select_task(task_type, emit=False)
        self.on_data_task_selected(task_type)
        self._win.select_module('data')

        self.open_directory(str(dataset_root))
        batch_root = Path(
            str(getattr(source, 'batch_root', '') or image_path)
        ).expanduser()
        target = (
            batch_root
            if image_path == batch_root / 'images'
            else image_path
        )
        if not self._tree.select_path(target):
            if not self._tree.select_path(batch_root):
                fallback_format = detect_format(
                    image_path, self._active_annotation_dir()
                )
                self._on_dir_selected(str(image_path), fallback_format)

        self._model_navigation_context = (model, source)
        self._win.show_data_source_context(model, source)
        self._win.status_bar.showMessage(
            f'已打开模型数据来源: {getattr(source, "batch_name", image_path.name)}',
            4000,
        )

    def return_to_model_details(self):
        """Return from a dataset batch to the originating model profile."""
        context = self._model_navigation_context
        if context is None:
            self._win.select_module('model')
            return
        model, _source = context
        self._win.select_module('model')
        if self._model_manager is not None:
            self._model_manager.show_model_details(model)
        self._win.hide_data_source_context()

    def open_training_dataset(self, batch_path: str, task_type: str):
        """Open a generated training batch in the shared review workspace."""
        batch = Path(batch_path).expanduser().resolve()
        if not batch.is_dir():
            QMessageBox.warning(
                self._win, '训练数据不可用', f'训练批次不存在:\n{batch}'
            )
            return
        training_root = batch.parent
        dataset_root = training_root.parent if training_root.name == 'training_data' else None
        if dataset_root is None or not (dataset_root / 'images').is_dir():
            QMessageBox.warning(
                self._win, '训练数据不可用', f'无法确定数据项目根目录:\n{batch}'
            )
            return

        self._win.select_task(task_type, emit=False)
        self.on_data_task_selected(task_type)
        self._win.select_module('data')
        self.open_directory(str(dataset_root))
        if not self._tree.select_path(batch, emit=True):
            self._on_dir_selected(
                str(batch), detect_format(batch, self._active_annotation_dir())
            )
        self._training_navigation_context = (batch, task_type)
        self._win.show_training_data_context(batch, task_type)
        self._win.status_bar.showMessage(
            f'已打开训练数据审查: {batch.name}', 3500
        )

    def return_to_training_task(self):
        """Return from shared data review to the active training workflow."""
        self._win.select_module('train')
        self._win.hide_data_source_context()

    def on_training_dataset_prepared(self, batch_path: str, task_type: str):
        """Refresh the project tree after the training center writes a batch."""
        batch = Path(batch_path).expanduser().resolve()
        if self._last_open_directory:
            try:
                opened = Path(self._last_open_directory).expanduser().resolve()
                dataset_root = batch.parent.parent
                if opened == dataset_root:
                    self._tree.refresh()
            except OSError:
                pass
        self._win.status_bar.showMessage(
            f'训练数据已生成: {batch.name}', 5000
        )

    def on_training_completed(self, repository_root: str, run_path: str):
        """Refresh the model registry without leaving the training monitor."""
        if self._model_manager is not None:
            self._model_manager.set_model_directory(repository_root)
        self._win.status_bar.showMessage(
            f'训练完成，模型产物已登记: {Path(run_path).name}', 6000
        )

    def open_training_result(self, repository_root: str, run_path: str):
        """Open a completed training run in the model profile workspace."""
        if self._model_manager is None:
            return
        if not self._model_manager.show_training_run(repository_root, run_path):
            QMessageBox.warning(
                self._win,
                '模型结果尚不可用',
                f'没有在模型仓库中识别到训练结果:\n{run_path}',
            )
            return
        self._win.select_module('model')
        self._win.status_bar.showMessage(
            f'已打开训练结果: {Path(run_path).name}', 4000
        )

    def open_evaluation(self, model_path: str, model_label: str):
        """Prefill the evaluation center for a model and switch module."""
        if self._evaluation_manager is not None:
            self._evaluation_manager.prefill_model(model_path, model_label)
        self._win.select_module('eval')
        self._win.status_bar.showMessage(
            f'已为 {model_label} 打开评估中心', 4000
        )

    def open_inference(self, model_path: str, model_label: str):
        """Prefill the inference workbench for a model and switch module."""
        if self._inference_manager is not None:
            self._inference_manager.prefill_model(model_path, model_label)
        self._win.select_module('infer')
        self._win.status_bar.showMessage(
            f'已为 {model_label} 打开推理中心', 4000
        )

    # Actions (public, connected from MainWindow)
    def on_copy(self):
        sources = self._get_selected_images()
        if not sources:
            return
        dest = self._pick_destination()
        if not dest:
            return
        errors = copy_images(
            sources, dest, self._sync_annotation, self._active_annotation_dir()
        )
        self._finish_op('复制', errors, sources)

    def on_move(self):
        sources = self._get_selected_images()
        if not sources:
            return
        dest = self._pick_destination()
        if not dest:
            return
        errors = move_images(
            sources, dest, self._sync_annotation, self._active_annotation_dir()
        )
        self._finish_op('移动', errors, sources)

    def on_delete(self):
        sources = self._get_selected_images()
        if not sources:
            return
        names = '\n'.join(p.name for p in sources[:10])
        more = f'\n... 还有 {len(sources) - 10} 张' if len(sources) > 10 else ''
        answer = QMessageBox.question(
            self._win, '确认删除',
            f'确定要删除以下 {len(sources)} 张图片吗？\n\n{names}{more}',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        errors = delete_images(
            sources, self._sync_annotation, self._active_annotation_dir()
        )
        self._finish_op('删除', errors, sources)

    def on_rename(self):
        if self._current_index < 0 or self._current_index >= len(self._images):
            return
        old = self._images[self._current_index]
        from app.views.dialogs import RenameDialog
        dlg = RenameDialog(old, self._win)
        if dlg.exec_():
            new_name = dlg.get_new_name()
            if new_name:
                err = rename_image(
                    old,
                    new_name,
                    self._sync_annotation,
                    self._active_annotation_dir(),
                )
                if err:
                    QMessageBox.warning(self._win, '重命名失败', err)
                self.refresh()

    def on_new_folder(self, parent_path: str = ''):
        if not parent_path:
            return
        from app.views.dialogs import NewFolderDialog
        dlg = NewFolderDialog(self._win)
        if dlg.exec_():
            name = dlg.get_folder_name()
            if name:
                err = fs_create_folder(parent_path, name)
                if err:
                    QMessageBox.warning(self._win, '创建文件夹失败', err)
                else:
                    self._tree.refresh()

    def on_open_label_tool(self):
        """Open the current image and annotation in X-AnyLabeling."""
        if self._current_index < 0 or self._current_index >= len(self._images):
            self._set_batch_annotate(False)
            QMessageBox.information(self._win, '没有图片', '请先选择一张图片。')
            return

        img_path = self._images[self._current_index]
        ann_path = self._find_annotation(img_path)
        if ann_path is None:
            expected = self._expected_annotation_path(img_path)
            config = current_pose_review_config()
            detail = (
                f'\n\n任务: {config.task_type}'
                f'\n标注集: {config.annotation_dir}'
            )
            if expected is not None:
                detail += f'\n期望路径: {expected}'
            QMessageBox.warning(
                self._win,
                '未找到标注文件',
                f'当前图片没有找到该任务对应的 JSON 标注文件:\n{img_path}'
                f'{detail}',
            )
            return

        try:
            command = build_xanylabeling_command(img_path, ann_path)
        except FileNotFoundError as exc:
            QMessageBox.warning(
                self._win,
                '未找到标注工具',
                f'{exc}\n\n'
                '可运行 deployment/install.sh 安装，或在 '
                'deployment/local.env 中配置独立环境的 Python 路径。',
            )
            return

        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(img_path.parent))
        process.setProcessChannelMode(QProcess.ForwardedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.remove('QT_QPA_PLATFORM_PLUGIN_PATH')
        environment.remove('QT_PLUGIN_PATH')
        process.setProcessEnvironment(environment)
        process.setProperty('image_path', str(img_path))
        process.setProperty('annotation_path', str(ann_path))
        process.setProperty(
            'annotation_fingerprint', annotation_file_fingerprint(ann_path)
        )
        process.setProperty('dataset_root', self._last_open_directory)
        process.setProperty('annotation_dir', self._active_annotation_dir())
        process.setProperty('command_text', command_to_text(command))
        process.finished.connect(
            lambda _exit_code, _status, p=process: self._on_label_tool_finished(p)
        )
        process.errorOccurred.connect(
            lambda _error, p=process: self._on_label_tool_error(p)
        )

        self._label_tool_processes.append(process)
        log(f'🧰 启动标注工具: {command_to_text(command)}')
        process.start()

        if not process.waitForStarted(4000):
            self._on_label_tool_error(process)
            return

        self._win.status_bar.showMessage(
            f'已打开标注工具: {img_path.name}', 3000
        )

    def on_annotate_folder(self):
        """标注本文件夹：X-AnyLabeling 单窗口 + 图库加载全部图片。

        ``--filename`` 接受目录时工具会自动加载整个文件夹；保存的 JSON
        始终写入标注集目录（--output），不污染图片目录。
        """
        if not self._images:
            QMessageBox.information(self._win, '没有图片', '请先选择一个图片文件夹。')
            return
        img_dir = self._images[0].parent
        ann_dir = annotation_set_dir_for_image(
            self._images[0], annotation_dir=self._active_annotation_dir(),
        )
        if ann_dir is None:
            QMessageBox.warning(
                self._win, '标注目录错误',
                f'无法定位标注集目录：{self._active_annotation_dir()}',
            )
            return
        try:
            ann_dir.mkdir(parents=True, exist_ok=True)
            command = build_xanylabeling_folder_command(
                img_dir, ann_dir,
            )
        except FileNotFoundError as exc:
            QMessageBox.warning(self._win, '未找到标注工具', str(exc))
            return

        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(img_dir))
        process.setProcessChannelMode(QProcess.ForwardedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.remove('QT_QPA_PLATFORM_PLUGIN_PATH')
        environment.remove('QT_PLUGIN_PATH')
        process.setProcessEnvironment(environment)
        process.setProperty('mode', 'folder')
        process.setProperty('annotation_set_dir', str(ann_dir))
        process.setProperty('dataset_root', self._last_open_directory)
        process.setProperty('annotation_dir', self._active_annotation_dir())
        process.finished.connect(
            lambda _code, _status, p=process: self._on_label_tool_finished(p)
        )
        process.errorOccurred.connect(
            lambda _error, p=process: self._on_label_tool_error(p)
        )
        self._label_tool_processes.append(process)
        log(f'🧰 启动文件夹标注: {command_to_text(command)}')
        process.start()
        if not process.waitForStarted(4000):
            self._on_label_tool_error(process)
            return
        self._win.status_bar.showMessage(
            f'已打开文件夹标注窗口: {img_dir.name}（图库加载全部图片）', 5000
        )

    def _on_annotation_folder_finished(self, process: QProcess):
        """Batch finish: sync changed JSONs to their replicas + refresh review."""
        ann_dir = Path(str(process.property('annotation_set_dir') or ''))
        root = process.property('dataset_root') or None
        if ann_dir.is_dir():
            synced, failed = synchronize_annotation_folder(
                ann_dir, dataset_root=root,
            )
            text = (
                f'文件夹标注完成：同步 {synced} 个标注副本'
                + (f"，{failed} 个失败" if failed else '')
            )
            self._win.status_bar.showMessage(text, 5000)
        self.refresh()
        self._refresh_review_decision_views('文件夹标注完成')

    def on_reorder_folder_keypoints(self):
        """Reorder keypoints for every annotation in the current folder."""
        if not self._images:
            QMessageBox.information(self._win, '没有图片', '请先选择一个图片文件夹。')
            return
        config = current_pose_review_config()
        if config.task_type != 'pose':
            QMessageBox.information(
                self._win,
                '当前任务不是 Pose',
                f'关键点重排序只适用于 Pose 任务。\n\n'
                f'当前任务: {config.task_type}',
            )
            return

        answer = QMessageBox.question(
            self._win,
            '确认批量重排序',
            '将对当前文件夹中所有图片对应的 JSON 标注文件执行关键点重排序。\n\n'
            f'图片数量: {len(self._images)}\n\n'
            '该操作会直接覆盖发生变化的 JSON 文件，确定继续吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        changed = 0
        unchanged = 0
        missing = 0
        failed = []
        total_groups = 0
        total_keypoints = 0

        self._win.status_bar.showMessage(
            f'正在重排当前文件夹关键点: {len(self._images)} 张图片...'
        )
        self._win.status_bar.repaint()

        for img_path in self._images:
            ann_path = self._find_annotation(img_path)
            if ann_path is None:
                missing += 1
                continue

            try:
                result = reorder_keypoints_file(ann_path)
            except Exception as exc:
                failed.append(f'{img_path.name}: {exc}')
                continue

            total_groups += result.groups
            total_keypoints += result.keypoints
            if result.changed:
                changed += 1
            else:
                unchanged += 1

        self._load_current()
        if self._detail:
            self._detail.clear_review_stats()

        summary = (
            f'批量重排序完成: 修改 {changed} 个, 已标准 {unchanged} 个, '
            f'缺失标注 {missing} 个, 失败 {len(failed)} 个; '
            f'{total_groups} 个人, {total_keypoints} 个关键点'
        )
        self._win.status_bar.showMessage(summary, 6000)

        if failed:
            QMessageBox.warning(
                self._win,
                '部分文件重排序失败',
                '\n'.join(failed[:20]),
            )

    def on_import_pose_review_config(self):
        """Open the review template manager/editor."""
        settings = QSettings('FilesProcessQT', 'ImageManager')
        from app.views.pose_template_dialog import PoseTemplateDialog

        current_config = current_pose_review_config()
        project_root = Path(__file__).resolve().parents[2]
        task_type = str(current_config.task_type or 'pose')
        dir_key = self._template_dir_setting_key(task_type)
        legacy_dir = (
            settings.value('poseReviewConfigDir')
            if task_type == 'pose' else None
        )
        template_dir = Path(
            settings.value(dir_key)
            or legacy_dir
            or project_root / 'resources' / 'review_templates' / task_type
        )
        plugin_dir = project_root / 'resources' / 'review_plugins' / task_type
        dlg = PoseTemplateDialog(
            pose_review_config_to_dict(current_config),
            template_dir,
            plugin_dir,
            project_root,
            self._win,
        )
        if not dlg.exec_():
            return

        saved_path = dlg.saved_template_path()
        if saved_path is None:
            return

        try:
            config = load_pose_review_config(saved_path)
        except ValueError as exc:
            QMessageBox.warning(
                self._win,
                '模板保存后导入失败',
                f'无法导入审查模板:\n{saved_path}\n\n{exc}',
            )
            return

        template_id = self._template_id_for_path(saved_path)
        self._upsert_pose_template_entry(template_id, config.name, str(saved_path))
        self._save_pose_template_entries()
        settings.setValue(self._template_dir_setting_key(config.task_type),
                          str(saved_path.parent))
        if config.task_type == 'pose':
            settings.setValue('poseReviewConfigDir', str(saved_path.parent))

        self._apply_pose_template(template_id)

        self._win.status_bar.showMessage(
            f'已保存并切换审查模板: {config.name}', 5000
        )

    def on_pose_review_template_selected(self, template_id: str):
        """Switch to a template from the template list."""
        if not template_id or template_id == self._active_pose_template_id:
            return
        self._apply_pose_template(template_id)

    def on_data_task_selected(self, task_type: str):
        """Switch the platform-wide task: data review + other centers."""
        template_id = self._template_id_for_task(task_type)
        self._apply_pose_template(template_id)
        # 全模块任务联动：训练中心(数据准备模板/目录) + 评估中心(任务徽章)
        if getattr(self, '_training_manager', None) is not None:
            self._training_manager.set_scope_task(task_type)
        if getattr(self, '_evaluation_manager', None) is not None:
            self._evaluation_manager.set_scope_task(task_type)

    def refresh(self):
        """Reload current directory listing and tree."""
        if self._images:
            parent = self._images[0].parent
            fmt = detect_format(parent, self._active_annotation_dir())
            self._images = list_images(parent, fmt)
            if self._current_index >= len(self._images):
                self._current_index = max(0, len(self._images) - 1)
            total = len(self._images)
            self._win.set_nav_enabled(total > 1)
            self._win.set_action_enabled(total > 0)
            if self._file_list:
                self._file_list.populate(self._images)
            self._load_current()
        self._tree.refresh()

    def _on_label_tool_finished(self, process: QProcess):
        self._discard_label_tool_process(process)
        if process.property('mode') == 'folder':
            self._on_annotation_folder_finished(process)
            return

        image_path = Path(process.property('image_path') or '')
        annotation_path = Path(process.property('annotation_path') or '')
        before = str(process.property('annotation_fingerprint') or '')
        after = annotation_file_fingerprint(annotation_path)
        annotation_changed = bool(before and after and before != after)
        sync_message = ''
        if annotation_changed:
            try:
                result = synchronize_annotation_replicas(
                    annotation_path,
                    process.property('dataset_root') or None,
                    str(process.property('annotation_dir') or 'annotations'),
                )
            except AnnotationSyncError as exc:
                sync_message = '标注已修改，但关联副本未同步'
                log(f'⚠️ 标注副本同步已停止: {exc}')
            else:
                if result.updated:
                    sync_message = f'已同步 {len(result.updated)} 个关联标注副本'
                    log(
                        '🔄 标注副本同步: '
                        + ', '.join(str(path) for path in result.updated)
                    )
                elif result.ambiguous:
                    sync_message = '检测到同名来源歧义，未同步其他标注'
                elif result.errors:
                    sync_message = '未找到可安全同步的关联标注'
                if result.errors:
                    for error in result.errors:
                        log(f'⚠️ 标注副本同步: {error}')
        current_is_edited = (
            0 <= self._current_index < len(self._images)
            and self._same_path(self._images[self._current_index], image_path)
        )
        if current_is_edited:
            self._load_current()
        same_review_folder = bool(
            self._images
            and self._same_path(self._images[0].parent, image_path.parent)
        )
        if (
            annotation_changed
            and same_review_folder
            and self._detail
            and self._detail.has_review_stats()
        ):
            self._scan_review_stats()
        message = (
            '标注工具已关闭，当前标注已刷新'
            if current_is_edited else
            f'标注工具已关闭: {image_path.name}'
        )
        if sync_message:
            message += f'；{sync_message}'
        self._win.status_bar.showMessage(message, 5000)

    def _on_label_tool_error(self, process: QProcess):
        if process.property('error_reported'):
            return
        process.setProperty('error_reported', True)
        self._discard_label_tool_process(process)

        command_text = process.property('command_text') or process.program()
        QMessageBox.warning(
            self._win,
            '标注工具启动失败',
            '无法启动 X-AnyLabeling。\n\n'
            f'命令:\n{command_text}\n\n'
            f'错误:\n{process.errorString()}',
        )

    def _discard_label_tool_process(self, process: QProcess):
        if process in self._label_tool_processes:
            self._label_tool_processes.remove(process)

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        try:
            return left.resolve() == right.resolve()
        except OSError:
            return str(left) == str(right)

    def _active_task_type(self) -> str:
        return current_pose_review_config().task_type

    def _active_annotation_dir(self) -> str:
        return current_pose_review_config().annotation_dir

    def _on_module_selected(self, module_id: str):
        if module_id in {'train', 'eval'}:
            self.on_data_task_selected(getattr(self._win, '_current_task_type', 'pose'))

    def _on_annotation_dir_changed(self, name: str):
        """User picked an annotation set: switch globally and reload."""
        from app.models.annotation_review import set_active_annotation_dir
        set_active_annotation_dir(name)
        self._load_current()
        self._refresh_review_decision_views('标注集已切换')

    def _find_annotation(self, image_path: Path) -> Path | None:
        return find_annotation(
            image_path,
            annotation_dir=self._active_annotation_dir(),
        )

    def _expected_annotation_path(self, image_path: Path) -> Path | None:
        return expected_annotation_path(
            image_path,
            annotation_dir=self._active_annotation_dir(),
        )

    # --- Private ---

    def _connect_own_signals(self):
        tree = self._tree
        tree.annotation_dir_changed.connect(self._on_annotation_dir_changed)
        # 进入训练/评估模块时再次同步全局任务（防时序遗漏）
        win = getattr(self, '_win', None)
        if win is not None and hasattr(win, 'module_selected'):
            win.module_selected.connect(self._on_module_selected)
        viewer = self._viewer
        win = self._win

        tree.directory_selected.connect(self._on_dir_selected)
        tree.new_folder_requested.connect(self.on_new_folder)
        tree.rename_folder_requested.connect(self._on_rename_folder)
        tree.delete_folder_requested.connect(self._on_delete_folder)
        tree.files_dropped.connect(self._on_files_dropped)

        win.btn_prev.clicked.connect(self._navigate_prev)
        win.btn_next.clicked.connect(self._navigate_next)
        win.btn_copy.clicked.connect(self.on_copy)
        win.btn_move.clicked.connect(self.on_move)
        win.btn_delete.clicked.connect(self.on_delete)
        win.btn_rename.clicked.connect(self.on_rename)
        win.btn_new_folder.clicked.connect(
            lambda: self.on_new_folder(tree.selected_path() or '')
        )
        win.btn_open_label_tool.clicked.connect(self.on_open_label_tool)
        win.action_open_label_tool.triggered.connect(self.on_open_label_tool)
        btn_batch = getattr(win, 'btn_batch_annotate', None)
        if btn_batch is not None:
            btn_batch.clicked.connect(self.on_annotate_folder)
        win.task_selected.connect(self.on_data_task_selected)
        win.model_return_requested.connect(self.return_to_model_details)
        if hasattr(win, 'training_return_requested'):
            win.training_return_requested.connect(self.return_to_training_task)
        if hasattr(win, 'about_to_close'):
            win.about_to_close.connect(self.save_last_directory)
            if self._training_manager is not None:
                win.about_to_close.connect(
                    self._training_manager.shutdown_training
                )

        if self._model_manager is not None:
            self._model_manager.dataset_source_requested.connect(
                self.open_model_dataset
            )

        if self._training_manager is not None:
            self._training_manager.review_dataset_requested.connect(
                self.open_training_dataset
            )
            self._training_manager.dataset_prepared.connect(
                self.on_training_dataset_prepared
            )
            self._training_manager.training_completed.connect(
                self.on_training_completed
            )
            self._training_manager.model_result_requested.connect(
                self.open_training_result
            )
            self._training_manager.status_message.connect(
                lambda message: self._win.status_bar.showMessage(message, 4000)
            )

        viewer.setContextMenuPolicy(Qt.CustomContextMenu)
        viewer.customContextMenuRequested.connect(self._show_viewer_context_menu)

        if self._file_list:
            self._file_list.current_changed.connect(self._jump_to_image)

        if self._detail:
            self._detail.point_toggled.connect(viewer.set_point_visible)
            self._detail.all_points_toggled.connect(viewer.set_all_points_visible)
            self._detail.review_issue_selected.connect(self._on_review_issue_selected)
            self._detail.review_stats_requested.connect(self._scan_review_stats)
            self._detail.manual_accept_current_requested.connect(
                self._accept_current_review_file
            )
            self._detail.manual_ignore_issue_requested.connect(
                self._ignore_current_review_issue
            )
            self._detail.manual_restore_current_requested.connect(
                self._restore_current_review_file
            )
            self._detail.review_file_selected.connect(self._jump_to_image)
            self._detail.folder_keypoints_reorder_requested.connect(
                self.on_reorder_folder_keypoints
            )
            self._detail.pose_config_import_requested.connect(
                self.on_import_pose_review_config
            )
            self._detail.pose_config_selected.connect(
                self.on_pose_review_template_selected
            )

    def _save_last_dir(self, path: str):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.setValue(LAST_DIRECTORY_KEY, path)
        settings.sync()

    def _save_selection_state(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        if self._last_selected_directory:
            settings.setValue(
                LAST_SELECTED_DIRECTORY_KEY, self._last_selected_directory
            )
        if self._last_selected_image:
            settings.setValue(LAST_SELECTED_IMAGE_KEY, self._last_selected_image)
        settings.sync()

    @staticmethod
    def _stored_path(key: str) -> str | None:
        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.sync()
        value = settings.value(key)
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return str(Path(text).expanduser())

    def _load_saved_pose_review_config(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        self._pose_template_entries = self._load_pose_template_entries(settings)
        selected_id = str(settings.value(POSE_TEMPLATE_SELECTED_KEY) or '')

        legacy_path = settings.value('poseReviewConfigPath')
        if legacy_path and Path(str(legacy_path)).is_file():
            legacy_id = self._template_id_for_path(str(legacy_path))
            if not self._has_pose_template(legacy_id):
                try:
                    legacy_config = load_pose_review_config(str(legacy_path))
                    self._upsert_pose_template_entry(
                        legacy_id, legacy_config.name, str(legacy_path)
                    )
                    self._save_pose_template_entries()
                except ValueError:
                    pass
            if not selected_id:
                selected_id = legacy_id

        if not selected_id or not self._has_pose_template(selected_id):
            selected_id = POSE_TEMPLATE_BUILTIN_ID

        try:
            self._apply_pose_template(
                selected_id,
                persist=False,
                refresh_current=False,
                show_error=False,
            )
        except ValueError as exc:
            self._win.status_bar.showMessage(
                f'上次审查模板加载失败，已切回内置模板: {exc}', 6000
            )
            self._apply_pose_template(
                POSE_TEMPLATE_BUILTIN_ID,
                persist=False,
                refresh_current=False,
                show_error=False,
            )

    def _apply_pose_template(self, template_id: str,
                             persist: bool = True,
                             refresh_current: bool = True,
                             show_error: bool = True):
        template_id = str(template_id or POSE_TEMPLATE_BUILTIN_ID)
        if not self._has_pose_template(template_id):
            template_id = POSE_TEMPLATE_BUILTIN_ID

        try:
            config = self._load_pose_template_config(template_id)
        except ValueError as exc:
            if show_error:
                QMessageBox.warning(
                    self._win,
                    '模板切换失败',
                    f'无法切换到该审查模板。\n\n{exc}',
                )
            raise

        self._active_pose_template_id = template_id
        self._review_policy_config = config
        if self._images:
            config = self._review_config_for_images(config)
            apply_pose_review_config(config)
        if persist:
            settings = QSettings('FilesProcessQT', 'ImageManager')
            settings.setValue(POSE_TEMPLATE_SELECTED_KEY, template_id)
            settings.setValue(
                'poseReviewConfigPath',
                '' if self._is_builtin_task_template_id(template_id)
                else str(config.path or ''),
            )

        self._sync_pose_template_ui(config)

        if self._detail:
            self._detail.clear_review_stats()

        if refresh_current and self._images:
            self._load_current()

        if persist:
            self._win.status_bar.showMessage(
                f'已切换审查模板: {config.name}', 4000
            )

    def _load_pose_template_config(self, template_id: str):
        task_type = (
            'pose' if template_id == POSE_TEMPLATE_BUILTIN_ID
            else self._task_type_from_template_id(template_id)
        )
        if task_type:
            builtin_path = self._builtin_task_template_path(task_type)
            if builtin_path.is_file():
                return load_and_apply_pose_review_config(builtin_path)
            config = default_task_review_config(task_type)
            apply_pose_review_config(config)
            return config

        entry = self._pose_template_entry(template_id)
        if not entry:
            raise ValueError(f'模板不存在: {template_id}')

        path = Path(str(entry.get('path') or ''))
        if not path.is_file():
            raise ValueError(f'模板文件不存在: {path}')
        return load_and_apply_pose_review_config(path)

    def _sync_pose_template_ui(self, config=None):
        if not self._detail:
            return
        if config is None:
            config = current_pose_review_config()
        template_entries = self._template_entries_for_task(
            config.task_type,
            self._active_pose_template_id,
        )
        self._detail.set_pose_review_templates(
            template_entries,
            self._active_pose_template_id,
        )
        self._detail.set_pose_review_config(
            config.name,
            config.path,
            config.task_type,
            config.annotation_dir,
        )
        if hasattr(self._win, 'set_task_context'):
            self._win.set_task_context(config.task_type, config.annotation_dir)

    def _load_pose_template_entries(self, settings: QSettings) -> list[dict]:
        entries = self._builtin_task_template_entries()
        for path in self._stored_pose_template_paths(settings):
            if not Path(path).is_file():
                continue
            try:
                config = load_pose_review_config(path)
            except ValueError:
                continue
            template_id = self._template_id_for_path(path)
            if not any(entry.get('id') == template_id for entry in entries):
                entries.append({
                    'id': template_id,
                    'name': config.name,
                    'path': str(Path(path)),
                    'source': str(Path(path)),
                    'task_type': config.task_type,
                    'annotation_dir': config.annotation_dir,
                    'builtin': False,
                })
        return entries

    def _stored_pose_template_paths(self, settings: QSettings) -> list[str]:
        raw_value = settings.value(POSE_TEMPLATE_LIST_KEY)
        paths = []
        if raw_value:
            try:
                parsed = json.loads(str(raw_value))
                if isinstance(parsed, list):
                    paths.extend(str(path) for path in parsed if path)
            except json.JSONDecodeError:
                paths.append(str(raw_value))

        legacy_path = settings.value('poseReviewConfigPath')
        if legacy_path:
            paths.append(str(legacy_path))

        deduped = []
        seen = set()
        for path in paths:
            template_id = self._template_id_for_path(path)
            if template_id in seen:
                continue
            seen.add(template_id)
            deduped.append(path)
        return deduped

    def _save_pose_template_entries(self):
        paths = [
            str(entry.get('path'))
            for entry in self._pose_template_entries
            if (
                not self._is_builtin_task_template_id(str(entry.get('id') or ''))
                and entry.get('path')
            )
        ]
        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.setValue(
            POSE_TEMPLATE_LIST_KEY,
            json.dumps(paths, ensure_ascii=False),
        )

    def _upsert_pose_template_entry(self, template_id: str, name: str, path: str):
        path_text = str(Path(path))
        try:
            config = load_pose_review_config(path)
            task_type = config.task_type
            annotation_dir = config.annotation_dir
        except ValueError:
            task_type = ''
            annotation_dir = ''
        for entry in self._pose_template_entries:
            if entry.get('id') == template_id:
                entry.update({
                    'name': name,
                    'path': path_text,
                    'source': path_text,
                    'task_type': task_type,
                    'annotation_dir': annotation_dir,
                    'builtin': False,
                })
                self._sync_pose_template_ui()
                return

        self._pose_template_entries.append({
            'id': template_id,
            'name': name,
            'path': path_text,
            'source': path_text,
            'task_type': task_type,
            'annotation_dir': annotation_dir,
            'builtin': False,
        })
        self._sync_pose_template_ui()

    def _has_pose_template(self, template_id: str) -> bool:
        return self._pose_template_entry(template_id) is not None

    def _pose_template_entry(self, template_id: str) -> dict | None:
        for entry in self._pose_template_entries:
            if entry.get('id') == template_id:
                return entry
        return None

    def _builtin_task_template_entries(self) -> list[dict]:
        return [
            self._builtin_task_template_entry(task_type)
            for task_type in DATA_TASK_ORDER
        ]

    def _builtin_task_template_entry(self, task_type: str) -> dict:
        path = self._builtin_task_template_path(task_type)
        try:
            fallback = default_task_review_config(task_type)
        except ValueError:
            fallback = default_task_review_config('pose')
        name = fallback.name
        source = '内置默认配置'
        config_task_type = fallback.task_type
        annotation_dir = fallback.annotation_dir
        if path.is_file():
            try:
                config = load_pose_review_config(path)
                name = config.name
                source = str(path)
                config_task_type = config.task_type
                annotation_dir = config.annotation_dir
            except ValueError:
                source = str(path)
        return {
            'id': self._template_id_for_task(task_type),
            'name': f'内置: {name}',
            'path': str(path) if path.is_file() else '',
            'source': source,
            'task_type': config_task_type,
            'annotation_dir': annotation_dir,
            'builtin': True,
        }

    def _template_entries_for_task(self, task_type: str,
                                   active_id: str) -> list[dict]:
        task_type = str(task_type or 'pose')
        entries = [
            entry for entry in self._pose_template_entries
            if str(entry.get('task_type') or '') == task_type
        ]
        if not any(entry.get('id') == active_id for entry in entries):
            active_entry = self._pose_template_entry(active_id)
            if active_entry:
                entries.append(active_entry)
        return entries

    @staticmethod
    def _builtin_pose_template_path() -> Path:
        return AppController._builtin_task_template_path('pose')

    @staticmethod
    def _builtin_task_template_path(task_type: str) -> Path:
        task_type = str(task_type or 'pose').strip()
        preset = TASK_PRESETS.get(task_type, TASK_PRESETS['pose'])
        file_name = str(
            preset.get('template_file') or f'{task_type}_review_template.json'
        )
        return Path(__file__).resolve().parents[2] / 'resources' / file_name

    @staticmethod
    def _template_dir_setting_key(task_type: str) -> str:
        task_type = str(task_type or 'pose').strip()
        return f'reviewConfigDir/{task_type}'

    @staticmethod
    def _template_id_for_path(path: str | Path) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except OSError:
            return str(Path(path).expanduser())

    @staticmethod
    def _template_id_for_task(task_type: str) -> str:
        task_type = str(task_type or 'pose').strip()
        if task_type == 'pose':
            return POSE_TEMPLATE_BUILTIN_ID
        return f'{TASK_TEMPLATE_PREFIX}{task_type}'

    @staticmethod
    def _task_type_from_template_id(template_id: str) -> str | None:
        template_id = str(template_id or '')
        if template_id.startswith(TASK_TEMPLATE_PREFIX):
            return template_id[len(TASK_TEMPLATE_PREFIX):]
        return None

    @staticmethod
    def _is_builtin_task_template_id(template_id: str) -> bool:
        return (
            template_id == POSE_TEMPLATE_BUILTIN_ID
            or template_id.startswith(TASK_TEMPLATE_PREFIX)
        )

    def _on_dir_selected(self, path: str, fmt: DirFormat):
        self._last_selected_directory = str(Path(path).expanduser().resolve())
        self._last_selected_image = ''
        self._save_selection_state()
        self._images = list_images(path, fmt)
        self._current_index = 0
        self._apply_current_data_schema()
        total = len(self._images)
        self._win.set_nav_enabled(total > 1)
        self._win.set_action_enabled(total > 0)
        if self._detail:
            self._detail.clear_review_stats()
        if self._file_list:
            self._file_list.populate(self._images)
        self._load_current()

    def _apply_current_data_schema(self):
        if not self._images:
            return
        policy = self._review_policy_config or current_pose_review_config()
        config = self._review_config_for_images(policy)
        apply_pose_review_config(config)
        self._sync_pose_template_ui(config)

    def _review_config_for_images(self, config):
        annotations = []
        for image_path in self._images:
            annotation = find_annotation(
                image_path,
                annotation_dir=config.annotation_dir,
            )
            if annotation is not None and annotation.is_file():
                annotations.append(annotation)
        return review_config_from_data(
            config,
            annotations,
            dataset_yaml=self._dataset_yaml_for_images(),
        )

    def _dataset_yaml_for_images(self) -> Path | None:
        if not self._images:
            return None
        current = self._images[0].parent
        for parent in (current, *current.parents):
            candidate = parent / 'dataset.yaml'
            if candidate.is_file():
                return candidate
            if self._last_open_directory and self._same_path(
                parent, Path(self._last_open_directory)
            ):
                break
        return None

    def _load_current(self):
        if not self._images or self._current_index < 0:
            self._viewer._pixmap = None
            self._viewer._label.clear()
            self._win.set_counter_text(0, 0)
            return

        total = len(self._images)
        idx = max(0, min(self._current_index, total - 1))
        self._current_index = idx
        img_path = self._images[idx]
        self._last_selected_image = str(img_path.expanduser().resolve())

        self._viewer.load_image(img_path)
        ann = self._find_annotation(img_path)
        expected_ann = self._expected_annotation_path(img_path)
        self._viewer.load_annotation(ann)
        review_result = self._review_result(img_path, ann)

        self._win.set_counter_text(idx + 1, total)

        if self._file_list:
            self._file_list.set_current_index(idx)
        if self._detail:
            config = current_pose_review_config()
            self._detail.show_image(
                img_path,
                ann,
                expected_ann,
                config.task_type,
                config.annotation_dir,
                review_result,
            )
            shape_indices, point_indices = self._detail.review_highlights()
            self._viewer.set_review_highlights(shape_indices, point_indices)

    def _navigate_prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._load_current()

    def _navigate_next(self):
        if self._current_index < len(self._images) - 1:
            self._current_index += 1
            self._load_current()

    def _jump_to_image(self, index: int):
        if 0 <= index < len(self._images):
            self._current_index = index
            self._load_current()

    def _on_review_issue_selected(self, shape_indices, point_indices):
        if shape_indices or point_indices:
            if self._viewer.annotation_mode() == 0:
                self._viewer.set_annotation_mode(2)
            self._viewer.set_review_highlights(shape_indices, point_indices)
            self._win.status_bar.showMessage('已高亮审查问题点', 2000)
        else:
            self._viewer.clear_review_highlights()

    def _review_result(self, image_path: Path,
                       annotation_path: Path | None) -> ReviewDecisionResult:
        if annotation_path is None or not annotation_path.is_file():
            return ReviewDecisionResult((), (), ())
        issues = tuple(review_annotation_file(annotation_path, image_path))
        if self._review_decision_store is None:
            return ReviewDecisionResult(issues, issues, ())
        try:
            return self._review_decision_store.evaluate(
                image_path,
                annotation_path,
                current_pose_review_config(),
                issues,
            )
        except OSError as exc:
            self._win.status_bar.showMessage(
                f'人工复核记录读取失败，暂按算法结果显示: {exc}', 5000
            )
            return ReviewDecisionResult(issues, issues, ())

    def _accept_current_review_file(self):
        self._accept_review_file_at(self._current_index)

    def _ignore_current_review_issue(self, issue):
        target = self._review_target(self._current_index)
        if target is None or self._review_decision_store is None:
            return
        img_path, ann_path, _issues = target
        try:
            added = self._review_decision_store.accept(
                img_path,
                ann_path,
                current_pose_review_config(),
                [issue],
                scope='issue',
            )
        except OSError as exc:
            self._show_review_decision_error(exc)
            return
        message = '已忽略选中的算法误报' if added else '该问题已被人工忽略'
        self._refresh_review_decision_views(message)

    def _restore_current_review_file(self):
        self._restore_review_file_at(self._current_index)

    def _accept_review_file_at(self, image_index: int):
        target = self._review_target(image_index)
        if target is None or self._review_decision_store is None:
            return
        img_path, ann_path, issues = target
        if not issues:
            self._win.status_bar.showMessage('当前文件没有可人工处理的算法问题', 3000)
            return
        try:
            added = self._review_decision_store.accept(
                img_path,
                ann_path,
                current_pose_review_config(),
                issues,
                scope='file',
            )
        except OSError as exc:
            self._show_review_decision_error(exc)
            return
        message = (
            f'已将 {img_path.name} 标记为人工通过，确认 {added} 个算法误报'
        )
        self._refresh_review_decision_views(message)

    def _restore_review_file_at(self, image_index: int):
        target = self._review_target(image_index)
        if target is None or self._review_decision_store is None:
            return
        img_path, ann_path, _issues = target
        try:
            removed = self._review_decision_store.revoke(
                img_path,
                ann_path,
                current_pose_review_config(),
            )
        except OSError as exc:
            self._show_review_decision_error(exc)
            return
        message = (
            f'已撤销 {img_path.name} 的人工复核结论'
            if removed else '当前文件没有可撤销的人工复核结论'
        )
        self._refresh_review_decision_views(message)

    def _review_target(self, image_index: int):
        if not (0 <= image_index < len(self._images)):
            return None
        img_path = self._images[image_index]
        ann_path = self._find_annotation(img_path)
        if ann_path is None or not ann_path.is_file():
            self._win.status_bar.showMessage(
                '缺失或无效标注不能标记为人工通过', 3500
            )
            return None
        issues = review_annotation_file(ann_path, img_path)
        return img_path, ann_path, issues

    def _refresh_review_decision_views(self, message: str):
        self._load_current()
        if self._detail and self._detail.has_review_stats():
            self._scan_review_stats()
        self._win.status_bar.showMessage(message, 5000)

    def _show_review_decision_error(self, exc: OSError):
        QMessageBox.warning(
            self._win,
            '人工复核记录保存失败',
            f'无法写入数据集的 .review/review_decisions.json:\n{exc}',
        )

    def _scan_review_stats(self):
        if not self._images:
            return

        rows = []
        total = len(self._images)
        folder_summary = self._new_review_folder_summary(total)
        annotation_dir = self._active_annotation_dir()
        task_type = self._active_task_type()
        annotation_set_dir = annotation_set_dir_for_image(
            self._images[0],
            annotation_dir=annotation_dir,
        )
        folder_summary['annotation_set_dir'] = (
            str(annotation_set_dir) if annotation_set_dir is not None else ''
        )
        folder_summary['annotation_set_missing'] = (
            annotation_set_dir is None or not annotation_set_dir.is_dir()
        )
        self._win.status_bar.showMessage(
            f'正在审查当前文件夹: {total} 张图片, '
            f'任务={task_type}, 标注集={annotation_dir}...'
        )
        self._win.status_bar.repaint()

        for idx, img_path in enumerate(self._images):
            filename = img_path.name
            self._add_metric_file(
                folder_summary, 'overview:images', idx, filename,
                status='图片', detail='当前文件夹中的图片',
            )
            ann = self._find_annotation(img_path)
            if ann is None:
                folder_summary['missing_annotations'] += 1
                self._add_metric_file(
                    folder_summary, 'overview:missing', idx, filename,
                    status='缺失标注', detail='未找到对应 JSON 标注文件',
                )
                self._add_metric_file(
                    folder_summary, 'quality:issue', idx, filename,
                    status='有问题', detail='缺失标注文件',
                )
                continue

            annotation_summary = self._add_annotation_summary(
                folder_summary, ann
            )
            annotation_status = '标注有效' if annotation_summary.valid else 'JSON 无效'
            self._add_metric_file(
                folder_summary, 'overview:annotations', idx, filename,
                status=annotation_status,
                detail=annotation_summary.error or '已找到对应 JSON 标注文件',
            )
            if not annotation_summary.valid:
                self._add_metric_file(
                    folder_summary, 'overview:invalid', idx, filename,
                    status='JSON 无效', detail=annotation_summary.error,
                )
                self._add_metric_file(
                    folder_summary, 'quality:issue', idx, filename,
                    status='有问题', detail=annotation_summary.error,
                )
                continue

            self._index_annotation_metrics(
                folder_summary, annotation_summary, idx, filename
            )
            review_result = self._review_result(img_path, ann)
            raw_issues = list(review_result.raw_issues)
            issues = list(review_result.active_issues)
            accepted_issues = list(review_result.accepted_issues)
            if not raw_issues:
                self._add_metric_file(
                    folder_summary, 'quality:ok', idx, filename,
                    status='规则通过', detail='当前已执行规则未发现标注问题',
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
                    'index': idx,
                    'filename': filename,
                    'issues': [],
                    'accepted_issues': accepted_issues,
                    'status': 'manual',
                })
                self._add_metric_file(
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
            self._add_issue_summary(folder_summary, issues)
            self._add_metric_file(
                folder_summary, 'quality:issue', idx, filename,
                count=len(issues), status='有问题',
                detail=f'发现 {len(issues)} 个审查问题',
            )
            for rule, count in Counter(issue.rule for issue in issues).items():
                self._add_metric_file(
                    folder_summary, f'rule:{rule}', idx, filename,
                    count=count, status=f'{count} 个问题',
                    detail='; '.join(
                        issue.message for issue in issues if issue.rule == rule
                    ),
                )

        if self._detail:
            self._detail.show_review_stats(total, rows, folder_summary)

        problem_images = min(
            total,
            folder_summary['issue_files']
            + folder_summary['missing_annotations']
            + folder_summary['invalid_annotations'],
        )
        self._win.status_bar.showMessage(
            f'审查统计完成: {problem_images} / {total} 张图片待处理, '
            f'{folder_summary["manual_pass_files"]} 张人工通过',
            4000,
        )

    def _new_review_folder_summary(self, total_images: int) -> dict:
        config = current_pose_review_config()
        return {
            'total_images': total_images,
            'task_type': config.task_type,
            'annotation_dir': config.annotation_dir,
            'annotation_set_dir': '',
            'annotation_set_missing': False,
            'annotation_files': 0,
            'missing_annotations': 0,
            'invalid_annotations': 0,
            'checked_true': 0,
            'checked_false': 0,
            'checked_unknown': 0,
            'total_shapes': 0,
            'person_boxes': 0,
            'keypoints': 0,
            'other_shapes': 0,
            'target_class_counts': {label: 0 for label in config.target_classes},
            'target_class_file_counts': {label: 0 for label in config.target_classes},
            'keypoint_counts': {label: 0 for label in config.keypoints},
            'keypoint_file_counts': {label: 0 for label in config.keypoints},
            'shape_type_counts': {},
            'shape_type_file_counts': {},
            'issue_files': 0,
            'issue_count': 0,
            'raw_issue_files': 0,
            'raw_issue_count': 0,
            'manual_pass_files': 0,
            'accepted_issue_count': 0,
            'stale_review_files': 0,
            'rule_counts': {},
            'raw_rule_counts': {},
            'severity_counts': {},
            'metric_files': {},
        }

    def _add_annotation_summary(self, folder_summary: dict, ann_path: Path):
        folder_summary['annotation_files'] += 1
        summary = summarize_annotation_file(ann_path)
        if not summary.valid:
            folder_summary['invalid_annotations'] += 1
            return summary

        folder_summary['total_shapes'] += summary.shapes
        folder_summary['person_boxes'] += summary.person_boxes
        folder_summary['keypoints'] += summary.keypoints
        folder_summary['other_shapes'] += summary.other_shapes

        if summary.checked is True:
            folder_summary['checked_true'] += 1
        elif summary.checked is False:
            folder_summary['checked_false'] += 1
        else:
            folder_summary['checked_unknown'] += 1

        for label, count in (summary.target_class_counts or {}).items():
            folder_summary['target_class_counts'][label] = (
                folder_summary['target_class_counts'].get(label, 0) + count
            )
        for label, count in (summary.keypoint_counts or {}).items():
            folder_summary['keypoint_counts'][label] = (
                folder_summary['keypoint_counts'].get(label, 0) + count
            )
        for shape_type, count in (summary.shape_type_counts or {}).items():
            folder_summary['shape_type_counts'][shape_type] = (
                folder_summary['shape_type_counts'].get(shape_type, 0) + count
            )
        return summary

    @staticmethod
    def _index_annotation_metrics(folder_summary: dict, summary,
                                  image_index: int, filename: str):
        for label, count in (summary.target_class_counts or {}).items():
            if count <= 0:
                continue
            folder_summary['target_class_file_counts'][label] = (
                folder_summary['target_class_file_counts'].get(label, 0) + 1
            )
            AppController._add_metric_file(
                folder_summary, f'class:{label}', image_index, filename,
                count=count, status=f'{count} 个实例',
                detail=f'该图片包含 {count} 个 {label}',
            )

        expected_keypoints = summary.person_boxes
        for label, count in (summary.keypoint_counts or {}).items():
            if count > 0:
                folder_summary['keypoint_file_counts'][label] = (
                    folder_summary['keypoint_file_counts'].get(label, 0) + 1
                )
            if count == expected_keypoints:
                continue
            missing = max(0, expected_keypoints - count)
            extra = max(0, count - expected_keypoints)
            parts = []
            if missing:
                parts.append(f'缺失 {missing}')
            if extra:
                parts.append(f'多出 {extra}')
            status = '，'.join(parts) or '数量异常'
            AppController._add_metric_file(
                folder_summary, f'keypoint:{label}', image_index, filename,
                count=count, expected=expected_keypoints, status=status,
                detail=(
                    f'{label}: 实际 {count}，按 {expected_keypoints} 个人框'
                    f'期望 {expected_keypoints}'
                ),
            )

        for shape_type, count in (summary.shape_type_counts or {}).items():
            if count <= 0:
                continue
            folder_summary['shape_type_file_counts'][shape_type] = (
                folder_summary['shape_type_file_counts'].get(shape_type, 0) + 1
            )
            AppController._add_metric_file(
                folder_summary, f'shape:{shape_type}', image_index, filename,
                count=count, status=f'{count} 个 shape',
                detail=f'该图片包含 {count} 个 {shape_type or "unknown"} shape',
            )

    @staticmethod
    def _add_metric_file(folder_summary: dict, metric_key: str,
                         image_index: int, filename: str, count: int = 1,
                         expected: int | None = None, status: str = '',
                         detail: str = ''):
        record = {
            'index': image_index,
            'filename': filename,
            'count': max(0, int(count)),
            'status': status,
            'detail': detail,
        }
        if expected is not None:
            record['expected'] = max(0, int(expected))
        folder_summary.setdefault('metric_files', {}).setdefault(
            metric_key, []
        ).append(record)

    def _add_issue_summary(self, folder_summary: dict, issues: list):
        folder_summary['issue_files'] += 1
        folder_summary['issue_count'] += len(issues)
        for issue in issues:
            folder_summary['rule_counts'][issue.rule] = (
                folder_summary['rule_counts'].get(issue.rule, 0) + 1
            )
            folder_summary['severity_counts'][issue.severity] = (
                folder_summary['severity_counts'].get(issue.severity, 0) + 1
            )

    def _get_selected_images(self) -> list[Path]:
        if self._file_list:
            selected = self._file_list.get_selected()
            if selected:
                return selected
        if 0 <= self._current_index < len(self._images):
            return [self._images[self._current_index]]
        return []

    def _pick_destination(self) -> str | None:
        dest = QFileDialog.getExistingDirectory(self._win, '选择目标目录')
        return dest if dest else None

    def _finish_op(self, op_name: str, errors: list[str],
                    affected: list[Path] | None = None):
        if errors:
            QMessageBox.warning(self._win, f'{op_name}出错',
                                '\n'.join(errors[:10]))
        else:
            count = len(affected) if affected else 0
            self._win.status_bar.showMessage(f'{op_name}成功: {count} 张图片')
            self._win.status_bar.repaint()
        self.refresh()

    def _show_viewer_context_menu(self, pos):
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self._viewer)
        if self._current_index >= 0:
            menu.addAction('复制', self.on_copy)
            menu.addAction('移动到...', self.on_move)
            menu.addAction('删除', self.on_delete)
            menu.addAction('重命名', self.on_rename)
        menu.exec_(self._viewer.mapToGlobal(pos))

    def _on_rename_folder(self, path: str):
        p = Path(path)
        from app.views.dialogs import RenameDialog
        dlg = RenameDialog(p, self._win)
        if dlg.exec_():
            new_name = dlg.get_new_name()
            if new_name:
                err = rename_image(p, new_name, False)
                if err:
                    QMessageBox.warning(self._win, '重命名失败', err)
                self._tree.refresh()

    def _on_files_dropped(self, dest: str, sources: list):
        """Handle drag-drop of image files onto a directory tree item."""
        # Filter to only image files
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        img_paths = [p for p in sources
                     if isinstance(p, Path) and p.suffix.lower() in img_exts]
        if not img_paths:
            return
        errors = move_images(
            img_paths, dest, self._sync_annotation, self._active_annotation_dir()
        )
        self._finish_op('移动', errors, img_paths)

    def _on_delete_folder(self, path: str):
        p = Path(path)
        if not p.is_dir():
            return
        contents = list(p.iterdir())
        if contents:
            QMessageBox.warning(
                self._win, '无法删除',
                f"文件夹 '{p.name}' 不为空，请先清空。"
            )
            return
        answer = QMessageBox.question(
            self._win, '确认删除',
            f"确定要删除文件夹 '{p.name}' 吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            try:
                p.rmdir()
            except OSError as e:
                QMessageBox.warning(self._win, '删除失败', str(e))
            self._tree.refresh()
