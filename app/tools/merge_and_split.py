"""Unified dataset management: add to test set + merge to training set.

Order: test data first, then training (auto-skips test stems).
"""

import os
import random
import shutil
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QSpinBox, QVBoxLayout,
)

from app.models.annotation_review import current_pose_review_config
from app.models.dataset_preparation import (
    DatasetPreparationRequest,
    prepare_dataset,
    scan_dataset,
)
from app.views.tool_dialog import ToolDialog, stored_dataset_path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SOURCES_FILE = "sources.txt"
TEST_LIST_FILE = "test_list.txt"


def _find_sub_dir(base, subdir_name, images_subdir_path):
    standard = base / subdir_name
    if standard.exists() and standard.is_dir():
        return standard
    inside = images_subdir_path / base.name
    if inside.exists() and inside.is_dir():
        return inside
    return None


def _list_available_subdirs(images_base, annotations_base):
    subdirs = []
    for item in sorted(images_base.iterdir()):
        if not item.is_dir() or item.name in {"annotations", "labels", "__pycache__"}:
            continue
        ann_dir = _find_sub_dir(annotations_base, item.name, item)
        img_count = sum(1 for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        ann_count = sum(1 for _ in ann_dir.glob("*.json")) if ann_dir else 0
        subdirs.append((item.name, img_count, ann_count, ann_dir is not None))
    return subdirs


def _load_test_stems(test_dir: Path) -> set[str]:
    """Load all stems currently in test set (from disk + test_list.txt)."""
    stems = set()
    img_dir = test_dir / "images"
    if img_dir.is_dir():
        for f in img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                stems.add(f.stem)
    test_list = test_dir / TEST_LIST_FILE
    if test_list.exists():
        with open(test_list) as f:
            for line in f:
                stem = line.split("#")[0].strip()
                if stem:
                    stems.add(stem)
    return stems


def _load_processed_sources(test_dir: Path) -> set[str]:
    """Load already-processed test source folders from sources.txt."""
    processed = set()
    sources_path = test_dir / SOURCES_FILE
    if sources_path.exists():
        with open(sources_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    processed.add(line)
    return processed


def _load_training_stems(td_base: Path) -> set[str]:
    """Load all stems from all training_data subdirectories."""
    stems = set()
    if td_base.is_dir():
        for td_date in td_base.iterdir():
            if td_date.is_dir():
                td_img = td_date / "images"
                if td_img.is_dir():
                    for f in td_img.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                            stems.add(f.stem)
    return stems


def run_pipeline(source: str, selected_dirs: list[str],
                 do_train: bool, do_test: bool,
                 train_date: str, test_ratio: float,
                 dry_run: bool, use_copy: bool,
                 val_ratio: float = 0.2, seed: int = 42,
                 task_type: str = 'pose',
                 annotation_dir: str = 'annotations',
                 label_dir: str = 'labels',
                 allow_background_without_label: bool = False,
                 class_names: tuple[str, ...] = (),
                 keypoints: tuple[str, ...] = (),
                 left_right_pairs: tuple[tuple[str, str], ...] = ()):
    s = Path(source)
    images_base = s / "images"
    annotations_base = s / annotation_dir
    labels_base = s / label_dir
    has_labels = labels_base.is_dir()

    if not selected_dirs:
        print("⚠️ 未选择任何子文件夹。")
        return

    test_dir = s / "test_data"
    copy_fn = shutil.copy2 if use_copy else os.link

    # Pre-load existing stems for dedup
    existing_test_stems = _load_test_stems(test_dir)
    processed_sources = _load_processed_sources(test_dir)
    training_stems = _load_training_stems(s / "training_data")

    print(f"📋 测试集已有: {len(existing_test_stems)} 张")
    print(f"📋 训练集已有: {len(training_stems)} 张")
    if processed_sources:
        print(f"📋 已处理源文件夹: {len(processed_sources)} 个")
    print()

    # --- Phase 1: Add test data ---
    test_new_stems = set()
    test_new_sources = []
    if do_test:
        print("=" * 50)
        print("📦 Phase 1: 添加测试集")
        print("=" * 50)

        new_test_dirs = [d for d in selected_dirs if d not in processed_sources]
        skipped = [d for d in selected_dirs if d in processed_sources]
        if skipped:
            print(f"⏭️ 跳过已处理的文件夹: {', '.join(skipped)}")

        if new_test_dirs:
            if test_ratio < 1.0:
                print(f"抽样比例: {test_ratio*100:.0f}%")

            target_timg = test_dir / "images"
            target_tann = test_dir / annotation_dir
            target_tlbl = test_dir / "labels"

            if not dry_run:
                test_dir.mkdir(parents=True, exist_ok=True)
                target_timg.mkdir(parents=True, exist_ok=True)
                target_tann.mkdir(parents=True, exist_ok=True)
                if has_labels:
                    target_tlbl.mkdir(parents=True, exist_ok=True)

            total_test_added = 0
            total_test_lbl = 0

            for dir_name in new_test_dirs:
                img_subdir = images_base / dir_name
                if not img_subdir.exists():
                    continue
                ann_dir = _find_sub_dir(annotations_base, dir_name, img_subdir)
                lbl_dir = _find_sub_dir(labels_base, dir_name, img_subdir) if has_labels else None

                image_files = sorted([
                    f for f in img_subdir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ])

                # Apply ratio
                if test_ratio < 1.0 and image_files:
                    sample_n = max(1, int(len(image_files) * test_ratio))
                    image_files = random.sample(image_files, sample_n)
                    print(f"   {dir_name}: 抽样 {len(image_files)}/{sum(1 for _ in img_subdir.iterdir() if _.suffix.lower() in IMAGE_EXTENSIONS)} 张")

                added = 0
                for img_path in image_files:
                    stem = img_path.stem
                    if stem in training_stems or stem in existing_test_stems or stem in test_new_stems:
                        continue
                    if ann_dir is None:
                        continue
                    ann_path = ann_dir / f"{stem}.json"
                    if not ann_path.exists():
                        continue

                    test_new_stems.add(stem)
                    added += 1

                    if not dry_run:
                        dest_img = target_timg / img_path.name
                        dest_ann = target_tann / ann_path.name
                        try:
                            if not dest_img.exists():
                                copy_fn(str(img_path), str(dest_img))
                            if not dest_ann.exists():
                                copy_fn(str(ann_path), str(dest_ann))
                            if lbl_dir:
                                lbl_path = lbl_dir / f"{stem}.txt"
                                if lbl_path.exists():
                                    copy_fn(str(lbl_path), str(target_tlbl / lbl_path.name))
                                    total_test_lbl += 1
                        except OSError as e:
                            print(f"  ⚠️ {img_path.name}: {e}")

                if added > 0:
                    print(f"   {dir_name}: +{added} 张")
                    test_new_sources.append(dir_name)
                total_test_added += added

            # Update sources.txt and test_list.txt
            if not dry_run and test_new_sources:
                all_sources = sorted(processed_sources | set(test_new_sources))
                with open(test_dir / SOURCES_FILE, 'w') as f:
                    f.write("# 已添加到测试集的源子文件夹（每行一个）\n")
                    for src_name in all_sources:
                        f.write(f"{src_name}\n")
                all_stems = sorted(existing_test_stems | test_new_stems)
                with open(test_dir / TEST_LIST_FILE, 'w') as f:
                    f.write("# 测试集图片 stem 列表\n")
                    for stem in all_stems:
                        f.write(f"{stem}\n")

            print(f"\n   测试集新增: {total_test_added} 张")
            if total_test_lbl:
                print(f"   TXT 标签: {total_test_lbl} 个")
            if dry_run:
                print(f"   🔍 dry-run，未写入")
        else:
            print("   没有新的子文件夹需要添加")

    # --- Phase 2: Merge and split training data ---
    train_added = 0
    if do_train:
        print("\n" + "=" * 50)
        print("📦 Phase 2: 合并并划分训练集")
        print("=" * 50)
        request = DatasetPreparationRequest(
            dataset_root=s,
            source_names=tuple(selected_dirs),
            target_name=train_date,
            task_type=task_type,
            annotation_dir=annotation_dir,
            label_dir=label_dir,
            val_ratio=val_ratio,
            seed=seed,
            use_copy=use_copy,
            exclude_test=True,
            allow_background_without_label=allow_background_without_label,
            class_names=class_names,
            keypoints=keypoints,
            left_right_pairs=left_right_pairs,
        )
        scan = scan_dataset(request)
        train_added = len(scan.samples)
        print(f"   任务: {task_type}")
        print(f"   JSON 标注目录: {annotation_dir}")
        print(f"   YOLO 标签目录: {label_dir}")
        print(f"   有效样本: {len(scan.samples)}")
        print(f"   测试集跳过: {len(scan.test_excluded)}")
        print(f"   重名跳过: {len(scan.duplicate_images)}")
        print(f"   缺少 JSON: {len(scan.missing_annotations)}")
        print(f"   缺少 TXT: {len(scan.missing_labels)}")
        if scan.background_without_labels:
            print(
                f"   疑似背景且无 TXT: {len(scan.background_without_labels)}"
            )
        if not scan.can_prepare:
            raise RuntimeError(scan.blocking_message())

        target_base = s / 'training_data' / train_date
        if dry_run:
            val_count = max(1, int(round(train_added * val_ratio)))
            val_count = min(val_count, max(0, train_added - 1))
            print("   🔍 dry-run，未写入")
            print(f"   目标: {target_base}")
            print(f"   预计 TRAIN: {train_added - val_count}")
            print(f"   预计 VAL: {val_count}")
        else:
            prepared = prepare_dataset(request, scan)
            print(f"\n✅ 训练数据准备完成: {prepared.batch_root}")
            print(f"   合计: {prepared.total_count}")
            print(f"   TRAIN: {prepared.train_count}")
            print(f"   VAL: {prepared.val_count}")
            print(f"   配置: {prepared.dataset_yaml}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("🎉 全部完成!")
    if do_test and test_new_stems:
        print(f"   测试集新增: {len(test_new_stems)} 张 (总计 {len(existing_test_stems) + len(test_new_stems)} 张)")
    if do_train:
        print(f"   训练集新增: {train_added} 张")
    print(f"{'='*50}")


def create_dialog(parent=None):
    dlg = ToolDialog('数据集管理（测试集 + 训练集）', parent)
    review_config = current_pose_review_config()

    dlg.edit_source = dlg._add_dir_picker('源数据根目录:',
        stored_dataset_path())

    lbl = QLabel('选择子文件夹:')
    dlg.param_widget.addWidget(lbl)

    list_row = QHBoxLayout()
    dlg.list_dirs = QListWidget()
    dlg.list_dirs.setSelectionMode(QListWidget.MultiSelection)
    dlg.list_dirs.setMaximumHeight(130)

    def _refresh_list():
        dlg.list_dirs.clear()
        src = Path(dlg.edit_source.text())
        imgs = src / "images"
        anns = src / review_config.annotation_dir
        processed = _load_processed_sources(src / "test_data")
        if imgs.exists():
            for name, ic, ac, ok in _list_available_subdirs(imgs, anns):
                status = "✅" if ok else "❌"
                tag = " [已处理]" if name in processed else ""
                item_text = f"{status} {name}{tag}  (图片:{ic}, 标注:{ac})"
                dlg.list_dirs.addItem(item_text)
                dlg.list_dirs.item(dlg.list_dirs.count() - 1).setData(
                    Qt.UserRole, name
                )

    btn_refresh = QPushButton('🔄 刷新列表')
    btn_refresh.clicked.connect(_refresh_list)
    list_row.addWidget(dlg.list_dirs)
    btn_col = QVBoxLayout()
    btn_col.addWidget(btn_refresh)
    list_row.addLayout(btn_col)
    dlg.param_widget.addLayout(list_row)

    # Target options
    dlg.cb_train = QCheckBox('添加到训练集')
    dlg.cb_train.setChecked(True)
    dlg.param_widget.addWidget(dlg.cb_train)

    task_row = QHBoxLayout()
    task_row.addWidget(QLabel('当前任务:'))
    task_row.addWidget(QLabel(
        f'{review_config.task_type} / {review_config.annotation_dir}'
    ))
    task_row.addStretch()
    dlg.param_widget.addLayout(task_row)

    row_date = QHBoxLayout()
    row_date.addWidget(QLabel('训练集日期:'))
    dlg.edit_date = QLineEdit(datetime.now().strftime('%Y-%m-%d'))
    row_date.addWidget(dlg.edit_date)
    row_date.addStretch()
    dlg.param_widget.addLayout(row_date)

    row_val = QHBoxLayout()
    row_val.addWidget(QLabel('验证集比例:'))
    dlg.sp_val_ratio = QDoubleSpinBox()
    dlg.sp_val_ratio.setRange(0.05, 0.5)
    dlg.sp_val_ratio.setValue(0.2)
    dlg.sp_val_ratio.setSingleStep(0.05)
    row_val.addWidget(dlg.sp_val_ratio)
    row_val.addWidget(QLabel('(默认 20%)'))
    row_val.addStretch()
    dlg.param_widget.addLayout(row_val)

    row_seed = QHBoxLayout()
    row_seed.addWidget(QLabel('随机种子:'))
    dlg.sp_seed = QSpinBox()
    dlg.sp_seed.setRange(0, 999999)
    dlg.sp_seed.setValue(42)
    row_seed.addWidget(dlg.sp_seed)
    row_seed.addStretch()
    dlg.param_widget.addLayout(row_seed)

    dlg.cb_background = QCheckBox(
        '确认将空 JSON 且无 TXT 的样本作为背景（生成空 TXT）'
    )
    dlg.param_widget.addWidget(dlg.cb_background)

    dlg.cb_test = QCheckBox('添加到测试集')
    dlg.cb_test.setChecked(True)
    dlg.param_widget.addWidget(dlg.cb_test)

    row_ratio = QHBoxLayout()
    row_ratio.addWidget(QLabel('测试集抽样比例:'))
    dlg.sp_ratio = QDoubleSpinBox()
    dlg.sp_ratio.setRange(0.01, 1.0)
    dlg.sp_ratio.setValue(0.15)
    dlg.sp_ratio.setSingleStep(0.05)
    row_ratio.addWidget(dlg.sp_ratio)
    row_ratio.addWidget(QLabel('(1.0=全部)'))
    row_ratio.addStretch()
    dlg.param_widget.addLayout(row_ratio)

    dlg.cb_dry = QCheckBox('预览模式 (dry-run, 不实际写入)')
    dlg.param_widget.addWidget(dlg.cb_dry)

    row_mode = QHBoxLayout()
    row_mode.addWidget(QLabel('写入模式:'))
    dlg.cb_mode = QComboBox()
    dlg.cb_mode.addItems(['硬链接 (节省空间)', '复制'])
    row_mode.addWidget(dlg.cb_mode)
    row_mode.addStretch()
    dlg.param_widget.addLayout(row_mode)

    def _get_selected():
        items = dlg.list_dirs.selectedItems()
        return [str(it.data(Qt.UserRole)) for it in items]

    dlg.set_runner(lambda: run_pipeline(
        dlg.edit_source.text(), _get_selected(),
        dlg.cb_train.isChecked(), dlg.cb_test.isChecked(),
        dlg.edit_date.text(), dlg.sp_ratio.value(),
        dlg.cb_dry.isChecked(), dlg.cb_mode.currentIndex() == 1,
        dlg.sp_val_ratio.value(), dlg.sp_seed.value(),
        review_config.task_type, review_config.annotation_dir, 'labels',
        dlg.cb_background.isChecked(), (), (), (),
    ))

    _refresh_list()
    return dlg
