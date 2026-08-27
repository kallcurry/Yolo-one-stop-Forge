"""Incremental test data tool — adds NEW subfolders to the test set.

Tracks which source subfolders have already been processed via
test_data/sources.txt. On each run, only subfolders NOT in
sources.txt are added to test_data/.
"""

import os
import random
import shutil
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout,
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


def run_add_test(source: str, selected_dirs: list[str],
                 dry_run: bool, use_copy: bool, test_ratio: float = 1.0):
    s = Path(source)
    test_dir = s / "test_data"
    sources_path = test_dir / SOURCES_FILE
    test_list_path = test_dir / TEST_LIST_FILE

    images_base = s / "images"
    annotations_base = s / "annotations"
    labels_base = s / "labels"
    has_labels = labels_base.is_dir()

    # Load already-processed sources
    processed = set()
    if sources_path.exists():
        with open(sources_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    processed.add(line)
        print(f"📋 已处理的源文件夹: {len(processed)} 个")
        for p in sorted(processed):
            print(f"   ✅ {p}")
    else:
        print("📋 首次添加测试集 (sources.txt 不存在)")

    # Filter: only new subfolders
    new_dirs = [d for d in selected_dirs if d not in processed]
    skipped = [d for d in selected_dirs if d in processed]

    if skipped:
        print(f"\n⏭️ 已跳过 {len(skipped)} 个已处理的文件夹:")
        for d in skipped:
            print(f"   ⏭️ {d}")

    if not new_dirs:
        print("\n⚠️ 没有新的子文件夹需要添加。所有选中的文件夹都已处理过。")
        return

    if test_ratio < 1.0:
        print(f"\n📦 将添加 {len(new_dirs)} 个新文件夹 (抽样比例: {test_ratio*100:.0f}%):")
    else:
        print(f"\n📦 将添加 {len(new_dirs)} 个新文件夹:")
    for d in new_dirs:
        print(f"   🆕 {d}")

    # Load existing test stems (from actual files + test_list.txt)
    existing_stems = set()
    test_images_dir = test_dir / "images"
    if test_images_dir.is_dir():
        for f in test_images_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                existing_stems.add(f.stem)
    if test_list_path.exists():
        with open(test_list_path) as f:
            for line in f:
                stem = line.split("#")[0].strip()
                if stem:
                    existing_stems.add(stem)

    # Also scan ALL training_data subfolders for stems to avoid
    training_stems = set()
    td_base = s / "training_data"
    if td_base.is_dir():
        for td_date in td_base.iterdir():
            if td_date.is_dir():
                td_img = td_date / "images"
                if td_img.is_dir():
                    for f in td_img.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                            training_stems.add(f.stem)
    if training_stems:
        print(f"📋 训练集已有 {len(training_stems)} 张图片，将自动跳过")

    # Prepare output dirs
    target_images = test_dir / "images"
    target_annotations = test_dir / "annotations"
    target_labels = test_dir / "labels"

    if not dry_run:
        test_dir.mkdir(parents=True, exist_ok=True)
        target_images.mkdir(parents=True, exist_ok=True)
        target_annotations.mkdir(parents=True, exist_ok=True)
        if has_labels:
            target_labels.mkdir(parents=True, exist_ok=True)

    copy_fn = shutil.copy2 if use_copy else os.link

    total_added = 0
    total_labels = 0
    total_skipped_existing = 0
    total_skipped_training = 0

    for dir_name in new_dirs:
        img_subdir = images_base / dir_name
        if not img_subdir.exists():
            print(f"  ⚠️ 跳过不存在的目录: {dir_name}")
            continue

        ann_dir = _find_sub_dir(annotations_base, dir_name, img_subdir)
        lbl_dir = _find_sub_dir(labels_base, dir_name, img_subdir) if has_labels else None

        image_files = sorted([
            f for f in img_subdir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        # Apply ratio: randomly sample a subset
        if test_ratio < 1.0:
            sample_n = max(1, int(len(image_files) * test_ratio))
            image_files = random.sample(image_files, sample_n)
            print(f"   {dir_name}: 抽样 {sample_n}/{len(image_files)} 张")

        added_from_dir = 0
        for img_path in image_files:
            stem = img_path.stem

            if stem in training_stems:
                total_skipped_training += 1
                continue

            if stem in existing_stems:
                total_skipped_existing += 1
                continue

            if ann_dir is None:
                continue
            ann_path = ann_dir / f"{stem}.json"
            if not ann_path.exists():
                continue

            existing_stems.add(stem)
            added_from_dir += 1
            total_added += 1

            if not dry_run:
                dest_img = target_images / img_path.name
                dest_ann = target_annotations / ann_path.name
                try:
                    if not dest_img.exists():
                        copy_fn(str(img_path), str(dest_img))
                    if not dest_ann.exists():
                        copy_fn(str(ann_path), str(dest_ann))
                    if lbl_dir:
                        lbl_path = lbl_dir / f"{stem}.txt"
                        if lbl_path.exists():
                            dest_lbl = target_labels / lbl_path.name
                            if not dest_lbl.exists():
                                copy_fn(str(lbl_path), str(dest_lbl))
                            total_labels += 1
                except OSError as e:
                    print(f"  ⚠️ {img_path.name}: {e}")

        print(f"   {dir_name}: +{added_from_dir} 张")

    # Write/update sources.txt and test_list.txt
    if not dry_run:
        with open(sources_path, 'w') as f:
            f.write("# 已添加到测试集的源子文件夹（每行一个）\n")
            all_sources = sorted(processed | set(new_dirs))
            for src_name in all_sources:
                f.write(f"{src_name}\n")

        with open(test_list_path, 'w') as f:
            f.write("# 测试集图片 stem 列表\n")
            for stem in sorted(existing_stems):
                f.write(f"{stem}\n")

    print(f"\n{'='*50}")
    print(f"📊 本次新增: {total_added} 张图片")
    if has_labels:
        print(f"   TXT 标签: {total_labels} 个")
    if total_skipped_training:
        print(f"   跳过(训练集已有): {total_skipped_training} 张")
    if total_skipped_existing:
        print(f"   跳过(测试集已有): {total_skipped_existing} 张")
    print(f"   测试集总计: {len(existing_stems)} 张图片")
    print(f"   已处理源文件夹: {len(processed) + len(new_dirs)} 个")

    if dry_run:
        print(f"\n🔍 dry-run 模式，未实际写入。")
    else:
        print(f"\n✅ 测试集已更新!")
        print(f"   图片: {target_images}")
        print(f"   标注: {target_annotations}")
        if has_labels:
            print(f"   标签: {target_labels}")
        print(f"   记录: {sources_path}")
        print(f"   清单: {test_list_path}")

    print(f"{'='*50}")


def create_dialog(parent=None):
    dlg = ToolDialog('添加测试集', parent)

    dlg.edit_source = dlg._add_dir_picker('源数据根目录:',
        stored_dataset_path())

    lbl = QLabel('选择要添加到测试集的子文件夹（已处理过的会自动跳过）:')
    dlg.param_widget.addWidget(lbl)

    list_row = QHBoxLayout()
    dlg.list_dirs = QListWidget()
    dlg.list_dirs.setSelectionMode(QListWidget.MultiSelection)
    dlg.list_dirs.setMaximumHeight(150)

    def _refresh_list():
        dlg.list_dirs.clear()
        src = Path(dlg.edit_source.text())
        imgs = src / "images"
        anns = src / "annotations"

        # Load processed sources
        processed = set()
        sources_path = src / "test_data" / SOURCES_FILE
        if sources_path.exists():
            with open(sources_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        processed.add(line)

        if imgs.exists():
            for name, ic, ac, ok in _list_available_subdirs(imgs, anns):
                status = "✅" if ok else "❌"
                tag = " [已处理]" if name in processed else ""
                dlg.list_dirs.addItem(f"{status} {name}{tag}  (图片:{ic}, 标注:{ac})")

    btn_refresh = QPushButton('🔄 刷新列表')
    btn_refresh.clicked.connect(_refresh_list)
    list_row.addWidget(dlg.list_dirs)
    btn_col = QVBoxLayout()
    btn_col.addWidget(btn_refresh)
    list_row.addLayout(btn_col)
    dlg.param_widget.addLayout(list_row)

    dlg.cb_dry = QCheckBox('预览模式 (dry-run, 不实际写入)')
    dlg.param_widget.addWidget(dlg.cb_dry)

    row_ratio = QHBoxLayout()
    row_ratio.addWidget(QLabel('抽样比例:'))
    dlg.sp_ratio = QDoubleSpinBox()
    dlg.sp_ratio.setRange(0.01, 1.0)
    dlg.sp_ratio.setValue(0.15)
    dlg.sp_ratio.setSingleStep(0.05)
    dlg.sp_ratio.setToolTip('从每个文件夹随机抽取的比例 (1.0 = 全部)')
    row_ratio.addWidget(dlg.sp_ratio)
    row_ratio.addWidget(QLabel(' (1.0=全部, 0.15=15%)'))
    row_ratio.addStretch()
    dlg.param_widget.addLayout(row_ratio)

    row_mode = QHBoxLayout()
    row_mode.addWidget(QLabel('写入模式:'))
    dlg.cb_mode = QComboBox()
    dlg.cb_mode.addItems(['硬链接 (节省空间)', '复制'])
    row_mode.addWidget(dlg.cb_mode)
    row_mode.addStretch()
    dlg.param_widget.addLayout(row_mode)

    def _get_selected():
        items = dlg.list_dirs.selectedItems()
        return [it.text().split()[1] for it in items]

    dlg.set_runner(lambda: run_add_test(
        dlg.edit_source.text(), _get_selected(),
        dlg.cb_dry.isChecked(), dlg.cb_mode.currentIndex() == 1,
        dlg.sp_ratio.value(),
    ))

    _refresh_list()
    return dlg
