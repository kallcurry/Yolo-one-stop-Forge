"""Swap labels tool — wraps swap_labels.py logic."""

import os
import shutil
from pathlib import Path

from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit

from app.views.tool_dialog import ToolDialog, stored_dataset_path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def run_swap(base_path: str, old_date: str, new_date: str,
             dry_run: bool, use_copy: bool):
    base = Path(base_path)
    old_dir = base / old_date
    new_dir = base / new_date

    if not old_dir.exists():
        print(f"❌ 原日期目录不存在: {old_dir}")
        return
    if not new_dir.exists():
        print(f"❌ 新日期目录不存在: {new_dir}")
        return

    new_images_src = new_dir / "images"
    new_labels_src = new_dir / "labels"

    if not new_labels_src.exists():
        print(f"❌ 新标签目录不存在: {new_labels_src}")
        return

    old_train_img = old_dir / "train_data" / "images" / "train"
    old_val_img = old_dir / "train_data" / "images" / "val"

    if not old_train_img.exists():
        print(f"❌ 原 train images 不存在: {old_train_img}")
        return

    new_train_img_dst = new_dir / "train_data" / "images" / "train"
    new_val_img_dst = new_dir / "train_data" / "images" / "val"
    new_train_lbl_dst = new_dir / "train_data" / "labels" / "train"
    new_val_lbl_dst = new_dir / "train_data" / "labels" / "val"

    copy_fn = shutil.copy2 if use_copy else os.link
    mode = "复制" if use_copy else "硬链接"

    print(f"📋 划分来源: {old_date}")
    print(f"🏷️ 标签来源: {new_date}")
    print(f"🔧 模式: {mode}")
    print()

    # Pre-build stem→filename mapping for new images
    stem_to_file = {}
    for f in new_images_src.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stem_to_file[f.stem] = f.name

    stats = {}

    for split, old_img_dir, img_dst, lbl_dst in [
        ("train", old_train_img, new_train_img_dst, new_train_lbl_dst),
        ("val", old_val_img, new_val_img_dst, new_val_lbl_dst),
    ]:
        if not old_img_dir.exists():
            print(f"⚠️ {split} 图片目录不存在，跳过: {old_img_dir}")
            stats[split] = {"img": 0, "lbl": 0, "miss": 0}
            continue

        if not dry_run:
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)

        old_images = sorted([
            f for f in old_img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        img_count = 0
        lbl_count = 0
        miss_count = 0

        for img_file in old_images:
            stem = img_file.stem
            matched_name = stem_to_file.get(stem)

            if matched_name is None and (new_images_src / img_file.name).exists():
                matched_name = img_file.name

            if matched_name is None:
                miss_count += 1
                continue

            new_img_path = new_images_src / matched_name
            dest_img = img_dst / matched_name

            if not dry_run and not dest_img.exists():
                try:
                    copy_fn(str(new_img_path), str(dest_img))
                except OSError as e:
                    print(f"  ⚠️ {matched_name}: {e}")
            img_count += 1

            # Copy label
            new_lbl_path = new_labels_src / f"{stem}.txt"
            dest_lbl = lbl_dst / f"{stem}.txt"
            if new_lbl_path.exists():
                if not dry_run and not dest_lbl.exists():
                    try:
                        copy_fn(str(new_lbl_path), str(dest_lbl))
                    except OSError:
                        pass
                lbl_count += 1

        stats[split] = {"img": img_count, "lbl": lbl_count, "miss": miss_count}

    print()
    print("=" * 50)
    for s in ["train", "val"]:
        st = stats[s]
        print(f"  {s.upper()}: 图片 {st['img']}, 标签 {st['lbl']}"
              + (f", 缺图片 {st['miss']}" if st['miss'] else ""))
    total_img = stats['train']['img'] + stats['val']['img']
    total_lbl = stats['train']['lbl'] + stats['val']['lbl']
    print(f"  合计: 图片 {total_img}, 标签 {total_lbl}")
    print(f"\n{'🔍 dry-run 模式' if dry_run else '✅ 完成!'}")
    if not dry_run:
        print(f"   输出: {new_dir / 'train_data'}")
    print("=" * 50)


def create_dialog(parent=None):
    dlg = ToolDialog('标签替换', parent)

    dlg.edit_base = dlg._add_dir_picker('数据目录:',
        stored_dataset_path('training_data'))

    row_old = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    row_old.addWidget(QLabel('原日期 (划分方案来源):'))
    dlg.edit_old = QLineEdit('2026-06-27')
    row_old.addWidget(dlg.edit_old)
    dlg.param_widget.addLayout(row_old)

    row_new = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    row_new.addWidget(QLabel('新日期 (标签来源):'))
    dlg.edit_new = QLineEdit('2026-06-29')
    row_new.addWidget(dlg.edit_new)
    dlg.param_widget.addLayout(row_new)

    dlg.cb_dry = QCheckBox('预览模式 (dry-run)')
    dlg.param_widget.addWidget(dlg.cb_dry)

    row_mode = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    row_mode.addWidget(QLabel('写入模式:'))
    dlg.cb_mode = QComboBox()
    dlg.cb_mode.addItems(['硬链接 (节省空间)', '复制'])
    row_mode.addWidget(dlg.cb_mode)
    row_mode.addStretch()
    dlg.param_widget.addLayout(row_mode)

    dlg.set_runner(lambda: run_swap(
        dlg.edit_base.text(), dlg.edit_old.text(), dlg.edit_new.text(),
        dlg.cb_dry.isChecked(), dlg.cb_mode.currentIndex() == 1
    ))
    return dlg
