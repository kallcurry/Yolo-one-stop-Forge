"""Find keypoint tool — wraps find_keypoint.py logic."""

import os
import json
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QSpinBox

from app.views.tool_dialog import ToolDialog, stored_dataset_path

KEYPOINTS = [
    "top_helmet", "left_helmet", "right_helmet", "nose", "jaw",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_shoulder_strap", "right_shoulder_strap",
    "left_waist", "right_waist", "left_hip", "right_hip", "left_thigh",
    "right_thigh", "left_knee", "right_knee", "left_ankle", "right_ankle"
]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _find_keypoint_in_json(json_path, keypoint):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False, []
    shapes = data.get('shapes', data.get('shapes ', []))
    details = []
    found = False
    for shape in shapes:
        label = shape.get('label', '').strip()
        shape_type = shape.get('shape_type', shape.get('shape_type ', '')).strip()
        if label == keypoint and shape_type == 'point':
            found = True
            difficult = shape.get('difficult', False)
            details.append({"difficult": difficult, "visible": 1 if difficult else 2})
    return found, details


def run_find_keypoint(base_dir: str, keypoint: str, date: str,
                      split: str, missing: bool, max_show: int, save_list: str):
    data_dir = Path(base_dir) / date
    ann_dir = data_dir / "annotations"
    if not ann_dir.exists():
        print(f"❌ 标注目录不存在: {ann_dir}")
        return

    mode_str = "缺少" if missing else "包含"
    print(f"🔍 查找 {mode_str}关键点 \"{keypoint}\"")
    print(f"   数据目录: {data_dir}")

    # Optional split filter
    split_stems = None
    if split != "all":
        split_img_dir = data_dir / "train_data" / "images" / split
        if split_img_dir.exists():
            split_stems = set()
            for f in split_img_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    split_stems.add(f.stem)
            print(f"   限定范围: {split} ({len(split_stems)} 张图片)")

    json_files = sorted(ann_dir.glob("*.json"))
    processed = 0
    matched = []

    for jf in json_files:
        if split_stems is not None and jf.stem not in split_stems:
            continue
        processed += 1
        has_kp, details = _find_keypoint_in_json(jf, keypoint)
        if missing:
            if not has_kp:
                matched.append((jf.stem, []))
        else:
            if has_kp:
                matched.append((jf.stem, details))

    total = processed
    print(f"\n📊 共扫描 {total} 个 JSON 标注")
    if missing:
        print(f"  ❌ 缺少 \"{keypoint}\": {len(matched)} 个标注")
    else:
        visible_count = sum(1 for _, d in matched if d)
        total_instances = sum(len(d) for _, d in matched)
        print(f"  ✅ 包含 \"{keypoint}\": {len(matched)} 个 ({len(matched)/total*100:.1f}%)")
        print(f"     至少一个实例可见: {visible_count}")
        print(f"     总实例数: {total_instances}")

    # Show results
    show_n = min(len(matched), max_show)
    print(f"\n📋 结果 (前 {show_n}/{len(matched)}):")
    for idx, (stem, details) in enumerate(matched[:show_n], 1):
        if missing:
            detail_str = "缺少此关键点"
        else:
            visible = sum(1 for d in details if not d["difficult"])
            diff = sum(1 for d in details if d["difficult"])
            parts = []
            if visible > 0:
                parts.append(f"可见×{visible}")
            if diff > 0:
                parts.append(f"遮挡×{diff}")
            detail_str = ", ".join(parts)
        print(f"  {idx:4d}. {stem:<45} {detail_str}")

    # Save list
    if save_list:
        with open(save_list, 'w', encoding='utf-8') as f:
            for stem, _ in matched:
                f.write(stem + "\n")
        print(f"\n💾 列表已保存: {save_list} ({len(matched)} 条)")

    print("\n✅ 查找完成!")


def create_dialog(parent=None):
    dlg = ToolDialog('查找关键点', parent)

    dlg.edit_base = dlg._add_dir_picker('数据目录:',
        stored_dataset_path('training_data'))

    # Keypoint combo
    row1 = QHBoxLayout()
    row1.addWidget(QLabel('关键点名称:'))
    dlg.cb_kpt = QComboBox()
    dlg.cb_kpt.addItems(KEYPOINTS)
    row1.addWidget(dlg.cb_kpt)
    row1.addStretch()
    dlg.param_widget.addLayout(row1)

    # Date
    row2 = QHBoxLayout()
    row2.addWidget(QLabel('日期目录:'))
    dlg.edit_date = QLineEdit('2026-06-29')
    row2.addWidget(dlg.edit_date)
    row2.addStretch()
    dlg.param_widget.addLayout(row2)

    # Split
    row3 = QHBoxLayout()
    row3.addWidget(QLabel('限定范围:'))
    dlg.cb_split = QComboBox()
    dlg.cb_split.addItems(['all', 'train', 'val'])
    row3.addWidget(dlg.cb_split)
    row3.addStretch()
    dlg.param_widget.addLayout(row3)

    # Options
    dlg.cb_missing = QCheckBox('查找缺少该关键点的标注')
    dlg.param_widget.addWidget(dlg.cb_missing)

    row4 = QHBoxLayout()
    row4.addWidget(QLabel('最多显示:'))
    dlg.sp_max = QSpinBox()
    dlg.sp_max.setRange(1, 10000)
    dlg.sp_max.setValue(50)
    row4.addWidget(dlg.sp_max)
    row4.addStretch()
    dlg.param_widget.addLayout(row4)

    dlg.edit_save = dlg._add_file_picker('保存列表到 (可选):', '')

    dlg.set_runner(lambda: run_find_keypoint(
        dlg.edit_base.text(), dlg.cb_kpt.currentText(), dlg.edit_date.text(),
        dlg.cb_split.currentText(), dlg.cb_missing.isChecked(),
        dlg.sp_max.value(), dlg.edit_save.text()
    ))
    return dlg
