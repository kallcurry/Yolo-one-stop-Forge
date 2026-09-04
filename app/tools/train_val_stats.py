"""Train/Val statistics tool — wraps train_val_stats.py core logic."""

import os
import json
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit

from app.models.annotation_schema import infer_annotation_schema
from app.views.tool_dialog import ToolDialog, stored_dataset_path

TARGET_CLASSES = []
KEYPOINTS = []
NUM_KPTS = 0
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _get_empty_pose_stats():
    return {
        'image_count': 0, 'json_count': 0, 'label_count': 0,
        'classes': {cls: {'instances': 0, 'kpt_0': [0]*NUM_KPTS, 'kpt_1': [0]*NUM_KPTS, 'kpt_2': [0]*NUM_KPTS} for cls in TARGET_CLASSES},
        'total_kpt_0': [0]*NUM_KPTS, 'total_kpt_1': [0]*NUM_KPTS, 'total_kpt_2': [0]*NUM_KPTS
    }


def _current_task_type() -> str:
    from app.models.task_context import current_task_type
    return current_task_type()


def _configure_schema(annotation_paths, dataset_yaml=None):
    global NUM_KPTS
    schema = infer_annotation_schema(
        annotation_paths,
        task_type=_current_task_type(),

        dataset_yaml=dataset_yaml,
    )
    TARGET_CLASSES[:] = list(schema.target_classes)
    KEYPOINTS[:] = list(schema.keypoints)
    NUM_KPTS = len(KEYPOINTS)


def _analyze_labelme_json(json_path, stats, source_instances=None):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    shapes = data.get('shapes', data.get('shapes ', []))
    instances_data = defaultdict(lambda: {'box_label': None, 'points': {}})
    for shape in shapes:
        label = shape.get('label', '').strip()
        group_id = shape.get('group_id', 0)
        shape_type = shape.get('shape_type', shape.get('shape_type ', '')).strip()
        difficult = shape.get('difficult', False)
        if label in TARGET_CLASSES and shape_type == 'rectangle':
            if instances_data[group_id]['box_label'] is None:
                instances_data[group_id]['box_label'] = label
        elif label in KEYPOINTS and shape_type == 'point':
            instances_data[group_id]['points'][label] = {'difficult': difficult}
    for gid, inst_data in instances_data.items():
        box_label = inst_data['box_label']
        if box_label not in TARGET_CLASSES:
            continue
        stats['classes'][box_label]['instances'] += 1
        if source_instances is not None:
            source_instances[box_label] = source_instances.get(box_label, 0) + 1
        points = inst_data['points']
        for i, kp_name in enumerate(KEYPOINTS):
            if kp_name in points:
                if points[kp_name]['difficult']:
                    stats['classes'][box_label]['kpt_1'][i] += 1
                    stats['total_kpt_1'][i] += 1
                else:
                    stats['classes'][box_label]['kpt_2'][i] += 1
                    stats['total_kpt_2'][i] += 1
            else:
                stats['classes'][box_label]['kpt_0'][i] += 1
                stats['total_kpt_0'][i] += 1


def run_train_val_stats(train_path: str, test_path: str, no_test: bool):
    train_dir = Path(train_path)
    if not train_dir.exists():
        print(f"❌ 训练集路径不存在: {train_dir}")
        return

    # Determine structure: already split vs flat
    train_split_img = train_dir / "train_data" / "images" / "train"
    val_split_img = train_dir / "train_data" / "images" / "val"
    flat_img = train_dir / "images"
    train_ann_dir = train_dir / "annotations"

    _configure_schema(
        train_ann_dir.glob('*.json'),
        dataset_yaml=train_dir / 'dataset.yaml',
    )

    is_split = train_split_img.is_dir()

    print(f"📊 训练集: {train_dir}")
    print(f"   数据结构: {'已拆分 (train/val)' if is_split else '未拆分 (全部图片)'}")

    if is_split:
        # Count train split
        train_imgs = sorted([f for f in train_split_img.iterdir()
                             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
        val_imgs = sorted([f for f in val_split_img.iterdir()
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]) if val_split_img.is_dir() else []
        all_imgs = sorted([f for f in flat_img.iterdir()
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]) if flat_img.is_dir() else []

        print(f"   图片: train={len(train_imgs)}, val={len(val_imgs)}, 合计={len(train_imgs)+len(val_imgs)}")
        if all_imgs:
            total_all = len(all_imgs)
            total_split = len(train_imgs) + len(val_imgs)
            if total_all != total_split:
                print(f"   ⚠️ images/ 总数={total_all}, train+val={total_split} (差{total_all-total_split})")

        # Analyze train split
        print(f"\n   === 训练集 (train) ===")
        stats = _count_split(train_split_img, train_ann_dir, train_dir / "train_data" / "labels" / "train")
        _print_split_report(stats, "TRAIN")

        # Analyze val split
        if val_split_img.is_dir():
            print(f"\n   === 验证集 (val) ===")
            vstats = _count_split(val_split_img, train_ann_dir, train_dir / "train_data" / "labels" / "val")
            _print_split_report(vstats, "VAL")
    else:
        # Flat: count all images
        image_files = sorted([f for f in flat_img.iterdir()
                              if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]) if flat_img.is_dir() else []
        print(f"   图片总数: {len(image_files)}")

        stats = _get_empty_pose_stats()
        stats['image_count'] = len(image_files)
        labels_dir = train_dir / "labels"
        stats['label_count'] = sum(1 for f in labels_dir.iterdir() if f.suffix == '.txt') if labels_dir.is_dir() else 0
        for img in image_files:
            json_path = train_ann_dir / f"{img.stem}.json"
            if json_path.exists():
                _analyze_labelme_json(json_path, stats)
                stats['json_count'] += 1
        _print_split_report(stats, "全部图片 (未拆分)")

    # Analyze test set
    if not no_test and test_path:
        test_dir = Path(test_path)
        if test_dir.exists():
            test_img_dir = test_dir / "images"
            test_ann_dir = test_dir / "annotations"
            if test_img_dir.exists():
                tstats = _get_empty_pose_stats()
                t_imgs = sorted([f for f in test_img_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
                tstats['image_count'] = len(t_imgs)
                for img in t_imgs:
                    json_path = test_ann_dir / f"{img.stem}.json"
                    if json_path.exists():
                        _analyze_labelme_json(json_path, tstats)
                        tstats['json_count'] += 1
                _print_split_report(tstats, "TEST (测试集)")

                # Cross validation: check test vs train+val
                if is_split:
                    train_stems = {f.stem for f in train_imgs} | {f.stem for f in val_imgs}
                else:
                    train_stems = {f.stem for f in image_files}
                test_stems = {f.stem for f in t_imgs}
                overlap = train_stems & test_stems
                if overlap:
                    print(f"❌ 训练/测试集交叉: {len(overlap)} 张重复!")
                else:
                    print(f"✅ 训练/测试集零交叉")

    print("\n✅ 统计完成!")


def _count_split(img_dir, ann_dir, lbl_dir):
    """Count stats for a single split directory."""
    stats = _get_empty_pose_stats()
    image_files = sorted([f for f in img_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
    stats['image_count'] = len(image_files)
    stats['label_count'] = sum(1 for f in lbl_dir.iterdir() if f.suffix == '.txt') if lbl_dir.is_dir() else 0
    for img in image_files:
        json_path = ann_dir / f"{img.stem}.json"
        if json_path.exists():
            _analyze_labelme_json(json_path, stats)
            stats['json_count'] += 1
    return stats


def _print_split_report(stats, name):
    total = sum(stats['classes'][c]['instances'] for c in TARGET_CLASSES)
    print(f"\n{'='*60}")
    print(f"  {name}: 图片={stats['image_count']}, 标注JSON={stats['json_count']}, 实例={total}")
    print(f"{'='*60}")
    for cls in TARGET_CLASSES:
        cnt = stats['classes'][cls]['instances']
        print(f"  {cls:<25}: {cnt}")
    if total > 0:
        bad_rates = []
        for i, kp in enumerate(KEYPOINTS):
            c0, c1, c2 = stats['total_kpt_0'][i], stats['total_kpt_1'][i], stats['total_kpt_2'][i]
            total_k = c0 + c1 + c2
            if total_k > 0:
                bad_rates.append((kp, (c0+c1)/total_k*100))
        bad_rates.sort(key=lambda x: -x[1])
        print(f"\n  🔴 不良率 TOP5:")
        for kp, rate in bad_rates[:5]:
            print(f"     {kp:<30} {rate:.1f}%")


def create_dialog(parent=None):
    dlg = ToolDialog('训练/测试集统计', parent)
    dlg.edit_train = dlg._add_dir_picker('训练集路径:',
        stored_dataset_path('training_data'))
    dlg.edit_test = dlg._add_dir_picker('测试集路径 (可选):',
        stored_dataset_path('test_data'))
    cb_no_test = QCheckBox('跳过测试集')
    dlg.param_widget.addWidget(cb_no_test)
    dlg.set_runner(lambda: run_train_val_stats(
        dlg.edit_train.text(), dlg.edit_test.text(), cb_no_test.isChecked()
    ))
    return dlg
