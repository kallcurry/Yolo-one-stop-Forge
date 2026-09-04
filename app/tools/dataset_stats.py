"""Dataset statistics report tool — wraps pose_dataset_excel_report logic."""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QLabel, QLineEdit

from app.models.annotation_schema import infer_annotation_schema
from app.views.tool_dialog import ToolDialog, stored_dataset_path

# --- Constants ---
TARGET_CLASSES = []
KEYPOINTS = []
NUM_KPTS = 0
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# --- Core logic adapted from pose_dataset_excel_report ---
def _get_empty_pose_stats():
    return {
        'image_count': 0, 'json_count': 0,
        'classes': {cls: {'instances': 0, 'kpt_0': [0]*NUM_KPTS, 'kpt_1': [0]*NUM_KPTS, 'kpt_2': [0]*NUM_KPTS} for cls in TARGET_CLASSES},
        'total_kpt_0': [0]*NUM_KPTS, 'total_kpt_1': [0]*NUM_KPTS, 'total_kpt_2': [0]*NUM_KPTS
    }


def _configure_schema(annotation_paths):
    global NUM_KPTS
    from app.models.task_context import current_task_type
    schema = infer_annotation_schema(
        annotation_paths, task_type=current_task_type(),
    )
    TARGET_CLASSES[:] = list(schema.target_classes)
    KEYPOINTS[:] = list(schema.keypoints)
    NUM_KPTS = len(KEYPOINTS)


def _add_pose_stats(target, source):
    target['image_count'] += source['image_count']
    target['json_count'] += source['json_count']
    for cls in TARGET_CLASSES:
        target['classes'][cls]['instances'] += source['classes'][cls]['instances']
        for i in range(NUM_KPTS):
            target['classes'][cls]['kpt_0'][i] += source['classes'][cls]['kpt_0'][i]
            target['classes'][cls]['kpt_1'][i] += source['classes'][cls]['kpt_1'][i]
            target['classes'][cls]['kpt_2'][i] += source['classes'][cls]['kpt_2'][i]
    for i in range(NUM_KPTS):
        target['total_kpt_0'][i] += source['total_kpt_0'][i]
        target['total_kpt_1'][i] += source['total_kpt_1'][i]
        target['total_kpt_2'][i] += source['total_kpt_2'][i]


def _analyze_labelme_json_pose(json_path, stats):
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


def _count_image_files(directory):
    if not os.path.exists(directory):
        return 0
    return sum(1 for f in os.listdir(directory)
               if os.path.isfile(os.path.join(directory, f))
               and Path(f).suffix.lower() in IMAGE_EXTENSIONS)


def _count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def _analyze_annotations_simple(annotations_dir):
    if not os.path.exists(annotations_dir):
        return 0, 0, 0
    total_labels = 0
    rectangle_count = 0
    point_count = 0
    for ann_file in os.listdir(annotations_dir):
        if ann_file.endswith('.json'):
            ann_path = os.path.join(annotations_dir, ann_file)
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                shapes = data.get('shapes', data.get('shapes ', []))
                for shape in shapes:
                    total_labels += 1
                    shape_type = shape.get('shape_type', shape.get('shape_type ', '')).strip()
                    if shape_type == 'rectangle':
                        rectangle_count += 1
                    elif shape_type == 'point':
                        point_count += 1
            except Exception:
                pass
    return total_labels, rectangle_count, point_count


# Matplotlib must use Agg BEFORE any pyplot import
import matplotlib
matplotlib.use('Agg')


def run_stats(base_dir: str, include_filter: str, output_dir: str,
              do_excel: bool, do_charts: bool, do_md: bool):
    """Run the complete stats pipeline with duplicate detection."""
    base = Path(base_dir)
    ann_base = base / "annotations"
    img_base = base / "images"

    if not ann_base.exists() or not img_base.exists():
        print(f"❌ 未找到 annotations/ 或 images/ 目录于: {base}")
        return

    # Determine subdirs
    sub_ann_dirs = [d for d in ann_base.iterdir() if d.is_dir()]
    if include_filter:
        names = {n.strip() for n in include_filter.split(',')}
        sub_ann_dirs = [d for d in sub_ann_dirs if d.name in names]

    _configure_schema(
        path for directory in sub_ann_dirs for path in directory.glob('*.json')
    )

    global_stats = _get_empty_pose_stats()
    folder_results = {}

    # Track stems → source subdirectories for duplicate detection
    stem_sources: dict[str, list[str]] = defaultdict(list)

    for sub_ann_dir in sorted(sub_ann_dirs):
        folder_name = sub_ann_dir.name
        sub_img_dir = img_base / folder_name
        folder_stats = _get_empty_pose_stats()

        if sub_img_dir.exists():
            folder_stats['image_count'] = _count_image_files(sub_img_dir)
        json_files = list(sub_ann_dir.glob("*.json"))
        folder_stats['json_count'] = len(json_files)

        # Collect stems for duplicate detection
        for jf in json_files:
            stem_sources[jf.stem].append(folder_name)

        print(f"🔍 处理: {folder_name} (图片:{folder_stats['image_count']}, JSON:{folder_stats['json_count']})")
        for jf in json_files:
            _analyze_labelme_json_pose(jf, folder_stats)
        _add_pose_stats(global_stats, folder_stats)
        folder_results[folder_name] = folder_stats

    # --- Duplicate detection ---
    duplicates = {stem: dirs for stem, dirs in stem_sources.items() if len(dirs) > 1}
    if duplicates:
        print(f"\n{'='*60}")
        print(f"⚠️ 发现 {len(duplicates)} 个跨子文件夹重复的图片")
        print(f"{'='*60}")

        # Count per folder-pair
        from collections import Counter
        dup_per_folder = Counter()
        for stem, dirs in duplicates.items():
            for d in dirs:
                dup_per_folder[d] += 1

        print("  各子文件夹中的重复图片数:")
        for d in sorted(dup_per_folder):
            count = dup_per_folder[d]
            print(f"    {d}: {count} 张 (与其他子文件夹重复)")

        # Show some examples
        show_n = min(20, len(duplicates))
        print(f"\n  示例 (前 {show_n} 个):")
        for i, (stem, dirs) in enumerate(sorted(duplicates.items())[:show_n], 1):
            print(f"    {i:3d}. {stem:<45} 出现在: {', '.join(dirs)}")
        if len(duplicates) > show_n:
            print(f"    ... 还有 {len(duplicates) - show_n} 个")
    else:
        print(f"\n✅ 未发现跨子文件夹重复图片")
    print(f"{'='*60}")

    # Print summary
    for folder_name, stats in folder_results.items():
        total = sum(stats['classes'][c]['instances'] for c in TARGET_CLASSES)
        print(f"\n📂 {folder_name}: 实例={total}")
        for cls in TARGET_CLASSES:
            print(f"     {cls}: {stats['classes'][cls]['instances']}")

    total_instances = sum(global_stats['classes'][c]['instances'] for c in TARGET_CLASSES)
    print(f"\n🏆 总计: {total_instances} 个目标实例, {len(stem_sources)} 唯一图片")
    for i, kp in enumerate(KEYPOINTS):
        c0, c1, c2 = global_stats['total_kpt_0'][i], global_stats['total_kpt_1'][i], global_stats['total_kpt_2'][i]
        total = c0 + c1 + c2
        if total > 0:
            bad_pct = (c0 + c1) / total * 100
            print(f"  {kp:<30} 未标注={c0:<6} 遮挡={c1:<6} 清晰={c2:<6} [不良率={bad_pct:.1f}%]")

    # Generate charts
    if do_charts and total_instances > 0:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            os.makedirs(output_dir, exist_ok=True)

            # Pie chart
            plt.figure(figsize=(8, 6))
            sizes = [global_stats['classes'][cls]['instances'] for cls in TARGET_CLASSES]
            colors = [plt.cm.tab20(index % 20) for index in range(len(TARGET_CLASSES))]
            filtered = [(l, s, c) for l, s, c in zip(TARGET_CLASSES, sizes, colors) if s > 0]
            if filtered:
                labels, sizes_f, colors_f = zip(*filtered)
                plt.pie(sizes_f, labels=labels, colors=colors_f, autopct='%1.1f%%', startangle=140)
                plt.title('Category Distribution')
                plt.savefig(os.path.join(output_dir, 'category_distribution.png'), dpi=150)
                print("✅ category_distribution.png")
            plt.close()

            # Stacked bar
            plt.figure(figsize=(14, 7))
            x = np.arange(NUM_KPTS)
            w = 0.6
            k0 = global_stats['total_kpt_0']
            k1 = global_stats['total_kpt_1']
            k2 = global_stats['total_kpt_2']
            plt.bar(x, k0, w, label='=0 (missing)', color='#cccccc')
            plt.bar(x, k1, w, bottom=k0, label='=1 (occluded)', color='#ff9800')
            bottom12 = [i + j for i, j in zip(k0, k1)]
            plt.bar(x, k2, w, bottom=bottom12, label='=2 (visible)', color='#4caf50')
            plt.xticks(x, [k[:15] for k in KEYPOINTS], rotation=45, ha='right', fontsize=8)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'keypoints_states.png'), dpi=150)
            print("✅ keypoints_states.png")
            plt.close()

            print("✅ 图表生成完毕")
        except ImportError as e:
            print(f"⚠️ 缺少依赖，跳过图表生成: {e}")
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")

    # Generate Markdown
    if do_md:
        md_path = os.path.join(output_dir, "labelme_stats_report.md")
        lines = [
            "# 姿态估计数据集统计报告",
            f"**生成时间**: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
            f"**数据目录**: {base_dir}\n",
        ]
        if duplicates:
            lines += [
                f"## ⚠️ 重复图片 ({len(duplicates)} 个)\n",
                "| STEM | 来源子文件夹 |",
                "|---|---|",
            ]
            for stem, dirs in sorted(duplicates.items()):
                lines.append(f"| {stem} | {', '.join(dirs)} |")
            lines.append("")

        for name, stats in folder_results.items():
            total = sum(stats['classes'][c]['instances'] for c in TARGET_CLASSES)
            lines += [f"## {name} (实例: {total})", "",
                      "| 类别 | 数量 |", "|---|---|"]
            for cls in TARGET_CLASSES:
                lines.append(f"| {cls} | {stats['classes'][cls]['instances']} |")
            lines.append("")
        if do_charts:
            lines += [
                "## 可视化\n",
                "### 类别分布",
                "![category](category_distribution.png)\n",
                "### 关键点状态",
                "![keypoints](keypoints_states.png)\n",
            ]
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ Markdown 报告: {md_path}")

    # Generate Excel
    if do_excel:
        try:
            import pandas as pd
            data = {
                '视频名称': [], '图片数': [], '标注数': [],
                '标签总数': [], '矩形框': [], '关键点': [],
                '重复图片数': [],
            }
            for sub_ann_dir in sorted(sub_ann_dirs):
                name = sub_ann_dir.name
                img_dir = img_base / name
                img_count = _count_files(img_dir)
                ann_count = _count_files(sub_ann_dir)
                total_lbl, rects, points = _analyze_annotations_simple(sub_ann_dir)
                dup_count = dup_per_folder.get(name, 0) if duplicates else 0
                data['视频名称'].append(name)
                data['图片数'].append(img_count)
                data['标注数'].append(ann_count)
                data['标签总数'].append(total_lbl)
                data['矩形框'].append(rects)
                data['关键点'].append(points)
                data['重复图片数'].append(dup_count)
            df = pd.DataFrame(data)
            xlsx_path = os.path.join(output_dir, "dataset_stats.xlsx")
            df.to_excel(xlsx_path, index=False)
            print(f"✅ Excel 报告: {xlsx_path}")
        except ImportError as e:
            print(f"⚠️ pandas/openpyxl 未安装: {e}")
        except Exception as e:
            print(f"⚠️ Excel 生成失败: {e}")

    print("\n✅ 统计完成!")


# --- Tool dialog factory ---
def create_dialog(parent=None):
    dlg = ToolDialog('数据集统计报告', parent)
    default_output = Path(__file__).resolve().parents[2] / 'reports' / 'dataset_stats'

    edit_base = dlg._add_dir_picker('数据根目录:',
        stored_dataset_path())
    edit_include = QLineEdit()
    edit_include.setPlaceholderText('留空=全部，用逗号分隔，如: AI_2026-04-20,Collect_2026-03-10')
    row = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    row.addWidget(QLabel('指定子批次:'))
    row.addWidget(edit_include)
    dlg.param_widget.addLayout(row)

    edit_output = dlg._add_dir_picker('输出目录:',
        str(default_output))

    cb_excel = QCheckBox('生成 Excel')
    cb_excel.setChecked(True)
    cb_charts = QCheckBox('生成图表')
    cb_charts.setChecked(True)
    cb_md = QCheckBox('生成 Markdown')
    cb_md.setChecked(True)
    row2 = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    row2.addWidget(cb_excel)
    row2.addWidget(cb_charts)
    row2.addWidget(cb_md)
    row2.addStretch()
    dlg.param_widget.addLayout(row2)

    dlg.set_runner(lambda: run_stats(
        edit_base.text(), edit_include.text(), edit_output.text(),
        cb_excel.isChecked(), cb_charts.isChecked(), cb_md.isChecked()
    ))
    return dlg
