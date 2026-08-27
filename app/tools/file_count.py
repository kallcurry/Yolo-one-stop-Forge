"""Image & annotation file count statistics tool."""

import os
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import QCheckBox, QLabel

from app.views.tool_dialog import ToolDialog, stored_dataset_path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def run_count(base_dir: str, do_excel: bool):
    base = Path(base_dir)
    print(f"📂 扫描目录: {base}\n")

    # Detect structure
    has_annotations = (base / "annotations").is_dir()
    has_images = (base / "images").is_dir()

    if has_annotations and has_images:
        # SEPARATED / DJI_PAIR structure
        print("检测到 images/ + annotations/ 结构\n")
        _count_separated(base, do_excel)
    else:
        # FLAT or DJI per-video structure
        print("检测到子文件夹结构 (每个子文件夹含 images/ + annotations/)\n")
        _count_flat(base, do_excel)

    print("\n✅ 统计完成!")


def _count_separated(base: Path, do_excel: bool):
    img_base = base / "images"
    ann_base = base / "annotations"
    lbl_base = base / "labels"

    # Get subdirectories
    img_subdirs = {d.name: d for d in img_base.iterdir() if d.is_dir()}
    ann_subdirs = {d.name: d for d in ann_base.iterdir() if d.is_dir()}
    lbl_subdirs = {}
    if lbl_base.is_dir():
        lbl_subdirs = {d.name: d for d in lbl_base.iterdir() if d.is_dir()}

    all_names = sorted(set(img_subdirs) | set(ann_subdirs) | set(lbl_subdirs))

    rows = []
    total_img = 0
    total_ann = 0
    total_lbl = 0

    print(f"{'子文件夹':<35} {'图片':>8} {'JSON标注':>8} {'TXT标签':>8} {'状态'}")
    print("-" * 75)

    for name in all_names:
        imgs = _count_files(img_subdirs.get(name), IMAGE_EXTENSIONS)
        anns = _count_files(ann_subdirs.get(name), {'.json'})
        lbls = _count_files(lbl_subdirs.get(name), {'.txt'})

        total_img += imgs
        total_ann += anns
        total_lbl += lbls

        # Status check
        status = "✅"
        if name not in img_subdirs:
            status = "⚠️ 无图片"
        elif name not in ann_subdirs:
            status = "⚠️ 无标注"
        elif imgs != anns:
            status = f"⚠️ 数量不匹配 (差{abs(imgs-anns)})"

        print(f"  {name:<33} {imgs:>8} {anns:>8} {lbls:>8}   {status}")
        rows.append((name, imgs, anns, lbls, status))

    print("-" * 75)
    print(f"  {'总计':<33} {total_img:>8} {total_ann:>8} {total_lbl:>8}")

    # Summary
    print(f"\n📊 汇总:")
    print(f"   图片子文件夹: {len(img_subdirs)} 个")
    print(f"   标注子文件夹: {len(ann_subdirs)} 个")
    print(f"   图片总数:     {total_img}")
    print(f"   JSON标注总数: {total_ann}")
    print(f"   TXT标签总数:  {total_lbl}")
    if total_img > 0 and total_ann > 0:
        print(f"   图片/标注比:   {total_img/total_ann:.2f}" if total_ann else "   图片/标注比: N/A")

    if do_excel:
        _write_excel(rows, base / "file_count_report.xlsx")


def _count_flat(base: Path, do_excel: bool):
    """Count per-video-folder structure (each subdir has images/ + annotations/)."""
    subdirs = sorted(d for d in base.iterdir() if d.is_dir())
    rows = []
    total_img = 0
    total_ann = 0

    print(f"{'文件夹':<40} {'图片':>8} {'JSON':>8} {'状态'}")
    print("-" * 65)

    for d in subdirs:
        img_dir = d / "images"
        ann_dir = d / "annotations"
        imgs = _count_files(img_dir, IMAGE_EXTENSIONS) if img_dir.is_dir() else 0
        anns = _count_files(ann_dir, {'.json'}) if ann_dir.is_dir() else 0
        total_img += imgs
        total_ann += anns

        status = "✅"
        if not ann_dir.is_dir():
            status = "⚠️ 无标注目录"
        elif imgs != anns:
            status = f"⚠️ 差{abs(imgs-anns)}"

        print(f"  {d.name:<38} {imgs:>8} {anns:>8}   {status}")
        rows.append((d.name, imgs, anns, status))

    print("-" * 65)
    print(f"  {'总计':<38} {total_img:>8} {total_ann:>8}")
    print(f"\n📊 文件夹总数: {len(subdirs)}, 图片: {total_img}, 标注: {total_ann}")


def _count_files(directory: Path | None, extensions: set[str]) -> int:
    if directory is None or not directory.is_dir():
        return 0
    try:
        return sum(1 for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions)
    except (OSError, PermissionError):
        return 0


def _write_excel(rows, path):
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=['子文件夹', '图片数', 'JSON标注数', 'TXT标签数', '状态'])
        df.to_excel(str(path), index=False)
        print(f"\n✅ Excel 报告: {path}")
    except ImportError:
        print("\n⚠️ pandas 未安装，跳过 Excel 导出")


def create_dialog(parent=None):
    dlg = ToolDialog('图片与标注数量统计', parent)
    dlg.edit_base = dlg._add_dir_picker('数据根目录:',
        stored_dataset_path())
    cb_excel = QCheckBox('导出 Excel 报告')
    dlg.param_widget.addWidget(cb_excel)
    dlg.set_runner(lambda: run_count(dlg.edit_base.text(), cb_excel.isChecked()))
    return dlg
