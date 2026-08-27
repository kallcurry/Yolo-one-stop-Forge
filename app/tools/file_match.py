"""Image-annotation matching check tool.

Scans subdirectories and reports:
  - Matched pairs (image <-> annotation)
  - Orphan images (no matching annotation)
  - Orphan annotations (no matching image)
"""

import os
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import QCheckBox, QLabel, QSpinBox, QHBoxLayout

from app.views.tool_dialog import ToolDialog, stored_dataset_path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def run_match(base_dir: str, check_json: bool, check_txt: bool,
              show_orphans_max: int):
    base = Path(base_dir)
    ann_base = base / "annotations"
    img_base = base / "images"
    lbl_base = base / "labels"

    has_separated = ann_base.is_dir() and img_base.is_dir()

    if has_separated:
        _match_separated(base, img_base, ann_base, lbl_base,
                         check_json, check_txt, show_orphans_max)
    else:
        _match_flat(base, check_json, show_orphans_max)

    print("\n✅ 匹配检查完成!")


def _match_separated(base, img_base, ann_base, lbl_base,
                     check_json, check_txt, show_max):
    img_subdirs = {d.name: d for d in img_base.iterdir() if d.is_dir()}
    ann_subdirs = {d.name: d for d in ann_base.iterdir() if d.is_dir()}
    lbl_subdirs = {}
    if lbl_base.is_dir():
        lbl_subdirs = {d.name: d for d in lbl_base.iterdir() if d.is_dir()}

    all_names = sorted(set(img_subdirs) | set(ann_subdirs))

    total_matched = 0
    total_orphan_img = 0
    total_orphan_ann = 0
    total_orphan_lbl = 0
    all_orphan_imgs = []
    all_orphan_anns = []
    all_orphan_lbls = []

    for name in all_names:
        img_dir = img_subdirs.get(name)
        ann_dir = ann_subdirs.get(name)
        lbl_dir = lbl_subdirs.get(name)

        img_stems = _get_stems(img_dir, IMAGE_EXTENSIONS) if img_dir else set()
        ann_stems = _get_stems(ann_dir, {'.json'}) if check_json and ann_dir else set()
        lbl_stems = _get_stems(lbl_dir, {'.txt'}) if check_txt and lbl_dir else set()

        # Matched: image stem in both annotation and label sets
        matched_json = img_stems & ann_stems if check_json else img_stems
        matched_txt = img_stems & lbl_stems if check_txt else set()

        # Orphan images: have no JSON or no TXT
        orphan_img_json = img_stems - ann_stems if check_json else set()
        orphan_img_txt = img_stems - lbl_stems if check_txt else set()
        orphan_img = orphan_img_json | orphan_img_txt

        # Orphan annotations: JSON without image
        orphan_ann = ann_stems - img_stems if check_json else set()

        # Orphan labels: TXT without image
        orphan_lbl = lbl_stems - img_stems if check_txt else set()

        n_img = len(img_stems)
        n_ann = len(ann_stems)
        n_lbl = len(lbl_stems)
        n_matched_json = len(matched_json) if check_json else n_img
        n_matched_txt = len(matched_txt) if check_txt else 0

        total_matched += n_matched_json
        total_orphan_img += len(orphan_img)
        total_orphan_ann += len(orphan_ann)
        total_orphan_lbl += len(orphan_lbl)

        # Status
        parts = []
        if check_json:
            parts.append(f"JSON匹配:{n_matched_json}/{n_img}" if n_img else f"JSON:{n_ann}")
        if check_txt:
            parts.append(f"TXT匹配:{n_matched_txt}/{n_img}" if n_img else f"TXT:{n_lbl}")

        has_issue = (len(orphan_img) > 0 or len(orphan_ann) > 0 or len(orphan_lbl) > 0)
        status = "⚠️" if has_issue else "✅"
        detail = ", ".join(parts) if parts else "-"

        print(f"\n📂 {name} {status}")
        print(f"   图片:{n_img}  JSON:{n_ann}  TXT:{n_lbl}  |  {detail}")

        if orphan_img:
            print(f"   ❌ 缺少标注的图片: {len(orphan_img)} 张")
            for s in sorted(orphan_img)[:show_max]:
                all_orphan_imgs.append((name, s))
                print(f"      {s}")
            if len(orphan_img) > show_max:
                print(f"      ... 还有 {len(orphan_img) - show_max} 张")

        if orphan_ann:
            print(f"   ❌ 缺少图片的标注: {len(orphan_ann)} 个")
            for s in sorted(orphan_ann)[:show_max]:
                all_orphan_anns.append((name, s))
                print(f"      {s}.json")
            if len(orphan_ann) > show_max:
                print(f"      ... 还有 {len(orphan_ann) - show_max} 个")

        if orphan_lbl:
            print(f"   ❌ 缺少图片的标签: {len(orphan_lbl)} 个")
            for s in sorted(orphan_lbl)[:show_max]:
                all_orphan_lbls.append((name, s))
                print(f"      {s}.txt")
            if len(orphan_lbl) > show_max:
                print(f"      ... 还有 {len(orphan_lbl) - show_max} 个")

        if not has_issue:
            print(f"   ✅ 全部匹配")

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 全局汇总")
    print(f"{'='*60}")
    print(f"  总匹配图片:     {total_matched}")
    print(f"  缺少标注的图片:  {total_orphan_img}")
    print(f"  缺少图片的标注:  {total_orphan_ann}")
    print(f"  缺少图片的标签:  {total_orphan_lbl}")

    if total_orphan_img == 0 and total_orphan_ann == 0:
        print(f"\n  🎉 所有图片和标注完全匹配!")


def _match_flat(base, check_json, show_max):
    """Match images and annotations in a flat directory structure (per-folder)."""
    subdirs = sorted(d for d in base.iterdir() if d.is_dir())

    for d in subdirs:
        img_dir = d / "images"
        ann_dir = d / "annotations"

        imgs = _get_stems(img_dir, IMAGE_EXTENSIONS) if img_dir.is_dir() else set()
        anns = _get_stems(ann_dir, {'.json'}) if check_json and ann_dir.is_dir() else set()

        matched = imgs & anns
        orphan_img = imgs - anns
        orphan_ann = anns - imgs

        status = "✅" if not orphan_img and not orphan_ann else "⚠️"
        print(f"\n📂 {d.name} {status}")
        print(f"   图片:{len(imgs)}  标注:{len(anns)}  匹配:{len(matched)}")
        if orphan_img:
            print(f"   ❌ 缺标注图片: {len(orphan_img)}")
            for s in sorted(orphan_img)[:show_max]:
                print(f"      {s}")
        if orphan_ann:
            print(f"   ❌ 缺图片标注: {len(orphan_ann)}")


def _get_stems(directory: Path | None, extensions: set[str]) -> set[str]:
    if directory is None or not directory.is_dir():
        return set()
    try:
        return {f.stem for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in extensions}
    except (OSError, PermissionError):
        return set()


def create_dialog(parent=None):
    dlg = ToolDialog('图片与标注匹配检查', parent)
    dlg.edit_base = dlg._add_dir_picker('数据根目录:',
        stored_dataset_path())

    cb_json = QCheckBox('检查 JSON 标注匹配')
    cb_json.setChecked(True)
    dlg.param_widget.addWidget(cb_json)

    cb_txt = QCheckBox('检查 TXT 标签匹配')
    cb_txt.setChecked(True)
    dlg.param_widget.addWidget(cb_txt)

    row = QHBoxLayout()
    row.addWidget(QLabel('每类最多显示:'))
    sp = QSpinBox()
    sp.setRange(1, 500)
    sp.setValue(20)
    row.addWidget(sp)
    row.addStretch()
    dlg.param_widget.addLayout(row)

    dlg.set_runner(lambda: run_match(
        dlg.edit_base.text(), cb_json.isChecked(), cb_txt.isChecked(), sp.value()
    ))
    return dlg
