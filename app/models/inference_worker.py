"""Inference workbench worker: capture -> predict -> annotate -> QImage.

The model is injected so unit tests can substitute a fake predictor.  The
worker emits ready-to-display frames plus lightweight stats; capturing,
inference and annotation all happen off the UI thread.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

# 图例 & 标签共用的关键点配色（与 ultralytics 点色近似，保证图例可辨识）
KEYPOINT_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (76, 230, 230), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40),
    (255, 250, 200), (128, 0, 0), (170, 255, 195), (0, 0, 128),
    (128, 128, 0), (255, 215, 180), (0, 64, 128), (64, 224, 208),
    (153, 51, 255), (255, 102, 178), (102, 255, 51),
]


def discover_keypoint_names(model) -> list[str] | None:
    """Recover keypoint names from the training dataset.yaml when possible.

    Models trained by this platform carry ``data`` pointing at the batch
    dataset.yaml which includes ``keypoint_names``; downloaded models fall
    back to numbered placeholders.
    """
    try:
        args = getattr(getattr(model, 'model', None), 'args', None) or {}
        data_path = args.get('data')
        kpt_shape = getattr(getattr(model, 'model', None), 'kpt_shape', None)
        if not data_path or not kpt_shape:
            return None
        import pathlib
        import yaml
        data_file = pathlib.Path(str(data_path))
        if not data_file.is_file():
            return None
        payload = yaml.safe_load(data_file.read_text(encoding='utf-8'))
        names = payload.get('keypoint_names') if isinstance(payload, dict) else None
        if not isinstance(names, list) or len(names) != int(kpt_shape[0]):
            return None
        return [str(name) for name in names]
    except Exception:  # noqa: BLE001 - best effort
        return None


class InferenceWorker(QThread):
    frame_ready = pyqtSignal(object)      # QImage
    stats_ready = pyqtSignal(object)      # dict
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    keypoints_ready = pyqtSignal(object)  # list[str] 关键点名称（Pose 图例）
    classes_ready = pyqtSignal(object)    # list[str] 类别名称（检测/分割/OBB 图例）
    task_ready = pyqtSignal(str)          # 模型任务类型 detect/segment/obb/pose

    def __init__(self, predictor, source: dict, parameters: dict,
                 parent=None, max_fps: float = 30.0):
        super().__init__(parent)
        self._predictor = predictor
        self._source = dict(source)
        self._parameters = dict(parameters)
        self._max_fps = float(max_fps or 30.0)
        self._stop = False
        self._pause = False
        self._recording = False
        self._recorder = None
        self._recorder_path: Path | None = None
        self._lock = None  # guarded by GIL for simple flags
        self._latest_frame = None
        self._latest_counts: dict[str, int] = {}
        self._latest_stats: dict = {}
        self._manual_stop = False
        self._kpt_names: list[str] = []
        self._enabled_kpts: set[str] = set()
        self._class_names: list[str] = []
        self._visible_classes: set[str] | None = None

    # ---- legends ----

    @property
    def kpt_names(self) -> list[str]:
        return list(self._kpt_names)

    def set_keypoint_labels(self, enabled_names):
        """Enable pose keypoint label rendering for the given names."""
        self._enabled_kpts = set(enabled_names or ())

    def set_visible_classes(self, visible: list | set | None):
        """Restrict detection/segmentation/OBB rendering to these classes."""
        self._visible_classes = (
            set(visible) if visible is not None else None
        )

    # ---- control ----

    def request_stop(self):
        self._stop = True
        self._pause = False

    def set_paused(self, paused: bool):
        self._pause = bool(paused)

    def start_recording(self, path: str | Path):
        self._recorder_path = Path(path).expanduser().resolve()
        self._recorder_path.parent.mkdir(parents=True, exist_ok=True)

    def stop_recording(self) -> Path | None:
        path = self._recorder_path
        self._recorder_path = None
        return path

    @property
    def is_recording(self) -> bool:
        return bool(self._recorder_path)

    @property
    def latest_frame(self):
        return self._latest_frame

    @property
    def latest_counts(self) -> dict[str, int]:
        return dict(self._latest_counts)

    @property
    def latest_stats(self) -> dict:
        return dict(self._latest_stats)

    # ---- run ----

    def run(self):
        try:
            capture = self._open_source()
            if capture is None:
                self.error_occurred.emit('无法打开输入源，请检查设备/地址/路径')
                return
            self._run_loop(capture)
        except Exception as exc:  # noqa: BLE001 - reported to UI
            self.error_occurred.emit(f'推理线程异常: {exc}')
        finally:
            self._close_source()
            self._release_recorder()
            self.status_changed.emit('已停止')

    def _open_source(self):
        kind = self._source.get('kind')
        value = self._source.get('value', '')
        if kind == 'camera':
            index = int(value or 0)
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        if kind == 'rtsp':
            return cv2.VideoCapture(str(value))
        if kind == 'video':
            return cv2.VideoCapture(str(value))
        if kind == 'images':
            return _ImageListSource(str(value))
        return None

    def _close_source(self):
        source = getattr(self, '_capture', None)
        if source is not None:
            try:
                source.release()
            except (AttributeError, Exception):  # noqa: BLE001
                pass

    def _predict_kwargs(self) -> dict:
        return {
            'conf': self._parameters.get('conf', 0.25),
            'iou': self._parameters.get('iou', 0.6),
            'imgsz': self._parameters.get('imgsz', 640),
            'device': self._parameters.get('device', 'auto'),
            'verbose': False,
        }

    def _apply_half_once(self):
        """Convert the model to fp16 exactly once (no per-frame 'half' arg).

        ``predict(half=...)`` is deprecated per-call in newer Ultralytics and
        warns on every frame; setting args + model precision once keeps the
        speedup without the warning spam.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            predictor = self._predictor
            predictor.args.half = True
            model = getattr(predictor, 'model', None)
            if model is not None:
                model = model.half()
        except Exception:  # noqa: BLE001
            pass

    def _run_loop(self, capture):
        self._capture = capture
        frame_index = 0
        fps_timer = time.time()
        fps_frames = 0
        fps_value = 0.0

        # 任务类型发现：ultralytics 模型自带 task（detect/segment/obb/pose）
        try:
            task = str(
                getattr(getattr(self._predictor, 'model', None), 'task', '')
                or ''
            )
        except (AttributeError, TypeError, ValueError):
            task = ''
        if task:
            self.task_ready.emit(task)

        # 结构发现（仅一次）：Pose → 关键点图例；检测/分割/OBB → 类别图例
        names = discover_keypoint_names(self._predictor)
        if not names:
            try:
                shape = getattr(
                    getattr(self._predictor, 'model', None), 'kpt_shape', None
                )
                if shape and int(shape[0]) > 1:
                    names = [f'kp_{index}' for index in range(int(shape[0]))]
            except (AttributeError, TypeError, ValueError):
                names = None
        if names:
            self._kpt_names = names
            self.keypoints_ready.emit(names)
        else:
            try:
                class_names = list(dict(self._predictor.names).values())
                class_names = [str(name) for name in class_names]
            except (AttributeError, TypeError, ValueError):
                class_names = []
            if class_names:
                self._class_names = class_names
                self._visible_classes = set(class_names)
                self.classes_ready.emit(class_names)

        # fp16 半精度：一次性转换（GPU 环境）
        self._apply_half_once()
        # 预热：消除冷启动（CUDA/内核编译）对帧率统计的污染
        try:
            warm = np.zeros((320, 480, 3), dtype=np.uint8)
            self._predictor.predict(warm, **self._predict_kwargs())
        except Exception:  # noqa: BLE001
            pass
        fps_timer = time.time()
        fps_frames = 0
        fps_value = 0.0

        while not self._stop:
            if self._pause:
                self.status_changed.emit('已暂停')
                time.sleep(0.05)
                continue
            started = time.time()
            ok, frame = capture.read()
            if not ok or frame is None:
                if self._source.get('kind') == 'images':
                    self.status_changed.emit('图片浏览完成，循环播放')
                else:
                    self.status_changed.emit('输入源结束')
                    if self._source.get('kind') == 'video':
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                break

            results = self._predictor.predict(frame, **self._predict_kwargs())
            infer_ms = (time.time() - started) * 1000.0
            try:
                _speed = dict(getattr(results[0], 'speed', None) or {})
                pre_ms = float(_speed.get('preprocess') or 0.0)
                post_ms = float(_speed.get('postprocess') or 0.0)
            except (AttributeError, TypeError, ValueError):
                pre_ms = post_ms = 0.0

            # 类别图例：先按可见类别过滤，再绘制（检测/分割/OBB）
            if self._visible_classes is not None and results is not None:
                results = _filter_results_classes(
                    results, self._visible_classes,
                )

            annotated = np.asarray(results[0].plot()) if results else frame
            counts = _count_targets(results, self._predictor)
            self._latest_counts = counts

            if self._enabled_kpts and results is not None:
                annotated = _draw_keypoint_labels(
                    annotated, results[0], self._kpt_names,
                    self._enabled_kpts,
                )

            frame_index += 1
            fps_frames += 1
            now = time.time()
            interval = now - fps_timer
            if interval >= 0.5:
                fps_value = fps_frames / interval
                fps_frames = 0
                fps_timer = now
            self._latest_stats = {
                'fps': round(fps_value, 1),
                'infer_ms': round(infer_ms, 1),
                'pre_ms': round(pre_ms, 1),
                'post_ms': round(post_ms, 1),
                'frame': frame_index,
                'counts': counts,
            }
            self.stats_ready.emit(self._latest_stats)

            hud = _draw_hud(annotated, self._latest_stats)
            self._latest_frame = hud
            self.frame_ready.emit(_to_qimage(hud))

            if self._recorder_path is not None:
                if self._recorder is None:
                    h, w = hud.shape[:2]
                    self._recorder = cv2.VideoWriter(
                        str(self._recorder_path),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        min(self._max_fps, 25.0),
                        (w, h),
                    )
                if self._recorder:
                    self._recorder.write(hud)

            elapsed = (time.time() - started) * 1000.0
            delay = max(0.0, (1000.0 / self._max_fps) - elapsed)
            time.sleep(delay / 1000.0)

    def _release_recorder(self):
        if self._recorder is not None:
            try:
                self._recorder.release()
            except Exception:  # noqa: BLE001
                pass
            self._recorder = None

    # ---- snapshots ----

    def save_snapshot(self, path: str | Path) -> bool:
        frame = self._latest_frame
        if frame is None:
            return False
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), frame)
        meta = target.with_suffix('.json')
        meta.write_text(
            json.dumps({
                'source': self._source,
                'parameters': self._parameters,
                'stats': self._latest_stats,
                'counts': self._latest_counts,
            }, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return True


class _ImageListSource:
    """Minimal read()/release() facade over a directory of images."""

    def __init__(self, directory: str):
        root = Path(directory).expanduser().resolve()
        self._files = sorted(
            str(path)
            for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in {
                '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'
            }
        )
        self._index = 0

    def read(self):
        if not self._files:
            return False, None
        frame = cv2.imread(self._files[self._index])
        self._index = (self._index + 1) % len(self._files)
        return (frame is not None), frame

    def release(self):
        pass


def _filter_results_classes(results, visible: set[str]):
    """Filter detections/segments/OBBs/pose instances per frame class set."""
    filtered = []
    names_map = {}
    for result in results:
        names_map = dict(getattr(result, 'names', None) or {})
        break
    for result in results:
        class_holder = (
            getattr(result, 'boxes', None)
            or getattr(result, 'obb', None)
        )
        if class_holder is None:
            filtered.append(result)
            continue
        try:
            classes = class_holder.cls
            classes_list = (
                classes.detach().cpu().numpy().astype(int).tolist()
                if hasattr(classes, 'detach') else list(classes)
            )
        except (AttributeError, TypeError):
            filtered.append(result)
            continue
        keep = [
            index for index, class_id in enumerate(classes_list)
            if str(names_map.get(int(class_id), int(class_id))) in visible
        ]
        if len(keep) == len(classes_list):
            filtered.append(result)
            continue
        for attr in ('boxes', 'masks', 'obb', 'keypoints'):
            holder = getattr(result, attr, None)
            if holder is None:
                continue
            try:
                setattr(result, attr, holder[keep])
            except (IndexError, TypeError):
                pass
        filtered.append(result)
    return filtered


def _draw_keypoint_labels(frame, result, names: list[str],
                          enabled: set[str]) -> np.ndarray:
    """Draw enabled keypoint names next to their dots.

    ``result.keypoints`` provides per-person ``xy`` (pixel) and ``conf``;
    only keypoints with confidence above 0.5 are labeled so occluded or
    missing joints do not clutter the view.
    """
    keypoints = getattr(result, 'keypoints', None)
    if keypoints is None or not names:
        return frame
    try:
        xy = keypoints.xy
        conf = keypoints.conf
        xy = xy.detach().cpu().numpy() if hasattr(xy, 'detach') else np.asarray(xy)
        conf = conf.detach().cpu().numpy() if hasattr(conf, 'detach') else np.asarray(conf)
    except (AttributeError, TypeError, ValueError):
        return frame
    if xy.ndim != 3:
        return frame
    for person_index in range(xy.shape[0]):
        for kpt_index in range(min(xy.shape[1], len(names))):
            name = names[kpt_index]
            if name not in enabled:
                continue
            try:
                is_visible = conf[person_index][kpt_index] > 0.5
            except (IndexError, TypeError):
                is_visible = False
            if not is_visible:
                continue
            x, y = xy[person_index][kpt_index]
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue
            color = KEYPOINT_COLORS[kpt_index % len(KEYPOINT_COLORS)]
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.rectangle(
                frame, (x + 5, y - 20), (x + 5 + 9 * len(name) + 6, y - 3),
                (20, 30, 45), -1,
            )
            cv2.putText(frame, name, (x + 9, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                        cv2.LINE_AA)
    return frame


def _count_targets(results, predictor) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not results:
        return counts
    names = getattr(results[0], 'names', None) or {}
    boxes = getattr(results[0], 'boxes', None)
    if boxes is None:
        return counts
    try:
        classes = boxes.cls
        classes = classes.detach().cpu().numpy().astype(int) if hasattr(
            classes, 'detach'
        ) else classes
        for class_id in classes:
            name = str(names.get(int(class_id), int(class_id)))
            counts[name] = counts.get(name, 0) + 1
    except (AttributeError, TypeError, ValueError):
        return counts
    return counts


def _draw_hud(frame, stats: dict) -> np.ndarray:
    """Draw translucent HUD (top-left stats, top-right counts) onto the frame."""
    text = (
        f"FPS {stats.get('fps', 0):.1f} · "
        f"pre {stats.get('pre_ms', 0):.1f} · infer {stats.get('infer_ms', 0):.1f} · "
        f"post {stats.get('post_ms', 0):.1f}ms · 帧 {stats.get('frame', 0)}"
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 44), (10, 22, 38), -1)
    cv2.rectangle(overlay, (10, 10), (330, 44), (54, 183, 255), 1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    counts = stats.get('counts') or {}
    if counts:
        height = 18 + len(counts) * 22
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 210, 10),
                      (frame.shape[1] - 10, height), (10, 22, 38), -1)
        cv2.rectangle(overlay, (frame.shape[1] - 210, 10),
                      (frame.shape[1] - 10, height), (69, 212, 131), 1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, '目标计数', (frame.shape[1] - 196, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (98, 232, 255), 1,
                    cv2.LINE_AA)
        for index, (name, count) in enumerate(
            sorted(counts.items(), key=lambda item: -item[1])
        ):
            y = 52 + index * 22
            cv2.putText(frame, f'{name} : {count}',
                        (frame.shape[1] - 196, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return frame


def _to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width,
                  QImage.Format_RGB888).copy()
