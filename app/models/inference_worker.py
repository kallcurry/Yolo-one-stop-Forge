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


class InferenceWorker(QThread):
    frame_ready = pyqtSignal(object)      # QImage
    stats_ready = pyqtSignal(object)      # dict
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

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

    def _run_loop(self, capture):
        self._capture = capture
        frame_index = 0
        fps_timer = time.time()
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

            results = self._predictor.predict(
                frame,
                conf=self._parameters.get('conf', 0.25),
                iou=self._parameters.get('iou', 0.6),
                imgsz=self._parameters.get('imgsz', 640),
                device=self._parameters.get('device', 'auto'),
                verbose=False,
            )
            infer_ms = (time.time() - started) * 1000.0

            annotated = np.asarray(results[0].plot()) if results else frame
            counts = _count_targets(results, self._predictor)
            self._latest_counts = counts

            frame_index += 1
            now = time.time()
            interval = now - fps_timer
            if interval >= 0.5:
                fps_value = 1.0 / interval
                fps_timer = now
            self._latest_stats = {
                'fps': round(fps_value, 1),
                'infer_ms': round(infer_ms, 1),
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
        f"FPS {stats.get('fps', 0):.1f} · {stats.get('infer_ms', 0):.1f}ms · "
        f"帧 {stats.get('frame', 0)}"
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
