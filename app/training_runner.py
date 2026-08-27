"""Project-local Ultralytics runner used by the Qt training center."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.training_job import (
    TRAIN_EVENT_PREFIX,
    TrainingJob,
    load_training_job,
    training_job_to_dict,
)


class TrainingCancelled(RuntimeError):
    pass


def emit_event(event_type: str, **payload):
    message = {'type': event_type, **payload}
    print(
        TRAIN_EVENT_PREFIX
        + json.dumps(message, ensure_ascii=False, separators=(',', ':')),
        flush=True,
    )


def run_training(job: TrainingJob, request_path: Path) -> Path:
    emit_event(
        'initializing', job_id=job.job_id, model=job.model,
        dataset=job.dataset_yaml, expected_run_dir=str(job.run_dir),
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            '当前 Python 环境未安装 Ultralytics，请在运行本应用的环境中安装依赖'
        ) from exc
    _install_ultralytics_plot_guard()

    job.project_dir.mkdir(parents=True, exist_ok=True)
    model_source = _prepare_model_source(job.model)
    model = YOLO(model_source, task=job.ultralytics_task)

    def on_train_start(trainer):
        save_dir = Path(trainer.save_dir).resolve()
        _write_job_snapshot(save_dir, job, request_path)
        emit_event(
            'started', job_id=job.job_id, save_dir=str(save_dir),
            epochs=int(getattr(trainer, 'epochs', 0) or 0),
            device=str(getattr(trainer, 'device', '-')),
        )

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, 'epoch', -1)) + 1
        epochs = int(getattr(trainer, 'epochs', 0) or 0)
        metrics: dict[str, Any] = {}
        tloss = getattr(trainer, 'tloss', None)
        if tloss is not None:
            try:
                metrics.update(trainer.label_loss_items(tloss))
            except Exception:
                if isinstance(tloss, dict):
                    metrics.update(tloss)
        current_metrics = getattr(trainer, 'metrics', None)
        if isinstance(current_metrics, dict):
            metrics.update(current_metrics)
        learning_rates = getattr(trainer, 'lr', None)
        if isinstance(learning_rates, dict):
            metrics.update(learning_rates)
        emit_event(
            'epoch', epoch=epoch, epochs=epochs,
            progress=(epoch / epochs * 100.0) if epochs else 0.0,
            metrics=_json_safe(metrics),
            fitness=_json_safe(getattr(trainer, 'fitness', None)),
            save_dir=str(Path(trainer.save_dir).resolve()),
        )

    def on_train_end(trainer):
        emit_event(
            'finalizing',
            save_dir=str(Path(trainer.save_dir).resolve()),
        )

    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    model.add_callback('on_train_end', on_train_end)

    model.train(
        data=job.dataset_yaml,
        project=str(job.project_dir),
        name=job.run_name,
        **job.parameters,
    )
    trainer = getattr(model, 'trainer', None)
    save_dir = Path(getattr(trainer, 'save_dir', job.run_dir)).resolve()
    _write_job_snapshot(save_dir, job, request_path)
    actual_epoch = _completed_epoch_count(save_dir, trainer)
    total_epochs = int(
        getattr(trainer, 'epochs', 0) or job.parameters.get('epochs') or 0
    )
    progress = (
        min(actual_epoch / total_epochs * 100.0, 100.0)
        if total_epochs else 0.0
    )
    _write_completion_marker(
        save_dir, actual_epoch=actual_epoch, total_epochs=total_epochs
    )
    emit_event(
        'completed', job_id=job.job_id, save_dir=str(save_dir),
        epoch=actual_epoch, epochs=total_epochs, progress=progress,
        early_stopped=bool(total_epochs and actual_epoch < total_epochs),
    )
    return save_dir


def _install_ultralytics_plot_guard():
    """Skip non-finite predicted keypoints in Ultralytics result images.

    Fresh custom pose heads can briefly emit infinite coordinates under AMP.
    Those predictions still belong in metric calculation, but OpenCV cannot
    convert them to integer pixels when Ultralytics renders preview images.
    """
    try:
        from ultralytics.utils.plotting import Annotator
    except (ImportError, AttributeError):
        return

    original = Annotator.kpts
    if getattr(original, '_files_process_finite_guard', False):
        return
    warning_emitted = False

    def guarded_kpts(annotator, keypoints, *args, **kwargs):
        nonlocal warning_emitted
        try:
            keypoints, invalid_count = _sanitize_plot_keypoints(keypoints)
            if invalid_count and not warning_emitted:
                warning_emitted = True
                print(
                    '[平台] 验证预测包含非有限关键点，已仅在结果图中'
                    '跳过这些点；训练与指标计算未被修改。',
                    flush=True,
                )
        except (TypeError, ValueError):
            pass
        return original(annotator, keypoints, *args, **kwargs)

    guarded_kpts._files_process_finite_guard = True
    Annotator.kpts = guarded_kpts


def _sanitize_plot_keypoints(keypoints):
    """Return a plotting copy with non-finite keypoint rows hidden."""
    import numpy as np

    array = np.asarray(keypoints)
    if array.ndim != 2 or array.shape[1] < 2:
        return keypoints, 0
    invalid = ~np.isfinite(array).all(axis=1)
    count = int(invalid.sum())
    if not count:
        return keypoints, 0
    sanitized = array.copy()
    sanitized[invalid] = 0
    return sanitized, count


def _prepare_model_source(value: str) -> str:
    """Keep downloadable Ultralytics weights in the project's models folder."""
    source = Path(value).expanduser()
    project_root = Path(__file__).resolve().parents[1]
    models_root = (project_root / 'models').resolve()
    if not source.is_absolute() and source.suffix.lower() == '.pt':
        source = models_root / source.name
    elif not source.is_absolute():
        return str(source)
    try:
        source.relative_to(models_root)
    except ValueError:
        return str(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    return str(source)


def _write_job_snapshot(save_dir: Path, job: TrainingJob,
                        request_path: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / 'training_request.json'
    try:
        shutil.copy2(request_path, target)
    except OSError:
        target.write_text(
            json.dumps(
                training_job_to_dict(job), ensure_ascii=False, indent=2
            ),
            encoding='utf-8',
        )


def _completed_epoch_count(save_dir: Path, trainer: Any) -> int:
    results_path = save_dir / 'results.csv'
    row_count = 0
    if results_path.is_file():
        try:
            with results_path.open(encoding='utf-8') as stream:
                row_count = max(0, sum(1 for _line in stream) - 1)
        except OSError:
            pass
    raw_epoch = getattr(trainer, 'epoch', -1)
    try:
        trainer_epoch = int(raw_epoch) + 1
    except (TypeError, ValueError):
        trainer_epoch = 0
    return max(row_count, trainer_epoch, 0)


def _write_completion_marker(save_dir: Path, *, actual_epoch: int,
                             total_epochs: int):
    marker = save_dir / 'training_complete.json'
    marker.write_text(
        json.dumps({
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'actual_epoch': int(actual_epoch),
            'total_epochs': int(total_epochs),
            'early_stopped': bool(total_epochs and actual_epoch < total_epochs),
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return float(value.item())
    except (AttributeError, TypeError, ValueError):
        return str(value)


def _raise_cancelled(_signum, _frame):
    raise TrainingCancelled('训练任务已由用户停止')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Files Process training runner')
    parser.add_argument('job', help='训练任务 JSON 文件')
    args = parser.parse_args(argv)
    request_path = Path(args.job).expanduser().resolve()
    models_root = Path(__file__).resolve().parents[1] / 'models'
    models_root.mkdir(parents=True, exist_ok=True)
    os.chdir(models_root)
    signal.signal(signal.SIGTERM, _raise_cancelled)
    signal.signal(signal.SIGINT, _raise_cancelled)
    try:
        job = load_training_job(request_path)
        run_training(job, request_path)
        return 0
    except TrainingCancelled as exc:
        emit_event('cancelled', message=str(exc))
        return 130
    except Exception as exc:
        emit_event('failed', message=str(exc))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
