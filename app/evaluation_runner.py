"""Ultralytics YOLO evaluation subprocess entry (evaluation center, phase 1).

Runs ``model.val(data=test_dataset.yaml)`` on a dedicated test batch and
writes the platform ``evaluation_result.json`` contract next to the
Ultralytics outputs.  Structured events use the ``EVAL_EVENT_PREFIX`` JSON
line protocol so the UI can monitor progress without polling.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# 支持直接运行: python app/evaluation_runner.py <spec.json>
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.evaluation_job import (
    EVAL_EVENT_PREFIX,
    EvaluationJob,
    METRIC_FAMILIES,
    load_evaluation_job,
)

RESULT_FILE = 'evaluation_result.json'


class EvaluationCancelled(RuntimeError):
    pass


def emit_event(event_type: str, **payload):
    message = {'type': event_type, **payload}
    print(
        EVAL_EVENT_PREFIX
        + json.dumps(message, ensure_ascii=False, separators=(',', ':'))
    )
    sys.stdout.flush()


def _raise_cancelled(_signum, _frame):
    raise EvaluationCancelled('收到终止信号')


def _task_family(task_type: str) -> str:
    return METRIC_FAMILIES.get(task_type, '(B)')


# --- pure metric helpers (testable without ultralytics) ---

def _pick(results_dict: dict, family: str, name: str) -> float | None:
    """Pick ``metrics/<name><family>`` first, then any ``metrics/<name>``."""
    for key in (
        f'metrics/{name}{family}',
        f'metrics/{name}',
    ):
        value = results_dict.get(key)
        if value is not None and isinstance(value, (int, float)):
            try:
                return None if math.isnan(float(value)) else float(value)
            except (TypeError, ValueError):
                return None
    return None


def extract_summary(results_dict: dict, task_type: str) -> dict[str, float | None]:
    family = _task_family(task_type)
    return {
        'mAP50-95': _pick(results_dict, family, 'mAP50-95'),
        'mAP50': _pick(results_dict, family, 'mAP50'),
        'mAP75': _pick(results_dict, family, 'mAP75'),
        'precision': _pick(results_dict, family, 'precision'),
        'recall': _pick(results_dict, family, 'recall'),
    }


def _metrics_block(metrics_obj, family: str,
                   names: dict | None = None) -> dict[str, Any] | None:
    ap = getattr(metrics_obj, 'ap', None)
    ap50 = getattr(metrics_obj, 'ap50', None)
    class_index = getattr(metrics_obj, 'ap_class_index', None)
    # ultralytics >= 8.4: val metric objects no longer carry ``names``;
    # fall back to the model's class names passed in from the runner.
    if not names:
        names = getattr(metrics_obj, 'names', None) or {}
    if ap is None or class_index is None:
        return None
    per_class: dict[str, dict[str, float]] = {}
    for index, class_id in enumerate(class_index):
        name = str(names.get(int(class_id), class_id))
        per_class[name] = {
            'mAP50-95': float(ap[index]) if index < len(ap) else None,
            'mAP50': float(ap50[index]) if ap50 is not None and index < len(ap50) else None,
        }
    return per_class


def extract_per_class(metrics_obj, task_type: str,
                      names: dict | None = None) -> dict[str, dict[str, float]]:
    family = _task_family(task_type)
    for attr in ('box', 'pose', 'segment'):
        block = _metrics_block(getattr(metrics_obj, attr, None), family, names)
        if block:
            return block
    return {}


def extract_latency(metrics_obj) -> dict[str, float | None]:
    speed = getattr(metrics_obj, 'speed', None) or {}
    preprocess = float(speed.get('preprocess') or 0)
    inference = float(speed.get('inference') or 0)
    postprocess = float(speed.get('postprocess') or 0)
    per_image = speed.get('per_image')
    if per_image is None:
        per_image = (
            (preprocess + inference + postprocess)
            if (preprocess or inference or postprocess) else None
        )
    return {
        'ms_per_image': float(per_image) if per_image else None,
        'fps': float(1000.0 / per_image) if per_image else None,
    }


def read_train_metrics(training_run_dir: str | None,
                       task_type: str) -> dict[str, float | None]:
    """Read the model's own validation curves from its training results.csv."""
    if not training_run_dir:
        return {}
    results_csv = Path(training_run_dir) / 'results.csv'
    if not results_csv.is_file():
        return {}
    family = _task_family(task_type)
    try:
        with results_csv.open('r', encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            last = None
            for last in reader:
                pass
    except (OSError, csv.Error):
        return {}
    if last is None:
        return {}

    def _last_value(name: str):
        for key in (
            f'metrics/{name}{family}',
            f'metrics/{name}',
        ):
            if key in last:
                try:
                    value = float(last[key])
                    return None if math.isnan(value) else value
                except (TypeError, ValueError):
                    return None
        return None

    return {
        'mAP50-95': _last_value('mAP50-95'),
        'mAP50': _last_value('mAP50'),
        'mAP75': _last_value('mAP75'),
    }


def build_result_payload(
    job: EvaluationJob,
    summary: dict,
    per_class: dict,
    train_metrics: dict,
    latency: dict,
    output_dir: Path,
) -> dict:
    gap = None
    train_map = train_metrics.get('mAP50-95')
    test_map = summary.get('mAP50-95')
    if train_map is not None and test_map is not None:
        gap = round(test_map - train_map, 6)
    return {
        'version': 1,
        'task_id': job.job_id,
        'model_path': job.model_path,
        'model_label': job.model_label,
        'task_type': job.task_type,
        'test_batch': job.test_batch,
        'test_manifest_sha256': job.test_manifest_sha256,
        'training_batch': job.training_batch,
        'training_run_dir': job.training_run_dir,
        'metrics': summary,
        'per_class': per_class,
        'train_metrics': train_metrics,
        'generalization_gap': gap,
        'latency': latency,
        'outputs': {
            'results_csv': str(output_dir / 'results.csv'),
            'results_png': str(output_dir / 'results.png'),
            'confusion_matrix_png': str(output_dir / 'confusion_matrix.png'),
        },
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def run_evaluation(job: EvaluationJob, spec_path: Path) -> Path:
    from ultralytics import YOLO

    emit_event(
        'initializing', job_id=job.job_id, model=job.model_path,
        test_batch=job.test_batch,
    )
    t0 = time.time()
    model = YOLO(job.model_path)
    parameters = dict(job.parameters)
    results = model.val(
        data=job.test_dataset_yaml,
        project=job.project_dir,
        name=job.run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
        **parameters,
    )
    summary = extract_summary(results.results_dict, job.task_type)
    per_class = extract_per_class(
        results, job.task_type, names=model.names,
    )
    latency = extract_latency(results)
    train_metrics = read_train_metrics(job.training_run_dir, job.task_type)
    output_dir = Path(job.project_dir) / job.run_name
    payload = build_result_payload(
        job, summary, per_class, train_metrics, latency, output_dir
    )
    result_path = output_dir / RESULT_FILE
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    emit_event(
        'completed', job_id=job.job_id,
        elapsed=round(time.time() - t0, 2),
        result_file=str(result_path),
        metrics=_json_safe(summary),
        generalization_gap=payload.get('generalization_gap'),
    )
    return result_path


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def main(argv=None) -> int:
    parser = __import__('argparse').ArgumentParser(
        description='Files Process evaluation runner'
    )
    parser.add_argument('job', help='评估任务 JSON 文件')
    args = parser.parse_args(argv)
    spec_path = Path(args.job).expanduser().resolve()
    signal.signal(signal.SIGTERM, _raise_cancelled)
    signal.signal(signal.SIGINT, _raise_cancelled)
    os.chdir(Path(__file__).resolve().parents[1])
    try:
        job = load_evaluation_job(spec_path)
        problems = job.validate()
        if problems:
            raise RuntimeError('; '.join(problems))
        run_evaluation(job, spec_path)
        return 0
    except EvaluationCancelled as exc:
        emit_event('cancelled', message=str(exc))
        return 130
    except Exception as exc:
        emit_event('failed', message=str(exc))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
