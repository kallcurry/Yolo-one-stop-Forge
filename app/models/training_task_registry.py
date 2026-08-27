"""Persistent task registry and recovery for the training center."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from app.models.training_config import MANAGED_PARAMETER_NAMES
from app.models.training_job import (
    TrainingJob,
    load_training_job,
    training_job_from_dict,
    training_job_to_dict,
)


ACTIVE_STATUSES = {'preparing', 'running', 'stopping'}
TERMINAL_STATUSES = {'completed', 'failed', 'cancelled', 'interrupted'}
TASK_STATUSES = {
    'draft', 'queued', 'preparing', 'running', 'stopping',
    'completed', 'failed', 'cancelled', 'interrupted', 'archived',
}

ULTRALYTICS_TO_TASK = {
    'pose': 'pose',
    'detect': 'detection',
    'segment': 'segmentation',
    'obb': 'obb',
}


class TrainingTaskRegistryError(RuntimeError):
    """Raised when a persisted training-task operation is invalid."""


@dataclass(frozen=True)
class TrainingTaskRecord:
    task_id: str
    display_name: str
    status: str
    task_type: str
    run_name: str
    project_name: str
    model: str
    batch_root: str
    dataset_yaml: str
    output_root: str
    run_dir: str
    parameters: dict[str, Any]
    job: TrainingJob
    request_path: str
    log_path: str
    progress: float
    current_epoch: int
    total_epochs: int
    pid: int | None
    created_at: str
    updated_at: str
    started_at: str
    finished_at: str
    error_message: str
    archived: bool
    source: str
    notes: str


@dataclass(frozen=True)
class TrainingRunInspection:
    status: str
    error_message: str
    current_epoch: int
    total_epochs: int
    progress: float


class TrainingTaskRegistry:
    """SQLite-backed registry; one connection is owned by one UI instance."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser().resolve())
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self):
        with self._lock:
            self._connection.close()

    def _initialize(self):
        with self._lock, self._connection:
            self._connection.execute('PRAGMA foreign_keys = ON')
            self._connection.execute('PRAGMA journal_mode = WAL')
            self._connection.execute('''
                CREATE TABLE IF NOT EXISTS training_tasks (
                    task_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    batch_root TEXT NOT NULL,
                    dataset_yaml TEXT NOT NULL,
                    output_root TEXT NOT NULL,
                    run_dir TEXT NOT NULL UNIQUE,
                    parameters_json TEXT NOT NULL,
                    job_json TEXT NOT NULL,
                    request_path TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL DEFAULT '',
                    progress REAL NOT NULL DEFAULT 0,
                    current_epoch INTEGER NOT NULL DEFAULT 0,
                    total_epochs INTEGER NOT NULL DEFAULT 0,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    archived INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'platform',
                    notes TEXT NOT NULL DEFAULT ''
                )
            ''')
            self._connection.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_tasks_status
                ON training_tasks(status, archived, created_at DESC)
            ''')

    def register_job(self, job: TrainingJob, *, status: str = 'draft',
                     request_path: str | Path = '', log_path: str | Path = '',
                     source: str = 'platform') -> TrainingTaskRecord:
        self._validate_status(status)
        now = _utc_now()
        job_payload = training_job_to_dict(job)
        values = {
            'task_id': job.job_id,
            'display_name': job.run_name,
            'status': status,
            'task_type': job.task_type,
            'run_name': job.run_name,
            'project_name': job.project_name,
            'model': job.model,
            'batch_root': job.batch_root,
            'dataset_yaml': job.dataset_yaml,
            'output_root': job.output_root,
            'run_dir': str(job.run_dir.resolve()),
            'parameters_json': _json_dump(job.parameters),
            'job_json': _json_dump(job_payload),
            'request_path': str(request_path or ''),
            'log_path': str(log_path or ''),
            'progress': 100.0 if status == 'completed' else 0.0,
            'current_epoch': (
                int(job.parameters.get('epochs') or 0)
                if status == 'completed' else 0
            ),
            'total_epochs': int(job.parameters.get('epochs') or 0),
            'created_at': job.created_at or now,
            'updated_at': now,
            'source': source,
        }
        columns = ', '.join(values)
        placeholders = ', '.join('?' for _ in values)
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    f'INSERT INTO training_tasks ({columns}) VALUES ({placeholders})',
                    tuple(values.values()),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.get_by_run_dir(job.run_dir)
            if existing is not None:
                return existing
            raise TrainingTaskRegistryError(str(exc)) from exc
        return self.get(job.job_id)

    def update_job(self, task_id: str, job: TrainingJob,
                   request_path: str | Path = '') -> TrainingTaskRecord:
        current = self.require(task_id)
        if current.status != 'draft':
            raise TrainingTaskRegistryError('只有草稿任务可修改训练参数')
        job = replace(
            job, job_id=current.task_id, created_at=current.created_at
        )
        now = _utc_now()
        try:
            with self._lock, self._connection:
                self._connection.execute('''
                    UPDATE training_tasks SET
                        display_name = ?, task_type = ?, run_name = ?,
                        project_name = ?, model = ?, batch_root = ?,
                        dataset_yaml = ?, output_root = ?, run_dir = ?,
                        parameters_json = ?, job_json = ?, request_path = ?,
                        total_epochs = ?, updated_at = ?
                    WHERE task_id = ?
                ''', (
                    job.run_name, job.task_type, job.run_name,
                    job.project_name, job.model, job.batch_root,
                    job.dataset_yaml, job.output_root, str(job.run_dir.resolve()),
                    _json_dump(job.parameters),
                    _json_dump(training_job_to_dict(job)), str(request_path or ''),
                    int(job.parameters.get('epochs') or 0), now, task_id,
                ))
        except sqlite3.IntegrityError as exc:
            raise TrainingTaskRegistryError('训练任务目录已被其他任务占用') from exc
        return self.require(task_id)

    def relocate_artifacts(self, task_id: str, job: TrainingJob, *,
                           request_path: str | Path,
                           log_path: str | Path) -> TrainingTaskRecord:
        """Move inspectable task files without changing lifecycle state."""
        current = self.require(task_id)
        if current.status in ACTIVE_STATUSES:
            raise TrainingTaskRegistryError('运行中的任务不能迁移任务文件')
        job = replace(
            job, job_id=current.task_id, created_at=current.created_at
        )
        if job.run_dir.resolve() != Path(current.run_dir).resolve():
            raise TrainingTaskRegistryError('迁移任务文件时不能修改训练输出目录')
        now = _utc_now()
        with self._lock, self._connection:
            self._connection.execute('''
                UPDATE training_tasks SET
                    model = ?, batch_root = ?, dataset_yaml = ?,
                    parameters_json = ?, job_json = ?, request_path = ?,
                    log_path = ?, updated_at = ?
                WHERE task_id = ?
            ''', (
                job.model, job.batch_root, job.dataset_yaml,
                _json_dump(job.parameters),
                _json_dump(training_job_to_dict(job)),
                str(Path(request_path).expanduser().resolve()),
                str(Path(log_path).expanduser().resolve()),
                now, task_id,
            ))
        return self.require(task_id)

    def get(self, task_id: str) -> TrainingTaskRecord | None:
        with self._lock:
            row = self._connection.execute(
                'SELECT * FROM training_tasks WHERE task_id = ?',
                (str(task_id),),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def require(self, task_id: str) -> TrainingTaskRecord:
        record = self.get(task_id)
        if record is None:
            raise TrainingTaskRegistryError(f'训练任务不存在: {task_id}')
        return record

    def get_by_run_dir(self, run_dir: str | Path) -> TrainingTaskRecord | None:
        path = str(Path(run_dir).expanduser().resolve())
        with self._lock:
            row = self._connection.execute(
                'SELECT * FROM training_tasks WHERE run_dir = ?', (path,)
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list_tasks(self, *, status: str = '', search: str = '',
                   include_archived: bool = False) -> list[TrainingTaskRecord]:
        clauses = []
        values: list[Any] = []
        if not include_archived:
            clauses.append('archived = 0')
        if status:
            if status == 'archived':
                clauses.append('archived = 1')
            else:
                clauses.append('status = ?')
                values.append(status)
        if search.strip():
            query = f'%{search.strip()}%'
            clauses.append('''(
                display_name LIKE ? OR project_name LIKE ? OR model LIKE ?
                OR batch_root LIKE ? OR task_type LIKE ?
            )''')
            values.extend([query] * 5)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with self._lock:
            rows = self._connection.execute(
                'SELECT * FROM training_tasks' + where
                + ' ORDER BY created_at DESC',
                tuple(values),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def next_queued(self) -> TrainingTaskRecord | None:
        with self._lock:
            row = self._connection.execute('''
                SELECT * FROM training_tasks
                WHERE status = 'queued' AND archived = 0
                ORDER BY created_at ASC LIMIT 1
            ''').fetchone()
        return _record_from_row(row) if row is not None else None

    def set_status(self, task_id: str, status: str, *,
                   error_message: str | None = None,
                   pid: int | None = None) -> TrainingTaskRecord:
        self._validate_status(status)
        now = _utc_now()
        fields: dict[str, Any] = {'status': status, 'updated_at': now}
        if error_message is not None:
            fields['error_message'] = str(error_message)
        if pid is not None:
            fields['pid'] = int(pid)
        if status == 'running':
            fields['started_at'] = now
            fields['finished_at'] = ''
        if status in TERMINAL_STATUSES:
            fields['finished_at'] = now
            fields['pid'] = None
        self._update_fields(task_id, fields)
        return self.require(task_id)

    def update_progress(self, task_id: str, *, epoch: int, epochs: int,
                        progress: float) -> TrainingTaskRecord:
        self._update_fields(task_id, {
            'current_epoch': max(0, int(epoch)),
            'total_epochs': max(0, int(epochs)),
            'progress': max(0.0, min(float(progress), 100.0)),
            'updated_at': _utc_now(),
        })
        return self.require(task_id)

    def rename(self, task_id: str, display_name: str) -> TrainingTaskRecord:
        name = str(display_name or '').strip()
        if not name:
            raise TrainingTaskRegistryError('任务显示名称不能为空')
        self._update_fields(task_id, {
            'display_name': name,
            'updated_at': _utc_now(),
        })
        return self.require(task_id)

    def set_notes(self, task_id: str, notes: str) -> TrainingTaskRecord:
        self._update_fields(task_id, {
            'notes': str(notes or ''),
            'updated_at': _utc_now(),
        })
        return self.require(task_id)

    def archive(self, task_id: str, archived: bool = True) -> TrainingTaskRecord:
        current = self.require(task_id)
        if current.status in ACTIVE_STATUSES:
            raise TrainingTaskRegistryError('运行中的任务不能归档')
        fields = {
            'archived': int(bool(archived)),
            'updated_at': _utc_now(),
        }
        # Older databases used ``archived`` as a lifecycle status. Preserve a
        # usable status when such a record is restored; new records retain their
        # original lifecycle while hidden.
        if not archived and current.status == 'archived':
            fields['status'] = 'failed'
        self._update_fields(task_id, fields)
        return self.require(task_id)

    def delete(self, task_id: str):
        current = self.require(task_id)
        if current.status in ACTIVE_STATUSES:
            raise TrainingTaskRegistryError('运行中的任务不能删除')
        with self._lock, self._connection:
            self._connection.execute(
                'DELETE FROM training_tasks WHERE task_id = ?', (task_id,)
            )

    def counts(self) -> dict[str, int]:
        result = {'total': 0, 'active': 0, 'queued': 0,
                  'completed': 0, 'failed': 0, 'archived': 0}
        with self._lock:
            rows = self._connection.execute('''
                SELECT status, archived, COUNT(*) AS count
                FROM training_tasks GROUP BY status, archived
            ''').fetchall()
        for row in rows:
            count = int(row['count'])
            result['total'] += count
            if row['archived']:
                result['archived'] += count
            elif row['status'] in ACTIVE_STATUSES:
                result['active'] += count
            elif row['status'] == 'queued':
                result['queued'] += count
            elif row['status'] == 'completed':
                result['completed'] += count
            elif row['status'] in {'failed', 'cancelled', 'interrupted'}:
                result['failed'] += count
        return result

    def recover(self, *, request_directory: str | Path,
                output_roots: Iterable[str | Path],
                extra_request_directories: Iterable[str | Path] = (),
                mark_active_interrupted: bool = True) -> dict[str, int]:
        recovered = (
            self._mark_active_interrupted() if mark_active_interrupted else 0
        )
        imported = 0
        request_dirs = _unique_paths(
            (request_directory, *extra_request_directories)
        )
        for requests in request_dirs:
            if not requests.is_dir():
                continue
            paths = set(requests.glob('*.json'))
            paths.update(requests.rglob('training_request.json'))
            for path in sorted(paths):
                if self._import_request(path):
                    imported += 1
        discovered = 0
        for root in _unique_paths(output_roots):
            if root.is_dir():
                discovered += self._discover_runs(root)
        return {
            'interrupted': recovered,
            'imported': imported,
            'discovered': discovered,
        }

    def _import_request(self, path: Path) -> bool:
        try:
            job = load_training_job(path)
        except Exception:
            return False
        if not Path(job.batch_root).is_dir() and not job.run_dir.is_dir():
            return False
        existing = self.get_by_run_dir(job.run_dir)
        run_exists = job.run_dir.is_dir()
        inspection = (
            inspect_training_run(job.run_dir)
            if run_exists else TrainingRunInspection(
                status='draft', error_message='', current_epoch=0,
                total_epochs=int(job.parameters.get('epochs') or 0),
                progress=0.0,
            )
        )
        log_path = (
            path.with_name('training.log')
            if path.name == 'training_request.json'
            else path.parent.parent / 'training_logs' / f'{job.job_id}.log'
        )
        if existing is not None:
            fields = {
                'request_path': str(path.resolve()),
                'log_path': str(log_path.resolve()),
                'updated_at': _utc_now(),
            }
            if run_exists and existing.status not in ACTIVE_STATUSES:
                fields.update({
                    'status': inspection.status,
                    'error_message': inspection.error_message,
                    'current_epoch': inspection.current_epoch,
                    'total_epochs': (
                        inspection.total_epochs or existing.total_epochs
                    ),
                    'progress': inspection.progress,
                })
            self._update_fields(existing.task_id, fields)
            return False
        self.register_job(
            job, status=inspection.status, request_path=path.resolve(),
            log_path=log_path.resolve(), source='recovered',
        )
        if run_exists:
            self._update_fields(job.job_id, {
                'error_message': inspection.error_message,
                'current_epoch': inspection.current_epoch,
                'total_epochs': inspection.total_epochs,
                'progress': inspection.progress,
            })
        return True

    def _discover_runs(self, output_root: Path) -> int:
        count = 0
        for args_path in sorted(output_root.rglob('args.yaml')):
            run_dir = args_path.parent.resolve()
            existing = self.get_by_run_dir(run_dir)
            inspection = inspect_training_run(run_dir)
            if existing is not None:
                if existing.status not in ACTIVE_STATUSES:
                    fields = {
                        'status': inspection.status,
                        'error_message': inspection.error_message,
                        'current_epoch': inspection.current_epoch,
                        'total_epochs': (
                            inspection.total_epochs or existing.total_epochs
                        ),
                        'progress': inspection.progress,
                        'updated_at': _utc_now(),
                    }
                    self._update_fields(existing.task_id, fields)
                continue
            job = _job_from_run(args_path, output_root)
            if job is None:
                continue
            self.register_job(
                job, status=inspection.status, source='discovered'
            )
            self._update_fields(job.job_id, {
                'error_message': inspection.error_message,
                'current_epoch': inspection.current_epoch,
                'total_epochs': inspection.total_epochs,
                'progress': inspection.progress,
            })
            count += 1
        return count

    def _mark_active_interrupted(self) -> int:
        now = _utc_now()
        placeholders = ', '.join('?' for _ in ACTIVE_STATUSES)
        with self._lock, self._connection:
            cursor = self._connection.execute(f'''
                UPDATE training_tasks
                SET status = 'interrupted', pid = NULL, finished_at = ?,
                    updated_at = ?,
                    error_message = CASE WHEN error_message = ''
                        THEN '应用关闭时任务尚未结束' ELSE error_message END
                WHERE status IN ({placeholders})
            ''', (now, now, *ACTIVE_STATUSES))
        return int(cursor.rowcount)

    def _update_fields(self, task_id: str, fields: dict[str, Any]):
        if not fields:
            return
        allowed = {
            'display_name', 'status', 'request_path', 'log_path', 'progress',
            'current_epoch', 'total_epochs', 'pid', 'updated_at', 'started_at',
            'finished_at', 'error_message', 'archived', 'notes',
        }
        invalid = set(fields).difference(allowed)
        if invalid:
            raise TrainingTaskRegistryError(
                '无法更新任务字段: ' + ', '.join(sorted(invalid))
            )
        assignments = ', '.join(f'{name} = ?' for name in fields)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f'UPDATE training_tasks SET {assignments} WHERE task_id = ?',
                (*fields.values(), task_id),
            )
        if cursor.rowcount == 0:
            raise TrainingTaskRegistryError(f'训练任务不存在: {task_id}')

    @staticmethod
    def _validate_status(status: str):
        if status not in TASK_STATUSES:
            raise TrainingTaskRegistryError(f'无效训练任务状态: {status}')


def _record_from_row(row: sqlite3.Row) -> TrainingTaskRecord:
    job_payload = json.loads(row['job_json'])
    return TrainingTaskRecord(
        task_id=row['task_id'],
        display_name=row['display_name'],
        status=row['status'],
        task_type=row['task_type'],
        run_name=row['run_name'],
        project_name=row['project_name'],
        model=row['model'],
        batch_root=row['batch_root'],
        dataset_yaml=row['dataset_yaml'],
        output_root=row['output_root'],
        run_dir=row['run_dir'],
        parameters=json.loads(row['parameters_json']),
        job=training_job_from_dict(job_payload),
        request_path=row['request_path'],
        log_path=row['log_path'],
        progress=float(row['progress']),
        current_epoch=int(row['current_epoch']),
        total_epochs=int(row['total_epochs']),
        pid=int(row['pid']) if row['pid'] is not None else None,
        created_at=row['created_at'],
        updated_at=row['updated_at'],
        started_at=row['started_at'],
        finished_at=row['finished_at'],
        error_message=row['error_message'],
        archived=bool(row['archived']),
        source=row['source'],
        notes=row['notes'],
    )


def _job_from_run(args_path: Path, output_root: Path) -> TrainingJob | None:
    try:
        args = yaml.safe_load(args_path.read_text(encoding='utf-8')) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(args, dict):
        return None
    run_dir = args_path.parent.resolve()
    task_type = ULTRALYTICS_TO_TASK.get(str(args.get('task') or ''))
    if task_type is None:
        return None
    data_value = str(args.get('data') or '')
    dataset_yaml = Path(data_value).expanduser()
    if not dataset_yaml.is_absolute():
        dataset_yaml = (run_dir / dataset_yaml).resolve()
    batch_root = dataset_yaml.parent
    root = output_root.resolve()
    project_name = run_dir.parent.name
    managed = set(MANAGED_PARAMETER_NAMES) | {
        'model', 'data', 'project', 'name', 'task', 'mode', 'save_dir',
    }
    parameters = {
        str(key): value for key, value in args.items()
        if key not in managed and _json_compatible(value)
    }
    created = datetime.fromtimestamp(
        args_path.stat().st_mtime, timezone.utc
    ).isoformat()
    identifier = uuid.uuid5(uuid.NAMESPACE_URL, str(run_dir)).hex
    return TrainingJob(
        job_id=identifier,
        created_at=created,
        task_type=task_type,
        ultralytics_task=str(args.get('task')),
        model=str(args.get('model') or ''),
        batch_root=str(batch_root),
        dataset_yaml=str(dataset_yaml),
        output_root=str(root),
        project_name=project_name,
        run_name=run_dir.name,
        parameters=parameters,
    )


def inspect_training_run(run_dir: str | Path) -> TrainingRunInspection:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        return TrainingRunInspection(
            'interrupted', '训练请求已创建，但运行目录不存在', 0, 0, 0.0
        )

    total_epochs = _read_total_epochs(run_dir)
    current_epoch = _read_completed_epochs(run_dir / 'results.csv')
    progress = (
        min(current_epoch / total_epochs * 100.0, 100.0)
        if total_epochs else 0.0
    )
    results = run_dir / 'results.csv'
    weights = run_dir / 'weights'
    has_weight = weights.is_dir() and any(weights.glob('*.pt'))
    completion_marker = run_dir / 'training_complete.json'
    normally_completed = completion_marker.is_file()
    legacy_completed = bool(
        results.is_file() and has_weight and total_epochs
        and current_epoch >= total_epochs
    )
    legacy_early_stop = bool(
        results.is_file() and has_weight and (run_dir / 'results.png').is_file()
    )
    if normally_completed or legacy_completed or legacy_early_stop:
        return TrainingRunInspection(
            'completed', '', current_epoch, total_epochs, progress
        )
    if current_epoch:
        expected = str(total_epochs) if total_epochs else '?'
        return TrainingRunInspection(
            'interrupted',
            f'训练在完成 {current_epoch}/{expected} 个 epoch 后中断，'
            '未检测到正常完成标记',
            current_epoch, total_epochs, progress,
        )
    if (run_dir / 'args.yaml').is_file() or any(run_dir.iterdir()):
        return TrainingRunInspection(
            'failed', '检测到未完成的训练产物', 0, total_epochs, 0.0
        )
    return TrainingRunInspection(
        'failed', '训练目录为空，任务未完成', 0, total_epochs, 0.0
    )


def _infer_run_status(run_dir: Path) -> tuple[str, str]:
    inspection = inspect_training_run(run_dir)
    return inspection.status, inspection.error_message


def _read_total_epochs(run_dir: Path) -> int:
    for path in (run_dir / 'args.yaml', run_dir / 'training_request.json'):
        if not path.is_file():
            continue
        try:
            if path.suffix == '.json':
                payload = json.loads(path.read_text(encoding='utf-8'))
                value = (payload.get('parameters') or {}).get('epochs')
            else:
                payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
                value = payload.get('epochs')
            return max(0, int(value or 0))
        except (OSError, UnicodeError, ValueError, TypeError,
                json.JSONDecodeError, yaml.YAMLError):
            continue
    return 0


def _read_completed_epochs(results_path: Path) -> int:
    if not results_path.is_file():
        return 0
    row_count = 0
    last_epoch = 0
    try:
        with results_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                row_count += 1
                raw = row.get('epoch') or row.get(' epoch')
                try:
                    last_epoch = max(last_epoch, int(float(str(raw).strip())))
                except (TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return 0
    # Some Ultralytics versions store zero-based epochs, while newer builds
    # write one-based values. The number of complete CSV rows is unambiguous.
    return max(row_count, last_epoch)


def _unique_paths(paths: Iterable[str | Path]) -> list[Path]:
    result = []
    seen = set()
    for value in paths:
        if not str(value or '').strip():
            continue
        path = Path(value).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _json_dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _json_compatible(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
