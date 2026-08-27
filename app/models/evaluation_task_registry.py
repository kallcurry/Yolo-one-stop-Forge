"""Evaluation task registry backed by sqlite3 (evaluation center, phase 1)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


@dataclass(frozen=True)
class EvaluationTaskRecord:
    task_id: str
    status: str
    created_at: str
    updated_at: str
    spec: dict
    output_dir: str = ''
    log_path: str = ''
    task_dir: str = ''
    error: str = ''
    summary: str = ''

    @property
    def metrics(self) -> dict:
        try:
            return json.loads(self.summary) or {}
        except (TypeError, ValueError):
            return {}


ACTIVE_STATUSES = ('queued', 'running')


class EvaluationTaskRegistry:
    """Minimal sqlite registry; a busy timeout keeps recovery writes safe."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute('PRAGMA busy_timeout=5000')
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    output_dir TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL DEFAULT '',
                    task_dir TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f'评估任务注册表不可写: {self.path} ({exc})。'
                '请检查目录权限后重试。'
            ) from exc

    def close(self):
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def create(self, spec: dict, task_dir: str = '', output_dir: str = '',
               log_path: str = '') -> EvaluationTaskRecord:
        record = EvaluationTaskRecord(
            task_id=str(spec['job_id']),
            status='queued',
            created_at=_now(),
            updated_at=_now(),
            spec=spec,
            task_dir=task_dir,
            output_dir=output_dir,
            log_path=log_path,
        )
        self._conn.execute(
            'INSERT OR REPLACE INTO evaluation_tasks '
            '(task_id, status, created_at, updated_at, spec, task_dir, '
            ' output_dir, log_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                record.task_id, record.status, record.created_at,
                record.updated_at, json.dumps(spec, ensure_ascii=False),
                record.task_dir, record.output_dir, record.log_path,
            ),
        )
        self._conn.commit()
        return record

    def update(self, task_id: str, **changes: Any) -> bool:
        columns = {
            'status', 'output_dir', 'log_path', 'task_dir', 'error', 'summary',
        }
        values = {key: value for key, value in changes.items() if key in columns}
        if not values:
            return False
        values['updated_at'] = _now()
        assignments = ', '.join(f'{key} = ?' for key in values)
        row = tuple(
            json.dumps(value, ensure_ascii=False)
            if key == 'summary' and not isinstance(value, str)
            else value
            for key, value in values.items()
        )
        self._conn.execute(
            f'UPDATE evaluation_tasks SET {assignments} '
            f'WHERE task_id = ?',
            (*row, task_id),
        )
        self._conn.commit()
        return True

    def get(self, task_id: str) -> EvaluationTaskRecord | None:
        row = self._conn.execute(
            'SELECT task_id, status, created_at, updated_at, spec, output_dir, '
            'log_path, task_dir, error, summary FROM evaluation_tasks '
            'WHERE task_id = ?',
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_all(self) -> list[EvaluationTaskRecord]:
        rows = self._conn.execute(
            'SELECT task_id, status, created_at, updated_at, spec, output_dir, '
            'log_path, task_dir, error, summary FROM evaluation_tasks '
            'ORDER BY created_at DESC'
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def next_queued(self) -> EvaluationTaskRecord | None:
        row = self._conn.execute(
            'SELECT task_id, status, created_at, updated_at, spec, output_dir, '
            'log_path, task_dir, error, summary FROM evaluation_tasks '
            'WHERE status = \'queued\' ORDER BY created_at ASC LIMIT 1'
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def running_or_queued(self) -> list[EvaluationTaskRecord]:
        rows = self._conn.execute(
            'SELECT task_id, status, created_at, updated_at, spec, output_dir, '
            'log_path, task_dir, error, summary FROM evaluation_tasks '
            "WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete(self, task_id: str) -> bool:
        cursor = self._conn.execute(
            'DELETE FROM evaluation_tasks WHERE task_id = ?', (task_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def recover_interrupted(self) -> int:
        """Mark tasks left 'running' after an app crash as 'interrupted'."""
        cursor = self._conn.execute(
            "UPDATE evaluation_tasks SET status = 'interrupted', "
            "updated_at = ? WHERE status = 'running'",
            (_now(),),
        )
        self._conn.commit()
        return cursor.rowcount

    @staticmethod
    def _record_from_row(row) -> EvaluationTaskRecord:
        return EvaluationTaskRecord(
            task_id=row[0],
            status=row[1],
            created_at=row[2],
            updated_at=row[3],
            spec=json.loads(row[4]),
            output_dir=row[5],
            log_path=row[6],
            task_dir=row[7],
            error=row[8],
            summary=row[9],
        )
