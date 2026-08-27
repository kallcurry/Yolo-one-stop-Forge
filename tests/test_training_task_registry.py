import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.models.training_job import TrainingJob, write_training_job
from app.models.training_task_registry import TrainingTaskRegistry


class TrainingTaskRegistryTest(unittest.TestCase):
    def _job(self, root: Path, name='pose-run') -> TrainingJob:
        batch = root / 'dataset' / 'training_data' / 'batch-a'
        batch.mkdir(parents=True, exist_ok=True)
        yaml_path = batch / 'dataset.yaml'
        yaml_path.write_text('train: images/train\nval: images/val\n', encoding='utf-8')
        return TrainingJob(
            job_id=f'job-{name}',
            created_at=datetime.now(timezone.utc).isoformat(),
            task_type='pose',
            ultralytics_task='pose',
            model='yolov8n-pose.pt',
            batch_root=str(batch),
            dataset_yaml=str(yaml_path),
            output_root=str(root / 'training_runs'),
            project_name='ShengSong',
            run_name=name,
            parameters={'epochs': 10, 'device': 'cpu'},
        )

    def test_registry_persists_lifecycle_progress_and_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / 'tasks.sqlite3'
            registry = TrainingTaskRegistry(database)
            job = self._job(root)

            created = registry.register_job(job, status='queued')
            registry.set_status(created.task_id, 'running', pid=123)
            registry.update_progress(
                created.task_id, epoch=4, epochs=10, progress=40.0
            )
            registry.set_status(created.task_id, 'completed')
            registry.close()

            reopened = TrainingTaskRegistry(database)
            record = reopened.require(created.task_id)
            self.assertEqual(record.status, 'completed')
            self.assertEqual(record.current_epoch, 4)
            self.assertEqual(record.progress, 40.0)
            self.assertIsNone(record.pid)
            self.assertTrue(record.started_at)
            self.assertTrue(record.finished_at)
            reopened.archive(record.task_id)
            self.assertEqual(reopened.list_tasks(), [])
            self.assertEqual(
                len(reopened.list_tasks(status='archived', include_archived=True)),
                1,
            )
            restored = reopened.archive(record.task_id, False)
            self.assertEqual(restored.status, 'completed')
            self.assertFalse(restored.archived)
            reopened.close()

    def test_recovery_imports_real_request_and_ignores_missing_temp_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / '.runtime' / 'training_jobs'
            job = self._job(root)
            job.run_dir.mkdir(parents=True)
            (job.run_dir / 'args.yaml').write_text(
                'task: pose\nmodel: yolov8n-pose.pt\n', encoding='utf-8'
            )
            write_training_job(job, runtime)

            stale = TrainingJob(
                **{
                    **job.__dict__,
                    'job_id': 'stale-job',
                    'batch_root': str(root / 'missing-batch'),
                    'output_root': str(root / 'missing-output'),
                    'run_name': 'stale-run',
                }
            )
            write_training_job(stale, runtime)

            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')
            result = registry.recover(
                request_directory=runtime,
                output_roots=(root / 'training_runs',),
            )

            records = registry.list_tasks()
            self.assertEqual(result['imported'], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, 'failed')
            self.assertIn('未完成', records[0].error_message)
            registry.close()

    def test_recovery_marks_partial_run_interrupted_with_actual_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = self._job(root)
            job.run_dir.mkdir(parents=True)
            (job.run_dir / 'weights').mkdir()
            (job.run_dir / 'weights' / 'last.pt').write_bytes(b'partial')
            (job.run_dir / 'args.yaml').write_text(
                'task: pose\nepochs: 10\nmodel: yolov8n-pose.pt\n'
                f'data: {job.dataset_yaml}\n',
                encoding='utf-8',
            )
            (job.run_dir / 'results.csv').write_text(
                'epoch,time\n1,1.0\n2,2.0\n3,3.0\n4,4.0\n',
                encoding='utf-8',
            )
            request = write_training_job(
                job, root / 'requests', filename='training_request.json'
            )
            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')

            registry.recover(
                request_directory=request.parent,
                output_roots=(root / 'training_runs',),
            )

            record = registry.require(job.job_id)
            self.assertEqual(record.status, 'interrupted')
            self.assertEqual(record.current_epoch, 4)
            self.assertEqual(record.total_epochs, 10)
            self.assertEqual(record.progress, 40.0)
            self.assertIn('4/10', record.error_message)
            registry.close()

    def test_recovery_accepts_explicit_early_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = self._job(root)
            job.run_dir.mkdir(parents=True)
            (job.run_dir / 'weights').mkdir()
            (job.run_dir / 'weights' / 'best.pt').write_bytes(b'complete')
            (job.run_dir / 'args.yaml').write_text(
                'task: pose\nepochs: 10\nmodel: yolov8n-pose.pt\n'
                f'data: {job.dataset_yaml}\n',
                encoding='utf-8',
            )
            (job.run_dir / 'results.csv').write_text(
                'epoch,time\n1,1.0\n2,2.0\n3,3.0\n4,4.0\n',
                encoding='utf-8',
            )
            (job.run_dir / 'training_complete.json').write_text(
                '{"actual_epoch": 4, "total_epochs": 10, '
                '"early_stopped": true}',
                encoding='utf-8',
            )
            request = write_training_job(
                job, root / 'requests', filename='training_request.json'
            )
            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')

            registry.recover(
                request_directory=request.parent,
                output_roots=(root / 'training_runs',),
            )

            record = registry.require(job.job_id)
            self.assertEqual(record.status, 'completed')
            self.assertEqual(record.current_epoch, 4)
            self.assertEqual(record.progress, 40.0)
            registry.close()

    def test_recovery_marks_previous_running_task_interrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')
            job = self._job(root)
            registry.register_job(job, status='running')

            result = registry.recover(
                request_directory=root / 'requests', output_roots=()
            )

            record = registry.require(job.job_id)
            self.assertEqual(result['interrupted'], 1)
            self.assertEqual(record.status, 'interrupted')
            registry.delete(record.task_id)
            self.assertIsNone(registry.get(record.task_id))
            registry.close()

    def test_regular_refresh_does_not_interrupt_running_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')
            job = self._job(root)
            registry.register_job(job, status='running')

            result = registry.recover(
                request_directory=root / 'requests', output_roots=(),
                mark_active_interrupted=False,
            )

            self.assertEqual(result['interrupted'], 0)
            self.assertEqual(registry.require(job.job_id).status, 'running')
            registry.close()

    def test_artifact_migration_preserves_task_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')
            job = self._job(root)
            registry.register_job(
                job, status='failed', request_path=root / 'legacy.json'
            )
            task_dir = root / 'training' / 'tasks' / job.job_id
            task_dir.mkdir(parents=True)
            dataset = task_dir / 'dataset.yaml'
            dataset.write_text('train: images/train\n', encoding='utf-8')
            bundled = TrainingJob(**{
                **job.__dict__, 'dataset_yaml': str(dataset),
            })

            migrated = registry.relocate_artifacts(
                job.job_id, bundled,
                request_path=task_dir / 'training_request.json',
                log_path=task_dir / 'training.log',
            )

            self.assertEqual(migrated.status, 'failed')
            self.assertEqual(Path(migrated.dataset_yaml), dataset)
            self.assertEqual(
                Path(migrated.request_path),
                task_dir / 'training_request.json',
            )
            registry.close()

    def test_recovery_keeps_saved_task_without_run_as_draft(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / 'training' / 'tasks' / 'job-pose-run'
            job = self._job(root)
            request = write_training_job(
                job, task_dir, filename='training_request.json'
            )

            registry = TrainingTaskRegistry(root / 'tasks.sqlite3')
            result = registry.recover(
                request_directory=root / 'legacy',
                extra_request_directories=(root / 'training' / 'tasks',),
                output_roots=(root / 'training' / 'runs',),
            )

            record = registry.require(job.job_id)
            self.assertEqual(result['imported'], 1)
            self.assertEqual(record.status, 'draft')
            self.assertEqual(Path(record.request_path), request.resolve())
            self.assertEqual(Path(record.log_path), task_dir / 'training.log')
            registry.close()


if __name__ == '__main__':
    unittest.main()
