import json
import tempfile
import unittest
from pathlib import Path

from app.models.training_job import (
    TrainingJobError,
    create_training_job,
    load_training_job,
    write_training_job,
)


class TrainingJobTest(unittest.TestCase):
    def _ready_batch(self, root: Path) -> Path:
        batch = root / 'training_data' / '2026-08-07'
        for relative in (
            'images', 'labels',
            'train_data/images/train', 'train_data/images/val',
            'train_data/labels/train', 'train_data/labels/val',
        ):
            (batch / relative).mkdir(parents=True, exist_ok=True)
        for stem, split in (('frame_train', 'train'), ('frame_val', 'val')):
            (batch / 'images' / f'{stem}.jpg').write_bytes(b'image')
            (batch / 'labels' / f'{stem}.txt').write_text(
                '0 0.5 0.5 0.2 0.2\n', encoding='utf-8'
            )
            (batch / 'train_data' / 'images' / split / f'{stem}.jpg').write_bytes(
                b'image'
            )
            (batch / 'train_data' / 'labels' / split / f'{stem}.txt').write_text(
                '0 0.5 0.5 0.2 0.2\n', encoding='utf-8'
            )
        (batch / 'dataset.yaml').write_text(
            'path: train_data\ntrain: images/train\nval: images/val\n'
            'names: {0: person}\n',
            encoding='utf-8',
        )
        (batch / 'preparation_manifest.json').write_text(
            json.dumps({'request': {
                'task_type': 'pose', 'annotation_dir': 'annotations',
            }}),
            encoding='utf-8',
        )
        return batch

    def test_job_round_trip_preserves_managed_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch = self._ready_batch(root)
            job = create_training_job(
                task_type='pose', model='yolov8n-pose.pt',
                batch_root=batch, output_root=root / 'runs',
                project_name='ShengSong', run_name='pose-run',
                parameters={'epochs': 2, 'device': 'cpu'},
            )
            path = write_training_job(job, root / 'requests')
            loaded = load_training_job(path)

            self.assertEqual(loaded, job)
            self.assertEqual(
                loaded.run_dir, root / 'runs' / 'ShengSong' / 'pose-run'
            )
            self.assertEqual(loaded.ultralytics_task, 'pose')
            self.assertEqual(
                Path(loaded.model),
                Path(__file__).resolve().parents[1] / 'models' / 'yolov8n-pose.pt',
            )

    def test_job_rejects_existing_run_without_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch = self._ready_batch(root)
            (root / 'runs' / 'ShengSong' / 'pose-run').mkdir(parents=True)
            with self.assertRaisesRegex(TrainingJobError, '已存在'):
                create_training_job(
                    task_type='pose', model='yolov8n-pose.pt',
                    batch_root=batch, output_root=root / 'runs',
                    project_name='ShengSong', run_name='pose-run',
                    parameters={'epochs': 2},
                )

    def test_job_resume_requires_last_checkpoint_in_same_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch = self._ready_batch(root)
            run_dir = root / 'runs' / 'ShengSong' / 'pose-run'
            checkpoint = run_dir / 'weights' / 'last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'checkpoint')

            job = create_training_job(
                task_type='pose', model=str(checkpoint),
                batch_root=batch, output_root=root / 'runs',
                project_name='ShengSong', run_name='pose-run',
                parameters={'epochs': 10, 'resume': True},
            )

            self.assertTrue(job.parameters['resume'])
            self.assertEqual(Path(job.model), checkpoint.resolve())

            with self.assertRaisesRegex(TrainingJobError, 'weights/last.pt'):
                create_training_job(
                    task_type='pose', model='yolov8n-pose.pt',
                    batch_root=batch, output_root=root / 'runs',
                    project_name='ShengSong', run_name='pose-run',
                    parameters={'epochs': 10, 'resume': True},
                )

    def test_bare_model_yaml_is_left_for_ultralytics_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch = self._ready_batch(root)
            job = create_training_job(
                task_type='pose', model='yolov8n-pose.yaml',
                batch_root=batch, output_root=root / 'runs',
                project_name='ShengSong', run_name='pose-yaml-run',
                parameters={'epochs': 2},
            )

            self.assertEqual(job.model, 'yolov8n-pose.yaml')


if __name__ == '__main__':
    unittest.main()
