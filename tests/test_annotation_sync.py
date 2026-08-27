import json
import tempfile
import unittest
from pathlib import Path

from app.models.annotation_sync import (
    AnnotationSyncError,
    annotation_file_fingerprint,
    synchronize_annotation_replicas,
)


def _annotation(label: str) -> str:
    return json.dumps({
        'version': '3.3.10',
        'shapes': [{'label': label, 'shape_type': 'point', 'points': [[1, 2]]}],
    }, ensure_ascii=False)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _write_manifest(batch: Path, canonical: Path,
                    annotation_dir: str = 'annotations'):
    batch.mkdir(parents=True, exist_ok=True)
    (batch / 'preparation_manifest.json').write_text(
        json.dumps({
            'version': 1,
            'request': {'annotation_dir': annotation_dir},
            'records': [{
                'stem': canonical.stem,
                'source_name': canonical.parent.name,
                'annotation_path': str(canonical),
            }],
        }),
        encoding='utf-8',
    )


class AnnotationSyncTest(unittest.TestCase):
    def test_manifest_identity_syncs_raw_training_and_test_replicas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'Dataset'
            (root / 'images' / 'source-a').mkdir(parents=True)
            canonical = root / 'annotations' / 'source-a' / 'frame_001.json'
            first = root / 'training_data' / 'batch-a'
            second = root / 'training_data' / 'batch-b'
            test_copy = root / 'test_data' / 'annotations' / 'frame_001.json'
            _write(canonical, _annotation('old'))
            _write(first / 'annotations' / 'frame_001.json', _annotation('new'))
            _write(second / 'annotations' / 'frame_001.json', _annotation('old'))
            _write(test_copy, _annotation('old'))
            _write_manifest(first, canonical)
            _write_manifest(second, canonical)

            result = synchronize_annotation_replicas(
                first / 'annotations' / 'frame_001.json', first
            )

            self.assertEqual(result.canonical, canonical.resolve())
            self.assertEqual(len(result.updated), 3)
            for path in (canonical, second / 'annotations' / 'frame_001.json', test_copy):
                self.assertIn('"new"', path.read_text(encoding='utf-8'))

    def test_task_specific_annotation_sets_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'Dataset'
            (root / 'images' / 'source-a').mkdir(parents=True)
            obb_source = (
                root / 'annotations-obb' / 'source-a' / 'frame_001.json'
            )
            pose_source = root / 'annotations' / 'source-a' / 'frame_001.json'
            obb_batch = root / 'training_data' / 'obb-batch'
            pose_batch = root / 'training_data' / 'pose-batch'
            _write(obb_source, _annotation('obb-old'))
            _write(pose_source, _annotation('pose-old'))
            _write(
                obb_batch / 'annotations-obb' / 'frame_001.json',
                _annotation('obb-new'),
            )
            _write(
                pose_batch / 'annotations' / 'frame_001.json',
                _annotation('pose-old'),
            )
            _write_manifest(obb_batch, obb_source, 'annotations-obb')
            _write_manifest(pose_batch, pose_source, 'annotations')

            synchronize_annotation_replicas(
                obb_batch / 'annotations-obb' / 'frame_001.json',
                root,
                'annotations-obb',
            )

            self.assertIn('obb-new', obb_source.read_text(encoding='utf-8'))
            self.assertIn('pose-old', pose_source.read_text(encoding='utf-8'))
            self.assertIn(
                'pose-old',
                (pose_batch / 'annotations' / 'frame_001.json').read_text(
                    encoding='utf-8'
                ),
            )

    def test_moved_dataset_rebuilds_canonical_path_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'MovedDataset'
            (root / 'images' / 'source-a').mkdir(parents=True)
            canonical = root / 'annotations' / 'source-a' / 'frame_001.json'
            batch = root / 'training_data' / 'batch-a'
            changed = batch / 'annotations' / 'frame_001.json'
            _write(canonical, _annotation('old'))
            _write(changed, _annotation('new'))
            _write_manifest(
                batch,
                Path('/old/computer/Dataset/annotations/source-a/frame_001.json'),
            )

            result = synchronize_annotation_replicas(changed, root)

            self.assertEqual(result.canonical, canonical.resolve())
            self.assertIn('new', canonical.read_text(encoding='utf-8'))

    def test_ambiguous_legacy_filename_does_not_overwrite_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'Dataset'
            (root / 'images').mkdir(parents=True)
            first_raw = root / 'annotations' / 'source-a' / 'same.json'
            second_raw = root / 'annotations' / 'source-b' / 'same.json'
            changed = root / 'training_data' / 'legacy' / 'annotations' / 'same.json'
            _write(first_raw, _annotation('first'))
            _write(second_raw, _annotation('second'))
            _write(changed, _annotation('changed'))

            result = synchronize_annotation_replicas(changed, root)

            self.assertTrue(result.ambiguous)
            self.assertEqual(result.updated, ())
            self.assertIn('first', first_raw.read_text(encoding='utf-8'))
            self.assertIn('second', second_raw.read_text(encoding='utf-8'))

    def test_invalid_json_is_not_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'Dataset'
            (root / 'images').mkdir(parents=True)
            changed = root / 'annotations' / 'source-a' / 'bad.json'
            _write(changed, '{bad json')

            with self.assertRaises(AnnotationSyncError):
                synchronize_annotation_replicas(changed, root)

    def test_fingerprint_only_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'annotation.json'
            _write(path, _annotation('first'))
            first = annotation_file_fingerprint(path)
            path.touch()
            self.assertEqual(annotation_file_fingerprint(path), first)
            _write(path, _annotation('second'))
            self.assertNotEqual(annotation_file_fingerprint(path), first)


if __name__ == '__main__':
    unittest.main()
