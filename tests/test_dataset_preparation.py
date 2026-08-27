import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from app.models.dataset_preparation import (
    DatasetPreparationError,
    DatasetPreparationRequest,
    ensure_training_dataset_yaml,
    inspect_training_batch,
    list_source_batches,
    prepare_existing_batch_split,
    prepare_dataset,
    scan_dataset,
)


class DatasetPreparationTest(unittest.TestCase):
    def _build_dataset(self, root: Path, count: int = 10,
                       source: str = 'Collect_A', start: int = 0):
        image_dir = root / 'images' / source
        annotation_dir = root / 'annotations' / source
        label_dir = root / 'labels' / source
        for directory in (image_dir, annotation_dir, label_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for index in range(start, start + count):
            stem = f'frame_{index:03d}'
            (image_dir / f'{stem}.png').write_bytes(b'image')
            (annotation_dir / f'{stem}.json').write_text(
                json.dumps({
                    'imageWidth': 100,
                    'imageHeight': 100,
                    'shapes': [
                        {
                            'label': 'person',
                            'shape_type': 'rectangle',
                            'group_id': 0,
                            'points': [[10, 10], [80, 90]],
                        },
                        {
                            'label': 'left',
                            'shape_type': 'point',
                            'group_id': 0,
                            'points': [[20, 30]],
                        },
                        {
                            'label': 'right',
                            'shape_type': 'point',
                            'group_id': 0,
                            'points': [[70, 30]],
                        },
                    ],
                }),
                encoding='utf-8',
            )
            (label_dir / f'{stem}.txt').write_text(
                '0 0.5 0.5 0.4 0.8 0.1 0.2 2 0.3 0.4 2\n',
                encoding='utf-8',
            )

    def _request(self, root: Path, target: str = '2026-08-07'):
        return DatasetPreparationRequest(
            dataset_root=root,
            source_names=('Collect_A',),
            target_name=target,
            task_type='pose',
            annotation_dir='annotations',
            label_dir='labels',
            val_ratio=0.2,
            seed=42,
            use_copy=True,
            class_names=('person',),
            keypoints=('left', 'right'),
            left_right_pairs=(('left', 'right'),),
        )

    def test_lists_raw_batches_with_pair_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)

            rows = list_source_batches(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'Collect_A')
        self.assertEqual(rows[0]['image_count'], 4)
        self.assertEqual(rows[0]['annotation_count'], 4)
        self.assertEqual(rows[0]['label_count'], 4)

    def test_scan_excludes_test_images_before_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=5)
            test_images = root / 'test_data' / 'images'
            test_images.mkdir(parents=True)
            (test_images / 'frame_002.png').write_bytes(b'test')

            result = scan_dataset(self._request(root))

        self.assertEqual(len(result.samples), 4)
        self.assertEqual([path.stem for path in result.test_excluded], ['frame_002'])
        self.assertTrue(result.can_prepare)

    def test_missing_txt_is_reported_and_blocks_preparation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            (root / 'labels' / 'Collect_A' / 'frame_001.txt').unlink()

            request = self._request(root)
            result = scan_dataset(request)

            self.assertEqual(len(result.missing_labels), 1)
            self.assertFalse(result.can_prepare)
            self.assertIn('缺少 YOLO TXT 标签', result.blocking_message())
            with self.assertRaises(DatasetPreparationError):
                prepare_dataset(request, result)

    def test_training_mode_skips_images_without_complete_label_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)
            (root / 'annotations' / 'Collect_A' / 'frame_001.json').unlink()
            (root / 'labels' / 'Collect_A' / 'frame_002.txt').unlink()
            request = replace(
                self._request(root),
                skip_incomplete_samples=True,
            )

            result = scan_dataset(request)
            self.assertEqual(len(result.samples), 2)
            self.assertEqual(len(result.missing_annotations), 1)
            self.assertEqual(len(result.missing_labels), 1)
            self.assertTrue(result.can_prepare)

            prepared = prepare_dataset(request, result)
            manifest = json.loads(prepared.manifest_path.read_text())
            self.assertEqual(prepared.total_count, 2)
            self.assertEqual(
                manifest['summary']['skipped_missing_annotations'], 1
            )
            self.assertEqual(manifest['summary']['skipped_missing_labels'], 1)

    def test_training_mode_can_deduplicate_overlapping_source_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3, source='Collect_A')
            self._build_dataset(root, count=3, source='Reviewed_A')
            request = replace(
                self._request(root),
                source_names=('Collect_A', 'Reviewed_A'),
                skip_duplicate_samples=True,
            )

            result = scan_dataset(request)

            self.assertEqual(len(result.samples), 3)
            self.assertEqual(len(result.duplicate_images), 3)
            self.assertTrue(result.can_prepare)
            prepared = prepare_dataset(request, result)
            manifest = json.loads(prepared.manifest_path.read_text())
            self.assertEqual(manifest['summary']['duplicates'], 3)

    def test_pose_label_column_mismatch_is_reported_before_preparation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            invalid = ' '.join(['0', '0.5', '0.5', '0.4', '0.8'] + ['0'] * 75)
            (root / 'labels' / 'Collect_A' / 'frame_001.txt').write_text(
                invalid + '\n', encoding='utf-8'
            )

            result = scan_dataset(self._request(root))

            self.assertEqual(len(result.invalid_labels), 1)
            self.assertEqual(result.observed_label_columns, {11, 80})
            self.assertFalse(result.can_prepare)
            self.assertIn('每行期望 11 列，实际发现 11, 80 列', result.blocking_message())

    def test_pose_label_non_finite_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            label = root / 'labels' / 'Collect_A' / 'frame_001.txt'
            label.write_text(
                '0 0.5 0.5 0.4 0.8 inf 0.2 2 0.3 0.4 2\n',
                encoding='utf-8',
            )

            result = scan_dataset(self._request(root))

            self.assertEqual(result.invalid_labels, [label])
            self.assertFalse(result.can_prepare)

    def test_pose_label_out_of_range_keypoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            label = root / 'labels' / 'Collect_A' / 'frame_001.txt'
            label.write_text(
                '0 0.5 0.5 0.4 0.8 1.2 0.2 2 0.3 0.4 2\n',
                encoding='utf-8',
            )

            result = scan_dataset(self._request(root))

            self.assertEqual(result.invalid_labels, [label])
            self.assertFalse(result.can_prepare)

    def test_pose_yaml_uses_actual_label_columns_instead_of_template_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)
            actual = ' '.join(['0', '0.5', '0.5', '0.4', '0.8'] + ['0'] * 75)
            for path in (root / 'labels' / 'Collect_A').glob('*.txt'):
                path.write_text(actual + '\n', encoding='utf-8')
            for path in (root / 'annotations' / 'Collect_A').glob('*.json'):
                document = json.loads(path.read_text(encoding='utf-8'))
                document['shapes'] = document['shapes'][:1] + [
                    {
                        'label': f'kp_{index:02d}',
                        'shape_type': 'point',
                        'group_id': 0,
                        'points': [[index, index]],
                    }
                    for index in range(25)
                ]
                path.write_text(json.dumps(document), encoding='utf-8')

            prepared = prepare_dataset(self._request(root))
            payload = yaml.safe_load(
                prepared.dataset_yaml.read_text(encoding='utf-8')
            )

            self.assertEqual(payload['kpt_shape'], [25, 3])
            self.assertEqual(len(payload['keypoint_names']), 25)
            self.assertTrue(inspect_training_batch(prepared.batch_root).is_ready)

    def test_dataset_yaml_adds_observed_class_missing_from_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)
            annotation = root / 'annotations' / 'Collect_A' / 'frame_001.json'
            document = json.loads(annotation.read_text(encoding='utf-8'))
            document['shapes'][0]['label'] = 'climbing_tower'
            annotation.write_text(json.dumps(document), encoding='utf-8')
            label = root / 'labels' / 'Collect_A' / 'frame_001.txt'
            label.write_text(
                label.read_text(encoding='utf-8').replace('0 ', '1 ', 1),
                encoding='utf-8',
            )

            prepared = prepare_dataset(self._request(root))
            payload = yaml.safe_load(
                prepared.dataset_yaml.read_text(encoding='utf-8')
            )

            self.assertEqual(
                payload['names'],
                {0: 'person', 1: 'climbing_tower'},
            )

    def test_dataset_yaml_supports_arbitrary_class_counts_from_data(self):
        for class_count in (4, 6, 8, 12):
            with self.subTest(class_count=class_count), \
                    tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                sample_count = class_count * 2
                self._build_dataset(root, count=sample_count)
                for index in range(sample_count):
                    stem = f'frame_{index:03d}'
                    class_id = index % class_count
                    label = root / 'labels' / 'Collect_A' / f'{stem}.txt'
                    label.write_text(
                        label.read_text(encoding='utf-8').replace(
                            '0 ', f'{class_id} ', 1
                        ),
                        encoding='utf-8',
                    )
                    annotation = (
                        root / 'annotations' / 'Collect_A' / f'{stem}.json'
                    )
                    document = json.loads(
                        annotation.read_text(encoding='utf-8')
                    )
                    document['shapes'][0]['label'] = (
                        f'data_class_{class_id}'
                    )
                    annotation.write_text(
                        json.dumps(document), encoding='utf-8'
                    )

                prepared = prepare_dataset(replace(
                    self._request(root),
                    class_names=('stale_0', 'stale_1', 'stale_2'),
                ))
                payload = yaml.safe_load(
                    prepared.dataset_yaml.read_text(encoding='utf-8')
                )

                self.assertEqual(
                    payload['names'],
                    {
                        index: f'data_class_{index}'
                        for index in range(class_count)
                    },
                )

    def test_batch_preflight_rejects_class_missing_from_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=10)
            prepared = prepare_dataset(self._request(root))
            train_label = next(
                (prepared.batch_root / 'train_data/labels/train').glob('*.txt')
            )
            train_label.write_text(
                train_label.read_text(encoding='utf-8').replace('0 ', '3 ', 1),
                encoding='utf-8',
            )
            payload = yaml.safe_load(
                prepared.dataset_yaml.read_text(encoding='utf-8')
            )
            payload['names'][3] = 'data_class_3'
            prepared.dataset_yaml.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding='utf-8',
            )

            summary = inspect_training_batch(prepared.batch_root)

            self.assertFalse(summary.is_ready)
            self.assertEqual(summary.missing_val_class_ids, (3,))
            self.assertIn('3:data_class_3', summary.readiness_message())

    def test_ensure_repairs_stale_class_map_and_clears_label_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)
            prepared = prepare_dataset(self._request(root))
            stem = 'frame_001'
            annotation = prepared.batch_root / 'annotations' / f'{stem}.json'
            document = json.loads(annotation.read_text(encoding='utf-8'))
            document['shapes'][0]['label'] = 'climbing_tower'
            annotation.write_text(json.dumps(document), encoding='utf-8')
            for label in (
                prepared.batch_root / 'labels' / f'{stem}.txt',
                prepared.batch_root / 'train_data' / 'labels' / 'train'
                / f'{stem}.txt',
                prepared.batch_root / 'train_data' / 'labels' / 'val'
                / f'{stem}.txt',
            ):
                if label.is_file():
                    label.write_text(
                        label.read_text(encoding='utf-8').replace('0 ', '1 ', 1),
                        encoding='utf-8',
                    )
            caches = (
                prepared.batch_root / 'train_data' / 'labels' / 'train.cache',
                prepared.batch_root / 'train_data' / 'labels' / 'val.cache',
            )
            for cache in caches:
                cache.write_bytes(b'stale')

            result = ensure_training_dataset_yaml(prepared.batch_root)
            payload = yaml.safe_load(result.read_text(encoding='utf-8'))

            self.assertEqual(
                payload['names'],
                {0: 'person', 1: 'climbing_tower'},
            )
            self.assertTrue(all(not cache.exists() for cache in caches))

    def test_existing_batch_can_generate_missing_split_without_touching_top_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=5)
            batch = root / 'training_data' / 'batch'
            for relative in ('images', 'labels'):
                (batch / relative).mkdir(parents=True)
            for path in (root / 'images' / 'Collect_A').iterdir():
                (batch / 'images' / path.name).write_bytes(path.read_bytes())
            for path in (root / 'labels' / 'Collect_A').iterdir():
                (batch / 'labels' / path.name).write_bytes(path.read_bytes())

            train_count, val_count = prepare_existing_batch_split(batch)
            self.assertEqual((train_count, val_count), (4, 1))
            self.assertEqual(len(list((batch / 'images').glob('*.png'))), 5)
            self.assertEqual(len(list((batch / 'train_data/images/train').glob('*.png'))), 4)
            self.assertEqual(len(list((batch / 'train_data/images/val').glob('*.png'))), 1)

    def test_duplicate_stems_across_sources_block_preparation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=2, source='Batch_A')
            self._build_dataset(root, count=2, source='Batch_B')
            request = replace(
                self._request(root),
                source_names=('Batch_A', 'Batch_B'),
            )

            result = scan_dataset(request)

            self.assertEqual(len(result.duplicate_images), 2)
            self.assertFalse(result.can_prepare)
            self.assertIn(
                '不同来源存在同名图片 2 个', result.blocking_message()
            )
            with self.assertRaises(DatasetPreparationError):
                prepare_dataset(request, result)

    def test_confirmed_empty_annotation_creates_background_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            annotation = root / 'annotations' / 'Collect_A' / 'frame_001.json'
            annotation.write_text('{"shapes": []}', encoding='utf-8')
            (root / 'labels' / 'Collect_A' / 'frame_001.txt').unlink()

            request = replace(
                self._request(root), allow_background_without_label=True
            )
            result = scan_dataset(request)
            prepared = prepare_dataset(request, result)

            generated = prepared.batch_root / 'labels' / 'frame_001.txt'
            self.assertTrue(generated.is_file())
            self.assertEqual(generated.read_text(encoding='utf-8'), '')

    def test_prepare_creates_merged_batch_split_manifest_and_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=10)

            prepared = prepare_dataset(self._request(root))
            summary = inspect_training_batch(prepared.batch_root)
            payload = yaml.safe_load(
                prepared.dataset_yaml.read_text(encoding='utf-8')
            )
            manifest = json.loads(
                prepared.manifest_path.read_text(encoding='utf-8')
            )

            self.assertEqual(prepared.total_count, 10)
            self.assertEqual(prepared.train_count, 8)
            self.assertEqual(prepared.val_count, 2)
            self.assertTrue(summary.is_ready)
            self.assertEqual(summary.image_count, 10)
            self.assertEqual(summary.annotation_count, 10)
            self.assertEqual(summary.label_count, 10)
            self.assertEqual(payload['train'], 'train_data/images/train')
            self.assertEqual(payload['val'], 'train_data/images/val')
            self.assertNotIn('path', payload)
            self.assertEqual(payload['kpt_shape'], [2, 3])
            self.assertEqual(payload['flip_idx'], [1, 0])
            self.assertEqual(manifest['summary']['train'], 8)
            self.assertEqual(manifest['summary']['val'], 2)
            self.assertEqual(len(manifest['records']), 10)

    def test_legacy_staging_yaml_is_repaired_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=4)
            prepared = prepare_dataset(self._request(root))
            payload = yaml.safe_load(
                prepared.dataset_yaml.read_text(encoding='utf-8')
            )
            payload['path'] = str(
                root / 'training_data' / '.2026-08-07.preparing-deadbeef'
                / 'train_data'
            )
            payload['train'] = 'images/train'
            payload['val'] = 'images/val'
            prepared.dataset_yaml.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding='utf-8',
            )

            result = ensure_training_dataset_yaml(prepared.batch_root)
            repaired = yaml.safe_load(result.read_text(encoding='utf-8'))

            self.assertNotIn('path', repaired)
            self.assertEqual(repaired['train'], 'train_data/images/train')
            self.assertEqual(repaired['val'], 'train_data/images/val')

    def test_same_seed_produces_same_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=12)

            first = prepare_dataset(self._request(root, 'batch-a'))
            second = prepare_dataset(self._request(root, 'batch-b'))
            first_manifest = json.loads(first.manifest_path.read_text())
            second_manifest = json.loads(second.manifest_path.read_text())

            first_split = {
                row['stem']: row['split'] for row in first_manifest['records']
            }
            second_split = {
                row['stem']: row['split'] for row in second_manifest['records']
            }
            self.assertEqual(first_split, second_split)

    def test_split_keeps_repeatable_rare_class_in_train_and_val(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=20)
            labels = root / 'labels' / 'Collect_A'
            for stem in ('frame_000', 'frame_001'):
                path = labels / f'{stem}.txt'
                path.write_text(
                    path.read_text(encoding='utf-8').replace('0 ', '3 ', 1),
                    encoding='utf-8',
                )

            prepared = prepare_dataset(self._request(root))
            manifest = json.loads(prepared.manifest_path.read_text())
            split_by_stem = {
                row['stem']: row['split'] for row in manifest['records']
            }

            self.assertEqual(
                {split_by_stem['frame_000'], split_by_stem['frame_001']},
                {'train', 'val'},
            )

    def test_split_applies_validation_ratio_per_source_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=5, source='Collect_A')
            self._build_dataset(
                root, count=10, source='Collect_B', start=100
            )
            request = replace(
                self._request(root),
                source_names=('Collect_A', 'Collect_B'),
            )

            prepared = prepare_dataset(request)
            manifest = json.loads(prepared.manifest_path.read_text())
            val_by_source = {}
            for row in manifest['records']:
                if row['split'] == 'val':
                    source = row['source_name']
                    val_by_source[source] = val_by_source.get(source, 0) + 1

            self.assertEqual(val_by_source, {'Collect_A': 1, 'Collect_B': 2})
            self.assertEqual(prepared.train_count, 12)
            self.assertEqual(prepared.val_count, 3)

    def test_task_annotation_directory_is_preserved_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._build_dataset(root, count=3)
            (root / 'annotations-obb').mkdir()
            (root / 'annotations' / 'Collect_A').rename(
                root / 'annotations-obb' / 'Collect_A'
            )
            request = replace(
                self._request(root),
                target_name='obb-batch',
                task_type='obb',
                annotation_dir='annotations-obb',
                keypoints=(),
                left_right_pairs=(),
            )

            prepared = prepare_dataset(request)
            summary = inspect_training_batch(prepared.batch_root)

            self.assertTrue(
                (prepared.batch_root / 'annotations-obb' / 'frame_000.json').is_file()
            )
            self.assertEqual(summary.task_type, 'obb')
            self.assertEqual(summary.annotation_dir, 'annotations-obb')
            self.assertEqual(summary.annotation_count, 3)


if __name__ == '__main__':
    unittest.main()
