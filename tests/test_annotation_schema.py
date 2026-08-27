import json
import tempfile
import unittest
from pathlib import Path

import yaml

from app.models.annotation_schema import infer_annotation_schema
from app.models.annotation_review import (
    PoseReviewConfig,
    review_config_from_data,
)


class AnnotationSchemaTest(unittest.TestCase):
    def _write_annotation(self, root: Path, index: int, class_id: int):
        path = root / f'frame_{index:03d}.json'
        points = ['nose', 'left_shoulder', 'right_shoulder']
        if index % 2:
            points.insert(0, 'top_helmet')
        path.write_text(json.dumps({
            'shapes': [
                {
                    'label': f'data_class_{class_id}',
                    'shape_type': 'rectangle',
                    'group_id': 0,
                    'points': [[0, 0], [10, 10]],
                },
                *[
                    {
                        'label': label,
                        'shape_type': 'point',
                        'group_id': 0,
                        'points': [[1, 1]],
                    }
                    for label in points
                ],
            ],
        }), encoding='utf-8')
        return path

    def test_schema_discovers_arbitrary_class_count_from_json(self):
        for class_count in (4, 6, 8, 12):
            with self.subTest(class_count=class_count), \
                    tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                paths = [
                    self._write_annotation(root, index, index)
                    for index in range(class_count)
                ]

                schema = infer_annotation_schema(paths, task_type='pose')

                self.assertEqual(
                    schema.target_classes,
                    tuple(
                        f'data_class_{index}' for index in range(class_count)
                    ),
                )

    def test_schema_uses_partial_json_sequences_for_keypoint_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                self._write_annotation(root, 0, 0),
                self._write_annotation(root, 1, 0),
            ]

            schema = infer_annotation_schema(paths, task_type='pose')

            self.assertEqual(
                schema.keypoints,
                ('top_helmet', 'nose', 'left_shoulder', 'right_shoulder'),
            )
            self.assertEqual(
                schema.left_right_pairs,
                (('left_shoulder', 'right_shoulder'),),
            )

    def test_dataset_yaml_is_treated_as_dataset_owned_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            yaml_path = root / 'dataset.yaml'
            yaml_path.write_text(yaml.safe_dump({
                'names': {index: f'class_{index}' for index in range(8)},
                'keypoint_names': ['left_eye', 'right_eye'],
                'flip_idx': [1, 0],
                'kpt_connections': [[0, 1]],
            }), encoding='utf-8')

            schema = infer_annotation_schema(
                [], task_type='pose', dataset_yaml=yaml_path
            )

            self.assertEqual(len(schema.target_classes), 8)
            self.assertEqual(schema.keypoints, ('left_eye', 'right_eye'))
            self.assertEqual(
                schema.left_right_pairs, (('left_eye', 'right_eye'),)
            )
            self.assertEqual(schema.kpt_connections, ((0, 1),))

    def test_review_policy_receives_data_schema_without_class_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                self._write_annotation(root, index, index)
                for index in range(8)
            ]
            policy = PoseReviewConfig(
                name='policy',
                task_type='pose',
                annotation_dir='annotations',
                keypoints=('stale_point',),
                target_classes=('stale_class',),
                kpt_connections=((0, 0),),
                left_right_pairs=(),
                rules={},
                thresholds={},
            )

            config = review_config_from_data(policy, paths)

            self.assertEqual(len(config.target_classes), 8)
            self.assertNotIn('stale_class', config.target_classes)
            self.assertNotIn('stale_point', config.keypoints)
            self.assertEqual(config.kpt_connections, ())

    def test_review_policy_keeps_template_skeleton_and_remaps_data_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / 'frame.json'
            path.write_text(json.dumps({
                'shapes': [
                    {
                        'label': 'class_a',
                        'shape_type': 'rectangle',
                        'group_id': 0,
                        'points': [[0, 0], [10, 10]],
                    },
                    {
                        'label': 'right',
                        'shape_type': 'point',
                        'group_id': 0,
                        'points': [[1, 1]],
                    },
                    {
                        'label': 'left',
                        'shape_type': 'point',
                        'group_id': 0,
                        'points': [[2, 2]],
                    },
                ],
            }), encoding='utf-8')
            policy = PoseReviewConfig(
                name='policy', task_type='pose', annotation_dir='annotations',
                keypoints=('left', 'right'), target_classes=('old',),
                kpt_connections=((0, 1),), left_right_pairs=(), rules={},
                thresholds={},
            )

            config = review_config_from_data(policy, [path])

            self.assertEqual(config.keypoints, ('right', 'left'))
            self.assertEqual(config.kpt_connections, ((1, 0),))

    def test_review_policy_keeps_template_left_right_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = self._write_annotation(root, 0, 0)
            policy = PoseReviewConfig(
                name='policy', task_type='pose', annotation_dir='annotations',
                keypoints=('nose', 'left_shoulder', 'right_shoulder'),
                target_classes=('old',), kpt_connections=(),
                left_right_pairs=(('left_shoulder', 'right_shoulder'),),
                rules={}, thresholds={},
            )

            config = review_config_from_data(policy, [path])

            self.assertEqual(
                config.left_right_pairs,
                (('left_shoulder', 'right_shoulder'),),
            )


if __name__ == '__main__':
    unittest.main()
