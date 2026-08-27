import json
import tempfile
import unittest
from pathlib import Path

from app.models.annotation_review import (
    KEYPOINTS,
    KPT_CONNECTION_LABELS,
    apply_pose_review_config,
    default_task_review_config,
    load_pose_review_config,
    pose_review_config_from_dict,
    pose_review_config_to_dict,
    reorder_keypoints_data,
    reorder_keypoints_file,
    reset_pose_review_config,
    review_annotation_data,
    summarize_annotation_file,
)


def _point(label, group_id, x, y):
    return {
        'label': label,
        'shape_type': 'point',
        'group_id': group_id,
        'points': [[x, y]],
    }


def _box(group_id, x1=0, y1=0, x2=120, y2=140):
    return {
        'label': 'person_dress_middle',
        'shape_type': 'rectangle',
        'group_id': group_id,
        'points': [[x1, y1], [x2, y2]],
    }


class AnnotationReviewTest(unittest.TestCase):
    def setUp(self):
        apply_pose_review_config(load_pose_review_config(
            Path(__file__).resolve().parents[1]
            / 'resources'
            / 'pose_review_template.json'
        ))

    def tearDown(self):
        reset_pose_review_config()

    def test_keypoint_schema_has_23_points(self):
        self.assertEqual(len(KEYPOINTS), 23)
        self.assertNotIn('left_main_safety_strap', KEYPOINTS)
        self.assertNotIn('right_main_safety_strap', KEYPOINTS)

        flattened_connections = {
            label
            for connection in KPT_CONNECTION_LABELS
            for label in connection
        }
        self.assertNotIn('left_main_safety_strap', flattened_connections)
        self.assertNotIn('right_main_safety_strap', flattened_connections)

    def test_pose_review_template_file_loads(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / 'resources'
            / 'pose_review_template.json'
        )

        config = load_pose_review_config(template_path)

        self.assertEqual(config.name, 'ShengSong Pose 23点审查模板')
        self.assertEqual(config.task_type, 'pose')
        self.assertEqual(config.annotation_dir, 'annotations')
        self.assertEqual(len(config.keypoints), 23)
        self.assertTrue(config.rules['missing_person_box'])
        self.assertEqual(config.kpt_connections[0], (0, 1))
        self.assertEqual(config.custom_rules, ())

    def test_builtin_task_review_template_files_load(self):
        resource_dir = Path(__file__).resolve().parents[1] / 'resources'
        cases = [
            (
                'detection_review_template.json',
                'detection',
                'annotations-det',
                'invalid_rectangle',
            ),
            (
                'segmentation_review_template.json',
                'segmentation',
                'annotations-seg',
                'invalid_polygon',
            ),
            (
                'obb_review_template.json',
                'obb',
                'annotations-obb',
                'invalid_rotation_box',
            ),
        ]

        for file_name, task_type, annotation_dir, rule_name in cases:
            with self.subTest(file_name=file_name):
                config = load_pose_review_config(resource_dir / file_name)

                self.assertEqual(config.task_type, task_type)
                self.assertEqual(config.annotation_dir, annotation_dir)
                self.assertEqual(config.keypoints, ())
                self.assertEqual(config.kpt_connections, ())
                self.assertTrue(config.rules[rule_name])
                self.assertTrue(config.rules['image_size_mismatch'])
                self.assertFalse(config.rules['duplicate_keypoint'])
                self.assertNotIn('Pose', config.name)

    def test_non_pose_template_export_hides_pose_only_fields(self):
        config = load_pose_review_config(
            Path(__file__).resolve().parents[1]
            / 'resources'
            / 'obb_review_template.json'
        )

        exported = pose_review_config_to_dict(config)

        self.assertEqual(exported['task_type'], 'obb')
        self.assertIn('invalid_rotation_box', exported['rules'])
        self.assertNotIn('duplicate_keypoint', exported['rules'])
        self.assertNotIn('keypoints', exported)
        self.assertNotIn('kpt_connections', exported)
        self.assertNotIn('left_right_pairs', exported)
        self.assertIn('thresholds', exported)
        self.assertIn('obb_min_area', exported['thresholds'])
        self.assertNotIn('box_margin_min', exported['thresholds'])
        self.assertNotIn('重排序', exported['description'])

    def test_detection_template_export_hides_other_task_thresholds(self):
        config = load_pose_review_config(
            Path(__file__).resolve().parents[1]
            / 'resources'
            / 'detection_review_template.json'
        )

        exported = pose_review_config_to_dict(config)

        self.assertEqual(exported['task_type'], 'detection')
        self.assertIn('bbox_small_area', exported['rules'])
        self.assertIn('bbox_duplicate', exported['rules'])
        self.assertIn('thresholds', exported)
        self.assertIn('bbox_duplicate_iou', exported['thresholds'])
        self.assertNotIn('obb_min_area', exported['thresholds'])
        self.assertNotIn('keypoints', exported)

    def test_segmentation_template_export_hides_other_task_thresholds(self):
        config = load_pose_review_config(
            Path(__file__).resolve().parents[1]
            / 'resources'
            / 'segmentation_review_template.json'
        )

        exported = pose_review_config_to_dict(config)

        self.assertEqual(exported['task_type'], 'segmentation')
        self.assertIn('polygon_self_intersection', exported['rules'])
        self.assertIn('thresholds', exported)
        self.assertIn('polygon_min_area', exported['thresholds'])
        self.assertNotIn('bbox_min_area', exported['thresholds'])
        self.assertNotIn('keypoints', exported)

    def test_pose_review_config_can_roundtrip_to_editable_json(self):
        config = pose_review_config_from_dict({
            'name': 'editable',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [['head', 'tail']],
            'left_right_pairs': [],
            'custom_rules': [
                {
                    'id': 'must_have_tail',
                    'name': '必须有 tail',
                    'type': 'required_keypoints',
                    'labels': ['tail'],
                }
            ],
        })

        exported = pose_review_config_to_dict(config)
        loaded = pose_review_config_from_dict(exported)

        self.assertEqual(loaded.name, 'editable')
        self.assertEqual(loaded.task_type, 'pose')
        self.assertEqual(loaded.annotation_dir, 'annotations')
        self.assertEqual(loaded.custom_rules[0]['id'], 'must_have_tail')

    def test_pose_review_config_accepts_task_type_and_annotation_dir(self):
        config = pose_review_config_from_dict({
            'name': 'obb task',
            'task_type': 'obb',
            'annotation_dir': 'annotations-obb',
            'classes': ['Person'],
        })

        self.assertEqual(config.task_type, 'obb')
        self.assertEqual(config.annotation_dir, 'annotations-obb')
        self.assertEqual(config.target_classes, ('Person',))
        self.assertEqual(config.keypoints, ())

    def test_default_non_pose_task_maps_annotation_dir(self):
        config = default_task_review_config('obb')

        self.assertEqual(config.task_type, 'obb')
        self.assertEqual(config.annotation_dir, 'annotations-obb')
        self.assertEqual(config.keypoints, ())
        self.assertTrue(config.rules['image_size_mismatch'])
        self.assertFalse(config.rules['duplicate_keypoint'])

    def test_default_detection_task_enables_detection_rules(self):
        config = default_task_review_config('detection')

        self.assertEqual(config.task_type, 'detection')
        self.assertEqual(config.annotation_dir, 'annotations-det')
        self.assertTrue(config.rules['empty_annotation'])
        self.assertTrue(config.rules['invalid_rectangle'])
        self.assertTrue(config.rules['bbox_outside_image'])
        self.assertTrue(config.rules['bbox_small_area'])
        self.assertTrue(config.rules['bbox_bad_aspect_ratio'])
        self.assertTrue(config.rules['bbox_duplicate'])
        self.assertTrue(config.rules['unknown_class'])
        self.assertTrue(config.rules['unexpected_shape_type'])
        self.assertFalse(config.rules['duplicate_keypoint'])

    def test_default_obb_task_enables_obb_rules(self):
        config = default_task_review_config('obb')

        self.assertEqual(config.task_type, 'obb')
        self.assertEqual(config.annotation_dir, 'annotations-obb')
        self.assertTrue(config.rules['empty_annotation'])
        self.assertTrue(config.rules['invalid_rotation_box'])
        self.assertTrue(config.rules['obb_outside_image'])
        self.assertTrue(config.rules['obb_duplicate_points'])
        self.assertTrue(config.rules['obb_corner_order'])
        self.assertTrue(config.rules['obb_small_area'])
        self.assertTrue(config.rules['obb_bad_aspect_ratio'])
        self.assertTrue(config.rules['unknown_class'])
        self.assertTrue(config.rules['unexpected_shape_type'])
        self.assertFalse(config.rules['duplicate_keypoint'])

    def test_default_segmentation_task_enables_polygon_rules(self):
        config = default_task_review_config('segmentation')

        self.assertEqual(config.task_type, 'segmentation')
        self.assertEqual(config.annotation_dir, 'annotations-seg')
        self.assertTrue(config.rules['empty_annotation'])
        self.assertTrue(config.rules['invalid_polygon'])
        self.assertTrue(config.rules['polygon_outside_image'])
        self.assertTrue(config.rules['polygon_duplicate_points'])
        self.assertTrue(config.rules['polygon_self_intersection'])
        self.assertTrue(config.rules['polygon_small_area'])
        self.assertTrue(config.rules['unknown_class'])
        self.assertTrue(config.rules['unexpected_shape_type'])
        self.assertFalse(config.rules['duplicate_keypoint'])

    def test_summarize_annotation_file_counts_template_labels(self):
        data = {
            'checked': True,
            'shapes': [
                _box(1),
                _point('nose', 1, 10, 20),
                {'label': 'other', 'shape_type': 'polygon', 'points': [[0, 0]]},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            summary = summarize_annotation_file(path)

        self.assertTrue(summary.valid)
        self.assertIs(summary.checked, True)
        self.assertEqual(summary.shapes, 3)
        self.assertEqual(summary.person_boxes, 1)
        self.assertEqual(summary.keypoints, 1)
        self.assertEqual(summary.other_shapes, 1)
        self.assertEqual(summary.target_class_counts['person_dress_middle'], 1)
        self.assertEqual(summary.keypoint_counts['nose'], 1)
        self.assertEqual(summary.shape_type_counts['rectangle'], 1)
        self.assertEqual(summary.shape_type_counts['point'], 1)
        self.assertEqual(summary.shape_type_counts['polygon'], 1)

    def test_detection_summary_counts_rectangle_classes_and_shape_types(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'checked': False,
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[0, 0], [10, 10]],
                },
                {
                    'label': 'glove',
                    'shape_type': 'rectangle',
                    'points': [[20, 20], [30, 30]],
                },
                {
                    'label': 'mask',
                    'shape_type': 'polygon',
                    'points': [[0, 0], [1, 1], [2, 0]],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            summary = summarize_annotation_file(path)

        self.assertTrue(summary.valid)
        self.assertEqual(summary.person_boxes, 2)
        self.assertEqual(summary.keypoints, 0)
        self.assertEqual(summary.other_shapes, 1)
        self.assertEqual(summary.target_class_counts['helmet'], 1)
        self.assertEqual(summary.target_class_counts['glove'], 1)
        self.assertEqual(summary.shape_type_counts['rectangle'], 2)
        self.assertEqual(summary.shape_type_counts['polygon'], 1)

    def test_obb_summary_counts_rotation_classes_and_shape_types(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[0, 0], [10, 0], [10, 20], [0, 20]],
                },
                {
                    'label': 'Car',
                    'shape_type': 'rotation',
                    'points': [[20, 20], [30, 20], [30, 35], [20, 35]],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            summary = summarize_annotation_file(path)

        self.assertTrue(summary.valid)
        self.assertEqual(summary.person_boxes, 2)
        self.assertEqual(summary.target_class_counts['Person'], 1)
        self.assertEqual(summary.target_class_counts['Car'], 1)
        self.assertEqual(summary.shape_type_counts['rotation'], 2)

    def test_segmentation_summary_counts_polygon_classes_and_shape_types(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[0, 0], [10, 0], [10, 20]],
                },
                {
                    'label': 'helmet',
                    'shape_type': 'polygon',
                    'points': [[20, 20], [30, 20], [30, 35]],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            summary = summarize_annotation_file(path)

        self.assertTrue(summary.valid)
        self.assertEqual(summary.person_boxes, 2)
        self.assertEqual(summary.target_class_counts['person'], 1)
        self.assertEqual(summary.target_class_counts['helmet'], 1)
        self.assertEqual(summary.shape_type_counts['polygon'], 2)

    def test_imported_pose_config_controls_review_rules_and_labels(self):
        config = pose_review_config_from_dict({
            'name': 'custom two point pose',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [['head', 'tail']],
            'left_right_pairs': [],
            'rules': {
                'missing_person_box': False,
            },
        })
        apply_pose_review_config(config)

        no_box_data = {
            'shapes': [
                _point('head', 1, 10, 20),
            ],
        }
        outside_data = {
            'shapes': [
                _point('head', 1, 200, 200),
                {
                    'label': 'human',
                    'shape_type': 'rectangle',
                    'group_id': 1,
                    'points': [[0, 0], [100, 100]],
                },
            ],
        }

        self.assertNotIn(
            'missing_person_box',
            {issue.rule for issue in review_annotation_data(no_box_data)},
        )
        self.assertIn(
            'keypoint_outside_box',
            {issue.rule for issue in review_annotation_data(outside_data)},
        )

    def test_unknown_builtin_rule_reports_unavailable_instead_of_import_error(self):
        config = pose_review_config_from_dict({
            'name': 'unknown rule',
            'target_classes': ['human'],
            'keypoints': ['head'],
            'kpt_connections': [],
            'left_right_pairs': [],
            'rules': {
                'rule_that_has_no_executor': True,
            },
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                {
                    'label': 'human',
                    'shape_type': 'rectangle',
                    'group_id': 1,
                    'points': [[0, 0], [100, 100]],
                },
            ],
        })

        self.assertIn('unavailable_rule', {issue.rule for issue in issues})

    def test_custom_required_keypoints_rule_reports_missing_labels(self):
        config = pose_review_config_from_dict({
            'name': 'required custom rule',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [],
            'left_right_pairs': [],
            'custom_rules': [
                {
                    'id': 'human_must_have_tail',
                    'name': 'human 必须有 tail',
                    'type': 'required_keypoints',
                    'target_classes': ['human'],
                    'labels': ['tail'],
                }
            ],
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                {
                    'label': 'human',
                    'shape_type': 'rectangle',
                    'group_id': 1,
                    'points': [[0, 0], [100, 100]],
                },
                _point('head', 1, 10, 20),
            ],
        })

        self.assertIn('human_must_have_tail', {issue.rule for issue in issues})

    def test_custom_relative_position_rule_reports_violation(self):
        config = pose_review_config_from_dict({
            'name': 'relative custom rule',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [],
            'left_right_pairs': [],
            'custom_rules': [
                {
                    'id': 'head_above_tail',
                    'name': 'head 必须在 tail 上方',
                    'type': 'relative_position',
                    'point_a': 'head',
                    'point_b': 'tail',
                    'relation': 'above',
                    'margin': 5,
                }
            ],
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                {
                    'label': 'human',
                    'shape_type': 'rectangle',
                    'group_id': 1,
                    'points': [[0, 0], [100, 100]],
                },
                _point('head', 1, 50, 90),
                _point('tail', 1, 50, 20),
            ],
        })

        self.assertIn('head_above_tail', {issue.rule for issue in issues})

    def test_custom_paired_keypoints_rule_reports_single_sided_pair(self):
        config = pose_review_config_from_dict({
            'name': 'paired custom rule',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [],
            'left_right_pairs': [],
            'custom_rules': [
                {
                    'id': 'head_tail_pair',
                    'type': 'paired_keypoints',
                    'pairs': [['head', 'tail']],
                }
            ],
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                _point('head', 1, 10, 20),
            ],
        })

        self.assertIn('head_tail_pair', {issue.rule for issue in issues})

    def test_custom_distance_range_rule_reports_far_points(self):
        config = pose_review_config_from_dict({
            'name': 'distance custom rule',
            'target_classes': ['human'],
            'keypoints': ['head', 'tail'],
            'kpt_connections': [],
            'left_right_pairs': [],
            'custom_rules': [
                {
                    'id': 'head_tail_distance',
                    'type': 'distance_range',
                    'point_a': 'head',
                    'point_b': 'tail',
                    'max_distance': 20,
                }
            ],
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                _point('head', 1, 0, 0),
                _point('tail', 1, 100, 0),
            ],
        })

        self.assertIn('head_tail_distance', {issue.rule for issue in issues})

    def test_python_custom_rule_plugin_can_return_issue_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = Path(tmp) / 'custom_rule.py'
            plugin.write_text(
                'def check(context, rule):\n'
                '    point = context.point(1, "head")\n'
                '    if point is None:\n'
                '        return []\n'
                '    return [context.issue(\n'
                '        rule_id=rule["id"],\n'
                '        severity="warning",\n'
                '        message="plugin issue",\n'
                '        group_id=1,\n'
                '        label="head",\n'
                '        shape_indices=[point.shape_idx],\n'
                '        point_indices=[(point.shape_idx, 0)],\n'
                '    )]\n',
                encoding='utf-8',
            )
            config_path = Path(tmp) / 'template.json'
            config = pose_review_config_from_dict({
                'name': 'python plugin custom rule',
                'target_classes': ['human'],
                'keypoints': ['head'],
                'kpt_connections': [],
                'left_right_pairs': [],
                'custom_rules': [
                    {
                        'id': 'plugin_head_rule',
                        'type': 'python',
                        'path': 'custom_rule.py',
                        'function': 'check',
                    }
                ],
            }, config_path)
            apply_pose_review_config(config)

            issues = review_annotation_data({
                'shapes': [
                    _point('head', 1, 10, 20),
                ],
            })

        self.assertIn('plugin_head_rule', {issue.rule for issue in issues})

    def test_imported_pose_config_controls_reorder_order(self):
        config = pose_review_config_from_dict({
            'name': 'reverse two point pose',
            'target_classes': ['human'],
            'keypoints': ['tail', 'head'],
            'kpt_connections': [['tail', 'head']],
            'left_right_pairs': [],
        })
        apply_pose_review_config(config)
        data = {
            'shapes': [
                {
                    'label': 'human',
                    'shape_type': 'rectangle',
                    'group_id': 1,
                    'points': [[0, 0], [100, 100]],
                },
                _point('head', 1, 10, 20),
                _point('tail', 1, 30, 40),
            ],
        }

        result = reorder_keypoints_data(data)

        self.assertTrue(result.changed)
        self.assertEqual(
            [shape['label'] for shape in data['shapes']],
            ['human', 'tail', 'head'],
        )

    def test_reorder_keypoints_groups_points_after_person_box(self):
        data = {
            'shapes': [
                _point('right_knee', 1, 20, 20),
                _box(1),
                _point('nose', 1, 10, 10),
                _point('left_knee', 1, 15, 15),
                _box(2),
                _point('right_wrist', 2, 40, 40),
                _point('left_shoulder', 2, 30, 30),
            ]
        }

        result = reorder_keypoints_data(data)

        self.assertTrue(result.changed)
        self.assertEqual(result.groups, 2)
        self.assertEqual(result.keypoints, 5)
        self.assertEqual(
            [shape['label'] for shape in data['shapes']],
            [
                'person_dress_middle',
                'nose',
                'left_knee',
                'right_knee',
                'person_dress_middle',
                'left_shoulder',
                'right_wrist',
            ],
        )

    def test_reorder_keypoints_without_box_uses_first_point_position(self):
        data = {
            'shapes': [
                {'label': 'other', 'shape_type': 'polygon', 'points': []},
                _point('right_ankle', 'ghost', 20, 20),
                _point('nose', 'ghost', 10, 10),
                {'label': 'tail', 'shape_type': 'polygon', 'points': []},
            ]
        }

        result = reorder_keypoints_data(data)

        self.assertTrue(result.changed)
        self.assertEqual(
            [shape['label'] for shape in data['shapes']],
            ['other', 'nose', 'right_ankle', 'tail'],
        )

    def test_reorder_keypoints_file_writes_json(self):
        data = {
            'version': 'test',
            'shapes': [
                _box(1),
                _point('right_knee', 1, 20, 20),
                _point('nose', 1, 10, 10),
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(data), encoding='utf-8')

            result = reorder_keypoints_file(path)
            saved = json.loads(path.read_text(encoding='utf-8'))

        self.assertTrue(result.changed)
        self.assertEqual(
            [shape['label'] for shape in saved['shapes']],
            ['person_dress_middle', 'nose', 'right_knee'],
        )

    def test_duplicate_keypoints_in_same_group(self):
        data = {
            'shapes': [
                {
                    'label': 'nose',
                    'shape_type': 'point',
                    'group_id': 1,
                    'points': [[10, 20]],
                },
                {
                    'label': 'nose',
                    'shape_type': 'point',
                    'group_id': 1,
                    'points': [[12, 22]],
                },
                _box(1),
            ]
        }

        issues = review_annotation_data(data)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule, 'duplicate_keypoint')
        self.assertEqual(issues[0].label, 'nose')
        self.assertEqual(issues[0].shape_indices, [0, 1])
        self.assertEqual(issues[0].point_indices, [(0, 0), (1, 0)])

    def test_same_keypoint_in_different_groups_is_allowed(self):
        data = {
            'shapes': [
                {
                    'label': 'nose',
                    'shape_type': 'point',
                    'group_id': 1,
                    'points': [[10, 20]],
                },
                {
                    'label': 'nose',
                    'shape_type': 'point',
                    'group_id': 2,
                    'points': [[12, 22]],
                },
                _box(1),
                _box(2),
            ]
        }

        self.assertEqual(review_annotation_data(data), [])

    def test_shape_type_with_trailing_space_key_is_supported(self):
        data = {
            'shapes ': [
                {
                    'label': 'left_wrist',
                    'shape_type ': 'point',
                    'group_id': 'person-1',
                    'points': [[1, 2]],
                },
                {
                    'label': 'left_wrist',
                    'shape_type ': 'point',
                    'group_id': 'person-1',
                    'points': [[3, 4]],
                },
                _box('person-1'),
            ]
        }

        issues = review_annotation_data(data)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].shape_indices, [0, 1])

    def test_suspected_left_right_swap_is_reported(self):
        data = {
            'shapes': [
                _point('left_hip', 1, 0, 0),
                _point('right_hip', 1, 100, 0),
                _point('left_thigh', 1, 0, 30),
                _point('right_thigh', 1, 100, 30),
                _point('left_knee', 1, 100, 60),
                _point('right_knee', 1, 0, 60),
                _point('left_ankle', 1, 0, 100),
                _point('right_ankle', 1, 100, 100),
                _box(1, -10, -10, 110, 110),
            ]
        }

        issues = review_annotation_data(data)

        swap_issues = [
            issue for issue in issues
            if issue.rule == 'suspected_left_right_swap'
        ]
        self.assertEqual(len(swap_issues), 1)
        self.assertEqual(swap_issues[0].label, 'left_knee/right_knee')

    def test_normal_left_right_chain_is_allowed(self):
        data = {
            'shapes': [
                _point('left_hip', 1, 0, 0),
                _point('right_hip', 1, 100, 0),
                _point('left_thigh', 1, 0, 30),
                _point('right_thigh', 1, 100, 30),
                _point('left_knee', 1, 0, 60),
                _point('right_knee', 1, 100, 60),
                _point('left_ankle', 1, 0, 100),
                _point('right_ankle', 1, 100, 100),
                _box(1, -10, -10, 110, 110),
            ]
        }

        self.assertEqual(review_annotation_data(data), [])

    def test_template_skeleton_detects_swapped_lower_body_pair(self):
        data = {
            'shapes': [
                _point('left_hip', 1, 20, 0),
                _point('right_hip', 1, 80, 0),
                _point('left_thigh', 1, 25, 30),
                _point('right_thigh', 1, 75, 30),
                _point('left_knee', 1, 75, 60),
                _point('right_knee', 1, 25, 60),
                _point('left_ankle', 1, 20, 100),
                _point('right_ankle', 1, 80, 100),
                _box(1, 0, -10, 100, 110),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn(
            'suspected_left_right_swap', {issue.rule for issue in issues}
        )

    def test_enabled_swap_rule_reports_missing_template_skeleton(self):
        config = pose_review_config_from_dict({
            'name': 'incomplete pose policy',
            'keypoints': ['left_knee', 'right_knee'],
            'target_classes': ['person_dress_middle'],
            'kpt_connections': [],
            'left_right_pairs': [
                ['left_knee', 'right_knee'],
            ],
            'rules': {
                'suspected_left_right_swap': True,
            },
        })
        apply_pose_review_config(config)

        issues = review_annotation_data({
            'shapes': [
                _point('left_knee', 1, 10, 10),
                _point('right_knee', 1, 20, 10),
            ]
        })

        self.assertIn('unavailable_rule', {issue.rule for issue in issues})

    def test_missing_person_box_is_reported(self):
        data = {
            'shapes': [
                _point('nose', 1, 10, 20),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn('missing_person_box', {issue.rule for issue in issues})

    def test_keypoint_outside_box_is_reported(self):
        data = {
            'shapes': [
                _point('nose', 1, 200, 200),
                _box(1, 0, 0, 100, 100),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn('keypoint_outside_box', {issue.rule for issue in issues})

    def test_keypoint_wrong_person_is_reported(self):
        data = {
            'shapes': [
                _point('nose', 1, 225, 50),
                _box(1, 0, 0, 100, 100),
                _box(2, 200, 0, 300, 100),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn('keypoint_wrong_person', {issue.rule for issue in issues})

    def test_group_id_missing_is_reported(self):
        data = {
            'shapes': [
                _point('nose', None, 10, 20),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn('group_id_missing', {issue.rule for issue in issues})

    def test_group_id_conflict_is_reported(self):
        data = {
            'shapes': [
                _box(1, 0, 0, 100, 100),
                _box(1, 200, 0, 300, 100),
            ]
        }

        issues = review_annotation_data(data)

        self.assertIn('group_id_conflict', {issue.rule for issue in issues})

    def test_image_size_mismatch_is_reported(self):
        data = {
            'imageWidth': 100,
            'imageHeight': 80,
            'shapes': [],
        }

        issues = review_annotation_data(data, image_size=(120, 80))

        self.assertEqual([issue.rule for issue in issues], ['image_size_mismatch'])

    def test_detection_empty_annotation_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))

        issues = review_annotation_data({'shapes': []}, image_size=(100, 100))

        self.assertIn('empty_annotation', {issue.rule for issue in issues})

    def test_detection_invalid_rectangle_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [10, 30]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        invalid = [issue for issue in issues if issue.rule == 'invalid_rectangle']
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].shape_indices, [0])

    def test_detection_bbox_outside_image_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[-1, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        outside = [issue for issue in issues if issue.rule == 'bbox_outside_image']
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0].shape_indices, [0])

    def test_detection_bbox_small_area_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [11, 11]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('bbox_small_area', {issue.rule for issue in issues})

    def test_detection_bbox_bad_aspect_ratio_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [110, 12]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(120, 30))

        self.assertIn('bbox_bad_aspect_ratio', {issue.rule for issue in issues})

    def test_detection_duplicate_bbox_is_reported_for_same_label(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [50, 50]],
                },
                {
                    'label': 'helmet',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [50, 50]],
                },
                {
                    'label': 'glove',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [50, 50]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        duplicates = [issue for issue in issues if issue.rule == 'bbox_duplicate']
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0].shape_indices, [0, 1])

    def test_detection_unknown_class_uses_template_classes(self):
        config = pose_review_config_from_dict({
            'name': 'detection whitelist',
            'task_type': 'detection',
            'annotation_dir': 'annotations-det',
            'classes': ['helmet'],
        })
        apply_pose_review_config(config)
        data = {
            'shapes': [
                {
                    'label': 'glove',
                    'shape_type': 'rectangle',
                    'points': [[10, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('unknown_class', {issue.rule for issue in issues})

    def test_detection_unexpected_shape_type_is_reported(self):
        apply_pose_review_config(default_task_review_config('detection'))
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'polygon',
                    'points': [[10, 10], [50, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('unexpected_shape_type', {issue.rule for issue in issues})

    def test_obb_valid_rotation_shape_passes_basic_review(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'imageWidth': 100,
            'imageHeight': 100,
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[10, 10], [50, 10], [50, 60], [10, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertEqual(issues, [])

    def test_obb_invalid_rotation_box_is_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[10, 10], [50, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        invalid = [issue for issue in issues if issue.rule == 'invalid_rotation_box']
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].shape_indices, [0])

    def test_obb_outside_image_is_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[-1, 10], [50, 10], [50, 60], [10, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        outside = [issue for issue in issues if issue.rule == 'obb_outside_image']
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0].point_indices, [(0, 0)])

    def test_obb_duplicate_points_are_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[0, 0], [10, 0], [10, 10], [10, 10]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        duplicate = [issue for issue in issues if issue.rule == 'obb_duplicate_points']
        self.assertEqual(len(duplicate), 1)
        self.assertIn((0, 2), duplicate[0].point_indices)
        self.assertIn((0, 3), duplicate[0].point_indices)

    def test_obb_corner_order_issue_is_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[0, 0], [10, 10], [10, 0], [0, 20]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('obb_corner_order', {issue.rule for issue in issues})

    def test_obb_small_area_is_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[0, 0], [3, 0], [3, 1], [0, 1]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('obb_small_area', {issue.rule for issue in issues})

    def test_obb_bad_aspect_ratio_is_reported(self):
        apply_pose_review_config(default_task_review_config('obb'))
        data = {
            'shapes': [
                {
                    'label': 'Person',
                    'shape_type': 'rotation',
                    'points': [[0, 0], [100, 0], [100, 2], [0, 2]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(120, 20))

        self.assertIn('obb_bad_aspect_ratio', {issue.rule for issue in issues})

    def test_obb_unknown_class_uses_template_classes(self):
        config = pose_review_config_from_dict({
            'name': 'obb whitelist',
            'task_type': 'obb',
            'annotation_dir': 'annotations-obb',
            'classes': ['Person'],
        })
        apply_pose_review_config(config)
        data = {
            'shapes': [
                {
                    'label': 'Car',
                    'shape_type': 'rotation',
                    'points': [[10, 10], [50, 10], [50, 60], [10, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('unknown_class', {issue.rule for issue in issues})

    def test_segmentation_valid_polygon_passes_basic_review(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'imageWidth': 100,
            'imageHeight': 100,
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[10, 10], [50, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertEqual(issues, [])

    def test_segmentation_invalid_polygon_is_reported(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[10, 10], [50, 10]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        invalid = [issue for issue in issues if issue.rule == 'invalid_polygon']
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].shape_indices, [0])

    def test_segmentation_polygon_outside_image_is_reported(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[-1, 10], [50, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        outside = [
            issue for issue in issues if issue.rule == 'polygon_outside_image'
        ]
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0].point_indices, [(0, 0)])

    def test_segmentation_duplicate_polygon_points_are_reported(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[0, 0], [10, 0], [10, 10], [10, 10], [0, 10]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        duplicate = [
            issue for issue in issues if issue.rule == 'polygon_duplicate_points'
        ]
        self.assertEqual(len(duplicate), 1)
        self.assertIn((0, 2), duplicate[0].point_indices)
        self.assertIn((0, 3), duplicate[0].point_indices)

    def test_segmentation_self_intersection_is_reported(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[0, 0], [10, 10], [0, 10], [10, 0]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('polygon_self_intersection', {issue.rule for issue in issues})

    def test_segmentation_small_area_is_reported(self):
        apply_pose_review_config(default_task_review_config('segmentation'))
        data = {
            'shapes': [
                {
                    'label': 'person',
                    'shape_type': 'polygon',
                    'points': [[0, 0], [3, 0], [1.5, 2.5]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('polygon_small_area', {issue.rule for issue in issues})

    def test_segmentation_unknown_class_uses_template_classes(self):
        config = pose_review_config_from_dict({
            'name': 'segmentation whitelist',
            'task_type': 'segmentation',
            'annotation_dir': 'annotations-seg',
            'classes': ['person'],
        })
        apply_pose_review_config(config)
        data = {
            'shapes': [
                {
                    'label': 'helmet',
                    'shape_type': 'polygon',
                    'points': [[10, 10], [50, 10], [50, 60]],
                },
            ],
        }

        issues = review_annotation_data(data, image_size=(100, 100))

        self.assertIn('unknown_class', {issue.rule for issue in issues})


if __name__ == '__main__':
    unittest.main()
