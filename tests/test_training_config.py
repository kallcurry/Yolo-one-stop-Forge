import tempfile
import unittest
from pathlib import Path

from app.models.training_config import (
    TRAINING_TASK_TYPES,
    default_training_config,
    list_training_template_paths,
    training_config_from_dict,
    training_config_to_dict,
)


class TrainingConfigTest(unittest.TestCase):
    def test_each_task_has_an_independent_builtin_template(self):
        models = set()
        for task_type in TRAINING_TASK_TYPES:
            config = default_training_config(task_type)
            self.assertEqual(config.task_type, task_type)
            self.assertGreaterEqual(len(config.parameters), 40)
            self.assertTrue(config.path.is_file())
            models.add(config.model)
        self.assertEqual(len(models), len(TRAINING_TASK_TYPES))

    def test_unknown_ultralytics_parameter_survives_round_trip(self):
        data = training_config_to_dict(default_training_config('pose'))
        data['parameters']['future_parameter'] = {'mode': 'experimental'}

        config = training_config_from_dict(data)
        exported = training_config_to_dict(config)

        self.assertEqual(
            exported['parameters']['future_parameter'],
            {'mode': 'experimental'},
        )

    def test_legacy_resume_parameter_is_removed_from_user_template(self):
        data = training_config_to_dict(default_training_config('pose'))
        data['parameters']['resume'] = True

        config = training_config_from_dict(data)

        self.assertNotIn('resume', config.parameters)

    def test_platform_managed_parameters_cannot_be_overridden(self):
        for name, value in (
            ('data', '/tmp/other.yaml'),
            ('classes', [0, 1, 2]),
            ('single_cls', True),
        ):
            with self.subTest(name=name):
                data = training_config_to_dict(default_training_config('pose'))
                data['parameters'][name] = value
                with self.assertRaisesRegex(ValueError, '由平台管理'):
                    training_config_from_dict(data)

    def test_invalid_probability_is_rejected(self):
        data = training_config_to_dict(default_training_config('detection'))
        data['parameters']['mosaic'] = 1.5
        with self.assertRaisesRegex(ValueError, '0 到 1'):
            training_config_from_dict(data)

    def test_custom_task_template_is_discovered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'custom.json'
            path.write_text(
                __import__('json').dumps(
                    training_config_to_dict(default_training_config('obb')),
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            paths = list_training_template_paths('obb', [path])
            self.assertIn(path, paths)


if __name__ == '__main__':
    unittest.main()
