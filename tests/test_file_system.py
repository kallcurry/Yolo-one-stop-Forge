import tempfile
import unittest
from pathlib import Path

from app.models.file_system import (
    DirFormat,
    annotation_set_dir_for_image,
    detect_format,
    expected_annotation_path,
    find_annotation,
    scan_tree,
)


class FileSystemAnnotationSetTest(unittest.TestCase):
    def test_find_annotation_uses_configured_annotation_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / 'images' / 'scene'
            ann_dir = root / 'annotations-obb' / 'scene'
            image_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            image_path = image_dir / 'frame_001.jpg'
            ann_path = ann_dir / 'frame_001.json'
            image_path.write_bytes(b'fake')
            ann_path.write_text('{}', encoding='utf-8')

            self.assertEqual(
                detect_format(root, 'annotations-obb'),
                DirFormat.SEPARATED,
            )
            self.assertEqual(
                find_annotation(image_path, annotation_dir='annotations-obb'),
                ann_path.resolve(),
            )

    def test_expected_annotation_path_is_available_when_set_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / 'images' / 'scene'
            image_dir.mkdir(parents=True)
            image_path = image_dir / 'frame_001.jpg'
            image_path.write_bytes(b'fake')

            expected = expected_annotation_path(
                image_path,
                annotation_dir='annotations-det',
            )

            self.assertEqual(
                expected,
                (root / 'annotations-det' / 'scene' / 'frame_001.json').resolve(),
            )
            self.assertFalse(annotation_set_dir_for_image(
                image_path,
                annotation_dir='annotations-det',
            ).is_dir())
            self.assertIsNone(
                find_annotation(image_path, annotation_dir='annotations-det')
            )

    def test_scan_project_tree_groups_raw_training_and_test_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ShengSong_Datasets'
            raw_images = root / 'images' / 'Collect_A'
            raw_annotations = root / 'annotations' / 'Collect_A'
            train_batch = root / 'training_data' / '2026-07-29'
            split_batch = root / 'training_data' / '2026-08-01'
            default_test = root / 'test_data'

            for path in (
                raw_images,
                raw_annotations,
                train_batch / 'images',
                train_batch / 'annotations',
                split_batch / 'images' / 'train',
                split_batch / 'annotations' / 'train',
                default_test / 'images',
                default_test / 'annotations',
            ):
                path.mkdir(parents=True)
            (raw_images / 'raw.jpg').write_bytes(b'image')
            (train_batch / 'images' / 'train.jpg').write_bytes(b'image')
            (split_batch / 'images' / 'train' / 'split.jpg').write_bytes(b'image')
            (default_test / 'images' / 'test.jpg').write_bytes(b'image')

            tree = scan_tree(root)

            self.assertFalse(tree['selectable'])
            scopes = {item['display_name']: item for item in tree['children']}
            self.assertEqual(
                set(scopes),
                {'原始数据', '训练数据', '验证 / 测试数据'},
            )
            self.assertFalse(scopes['训练数据']['selectable'])
            training = scopes['训练数据']['children']
            self.assertEqual(
                [Path(item['path']).name for item in training],
                ['2026-07-29', '2026-08-01'],
            )
            self.assertEqual(
                [Path(item['path']).name for item in training[1]['children']],
                ['train'],
            )
            test_batches = scopes['验证 / 测试数据']['children']
            self.assertEqual(len(test_batches), 1)
            self.assertEqual(test_batches[0]['display_name'], '默认测试集')
            self.assertEqual(Path(test_batches[0]['path']), default_test.resolve())


if __name__ == '__main__':
    unittest.main()
