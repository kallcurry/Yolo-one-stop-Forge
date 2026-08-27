import tempfile
import unittest
from pathlib import Path

from app.models.dataset_duplicates import (
    delete_duplicate_files,
    resolve_raw_dataset_root,
    scan_raw_duplicates,
)


class DatasetDuplicatesTest(unittest.TestCase):
    def test_batch_directory_resolves_to_generic_project_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'AnyProject'
            for directory in ('images/batch-a', 'annotations/batch-a'):
                (root / directory).mkdir(parents=True)

            self.assertEqual(
                resolve_raw_dataset_root(root / 'images' / 'batch-a'),
                root.resolve(),
            )
            result = scan_raw_duplicates(root / 'annotations' / 'batch-a')
            self.assertEqual(result.root, root.resolve())

    def test_derived_training_tree_is_not_accepted_as_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'AnyProject' / 'training_data' / 'run'
            (root / 'images').mkdir(parents=True)
            (root / 'annotations').mkdir()

            with self.assertRaises(ValueError):
                resolve_raw_dataset_root(root)

    def test_scans_all_raw_trees_and_ignores_derived_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ('images/a', 'images/b', 'annotations/a',
                              'annotations/b', 'labels/a', 'labels/b',
                              'training_data/run/images'):
                (root / directory).mkdir(parents=True)

            (root / 'images/a/frame.jpg').write_bytes(b'same image')
            (root / 'images/b/other.jpg').write_bytes(b'same image')
            (root / 'images/a/unique.png').write_bytes(b'unique')
            (root / 'annotations/a/frame.json').write_bytes(b'{"x": 1}')
            (root / 'annotations/b/other.json').write_bytes(b'{"x": 1}')
            (root / 'labels/a/frame.txt').write_text('0 0.5 0.5\n')
            (root / 'labels/b/other.txt').write_text('0 0.5 0.5\n')
            (root / 'training_data/run/images/ignored.jpg').write_bytes(
                b'same image'
            )

            result = scan_raw_duplicates(root)

            self.assertEqual(result.scanned_counts,
                             {'image': 3, 'annotation': 2, 'label': 2})
            self.assertEqual(result.duplicate_group_count, 3)
            self.assertEqual(result.duplicate_file_count, 3)
            self.assertEqual(result.groups['image'][0].keeper,
                             (root / 'images/a/frame.jpg').resolve())

    def test_same_size_non_duplicates_are_not_marked_as_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images/a').mkdir(parents=True)
            (root / 'images/b').mkdir(parents=True)
            (root / 'images/a/one.jpg').write_bytes(b'1234567890')
            (root / 'images/b/two.jpg').write_bytes(b'abcdefghij')

            result = scan_raw_duplicates(root)

            self.assertEqual(result.duplicate_group_count, 0)
            self.assertEqual(result.duplicate_file_count, 0)

    def test_deletes_only_duplicate_members_and_keeps_canonical_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images/a').mkdir(parents=True)
            (root / 'images/b').mkdir(parents=True)
            keeper = root / 'images/a/frame.jpg'
            duplicate = root / 'images/b/frame.jpg'
            keeper.write_bytes(b'same')
            duplicate.write_bytes(b'same')

            result = scan_raw_duplicates(root)
            deleted = delete_duplicate_files(result, use_trash=False)

            self.assertEqual(deleted.errors, ())
            self.assertEqual(deleted.deleted, (duplicate.resolve(),))
            self.assertTrue(keeper.exists())
            self.assertFalse(duplicate.exists())


if __name__ == '__main__':
    unittest.main()
