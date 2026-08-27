import tempfile
import unittest
from pathlib import Path

from app.models.operations import copy_images, rename_image


class OperationsAnnotationSetTest(unittest.TestCase):
    def test_copy_images_syncs_configured_annotation_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_img_dir = root / 'images' / 'source'
            target_img_dir = root / 'images' / 'target'
            source_ann_dir = root / 'annotations-obb' / 'source'
            source_img_dir.mkdir(parents=True)
            target_img_dir.mkdir(parents=True)
            source_ann_dir.mkdir(parents=True)
            image = source_img_dir / 'frame_001.jpg'
            annotation = source_ann_dir / 'frame_001.json'
            image.write_bytes(b'image')
            annotation.write_text('{"shapes": []}', encoding='utf-8')

            errors = copy_images(
                [image],
                target_img_dir,
                sync_annotation=True,
                annotation_dir='annotations-obb',
            )

            self.assertEqual(errors, [])
            self.assertTrue((target_img_dir / 'frame_001.jpg').is_file())
            self.assertTrue(
                (root / 'annotations-obb' / 'target' / 'frame_001.json').is_file()
            )

    def test_rename_image_syncs_configured_annotation_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / 'images' / 'source'
            ann_dir = root / 'annotations-det' / 'source'
            image_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            image = image_dir / 'frame_001.jpg'
            annotation = ann_dir / 'frame_001.json'
            image.write_bytes(b'image')
            annotation.write_text('{"shapes": []}', encoding='utf-8')

            error = rename_image(
                image,
                'frame_renamed.jpg',
                rename_annotation=True,
                annotation_dir='annotations-det',
            )

            self.assertIsNone(error)
            self.assertTrue((image_dir / 'frame_renamed.jpg').is_file())
            self.assertTrue((ann_dir / 'frame_renamed.json').is_file())
            self.assertFalse(annotation.exists())


if __name__ == '__main__':
    unittest.main()
