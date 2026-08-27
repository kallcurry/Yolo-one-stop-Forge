import unittest

import numpy as np

from app.training_runner import _sanitize_plot_keypoints


class TrainingRunnerTest(unittest.TestCase):
    def test_plot_guard_hides_non_finite_keypoints(self):
        result, count = _sanitize_plot_keypoints(
            np.array([[np.inf, 12.0, 0.9], [4.0, 5.0, 0.8]])
        )

        self.assertEqual(count, 1)
        self.assertTrue(np.array_equal(result[0], [0.0, 0.0, 0.0]))
        self.assertTrue(np.array_equal(result[1], [4.0, 5.0, 0.8]))


if __name__ == '__main__':
    unittest.main()
