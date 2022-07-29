import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import surface_distance
from surface_distance import metrics


class SurfaceDistanceTest(parameterized.TestCase, absltest.TestCase):

    def _assert_almost_equal(self, expected, actual, places):
        """Assertion wrapper correctly handling NaN equality."""
        if np.isnan(expected) and np.isnan(actual):
            return
        self.assertAlmostEqual(expected, actual, places)

    def _assert_metrics(self,
                        surface_distances, mask_gt, mask_pred,
                        expected_average_surface_distance,
                        expected_hausdorff_100,
                        expected_hausdorff_95,
                        expected_surface_overlap_at_1mm,
                        expected_surface_dice_at_1mm,
                        expected_volumetric_dice,
                        places=3):
        actual_average_surface_distance = (
            surface_distance.compute_average_surface_distance(surface_distances))
        for i in range(2):
            self._assert_almost_equal(
                expected_average_surface_distance[i],
                actual_average_surface_distance[i],
                places=places)

        self._assert_almost_equal(
            expected_hausdorff_100,
            surface_distance.compute_robust_hausdorff(surface_distances, 100),
            places=places)

        self._assert_almost_equal(
            expected_hausdorff_95,
            surface_distance.compute_robust_hausdorff(surface_distances, 95),
            places=places)

        actual_surface_overlap_at_1mm = (
            surface_distance.compute_surface_overlap_at_tolerance(
                surface_distances, tolerance_mm=1))
        for i in range(2):
            self._assert_almost_equal(
                expected_surface_overlap_at_1mm[i],
                actual_surface_overlap_at_1mm[i],
                places=places)

        self._assert_almost_equal(
            expected_surface_dice_at_1mm,
            surface_distance.compute_surface_dice_at_tolerance(
                surface_distances, tolerance_mm=1),
            places=places)

        self._assert_almost_equal(
            expected_volumetric_dice,
            surface_distance.compute_dice_coefficient(mask_gt, mask_pred),
            places=places)

    @parameterized.parameters((
            np.zeros([2, 2, 2], dtype=np.bool),
            np.zeros([2, 2], dtype=np.bool),
            [1, 1],
    ), (
            np.zeros([2, 2], dtype=np.bool),
            np.zeros([2, 2, 2], dtype=np.bool),
            [1, 1],
    ), (
            np.zeros([2, 2], dtype=np.bool),
            np.zeros([2, 2], dtype=np.bool),
            [1, 1, 1],
    ))
    def test_compute_surface_distances_raises_on_incompatible_shapes(
            self, mask_gt, mask_pred, spacing_mm):
        with self.assertRaisesRegex(ValueError,
                                    'The arguments must be of compatible shape'):
            surface_distance.compute_surface_distances(mask_gt, mask_pred, spacing_mm)

    @parameterized.parameters((
            np.zeros([2], dtype=np.bool),
            np.zeros([2], dtype=np.bool),
            [1],
    ), (
            np.zeros([2, 2, 2, 2], dtype=np.bool),
            np.zeros([2, 2, 2, 2], dtype=np.bool),
            [1, 1, 1, 1],
    ))
    def test_compute_surface_distances_raises_on_invalid_shapes(
            self, mask_gt, mask_pred, spacing_mm):
        with self.assertRaisesRegex(ValueError,
                                    'Only 2D and 3D masks are supported'):
            surface_distance.compute_surface_distances(mask_gt, mask_pred, spacing_mm)


class SurfaceDistance3DTest(SurfaceDistanceTest):

    def test_on_2_pixels_2mm_away(self):
        mask_gt = np.zeros((128, 128, 128), np.bool)
        mask_pred = np.zeros((128, 128, 128), np.bool)
        mask_gt[50, 60, 70] = 1
        mask_pred[50, 60, 72] = 1
        surface_distances = surface_distance.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=(3, 2, 1))
        self._assert_metrics(surface_distances, mask_gt, mask_pred,
                             expected_average_surface_distance=(1.5, 1.5),
                             expected_hausdorff_100=2.0,
                             expected_hausdorff_95=2.0,
                             expected_surface_overlap_at_1mm=(0.5, 0.5),
                             expected_surface_dice_at_1mm=0.5,
                             expected_volumetric_dice=0.0)

    def test_two_cubes_shifted_by_one_pixel(self):
        mask_gt = np.zeros((100, 100, 100), np.bool_)
        mask_pred = np.zeros((100, 100, 100), np.bool_)
        mask_gt[0:50, :, :] = 1
        mask_pred[0:51, :, :] = 1
        surface_distances = surface_distance.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=(2, 1, 1))
        # self._assert_metrics(
        #     surface_distances, mask_gt, mask_pred,
        #     expected_average_surface_distance=(0.322, 0.339),
        #     expected_hausdorff_100=2.0,
        #     expected_hausdorff_95=2.0,
        #     expected_surface_overlap_at_1mm=(0.842, 0.830),
        #     expected_surface_dice_at_1mm=0.836,
        #     expected_volumetric_dice=0.990)
        print(surface_distance.compute_average_surface_distance(surface_distances))

    def test_empty_prediction_mask(self):
        mask_gt = np.zeros((128, 128, 128), np.bool)
        mask_pred = np.zeros((128, 128, 128), np.bool)
        mask_gt[50, 60, 70] = 1
        surface_distances = surface_distance.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=(3, 2, 1))
        self._assert_metrics(
            surface_distances, mask_gt, mask_pred,
            expected_average_surface_distance=(np.inf, np.nan),
            expected_hausdorff_100=np.inf,
            expected_hausdorff_95=np.inf,
            expected_surface_overlap_at_1mm=(0.0, np.nan),
            expected_surface_dice_at_1mm=0.0,
            expected_volumetric_dice=0.0)

    def test_empty_ground_truth_mask(self):
        mask_gt = np.zeros((128, 128, 128), np.bool)
        mask_pred = np.zeros((128, 128, 128), np.bool)
        mask_pred[50, 60, 72] = 1
        surface_distances = surface_distance.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=(3, 2, 1))
        self._assert_metrics(
            surface_distances, mask_gt, mask_pred,
            expected_average_surface_distance=(np.nan, np.inf),
            expected_hausdorff_100=np.inf,
            expected_hausdorff_95=np.inf,
            expected_surface_overlap_at_1mm=(np.nan, 0.0),
            expected_surface_dice_at_1mm=0.0,
            expected_volumetric_dice=0.0)

    def test_both_empty_masks(self):
        mask_gt = np.zeros((128, 128, 128), np.bool)
        mask_pred = np.zeros((128, 128, 128), np.bool)
        surface_distances = surface_distance.compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=(3, 2, 1))
        self._assert_metrics(
            surface_distances, mask_gt, mask_pred,
            expected_average_surface_distance=(np.nan, np.nan),
            expected_hausdorff_100=np.inf,
            expected_hausdorff_95=np.inf,
            expected_surface_overlap_at_1mm=(np.nan, np.nan),
            expected_surface_dice_at_1mm=np.nan,
            expected_volumetric_dice=np.nan)
