import numpy as np
import unittest
from unittest import mock

from detectron2.data.transforms.transform import NoOpTransform, RotationTransform
from detectron2.data.transforms.transform_gen import RandomRotation


class TestRotationTransform(unittest.TestCase):
    def assertEqualsArrays(self, a1, a2):
        self.assertTrue(np.allclose(a1, a2))

    def randomData(self, h=5, w=5):
        image = np.random.rand(h, w)
        coords = np.array([[i, j] for j in range(h + 1) for i in range(w + 1)], dtype=float)
        return image, coords, h, w

    def test180(self):
        image, coords, h, w = self.randomData(6, 6)
        rot = RotationTransform(h, w, 180, expand=False, center=None)
        self.assertEqualsArrays(rot.apply_image(image), image[::-1, ::-1])
        rotated_coords = [[w - c[0], h - c[1]] for c in coords]
        self.assertEqualsArrays(rot.apply_coords(coords), rotated_coords)

    def test45_coords(self):
        _, coords, h, w = self.randomData(4, 6)
        rot = RotationTransform(h, w, 45, expand=False, center=None)
        rotated_coords = [
            [(x + y - (h + w) / 2) / np.sqrt(2) + w / 2, h / 2 + (y + (w - h) / 2 - x) / np.sqrt(2)]
            for (x, y) in coords
        ]
        self.assertEqualsArrays(rot.apply_coords(coords), rotated_coords)

    def test90(self):
        image, coords, h, w = self.randomData()
        rot = RotationTransform(h, w, 90, expand=False, center=None)
        self.assertEqualsArrays(rot.apply_image(image), image.T[::-1])
        rotated_coords = [[c[1], w - c[0]] for c in coords]
        self.assertEqualsArrays(rot.apply_coords(coords), rotated_coords)

    def test90_expand(self):  # non-square image
        image, coords, h, w = self.randomData(h=5, w=8)
        rot = RotationTransform(h, w, 90, expand=True, center=None)
        self.assertEqualsArrays(rot.apply_image(image), image.T[::-1])
        rotated_coords = [[c[1], w - c[0]] for c in coords]
        self.assertEqualsArrays(rot.apply_coords(coords), rotated_coords)

    def test_center_expand(self):
        # center has no effect if expand=True because it only affects shifting
        image, coords, h, w = self.randomData(h=5, w=8)
        angle = np.random.randint(360)
        rot1 = RotationTransform(h, w, angle, expand=True, center=None)
        rot2 = RotationTransform(h, w, angle, expand=True, center=(0, 0))
        rot3 = RotationTransform(h, w, angle, expand=True, center=(h, w))
        rot4 = RotationTransform(h, w, angle, expand=True, center=(2, 5))
        for r1 in [rot1, rot2, rot3, rot4]:
            for r2 in [rot1, rot2, rot3, rot4]:
                self.assertEqualsArrays(r1.apply_image(image), r2.apply_image(image))
                self.assertEqualsArrays(r1.apply_coords(coords), r2.apply_coords(coords))

    def test_random_transform_prob_out_of_range_check(self):
        # GIVEN
        test_probabilities = {0.0: True, 0.5: True, 1.0: True, -0.01: False, 1.01: False}

        # WHEN
        for given_probability, is_valid in test_probabilities.items():
            # THEN
            if not is_valid:
                self.assertRaises(AssertionError, RandomRotation, angle=42, prob=given_probability)
            else:
                RandomRotation(angle=42, prob=given_probability)

    def test_random_transform_probability_occured_evaluation(self):
        # GIVEN
        test_probability = 0.001
        test_image, *_ = self.randomData()
        test_angle = 42
        random_rotation = RandomRotation(
            angle=[test_angle], sample_style="choice", prob=test_probability
        )

        # WHEN
        with mock.patch.object(random_rotation, "_rand_range") as rand_range_mock:
            rand_range_mock.return_value = 0.0001
            rotation_transform = random_rotation.get_transform(test_image)

        # THEN
        assert rotation_transform.angle == test_angle

    def test_random_transform_probability_not_occured_evaluation(self):
        # GIVEN
        test_probability = 0.001
        test_image, *_ = self.randomData()
        random_rotation = RandomRotation(angle=[42], sample_style="choice", prob=test_probability)

        # WHEN
        with mock.patch.object(random_rotation, "_rand_range") as rand_range_mock:
            rand_range_mock.return_value = 0.9
            rotation_transform = random_rotation.get_transform(test_image)

        # THEN
        assert isinstance(rotation_transform, NoOpTransform)


if __name__ == "__main__":
    unittest.main()
