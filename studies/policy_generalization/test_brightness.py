import unittest
import torch

from brightness import apply_brightness


class TestBrightness(unittest.TestCase):
    def setUp(self):
        self.image = torch.full((1, 3, 4, 4), 0.8)
        self.state = torch.tensor([[1.0, 2.0]])

        self.observation = {
            "observation.images.camera1": self.image,
            "observation.state": self.state,
        }

    def test_factor_one(self):
        result = apply_brightness(self.observation, 1.0)

        torch.testing.assert_close(
            result["observation.images.camera1"],
            self.image
        )

    def test_darken_without_modifying_state(self):
        result = apply_brightness(self.observation, 0.5)

        expected = torch.full_like(self.image, 0.4)

        torch.testing.assert_close(
            result["observation.images.camera1"],
            expected
        )

        torch.testing.assert_close(
            result["observation.state"],
            self.state
        )

        # 原始图像没有被就地修改
        torch.testing.assert_close(
            self.observation["observation.images.camera1"],
            self.image
        )

    def test_invalid_factor(self):
        with self.assertRaises(ValueError):
            apply_brightness(self.observation, 0)


if __name__ == "__main__":
    unittest.main()
