import numpy as np
import pytest

from simnibs.segmentation._sanlm import sanlm


@pytest.fixture
def cube_image():
    img = np.zeros((50, 60, 70), dtype=float)
    img[10:40, 10:40, 10:40] = 1
    return img


class TestSanlm:
    def test_cube(self, cube_image):
        img = cube_image
        noisy = cube_image + 0.1 * np.random.normal(size=cube_image.shape)
        filtered = sanlm(noisy, 3, 1)
        assert np.linalg.norm(filtered - img) < np.linalg.norm(noisy - img)
