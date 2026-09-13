import numpy as np

from rice_disease_cv.quality import assess_image


def test_rejects_small_image():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = assess_image(image)
    assert not result.accepted
    assert "resolution_too_low" in result.issues


def test_rejects_unreadable_image():
    result = assess_image(None)
    assert not result.accepted
    assert result.issues == ["unreadable_image"]
