"""Tests for qrate.analyze module."""

from pathlib import Path

import numpy as np
from PIL import Image

from qrate.analyze import (
    compute_exposure_score,
    compute_file_hash,
    compute_perceptual_hash,
    compute_sharpness,
    estimate_noise,
    hamming_distance,
)


def make_test_image(
    w: int = 100, h: int = 100, color: tuple = (128, 128, 128)
) -> Image.Image:
    """Create a test image."""
    return Image.new("RGB", (w, h), color)


def make_gradient_image(w: int = 100, h: int = 100) -> Image.Image:
    """Create a gradient test image."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = [x * 255 // w, y * 255 // h, 128]
    return Image.fromarray(arr)


class TestComputeSharpness:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_sharpness(img)
        assert score >= 0

    def test_gradient_sharper(self):
        uniform = make_test_image()
        gradient = make_gradient_image()
        uniform_score = compute_sharpness(uniform)
        gradient_score = compute_sharpness(gradient)
        # Gradient has edges, should be "sharper"
        assert gradient_score > uniform_score

    def test_from_path(self, tmp_path: Path):
        img = make_test_image()
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")
        score = compute_sharpness(path)
        assert score >= 0


class TestComputeExposureScore:
    def test_mid_gray(self):
        img = make_test_image(color=(128, 128, 128))
        score = compute_exposure_score(img)
        assert 0 <= score <= 1

    def test_black_image(self):
        img = make_test_image(color=(0, 0, 0))
        score = compute_exposure_score(img)
        assert 0 <= score <= 1

    def test_white_image(self):
        img = make_test_image(color=(255, 255, 255))
        score = compute_exposure_score(img)
        assert 0 <= score <= 1


class TestEstimateNoise:
    def test_clean_image(self):
        img = make_test_image()
        noise = estimate_noise(img)
        assert 0 <= noise <= 1

    def test_with_iso(self):
        img = make_test_image()
        noise = estimate_noise(img, iso=6400)
        assert 0 <= noise <= 1


class TestComputeFileHash:
    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"test content")
        f2.write_bytes(b"test content")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content 1")
        f2.write_bytes(b"content 2")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestComputePerceptualHash:
    def test_returns_hex_string(self):
        img = make_test_image()
        phash = compute_perceptual_hash(img)
        assert isinstance(phash, str)
        assert len(phash) == 16  # 64 bits = 16 hex chars

    def test_similar_images_similar_hash(self):
        img1 = make_test_image(color=(128, 128, 128))
        img2 = make_test_image(color=(130, 130, 130))
        h1 = compute_perceptual_hash(img1)
        h2 = compute_perceptual_hash(img2)
        dist = hamming_distance(h1, h2)
        assert dist < 10  # Similar images should have low distance


class TestHammingDistance:
    def test_identical(self):
        dist = hamming_distance("0000000000000000", "0000000000000000")
        assert dist == 0

    def test_one_bit_diff(self):
        dist = hamming_distance("0000000000000000", "0000000000000001")
        assert dist == 1

    def test_all_different(self):
        dist = hamming_distance("0000000000000000", "ffffffffffffffff")
        assert dist == 64
