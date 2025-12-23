"""Tests for qrate.score module."""

import numpy as np
from PIL import Image

from qrate.score import (
    CompositionScores,
    ColorScores,
    ExhibitionScore,
    TechnicalScores,
    auto_orient,
    compute_color_contrast,
    compute_color_harmony,
    compute_dynamic_range,
    compute_negative_space,
    compute_obstruction,
    compute_saturation_balance,
    compute_simplicity,
    compute_subject_clarity,
    compute_subject_sharpness,
    compute_thirds_alignment,
    compute_visual_balance,
    score_image,
    score_uniqueness,
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


class TestTechnicalScores:
    def test_dataclass_defaults(self):
        t = TechnicalScores()
        assert t.sharpness == 0.0
        assert t.exposure == 0.0
        assert t.noise == 0.0
        assert t.dynamic_range == 0.0
        assert t.subject_sharpness == 0.0

    def test_custom_values(self):
        t = TechnicalScores(sharpness=100, exposure=0.8, noise=0.1)
        assert t.sharpness == 100
        assert t.exposure == 0.8
        assert t.noise == 0.1


class TestCompositionScores:
    def test_dataclass_defaults(self):
        c = CompositionScores()
        assert c.thirds_alignment == 0.0
        assert c.balance == 0.0
        assert c.simplicity == 0.0
        assert c.negative_space == 0.0
        assert c.obstruction == 0.0
        assert c.subject_clarity == 0.0


class TestColorScores:
    def test_dataclass_defaults(self):
        c = ColorScores()
        assert c.harmony == 0.0
        assert c.saturation_balance == 0.0
        assert c.color_contrast == 0.0


class TestExhibitionScore:
    def test_defaults(self):
        s = ExhibitionScore()
        assert s.technical_score >= 0
        assert s.composition_score >= 0
        assert s.color_score >= 0
        assert s.uniqueness >= 0  # Default may vary

    def test_final_score_range(self):
        s = ExhibitionScore()
        s.technical = TechnicalScores(sharpness=1000, exposure=0.8, noise=0.1)
        s.composition = CompositionScores(
            thirds_alignment=0.7,
            balance=0.8,
            simplicity=0.6,
            negative_space=0.5,
            obstruction=0.9,
            subject_clarity=0.8,
        )
        s.color = ColorScores(harmony=0.7, saturation_balance=0.8, color_contrast=0.6)
        score = s.final_score
        assert 0 <= score <= 100

    def test_technical_score_uses_subject_sharpness(self):
        s = ExhibitionScore()
        s.technical = TechnicalScores(
            sharpness=2000, subject_sharpness=0.8, exposure=0.7, noise=0.1
        )
        # Should use subject_sharpness when > 0
        score = s.technical_score
        assert score > 0


class TestAutoOrient:
    def test_no_exif(self):
        img = make_test_image()
        result = auto_orient(img)
        assert result.size == img.size

    def test_returns_image(self):
        img = make_test_image()
        result = auto_orient(img)
        assert isinstance(result, Image.Image)


class TestDynamicRange:
    def test_uniform_image(self):
        img = make_test_image(color=(128, 128, 128))
        dr = compute_dynamic_range(img)
        assert 0 <= dr <= 1

    def test_gradient_image(self):
        img = make_gradient_image()
        dr = compute_dynamic_range(img)
        assert dr > 0


class TestThirdsAlignment:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_thirds_alignment(img)
        assert 0 <= score <= 1

    def test_returns_float(self):
        img = make_gradient_image()
        score = compute_thirds_alignment(img)
        assert isinstance(score, float)


class TestVisualBalance:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_visual_balance(img)
        assert 0 <= score <= 1

    def test_centered_should_score_high(self):
        # Uniform image should be balanced
        img = make_test_image()
        score = compute_visual_balance(img)
        assert score >= 0.5


class TestSimplicity:
    def test_uniform_image_is_simple(self):
        img = make_test_image()
        score = compute_simplicity(img)
        assert score > 0.5  # Uniform = simple

    def test_gradient_less_simple(self):
        img = make_gradient_image()
        score = compute_simplicity(img)
        assert 0 <= score <= 1


class TestNegativeSpace:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_negative_space(img)
        assert 0 <= score <= 1


class TestObstruction:
    def test_clean_image(self):
        img = make_test_image()
        score = compute_obstruction(img)
        assert 0 <= score <= 1

    def test_returns_float(self):
        img = make_gradient_image()
        score = compute_obstruction(img)
        assert isinstance(score, float)


class TestSubjectClarity:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_subject_clarity(img)
        assert 0 <= score <= 1


class TestSubjectSharpness:
    def test_uniform_image(self):
        img = make_test_image()
        score = compute_subject_sharpness(img)
        assert 0 <= score <= 1

    def test_gradient_image(self):
        img = make_gradient_image()
        score = compute_subject_sharpness(img)
        assert 0 <= score <= 1


class TestColorHarmony:
    def test_grayscale(self):
        img = make_test_image(color=(128, 128, 128))
        score = compute_color_harmony(img)
        assert 0 <= score <= 1

    def test_colored_image(self):
        img = make_test_image(color=(255, 100, 50))
        score = compute_color_harmony(img)
        assert 0 <= score <= 1


class TestSaturationBalance:
    def test_grayscale(self):
        img = make_test_image(color=(128, 128, 128))
        score = compute_saturation_balance(img)
        assert 0 <= score <= 1

    def test_saturated(self):
        img = make_test_image(color=(255, 0, 0))
        score = compute_saturation_balance(img)
        assert 0 <= score <= 1


class TestColorContrast:
    def test_uniform(self):
        img = make_test_image()
        score = compute_color_contrast(img)
        assert 0 <= score <= 1

    def test_gradient(self):
        img = make_gradient_image()
        score = compute_color_contrast(img)
        assert 0 <= score <= 1


class TestScoreUniqueness:
    def test_near_duplicates_low_score(self):
        # Very similar hashes (low hamming distance) = not unique
        hashes = ["0000000000000000", "0000000000000001", "0000000000000002"]
        score = score_uniqueness("0000000000000000", hashes)
        assert score < 0.5

    def test_unique_hash(self):
        # Very different = unique (use valid 16-char hex)
        hashes = ["0000000000000000", "1111111111111111", "2222222222222222"]
        score = score_uniqueness("ffffffffffffffff", hashes)
        assert score > 0

    def test_empty_list(self):
        score = score_uniqueness("0000000000000000", [])
        assert score == 0.5

    def test_only_self(self):
        # When all hashes are same as target, no others to compare = unique
        hashes = ["0000000000000000"] * 5
        score = score_uniqueness("0000000000000000", hashes)
        assert score == 1.0


class TestScoreImage:
    def test_from_pil_image(self):
        img = make_gradient_image()
        score = score_image(img)
        assert isinstance(score, ExhibitionScore)
        assert 0 <= score.final_score <= 100

    def test_resizes_large_images(self):
        img = make_test_image(2000, 2000)
        score = score_image(img)
        assert isinstance(score, ExhibitionScore)

    def test_handles_rgba(self):
        img = Image.new("RGBA", (100, 100), (128, 128, 128, 255))
        score = score_image(img)
        assert isinstance(score, ExhibitionScore)

    def test_with_existing_technical(self):
        img = make_test_image()
        existing = TechnicalScores(sharpness=500, exposure=0.7, noise=0.2)
        score = score_image(img, existing_technical=existing)
        assert score.technical.sharpness == 500

    def test_from_path(self, tmp_path):
        img = make_test_image()
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")
        score = score_image(path)
        assert isinstance(score, ExhibitionScore)

    def test_from_string_path(self, tmp_path):
        img = make_test_image()
        path = tmp_path / "test.jpg"
        img.save(path, "JPEG")
        score = score_image(str(path))
        assert isinstance(score, ExhibitionScore)
