"""Score dataclasses for the multi-pass scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TechnicalScores:
    """Technical quality metrics."""

    sharpness: float = 0.0  # Laplacian variance (higher = sharper)
    exposure: float = 0.0  # 0-1, 1 = well exposed
    noise: float = 0.0  # 0-1, 0 = clean
    dynamic_range: float = 0.0  # 0-1, 1 = full range used
    subject_sharpness: float = 0.0  # Sharpness weighted by saliency (0-1)


@dataclass
class CompositionScores:
    """Compositional/aesthetic metrics."""

    thirds_alignment: float = 0.0  # 0-1, subject on power points
    balance: float = 0.0  # 0-1, visual weight distribution
    simplicity: float = 0.0  # 0-1, low clutter
    negative_space: float = 0.0  # 0-1, effective use of empty areas
    obstruction: float = 0.0  # 0-1, 1 = no foreground obstruction
    subject_clarity: float = 0.0  # 0-1, main subject unobstructed


@dataclass
class ColorScores:
    """Color aesthetic metrics."""

    harmony: float = 0.0  # 0-1, complementary/analogous colors
    saturation_balance: float = 0.0  # 0-1, not over/under saturated
    color_contrast: float = 0.0  # 0-1, effective color separation


@dataclass
class RenaissanceScores:
    """Renaissance art analysis metrics (da Vinci perspective)."""

    geometry_strength: float = 0.0  # 0-1, strong geometric patterns, lines, shapes
    focal_hierarchy: float = 0.0  # 0-1, clear visual hierarchy, subject prominence
    light_directional: float = 0.0  # 0-1, directional lighting, chiaroscuro
    subject_separation: float = 0.0  # 0-1, subject separated from background
    emotion_subdued: float = 0.0  # 0-1, subdued, contemplative emotion
    distance_readable: float = 0.0  # 0-1, readable depth, spatial relationships
    # Penalties (0-1, where 1 = no penalty, 0 = severe penalty)
    highlight_clipping_penalty: float = 1.0  # 1 = no clipping, 0 = severe clipping
    motion_blur_penalty: float = 1.0  # 1 = sharp, 0 = severe blur
    clutter_penalty: float = 1.0  # 1 = clean, 0 = cluttered


@dataclass
class ExhibitionScore:
    """Combined exhibition-worthiness score."""

    technical: TechnicalScores = field(default_factory=TechnicalScores)
    composition: CompositionScores = field(default_factory=CompositionScores)
    color: ColorScores = field(default_factory=ColorScores)
    renaissance: RenaissanceScores = field(default_factory=RenaissanceScores)
    uniqueness: float = 0.0  # 0-1, distinctiveness in collection

    # Weights for final score (Renaissance art priorities)
    WEIGHT_RENAISSANCE: float = 0.40  # Renaissance principles are primary
    WEIGHT_TECHNICAL: float = 0.20  # Technical excellence is table stakes
    WEIGHT_COMPOSITION: float = 0.20  # Composition matters
    WEIGHT_COLOR: float = 0.10  # Color harmony elevates work
    WEIGHT_UNIQUENESS: float = 0.10  # Unique perspectives are valued

    @property
    def technical_score(self) -> float:
        """Aggregate technical score (0-1)."""
        t = self.technical
        # Use subject_sharpness (saliency-weighted) for exhibition scoring
        # Falls back to raw sharpness if subject_sharpness not computed
        if t.subject_sharpness > 0:
            sharp_score = t.subject_sharpness  # Already normalized 0-1
        else:
            sharp_score = min(1.0, t.sharpness / 2000.0)
        # Noise is a penalty
        noise_penalty = 1.0 - t.noise
        return (
            sharp_score * 0.4
            + t.exposure * 0.3
            + noise_penalty * 0.15
            + t.dynamic_range * 0.15
        )

    @property
    def composition_score(self) -> float:
        """Aggregate composition score (0-1)."""
        c = self.composition
        return (
            c.thirds_alignment * 0.20
            + c.balance * 0.15
            + c.simplicity * 0.15
            + c.negative_space * 0.15
            + c.obstruction * 0.20  # Heavy weight - obstructions kill images
            + c.subject_clarity * 0.15
        )

    @property
    def color_score(self) -> float:
        """Aggregate color score (0-1)."""
        col = self.color
        return (
            col.harmony * 0.4 + col.saturation_balance * 0.3 + col.color_contrast * 0.3
        )

    @property
    def renaissance_score(self) -> float:
        """Aggregate Renaissance art analysis score (0-1).

        How would da Vinci evaluate this image?
        """
        r = self.renaissance

        # Core Renaissance principles
        base_score = (
            r.geometry_strength * 0.20
            + r.focal_hierarchy * 0.20
            + r.light_directional * 0.15
            + r.subject_separation * 0.15
            + r.emotion_subdued * 0.15
            + r.distance_readable * 0.15
        )

        # Apply penalties (less harsh: average penalty instead of multiplicative)
        # Multiplicative was too harsh - a single bad penalty killed the score
        avg_penalty = (
            r.highlight_clipping_penalty + r.motion_blur_penalty + r.clutter_penalty
        ) / 3.0

        # Apply penalty more gently: base_score * (0.5 + 0.5 * avg_penalty)
        # This means even with 0 penalty, score is halved, not zeroed
        return base_score * (0.5 + 0.5 * avg_penalty)

    @property
    def final_score(self) -> float:
        """Final exhibition score (0-100).

        When obstructions are detected, composition matters more.
        A technically perfect shot ruined by a sign is worthless for exhibition.
        """
        # Base weights
        w_tech = self.WEIGHT_TECHNICAL
        w_comp = self.WEIGHT_COMPOSITION

        # If obstruction detected, shift weight from technical to composition
        # Rationale: sharp signs don't make good art
        obstruction_penalty = (
            1.0 - self.composition.obstruction
        )  # 0 = clean, higher = obstructed
        if obstruction_penalty > 0.15:  # Significant obstruction
            # Reduce tech weight, boost composition weight
            shift = obstruction_penalty * 0.15  # Up to 15% shift
            w_tech = max(0.15, w_tech - shift)
            w_comp = min(0.45, w_comp + shift)

        raw = (
            self.renaissance_score * self.WEIGHT_RENAISSANCE
            + self.technical_score * w_tech
            + self.composition_score * w_comp
            + self.color_score * self.WEIGHT_COLOR
            + self.uniqueness * self.WEIGHT_UNIQUENESS
        )
        return raw * 100
