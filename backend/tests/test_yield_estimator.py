import pytest

from app.services.plant_size_analyzer import PlantSizeStats
from app.services.yield_estimator import YieldEstimator


def _size_stats(mean_area_cm2: float) -> PlantSizeStats:
    return PlantSizeStats(
        plant_count=10, mean_area_cm2=mean_area_cm2, median_area_cm2=mean_area_cm2,
        min_area_cm2=mean_area_cm2, max_area_cm2=mean_area_cm2, mean_aspect_ratio=1.1,
        size_uniformity_score=80.0,
    )


@pytest.fixture
def estimator():
    return YieldEstimator()


class TestResolvePerPlantKg:
    @pytest.mark.parametrize(
        "crop_type,expected",
        [
            ("wheat", 0.02), ("Wheat", 0.02), ("CORN", 0.18), ("rice", 0.025),
            ("Soybean", 0.015), ("tomato", 3.0),
        ],
    )
    def test_known_crop_uses_its_own_baseline(self, estimator, crop_type, expected):
        assert estimator.resolve_per_plant_kg(crop_type) == expected

    def test_unknown_crop_uses_default_baseline(self, estimator):
        assert estimator.resolve_per_plant_kg("unobtainium") == YieldEstimator.YIELD_PER_PLANT_KG["_default"]

    def test_explicit_override_takes_priority_over_baseline(self, estimator):
        assert estimator.resolve_per_plant_kg("corn", average_kg=0.5) == 0.5

    def test_zero_or_none_override_falls_back_to_baseline(self, estimator):
        assert estimator.resolve_per_plant_kg("corn", average_kg=0.0) == 0.18
        assert estimator.resolve_per_plant_kg("corn", average_kg=None) == 0.18


class TestEstimate:
    def test_uses_crop_baseline_when_no_override_given(self, estimator):
        # No average_kg supplied -- must resolve corn's own 0.18 kg/plant
        # baseline, not silently fall back to some generic constant.
        result = estimator.estimate(plant_count=100, crop_type="corn", coverage=100, health=100)
        assert result == pytest.approx(100 * 0.18, rel=1e-6)

    def test_explicit_average_kg_overrides_the_baseline(self, estimator):
        result = estimator.estimate(plant_count=100, crop_type="corn", coverage=100, health=100, average_kg=2.0)
        assert result == pytest.approx(200.0, rel=1e-6)

    def test_condition_factor_spans_065_to_1(self, estimator):
        worst = estimator.estimate(plant_count=100, crop_type="wheat", coverage=0, health=0, average_kg=1.0)
        best = estimator.estimate(plant_count=100, crop_type="wheat", coverage=100, health=100, average_kg=1.0)
        assert worst == pytest.approx(100 * 0.65, rel=1e-6)
        assert best == pytest.approx(100.0, rel=1e-6)
        assert worst < best

    def test_zero_plants_yields_zero(self, estimator):
        assert estimator.estimate(0, "wheat", 100, 100) == 0.0

    def test_matches_documented_formula_exactly(self, estimator):
        plant_count, crop_type, coverage, health, average_kg = 333, "corn", 61.3, 44.7, 0.037
        condition_factor = 0.65 + 0.35 * ((coverage + health) / 200)
        expected = round(plant_count * average_kg * condition_factor, 2)

        result = estimator.estimate(plant_count, crop_type, coverage, health, average_kg)

        assert result == expected

    def test_result_is_rounded_to_two_decimals(self, estimator):
        result = estimator.estimate(plant_count=333, crop_type="corn", coverage=61.3, health=44.7, average_kg=0.037)
        assert result == round(result, 2)

    def test_healthier_field_yields_more_at_same_plant_count(self, estimator):
        # The whole point of Phase L: yield reflects what the model actually
        # saw (health/coverage), not just a flat crop-name lookup.
        sickly = estimator.estimate(plant_count=200, crop_type="soybean", coverage=30, health=25)
        thriving = estimator.estimate(plant_count=200, crop_type="soybean", coverage=95, health=90)
        assert thriving > sickly

    def test_no_size_stats_leaves_estimate_unchanged_from_pre_size_behavior(self, estimator):
        # Backward compatibility: every caller that doesn't pass
        # plant_size_stats/area_ha (e.g. /recompute, which never re-runs
        # segmentation) must get exactly the old flat-per-plant number.
        without_size = estimator.estimate(plant_count=100, crop_type="corn", coverage=80, health=70, average_kg=0.2)
        with_no_area = estimator.estimate(
            plant_count=100, crop_type="corn", coverage=80, health=70, average_kg=0.2,
            plant_size_stats=_size_stats(500), area_ha=None,
        )
        assert without_size == with_no_area


class TestSizeAdjustment:
    def test_no_size_stats_gives_neutral_factor_and_no_note(self, estimator):
        factor, note = estimator.size_adjustment(None, plant_count=100, area_ha=1.0)
        assert factor == 1.0
        assert note is None

    def test_zero_plant_count_gives_neutral_factor(self, estimator):
        factor, note = estimator.size_adjustment(_size_stats(500), plant_count=0, area_ha=1.0)
        assert factor == 1.0
        assert note is None

    def test_missing_area_gives_neutral_factor(self, estimator):
        factor, note = estimator.size_adjustment(_size_stats(500), plant_count=100, area_ha=None)
        assert factor == 1.0
        assert note is None

    def test_plants_bigger_than_their_allotted_spacing_nudge_factor_up(self, estimator):
        # 1 hectare = 1e8 cm^2; split across 100 plants -> 1,000,000 cm^2
        # allotted per plant. A measured 4,000,000 cm^2 canopy is 4x that.
        factor, note = estimator.size_adjustment(_size_stats(4_000_000), plant_count=100, area_ha=1.0)
        assert factor > 1.0
        assert note is not None
        assert "nudged up" in note

    def test_plants_smaller_than_their_allotted_spacing_nudge_factor_down(self, estimator):
        factor, note = estimator.size_adjustment(_size_stats(100_000), plant_count=100, area_ha=1.0)
        assert factor < 1.0
        assert note is not None
        assert "nudged down" in note

    def test_factor_is_bounded_even_for_extreme_fill_ratios(self, estimator):
        huge, _ = estimator.size_adjustment(_size_stats(10_000_000), plant_count=100, area_ha=1.0)
        tiny, _ = estimator.size_adjustment(_size_stats(0.001), plant_count=100, area_ha=1.0)
        assert huge == YieldEstimator.SIZE_FACTOR_MAX
        assert tiny == YieldEstimator.SIZE_FACTOR_MIN

    def test_typical_fill_ratio_gives_no_note(self, estimator):
        # Canopy area matching the allotted spacing almost exactly shouldn't
        # be reported as a meaningful adjustment.
        factor, note = estimator.size_adjustment(_size_stats(1_000_000), plant_count=100, area_ha=1.0)
        assert factor == pytest.approx(1.0, abs=0.05)
        assert note is None

    def test_estimate_applies_the_size_factor_to_the_final_number(self, estimator):
        base = estimator.estimate(plant_count=100, crop_type="corn", coverage=80, health=70, average_kg=0.2)
        bigger_plants = estimator.estimate(
            plant_count=100, crop_type="corn", coverage=80, health=70, average_kg=0.2,
            plant_size_stats=_size_stats(4_000_000), area_ha=1.0,
        )
        assert bigger_plants > base
