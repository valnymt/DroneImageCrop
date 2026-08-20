import pytest

from app.services.yield_estimator import YieldEstimator


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
