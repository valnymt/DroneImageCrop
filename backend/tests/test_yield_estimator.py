import pytest

from app.services.yield_estimator import YieldEstimator


@pytest.fixture
def estimator():
    return YieldEstimator()


class TestYieldEstimator:
    @pytest.mark.parametrize(
        "crop_type,factor",
        [("wheat", 1.0), ("Wheat", 1.0), ("CORN", 1.05), ("rice", 0.95), ("Soybean", 0.9), ("tomato", 1.1)],
    )
    def test_crop_factor_is_case_insensitive(self, estimator, crop_type, factor):
        # At coverage=100, health=100 the condition factor is exactly 1.0,
        # isolating the crop factor for a direct assertion.
        result = estimator.estimate(plant_count=100, crop_type=crop_type, average_kg=1.0, coverage=100, health=100)
        assert result == pytest.approx(100 * factor, rel=1e-6)

    def test_unknown_crop_defaults_to_factor_one(self, estimator):
        result = estimator.estimate(plant_count=100, crop_type="unobtainium", average_kg=1.0, coverage=100, health=100)
        assert result == pytest.approx(100.0, rel=1e-6)

    def test_condition_factor_spans_065_to_1(self, estimator):
        worst = estimator.estimate(plant_count=100, crop_type="wheat", average_kg=1.0, coverage=0, health=0)
        best = estimator.estimate(plant_count=100, crop_type="wheat", average_kg=1.0, coverage=100, health=100)
        assert worst == pytest.approx(100 * 0.65, rel=1e-6)
        assert best == pytest.approx(100.0, rel=1e-6)
        assert worst < best

    def test_zero_plants_yields_zero(self, estimator):
        assert estimator.estimate(0, "wheat", 1.0, 100, 100) == 0.0

    def test_matches_documented_formula_exactly(self, estimator):
        plant_count, crop_type, average_kg, coverage, health = 333, "corn", 0.037, 61.3, 44.7
        crop_factor = 1.05
        condition_factor = 0.65 + 0.35 * ((coverage + health) / 200)
        expected = round(plant_count * average_kg * crop_factor * condition_factor, 2)

        result = estimator.estimate(plant_count, crop_type, average_kg, coverage, health)

        assert result == expected

    def test_result_is_rounded_to_two_decimals(self, estimator):
        result = estimator.estimate(plant_count=333, crop_type="corn", average_kg=0.037, coverage=61.3, health=44.7)
        assert result == round(result, 2)
