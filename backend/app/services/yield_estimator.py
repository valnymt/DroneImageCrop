class YieldEstimator:
    CROP_FACTORS = {"wheat": 1.0, "corn": 1.05, "rice": 0.95, "soybean": 0.9, "tomato": 1.1}

    def estimate(self, plant_count: int, crop_type: str, average_kg: float, coverage: float, health: float) -> float:
        crop_factor = self.CROP_FACTORS.get(crop_type.lower(), 1.0)
        condition_factor = 0.65 + 0.35 * ((coverage + health) / 200)
        return round(plant_count * average_kg * crop_factor * condition_factor, 2)
