class YieldEstimator:
    # Typical harvested weight per single plant, in kilograms -- these are
    # the pipeline's own baseline now (previously supplied by the caller as
    # a required "average_kg" input, which the frontend sourced from a
    # static per-crop lookup of its own; that duplicated this table one
    # layer up instead of replacing it). average_kg on estimate() is still
    # accepted as an override for when a real farm-record value is known,
    # it just no longer needs to be supplied for every call.
    YIELD_PER_PLANT_KG = {
        "wheat": 0.02,
        "corn": 0.18,
        "rice": 0.025,
        "soybean": 0.015,
        "tomato": 3.0,
        "_default": 0.05,
    }

    def resolve_per_plant_kg(self, crop_type: str, average_kg: float | None = None) -> float:
        if average_kg and average_kg > 0:
            return average_kg
        return self.YIELD_PER_PLANT_KG.get(crop_type.lower(), self.YIELD_PER_PLANT_KG["_default"])

    def estimate(
        self, plant_count: int, crop_type: str, coverage: float, health: float, average_kg: float | None = None,
    ) -> float:
        per_plant_kg = self.resolve_per_plant_kg(crop_type, average_kg)
        # Rewards fields that are both well-covered AND healthy-looking,
        # penalizes the opposite, without ever zeroing out a real plant
        # count -- 0.65 is the floor even at coverage=health=0.
        condition_factor = 0.65 + 0.35 * ((coverage + health) / 200)
        return round(plant_count * per_plant_kg * condition_factor, 2)
