from app.services.plant_size_analyzer import PlantSizeStats


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

    # How far a plant's measured canopy area can nudge its yield weight
    # away from the flat per-plant baseline, in either direction. Bounded
    # tightly (not wide open) because this is a heuristic response to a 2D
    # canopy silhouette, not a calibrated area-to-biomass regression -- no
    # destructive-harvest ground truth exists to fit one, and we're not
    # inventing per-crop coefficients to fake that precision.
    SIZE_FACTOR_MIN = 0.7
    SIZE_FACTOR_MAX = 1.4
    # fill_ratio (measured canopy area / the plant's own allotted land
    # share) is clamped before the sqrt so one badly-segmented outlier
    # plant can't blow the whole field's size_factor out to an extreme.
    FILL_RATIO_MIN = 0.3
    FILL_RATIO_MAX = 3.0

    def resolve_per_plant_kg(self, crop_type: str, average_kg: float | None = None) -> float:
        if average_kg and average_kg > 0:
            return average_kg
        return self.YIELD_PER_PLANT_KG.get(crop_type.lower(), self.YIELD_PER_PLANT_KG["_default"])

    def size_adjustment(
        self, plant_size_stats: PlantSizeStats | None, plant_count: int, area_ha: float | None,
    ) -> tuple[float, str | None]:
        """How much to scale the flat per-plant yield baseline based on
        each plant's actually-measured canopy area (from SAM's own
        instance masks, see plant_size_analyzer.py) instead of treating
        every detected plant as worth the same fixed weight.

        There's no external "expected mature canopy area" table for any
        crop here (that would be exactly the kind of invented number this
        project avoids elsewhere) -- so canopy area is compared against
        the field's OWN internally-derived reference instead: the average
        land area each plant would occupy if area_ha were divided evenly
        across plant_count. A field whose plants measure bigger than that
        implied spacing gets nudged up; smaller, nudged down. Real, already
        -measured inputs only -- no fabricated per-crop constant added.

        Returns (factor, note) -- factor is 1.0 with note=None whenever
        there isn't enough data to say anything (no size stats, no area,
        zero plants), matching the pipeline's existing behavior before
        this adjustment existed.
        """
        if plant_size_stats is None or not area_ha or area_ha <= 0 or plant_count <= 0:
            return 1.0, None

        allocated_area_cm2 = (area_ha * 1e8) / plant_count
        if allocated_area_cm2 <= 0:
            return 1.0, None

        fill_ratio = plant_size_stats.mean_area_cm2 / allocated_area_cm2
        clamped_ratio = max(self.FILL_RATIO_MIN, min(self.FILL_RATIO_MAX, fill_ratio))
        # sqrt, not a direct linear scale: canopy area is a 2D projection,
        # not a verified linear proxy for plant mass, so the response is
        # deliberately damped rather than claiming precision it can't back.
        factor = max(self.SIZE_FACTOR_MIN, min(self.SIZE_FACTOR_MAX, clamped_ratio**0.5))

        note = None
        if factor > 1.05:
            note = (
                f"Measured plants average {plant_size_stats.mean_area_cm2:.0f}cm² of canopy -- larger than the "
                f"~{allocated_area_cm2:.0f}cm² their density implies -- so yield was nudged up ({factor:.2f}x). "
                "A heuristic based on measured canopy size, not a calibrated biomass model."
            )
        elif factor < 0.95:
            note = (
                f"Measured plants average {plant_size_stats.mean_area_cm2:.0f}cm² of canopy -- smaller than the "
                f"~{allocated_area_cm2:.0f}cm² their density implies -- so yield was nudged down ({factor:.2f}x). "
                "A heuristic based on measured canopy size, not a calibrated biomass model."
            )
        return round(factor, 3), note

    def estimate(
        self,
        plant_count: int,
        crop_type: str,
        coverage: float,
        health: float,
        average_kg: float | None = None,
        plant_size_stats: PlantSizeStats | None = None,
        area_ha: float | None = None,
    ) -> float:
        per_plant_kg = self.resolve_per_plant_kg(crop_type, average_kg)
        # Rewards fields that are both well-covered AND healthy-looking,
        # penalizes the opposite, without ever zeroing out a real plant
        # count -- 0.65 is the floor even at coverage=health=0.
        condition_factor = 0.65 + 0.35 * ((coverage + health) / 200)
        size_factor, _ = self.size_adjustment(plant_size_stats, plant_count, area_ha)
        return round(plant_count * per_plant_kg * condition_factor * size_factor, 2)
