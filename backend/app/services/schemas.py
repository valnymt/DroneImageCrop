from pydantic import BaseModel, Field


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


class PlantSizeStats(BaseModel):
    """Per-plant size/shape distribution from SAM's own instance masks
    (see app/services/plant_size_analyzer.py) -- a second axis of analysis
    beyond plant_count: two fields with the same count and health score can
    still have very differently distributed individual plants."""

    plant_count: int = Field(ge=0)
    mean_area_cm2: float = Field(ge=0)
    median_area_cm2: float = Field(ge=0)
    min_area_cm2: float = Field(ge=0)
    max_area_cm2: float = Field(ge=0)
    mean_aspect_ratio: float = Field(ge=1)
    size_uniformity_score: float = Field(ge=0, le=100)


class AnalysisResult(BaseModel):
    plant_count: int = Field(ge=0)
    # "fine_tuned" (normal case) or "general_fallback" -- the fine-tuned
    # YOLO checkpoint found literally nothing despite real vegetation
    # coverage (see open_vocab_detector.py's docstring for why: it
    # generalizes poorly outside its ~823-image training set), so a
    # zero-shot open-vocabulary detector ran instead. Less precise than
    # the fine-tuned model in-distribution, but real detections instead
    # of a guaranteed zero on an unfamiliar photo. detection_note always
    # explains what happened either way.
    detection_method: str
    detection_note: str
    crop_density: float = Field(ge=0)
    crop_coverage: float = Field(ge=0, le=100)
    vegetation_score: float = Field(ge=0, le=100)
    health_score: float = Field(ge=0, le=100)
    # GLCM/Haralick texture uniformity of the vegetation region (see
    # app/services/texture_analyzer.py) -- a signal color alone can't give:
    # a uniformly discolored field (drought, nutrient deficiency) keeps a
    # smooth/uniform texture even as its color health drops, while disease
    # or pest damage tends to look patchy at the same color health. 50/
    # "mixed" is the neutral default when there's too little vegetation to
    # measure texture from at all.
    texture_uniformity_score: float = Field(ge=0, le=100)
    texture_pattern: str
    # Whether a perspective/tilt correction (see
    # app/services/tilt_corrector.py) was applied before analysis -- False
    # is the common, expected case for a photo that was already close to
    # straight-down, not a failure. tilt_correction_note always explains
    # what happened either way (why it was/wasn't applied).
    tilt_corrected: bool
    tilt_correction_note: str
    # Only set when tilt_corrected is True -- the actual analyzed frame
    # (data: URL), which detections/image_width/image_height are relative
    # to and which the caller's own original upload no longer matches.
    # None the rest of the time; callers should fall back to their own
    # original image, which is identical in that case anyway.
    analyzed_image: str | None = None
    # None whenever segmentation refinement was off or SAM wasn't
    # available (the plain Excess Green mask has no per-instance
    # boundaries to measure) -- not fabricated from plant_count alone.
    plant_size_stats: PlantSizeStats | None = None
    estimated_yield: float = Field(ge=0)
    # The per-plant yield actually used to compute estimated_yield --
    # either a caller-supplied override or the pipeline's own crop-specific
    # baseline (see YieldEstimator.YIELD_PER_PLANT_KG). Surfaced so callers
    # can persist/display the real number used rather than a guess of
    # their own.
    average_yield_per_plant_kg: float = Field(ge=0)
    confidence_score: float = Field(ge=0, le=100)
    detections: list[Detection]
    # Pixel space that `detections` coordinates are relative to -- the
    # preprocessed/resized image the pipeline actually analyzed, NOT
    # necessarily the original upload's raw dimensions (load_and_preprocess
    # caps at 1600px). The frontend needs this to scale boxes correctly
    # onto the displayed (original) photo.
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    # data: URLs (base64 PNG), ready to drop straight into an <img src>.
    segmentation_overlay: str
    heatmap_overlay: str


class RecomputeResult(BaseModel):
    """Response for POST /api/recompute -- crop_density and estimated_yield
    recalculated for a corrected crop type and/or field area, without
    re-running the CV pipeline (plant_count, coverage, and health don't
    change just because the user corrected a wrong AI guess about crop
    type or area; only the two values derived from them do)."""

    crop_density: float = Field(ge=0)
    estimated_yield: float = Field(ge=0)
    average_yield_per_plant_kg: float = Field(ge=0)


class ComparisonResult(BaseModel):
    """Response for POST /api/compare -- ORB feature matching + homography
    to align two photos of the same field taken at different times, then a
    vegetation-mask diff between them (see
    app/services/flight_comparator.py). alignment_ok is False whenever the
    two photos couldn't be confidently aligned (e.g. not the same field,
    too little overlap, too few distinguishing features) -- growth/loss
    percentages are 0 in that case, not a guess."""

    alignment_ok: bool
    keypoints_matched: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=100)
    growth_percent: float = Field(ge=0, le=100)
    loss_percent: float = Field(ge=0, le=100)
    unchanged_percent: float
    diff_overlay: str  # data: URL (base64 PNG)
    warning: str | None = None


class InspectResult(BaseModel):
    """Best-effort, low-confidence suggestions for pre-filling the analyze
    form -- never authoritative. crop_type comes from zero-shot CLIP
    classification (see app/services/crop_classifier.py; ~69% accurate,
    see backend/training/CLIP_CLASSIFIER_EVAL_REPORT.md for exactly where
    it's wrong). estimated_area_hectares is null whenever the photo has no
    altitude metadata to derive it from -- that is the common case, not an
    error."""

    crop_type: str
    confidence: float = Field(ge=0, le=100)
    estimated_area_hectares: float | None = Field(default=None, ge=0)
    area_source: str
