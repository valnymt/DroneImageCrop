from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

from app.services.pipeline import CropAnalysisPipeline, _segmentation_overlay
from app.services.schemas import Detection


class TestSegmentationOverlay:
    def test_tints_only_the_masked_region(self):
        image = np.full((40, 40, 3), (90, 140, 90), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:20, :] = 255  # top half masked, bottom half not

        overlay = _segmentation_overlay(image, mask)

        assert not np.array_equal(overlay[:20], image[:20])
        assert np.array_equal(overlay[20:], image[20:])

    def test_empty_mask_leaves_image_unchanged(self):
        image = np.full((20, 20, 3), (90, 140, 90), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)

        overlay = _segmentation_overlay(image, mask)

        assert np.array_equal(overlay, image)


def _mocked_pipeline() -> CropAnalysisPipeline:
    # Every sub-service mocked out -- these tests are about analyze()'s own
    # orchestration (does it call the right things with the right settings),
    # not the sub-services' real behavior (covered by their own test files).
    pipeline = CropAnalysisPipeline()
    pipeline.cv = MagicMock()
    pipeline.cv.load_and_preprocess.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
    pipeline.cv.vegetation_metrics.return_value = SimpleNamespace(
        green_mask=np.zeros((50, 50), dtype=np.uint8),
        coverage_percent=10.0, vegetation_score=20.0, health_score=30.0,
        heatmap=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    pipeline.detector = MagicMock()
    pipeline.detector.detect.return_value = []
    # Default coverage_percent (10.0, above FALLBACK_COVERAGE_THRESHOLD)
    # combined with an empty detector.detect() would otherwise trigger the
    # real (slow, network-loading) OpenVocabDetector on every test that
    # uses this fixture -- mocked here to also find nothing, keeping the
    # old "zero detections" behavior deterministic; TestFallbackDetection
    # below overrides this to exercise the real trigger logic.
    pipeline.fallback_detector = MagicMock()
    pipeline.fallback_detector.detect.return_value = []
    pipeline.segmenter = MagicMock()
    # Real SAMSegmenter.union_masks/refine semantics: [] of instance masks
    # unions to an all-empty coverage mask, same as the old default mock.
    pipeline.segmenter.segment_instances.return_value = []
    pipeline.segmenter.union_masks.side_effect = lambda instances, shape: np.zeros(shape, dtype=np.uint8)
    return pipeline


class TestPipelineSettings:
    def test_refine_segmentation_false_skips_sam(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, refine_segmentation=False)

        pipeline.segmenter.segment_instances.assert_not_called()

    def test_refine_segmentation_true_calls_sam(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, refine_segmentation=True)

        pipeline.segmenter.segment_instances.assert_called_once()

    def test_passes_enhance_and_conf_threshold_through(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0, 0.02, enhance=False, conf_threshold=0.4)

        _, preprocess_kwargs = pipeline.cv.load_and_preprocess.call_args
        assert preprocess_kwargs["enhance"] is False
        _, detect_kwargs = pipeline.detector.detect.call_args
        assert detect_kwargs["conf_threshold"] == 0.4


class TestTextureAffectsHealthScore:
    def test_patchy_texture_scores_lower_than_uniform_texture_at_same_color_health(self, tmp_path):
        # Same vegetation/coverage color signal in both cases -- only the
        # actual pixel texture differs (uniform green vs. real GLCM-patchy
        # noise) -- confirms texture is a real, non-token input to
        # health_score, not just decoration on the response.
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8))

        full_coverage_mask = np.full((80, 80), 255, dtype=np.uint8)

        uniform_pipeline = _mocked_pipeline()
        uniform_pipeline.cv.load_and_preprocess.return_value = np.full((80, 80, 3), (60, 150, 60), dtype=np.uint8)
        uniform_pipeline.segmenter.union_masks.side_effect = lambda instances, shape: full_coverage_mask
        uniform_result = uniform_pipeline.analyze(path, "Wheat", 2.0)

        rng = np.random.default_rng(0)
        noisy = np.full((80, 80, 3), (60, 150, 60), dtype=np.uint8)
        noise = rng.integers(-80, 80, size=noisy.shape).astype(np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        patchy_pipeline = _mocked_pipeline()
        patchy_pipeline.cv.load_and_preprocess.return_value = noisy
        patchy_pipeline.segmenter.union_masks.side_effect = lambda instances, shape: full_coverage_mask
        patchy_result = patchy_pipeline.analyze(path, "Wheat", 2.0)

        assert uniform_result.texture_pattern == "uniform"
        assert patchy_result.texture_pattern == "patchy"
        assert patchy_result.health_score < uniform_result.health_score

    def test_texture_fields_present_on_result(self, tmp_path):
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        assert result.texture_pattern == "mixed"  # empty mask -> neutral default
        assert result.texture_uniformity_score == 50.0


class TestTiltCorrectionWiring:
    def test_correct_tilt_false_skips_tilt_correction_entirely(self, tmp_path):
        pipeline = _mocked_pipeline()
        pipeline.tilt = MagicMock()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0, correct_tilt=False)

        pipeline.tilt.correct.assert_not_called()
        assert result.tilt_corrected is False
        assert "disabled" in result.tilt_correction_note.lower()

    def test_correct_tilt_true_by_default_calls_tilt_corrector(self, tmp_path):
        pipeline = _mocked_pipeline()
        pipeline.tilt = MagicMock()
        pipeline.tilt.correct.return_value = SimpleNamespace(
            corrected=False, image=np.zeros((50, 50, 3), dtype=np.uint8), note="Not enough visible line structure.",
        )
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.tilt.correct.assert_called_once()
        assert result.tilt_corrected is False
        assert result.tilt_correction_note == "Not enough visible line structure."

    def test_downstream_analysis_uses_the_corrected_image_when_tilt_was_corrected(self, tmp_path):
        # The whole point of running this before everything else: once a
        # correction is applied, vegetation/detection must see the
        # corrected geometry, not the original tilted photo.
        pipeline = _mocked_pipeline()
        corrected_image = np.full((64, 96, 3), (60, 150, 60), dtype=np.uint8)
        pipeline.segmenter.union_masks.side_effect = lambda instances, shape: np.zeros((64, 96), dtype=np.uint8)
        pipeline.tilt = MagicMock()
        pipeline.tilt.correct.return_value = SimpleNamespace(corrected=True, image=corrected_image, note="Perspective corrected.")
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        veg_call_image = pipeline.cv.vegetation_metrics.call_args[0][0]
        assert veg_call_image.shape == corrected_image.shape
        assert np.array_equal(veg_call_image, corrected_image)
        assert result.tilt_corrected is True
        assert result.image_width == 96 and result.image_height == 64
        assert result.tilt_correction_note == "Perspective corrected."


class TestPlantSizeStatsWiring:
    def test_refine_segmentation_false_means_no_plant_size_stats(self, tmp_path):
        # No per-instance masks exist without SAM refinement -- must not
        # fabricate size stats from plant_count alone.
        pipeline = _mocked_pipeline()
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0, refine_segmentation=False)

        assert result.plant_size_stats is None
        pipeline.segmenter.segment_instances.assert_not_called()

    def test_sam_unavailable_means_no_plant_size_stats(self, tmp_path):
        # segment_instances returning None (SAM unavailable / no
        # detections) must propagate to no stats, not an empty/zeroed one.
        pipeline = _mocked_pipeline()
        pipeline.segmenter.segment_instances.return_value = None
        pipeline.segmenter.union_masks.side_effect = lambda instances, shape: np.zeros(shape, dtype=np.uint8)
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0, refine_segmentation=True)

        assert result.plant_size_stats is None

    def test_real_instance_masks_produce_real_plant_size_stats(self, tmp_path):
        pipeline = _mocked_pipeline()
        h, w = 100, 100
        pipeline.cv.load_and_preprocess.return_value = np.zeros((h, w, 3), dtype=np.uint8)
        pipeline.cv.vegetation_metrics.return_value = SimpleNamespace(
            green_mask=np.zeros((h, w), dtype=np.uint8),
            coverage_percent=10.0, vegetation_score=20.0, health_score=30.0,
            heatmap=np.zeros((h, w, 3), dtype=np.uint8),
        )
        small = np.zeros((h, w), dtype=bool)
        cv2.circle(small.view(np.uint8), (20, 20), 8, 1, -1)
        large = np.zeros((h, w), dtype=bool)
        cv2.circle(large.view(np.uint8), (70, 70), 8, 1, -1)
        pipeline.segmenter.segment_instances.return_value = [small, large]
        pipeline.segmenter.union_masks.side_effect = lambda instances, shape: np.zeros(shape, dtype=np.uint8)
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))

        # area_ha=1 over a 100x100 frame -> exactly 1 cm^2/px, so the
        # returned areas are directly checkable against pixel counts.
        result = pipeline.analyze(path, "Wheat", area_ha=0.0001)

        assert result.plant_size_stats is not None
        assert result.plant_size_stats.plant_count == 2
        assert result.plant_size_stats.mean_area_cm2 > 0

    def test_yield_size_factor_reflects_measured_plant_size(self, tmp_path):
        # Same setup as above, but with plant_count actually matching the 2
        # instance masks (detector.detect() is [] by default in
        # _mocked_pipeline, which would make size_adjustment's plant_count
        # guard short-circuit to a neutral 1.0 regardless of the real
        # canopy-size math -- this confirms the pipeline threads
        # plant_size_stats/area_ha into YieldEstimator.size_adjustment for
        # real, not just computes plant_size_stats and discards it).
        pipeline = _mocked_pipeline()
        h, w = 100, 100
        pipeline.cv.load_and_preprocess.return_value = np.zeros((h, w, 3), dtype=np.uint8)
        pipeline.cv.vegetation_metrics.return_value = SimpleNamespace(
            green_mask=np.zeros((h, w), dtype=np.uint8),
            coverage_percent=10.0, vegetation_score=20.0, health_score=30.0,
            heatmap=np.zeros((h, w, 3), dtype=np.uint8),
        )
        pipeline.detector.detect.return_value = [_fake_detection(), _fake_detection()]
        small = np.zeros((h, w), dtype=bool)
        cv2.circle(small.view(np.uint8), (20, 20), 8, 1, -1)
        large = np.zeros((h, w), dtype=bool)
        cv2.circle(large.view(np.uint8), (70, 70), 8, 1, -1)
        pipeline.segmenter.segment_instances.return_value = [small, large]
        pipeline.segmenter.union_masks.side_effect = lambda instances, shape: np.zeros(shape, dtype=np.uint8)
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))

        # area_ha=0.0001 -> 1cm^2/px over this 100x100 frame, split across 2
        # plants -> 5,000cm^2 allotted each. Both circles measure ~200cm^2,
        # far under that -- the factor should be nudged down accordingly.
        result = pipeline.analyze(path, "Wheat", area_ha=0.0001)

        assert result.yield_size_factor < 1.0
        assert result.yield_size_note is not None
        assert "nudged down" in result.yield_size_note


def _fake_detection() -> Detection:
    return Detection(x1=1.0, y1=1.0, x2=10.0, y2=10.0, confidence=0.2, label="plant (general detector)")


class TestFallbackDetection:
    def test_fallback_triggers_when_fine_tuned_finds_nothing_and_coverage_is_real(self, tmp_path):
        pipeline = _mocked_pipeline()  # default coverage_percent=10.0, detector.detect returns []
        pipeline.fallback_detector.detect.return_value = [_fake_detection(), _fake_detection()]
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.fallback_detector.detect.assert_called_once()
        assert result.detection_method == "general_fallback"
        assert result.plant_count == 2
        assert "fine-tuned model found nothing" in result.detection_note

    def test_fallback_does_not_trigger_when_fine_tuned_finds_plants(self, tmp_path):
        pipeline = _mocked_pipeline()
        pipeline.detector.detect.return_value = [_fake_detection()]
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.fallback_detector.detect.assert_not_called()
        assert result.detection_method == "fine_tuned"

    def test_fallback_does_not_trigger_below_coverage_threshold(self, tmp_path):
        # Zero detections on a photo with barely any vegetation at all is
        # far more likely genuinely correct (bare soil) than a
        # domain-mismatch failure -- must not pay for the slow fallback
        # (or claim one ran) on every ordinary "no crop here" photo.
        pipeline = _mocked_pipeline()
        pipeline.cv.vegetation_metrics.return_value = SimpleNamespace(
            green_mask=np.zeros((50, 50), dtype=np.uint8),
            coverage_percent=2.0, vegetation_score=5.0, health_score=10.0,
            heatmap=np.zeros((50, 50, 3), dtype=np.uint8),
        )
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.fallback_detector.detect.assert_not_called()
        assert result.detection_method == "fine_tuned"
        assert result.plant_count == 0

    def test_detection_note_when_both_detectors_find_nothing(self, tmp_path):
        pipeline = _mocked_pipeline()  # fallback_detector.detect already mocked to return []
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        result = pipeline.analyze(path, "Wheat", 2.0)

        pipeline.fallback_detector.detect.assert_called_once()
        assert result.detection_method == "fine_tuned"  # nothing to fall back TO
        assert result.plant_count == 0
        assert "No plants detected by either" in result.detection_note

    def test_fallback_boxes_are_used_as_sam_prompts_too(self, tmp_path):
        # The fallback's boxes should flow through exactly like the
        # fine-tuned model's would -- SAM refinement, plant_count, density
        # -- not a special second-class path.
        pipeline = _mocked_pipeline()
        pipeline.fallback_detector.detect.return_value = [_fake_detection()]
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))

        pipeline.analyze(path, "Wheat", 2.0)

        passed_detections = pipeline.segmenter.segment_instances.call_args[0][1]
        assert len(passed_detections) == 1
        assert passed_detections[0].label == "plant (general detector)"
