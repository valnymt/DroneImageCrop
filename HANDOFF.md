# AgriSight — Session Handoff

Drone crop intelligence app. Next.js/vinext frontend (`app/page.tsx`, one file, ~600 lines) +
FastAPI computer-vision backend (`backend/`) + Cloudflare D1 for history persistence.
Repo: `https://github.com/valnymt/DroneImageCrop.git` (remote `origin`, on branch `main`).

## Current state

Working tree is clean. Everything through Phase N was pushed to `origin/main`. Phase O
(documentation-only, no code), Phase P (texture analysis), and Phase Q (flight comparison) are
committed locally, not yet pushed — push when the user asks for it.

## How to run it

**Frontend** (from repo root):
```powershell
npm run db:generate         # first time only, or after editing db/schema.ts
npx vinext dev               # NOT `npm run dev` -- breaks on Windows, see below
```
Then in a second terminal, once the dev server has started once: `npm run db:migrate:local`.

**Backend** (from `backend/`):
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --port 8000
```

**Gotchas hit this session (don't reintroduce):**
- `npm run dev` doesn't work on Windows — npm runs package.json scripts through `cmd.exe`,
  which chokes on the script's `VAR=value command` syntax. Use `npx vinext dev` directly.
- The dev-server-launching tool (`preview_start` equivalent) can get stuck launching the
  frontend bound to Node's debug port (9229) instead of 3000, or hit `ECONNRESET` during
  Vite's dependency bundling if a stale `HTTP_PROXY`/`HTTPS_PROXY` env var is set. If that
  happens, kill the stuck node processes and start directly: `HTTP_PROXY= HTTPS_PROXY= npx vinext dev`.
- Both dev servers routinely leave an **orphaned child process** when stopped (`npx vinext dev`
  spawns a real vinext server as a child; killing the wrapper doesn't kill the child — same
  pattern on the Python side with `--reload`'s multiprocessing workers). If clicks stop
  registering in the browser or a port bind fails with "address already in use", check
  `tasklist` for extra `node.exe`/`python.exe` beyond the expected one and kill them all before
  restarting.
- The backend loads real models (YOLO, MobileSAM, CLIP) lazily on first request — first
  `/inspect` or `/analyze` call after a restart is slow, subsequent ones are fast.

## Architecture

```
app/page.tsx (React, all views) --POST /api/analyze,/inspect,/report--> backend/ (FastAPI)
        |                                                                    |
        +--POST/GET /api/analyses--> app/api/analyses/route.ts --> D1 (db/schema.ts)
```
Backend never persists anything itself — the frontend persists results to D1 after a
successful `/analyze` call. Two independent processes; both must be running.

## What's been built, phase by phase

- **Phase A** — Crop-species classifier exploration (fine-tuned yolo11n-cls vs zero-shot CLIP).
  Fine-tuned version hit 100% accuracy via dataset-fingerprinting (not real), documented as a
  negative result. CLIP (68.9%, honest) is what's actually used. See
  `backend/training/CLASSIFIER_EVAL_REPORT.md` / `CLIP_CLASSIFIER_EVAL_REPORT.md`.
- **Phase B** — `POST /api/inspect`: CLIP crop-type suggestion + EXIF/XMP field-area estimate
  (`backend/app/services/crop_classifier.py`, `field_area_estimator.py`).
- **Phase C** — Wired `/inspect` into the upload flow (auto-fill suggestions, "Suggested: X —
  confirm or change" labels, silent fallback on failure).
- **Tiling fix** — YOLO was missing plants on large aerial photos because the model was
  fine-tuned exclusively on 640×640 images and a full-size upload got downscaled too far.
  Fixed with SAHI tiled inference (`backend/app/services/yolo_detector.py`). Verified with a
  before/after test: 3 detections → 1 on a large image without tiling, 4 with it.
- **Phase E** — Real PDF export (`report_generator.py`, `POST /api/report`) — draws actual
  detection boxes on the image, not a stub.
- **Phase F** — Made the Results panel honest: real bounding boxes (not hardcoded CSS
  positions), real segmentation overlay (magenta tint — green was invisible on green
  vegetation), real vegetation-density heatmap (TURBO colormap on Excess Green values).
  Added `image_width`/`image_height` to `AnalysisResult` — required because the frontend
  displays the *original* upload but detection coordinates are relative to the *preprocessed*
  (resized) image.
- **Phase G** — Real dashboard analytics: trend chart bucketed by month from actual history
  (SVG, not hardcoded dots), real period-over-period stat deltas (omitted, not faked, when
  there's no prior-period baseline), working Time period filter.
- **Phase H** — Real History search (by name/crop) + crop-type filter dropdown.
- **Phase I** — Made Settings real: enhancement toggle gates `opencv_processor`'s denoising,
  segmentation toggle gates whether `sam_segmenter.refine()` runs at all, model profile maps to
  real YOLO confidence thresholds (0.15/0.25/0.4), area unit is a real ha/acres conversion
  across Upload/Results/History, persisted to `localStorage`.

- **Phase J** — Removes the manual crop/area/yield form entirely; `/inspect`'s CLIP + EXIF/XMP
  prediction feeds `/analyze` directly with no confirmation step. `analyze()` awaits any
  in-flight inspection so it can't submit stale defaults while a prediction is still loading.
  Worth noting: this reverses Phase C's original design intent (CLIP is ~69% accurate, which is
  why Phase C deliberately made it a dismissible *suggestion* rather than authoritative) — Phase
  M (below) is the mitigation for that tradeoff.
- **Phase K** — Field-area estimation no longer depends on EXIF/XMP alone. Falls back, in
  order, to: a caller-supplied manual altitude (`/inspect`'s new `manual_altitude_m` field —
  the one last-resort input the UI shows only after a first call comes back "unavailable"), then
  a row-spacing heuristic that measures a genuinely repeating crop-row pattern via
  autocorrelation and converts it to area using a typical row spacing for the detected crop.
  The row-spacing detector requires both a real local-peak-with-prominence AND a matching
  harmonic near 2x that lag — an earlier FFT-magnitude version false-positived (0.001 ha) on an
  ordinary photo of grass with zero row structure; verified against every real photo in
  `images/` that all correctly now return "unavailable" instead.
- **Phase L** — `estimated_yield` no longer comes from a flat per-crop constant picked by name.
  `YieldEstimator` now owns one real per-plant-kg baseline per crop and applies it directly
  (folding away a previously redundant second crop-factor multiplier), still adjusted by the
  same coverage/health condition factor. `average_yield_per_plant_kg` is an optional override on
  `/analyze` now, not a required input; the resolved value comes back on `AnalysisResult` so
  the frontend persists what was actually used instead of a guess of its own.
- **Phase M** — Results now shows two confidence badges: CLIP's crop-type confidence, and how
  the field area was actually derived (measured altitude / manual altitude / row-spacing
  estimate / unmeasured default), each color-coded high/medium/low. Directly addresses Phase
  J's tradeoff — a wrong or low-confidence AI guess now reads differently from a verified one.
- **Phase N** — A collapsed "Adjust" link on Results lets the user correct a wrong crop-type or
  area guess after the fact. Only `crop_density`/`estimated_yield` recalculate (via a new
  `POST /api/recompute`, no image, no re-running YOLO/SAM/OpenCV — plant_count/coverage/health
  don't change just because the label was wrong); reuses `YieldEstimator`'s baseline table
  instead of duplicating it in the frontend a second time. A saved correction is shown at 100%
  confidence, "manually corrected by you."
- **Phase P** — Real GLCM/Haralick texture analysis (see the dedicated section above) as a
  second signal alongside color for health scoring — distinguishes uniformly-discolored fields
  (drought/nutrient stress) from patchy ones (disease/pest damage), which color alone can't.

All of Phases A–P have real backend + frontend tests and were verified live in a browser, not
just unit-tested (110 backend tests passing) — see individual commit-message-style descriptions
in the conversation this file came from if you need the detail.

## Phase O — Phase D zero-detection question, revisited (settled as far as it can be without the original file)

**Confirmed from the repo itself**: the fine-tuned YOLO checkpoint's training set
(`training/data/yolo/`, 823 train images) is a generic Roboflow "weed-crop-aerial" dataset —
classes are `weed-crop-aerial`/`crop`/`weed`, not tied to any specific crop species, and
nothing rice-specific. The only rice imagery anywhere in this repo
(`training/data/classify/train/Rice/`, used for the unrelated CLIP crop-type classifier) is
ground-level single-seedling close-ups, not aerial paddy shots — so there has never been any
rice-paddy-style imagery in this pipeline's training data at all.

**Diagnostic run** (per the prior handoff's own recommendation — `YOLODetector.detect()` at
`conf_threshold=0.01`, i.e. effectively unfiltered) against every real photo in this repo that
is *not* part of the fine-tuning distribution (`images/*.jpg`, plus two rice seedling
close-ups): max raw confidence per image was **0.01–0.05 in 7 of 9 cases**, only one image (a
real aerial soybean-farm photo) reaching 0.25. That is not "borderline misses clustering just
under the 0.15-0.25 threshold" — it's scores pinned at the noise floor, the signature of the
model having no learned response to that visual style at all. That is the same failure
signature Phase D observed on the user's rice photo (0 detections, confirmed not a
tiling/scale issue).

**What this settles**: the domain-mismatch explanation is no longer just plausible, it's
reproducible on other real out-of-distribution photos using data already in this repo.
**What it doesn't settle**: whether *that specific* rice-paddy photo would score in this same
near-zero range — that still needs the actual file (only screenshots were ever provided). If
the user provides it: `python -c "from app.services.yolo_detector import YOLODetector; import cv2; d=YOLODetector(); print([round(x.confidence,3) for x in d.detect(cv2.imread('PATH'), conf_threshold=0.01)])"` from `backend/`, with a score profile in this same 0.01-0.05 range confirming it definitively rather than by inference from other photos.

**If pursued further**: the real fix is a rice-inclusive training set (Roboflow Universe has
rice-paddy detection projects — see `training/README.md`'s "Pick a dataset" section), not a
threshold tweak; lowering `conf_threshold` further would not help since these aren't
near-threshold detections being missed, there's nothing there to threshold in.

## Phase P — GLCM/Haralick texture analysis, a real second signal beyond color

Health scoring was color-only, and the project's own RGB-screening disclaimer already admitted
it "cannot distinguish disease from drought, mature crops, harvest residue, shadows, or soil."
`backend/app/services/texture_analyzer.py` adds a genuinely classical-CV (not deep learning)
second signal: GLCM (gray-level co-occurrence matrix) computed via `scikit-image`, cropped to
the vegetation region's bounding box, averaged across 3 distances × 4 angles for a
scale/orientation-invariant reading. `homogeneity` + `energy` combine into a 0-100
`texture_uniformity_score` and a `"uniform" | "mixed" | "patchy"` label.

**Why this actually matters, not just decoration**: a uniformly discolored field (drought,
nutrient deficiency) keeps a *smooth* texture even as color health drops — disease or pest
damage tends to look *patchy* at the same color-based health. `pipeline.py`'s `health_score` is
now `0.40×vegetation_score + 0.35×coverage + 0.25×texture_uniformity` (previously
`0.55×vegetation + 0.45×coverage`, no texture input at all), and Results' recommendation text
branches on `texture_pattern` once health drops below 80 — patchy points at "inspect specific
irregular zones for disease/pests," uniform points at "check irrigation/nutrients field-wide."
Verified this is a real, non-token effect: same mocked color/coverage inputs, only the actual
image pixels differ between a uniform-texture and GLCM-patchy-noise test case, and
`patchy_result.health_score < uniform_result.health_score` (see
`tests/test_pipeline.py::TestTextureAffectsHealthScore`).

**Scope boundary**: `texture_uniformity_score`/`texture_pattern` are on `AnalysisResult` and
shown live on the Results screen (insight row + a texture-specific method-warning disclaimer +
the branched recommendation) and in the PDF report, but are **not** persisted to D1 history —
same boundary as the Phase M confidence badges, since `db/schema.ts` has no column for them and
adding one is a schema migration, not a CV feature.

10 new backend tests (7 for `TextureAnalyzer` in isolation, including that a noisy background
*outside* the vegetation mask doesn't leak into the reading; 2 for the pipeline wiring; plus
existing `AnalysisResult` fixtures updated for the two new required fields) — 110 backend tests
passing total. Verified end-to-end in the browser against a real photo with genuinely irregular
grass texture: correctly scored 23%/"patchy", visibly lowered `health_score` from what color
alone gave it, and the recommendation text correctly switched to the disease/pest-specific
message.

Also deleted `models/yolo/classifier.pt` (was never git-tracked, never loaded by any running
code — a leftover from the Phase A negative-result experiment; see `CLASSIFIER_EVAL_REPORT.md`
for why it was abandoned in favor of CLIP). Nothing wires it up; don't reintroduce it.

## Phase Q — flight-to-flight change detection (ORB + homography)

New standalone "Compare flights" view (its own nav item, not folded into History or Analyze).
`backend/app/services/flight_comparator.py`: ORB keypoint detection + Lowe's-ratio-filtered
BFMatcher + RANSAC homography aligns two photos of the same field taken at different times
(different camera angle/position tolerated — verified through an 8° rotation + 5% scale +
translation in tests), then diffs their vegetation masks directly: green overlay = new growth,
red = vegetation lost, dimmed = outside the overlap region (never actually compared, so not
shown as analyzed).

**Chose ORB over SIFT**: patent history aside (SIFT's has since expired), ORB is faster on CPU
and its rotation-invariant binary descriptors are already a comfortable match for two aerial
photos of the same field that share scale/orientation closely — no accuracy need here justifies
SIFT's extra cost.

**No image storage needed or added**: `db/schema.ts`'s `imagePath` column only ever stored a
filename string, never image bytes (confirmed before starting this) — so "compare against a
history entry" isn't actually wireable without a real storage layer (R2, out of scope). Scoped
instead as a direct two-photo upload tool, which needs nothing new on the persistence side.

**Alignment failure is a first-class, honest outcome, not an error to hide**: too few
distinguishing features, or good matches that don't agree on one consistent transform (low
RANSAC inlier ratio) both return `alignment_ok: false` with a specific reason — no diff is
shown, matching the project's existing "never fabricate a number" pattern (`field_area_estimator`'s "unavailable", `crop_classifier`'s confidence score). Verified this doesn't
just silently succeed on two genuinely unrelated real photos (5 matched features, correctly
rejected) and on two random-noise images (2 matched features, correctly rejected).

Small-patch filtering: connected-component filtering drops changed regions under ~0.15% of the
frame — otherwise single misaligned pixels at the warp's edge would read as "detected changes."
Caught and fixed a test case during verification where an intentionally-injected bare patch
was *correctly* filtered out for being just under that floor, then confirmed real detection
works once the patch is a realistic size (1.56% loss detected end-to-end through the actual
`/api/compare` HTTP layer, then again live in the browser with the diff overlay image rendering
at full resolution and stats matching the backend exactly).

9 new `FlightComparator` tests, 4 new `/api/compare` contract tests — 123 backend tests passing
total. `POST /api/compare` takes two images directly (`image_before`, `image_after`), no
history/database involved at all.

## Key files

| File | Purpose |
|---|---|
| `app/page.tsx` | Entire frontend — one file, all views/components |
| `app/globals.css` | All styling (Tailwind + one large custom CSS block) |
| `backend/app/services/pipeline.py` | Orchestrates the CV pipeline end to end |
| `backend/app/services/yolo_detector.py` | Detection + tiling logic |
| `backend/app/services/opencv_processor.py` | Vegetation indices, heatmap, preprocessing |
| `backend/app/services/texture_analyzer.py` | GLCM/Haralick texture uniformity (Phase P) |
| `backend/app/services/flight_comparator.py` | ORB + homography flight-to-flight diff (Phase Q) |
| `backend/app/services/sam_segmenter.py` | Box-prompted SAM refinement |
| `backend/app/services/report_generator.py` | PDF export |
| `backend/app/api/analysis.py` | All three HTTP endpoints |
| `backend/tests/` | 78 tests, run with `pytest` from `backend/` |
