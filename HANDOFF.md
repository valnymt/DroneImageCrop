# AgriSight — Session Handoff

Drone crop intelligence app. Next.js/vinext frontend (`app/page.tsx`, one file, ~600 lines) +
FastAPI computer-vision backend (`backend/`) + Cloudflare D1 for history persistence.
Repo: `https://github.com/valnymt/DroneImageCrop.git` (remote `origin`, on branch `main`).

## Current state

Working tree is clean. Everything through Phase I is committed and pushed to `origin/main`
(commit `cb23660`). **Phase J is also committed and pushed** (`127bb8c`) — see below; it was
made directly by the user outside this session, so it hasn't been tested/verified the way
Phases A–I were.

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

- **Phase J** *(not done in this session — see caveat above)* — Removes the manual
  crop/area/yield form entirely; `/inspect`'s CLIP + EXIF/XMP prediction feeds `/analyze`
  directly with no confirmation step. `analyze()` now awaits any in-flight inspection so it
  can't submit stale defaults while a prediction is still loading. Results still shows the
  AI's prediction read-only, so a wrong guess is visible rather than silently authoritative.
  Worth noting: this reverses Phase C's original design intent (CLIP is ~69% accurate, which
  is why Phase C deliberately made it a dismissible *suggestion* rather than authoritative) —
  if picking this back up, that tradeoff is worth understanding before extending it further.

All of Phases A–I have real backend + frontend tests (78 backend tests passing) and were
verified live in a browser, not just unit-tested — see individual commit-message-style
descriptions in the conversation this file came from if you need the detail. Phase J has
neither backend nor frontend test coverage and hasn't been browser-verified.

## One open item

**Phase D leftover**: the user's real rice-paddy test photo scored 0 plant detections even
after the tiling fix, because (confirmed via backend logs) the photo never triggered the tiled
path at all — it's small enough post-preprocessing that it took the single-pass path, same as
the in-distribution validation images that work fine. That means this specific photo's zero
result is **not** a scale/tiling problem — it's most likely genuine domain mismatch (the
fine-tuned YOLO checkpoint only ever saw 823 training images of one dataset's style). This was
never conclusively settled because the user never provided the actual image file — only
screenshots. If picking this back up: ask for the file, run `YOLODetector.detect()` on it
directly with the confidence threshold near zero to see the raw score distribution.

## Key files

| File | Purpose |
|---|---|
| `app/page.tsx` | Entire frontend — one file, all views/components |
| `app/globals.css` | All styling (Tailwind + one large custom CSS block) |
| `backend/app/services/pipeline.py` | Orchestrates the CV pipeline end to end |
| `backend/app/services/yolo_detector.py` | Detection + tiling logic |
| `backend/app/services/opencv_processor.py` | Vegetation indices, heatmap, preprocessing |
| `backend/app/services/sam_segmenter.py` | Box-prompted SAM refinement |
| `backend/app/services/report_generator.py` | PDF export |
| `backend/app/api/analysis.py` | All three HTTP endpoints |
| `backend/tests/` | 78 tests, run with `pytest` from `backend/` |
