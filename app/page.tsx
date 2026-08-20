"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type View = "dashboard" | "analyze" | "results" | "history" | "compare" | "settings";

type Settings = {
  areaUnit: "ha" | "acres";
  enhancement: boolean;
  segmentationRefinement: boolean;
  perspectiveCorrection: boolean;
  modelProfile: "sensitive" | "balanced" | "precise";
};

const DEFAULT_SETTINGS: Settings = {
  areaUnit: "ha",
  enhancement: true,
  segmentationRefinement: true,
  perspectiveCorrection: true,
  modelProfile: "balanced",
};

const SETTINGS_STORAGE_KEY = "agrisight-settings";

// Mirrors backend/app/services/yolo_detector.py's CONF_THRESHOLD (0.25,
// "balanced") -- sensitivity trades false negatives for false positives.
const MODEL_PROFILE_CONF: Record<Settings["modelProfile"], number> = {
  sensitive: 0.15,
  balanced: 0.25,
  precise: 0.4,
};

const ACRES_PER_HA = 2.47105;
const HA_PER_ACRE = 1 / ACRES_PER_HA;

// Only ever called client-side (useEffect / event handlers), never during
// the initial render, so this can't produce a server/client hydration
// mismatch -- the first render always uses DEFAULT_SETTINGS.
function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function persistSettings(settings: Settings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Best-effort -- e.g. private browsing with storage disabled. The
    // in-memory setting for this session still applies either way.
  }
}

type Detection = { x1: number; y1: number; x2: number; y2: number; confidence: number; label: string };

type AnalysisResult = {
  plant_count: number;
  crop_density: number;
  crop_coverage: number;
  vegetation_score: number;
  health_score: number;
  // GLCM/Haralick texture uniformity of the segmented canopy, independent
  // of color -- lets a uniformly discolored field (drought/nutrient
  // stress) read differently from a patchy one (disease/pest damage) at
  // the same color-based health score. Not persisted to D1 history (see
  // HistoryRow below) -- shown live on the just-computed Results screen
  // only, same scope boundary as the confidence badges (Phase M).
  texture_uniformity_score: number;
  texture_pattern: "uniform" | "mixed" | "patchy";
  // Whether a perspective/tilt correction ran before analysis -- false is
  // the common, expected case for an already-nadir photo, not a failure.
  // tilt_correction_note always explains what happened either way.
  tilt_corrected: boolean;
  tilt_correction_note: string;
  // Set only when tilt_corrected is true -- the frame detections/
  // image_width/image_height are actually relative to, which the app's
  // own original upload preview no longer matches at that point.
  analyzed_image: string | null;
  // Per-plant size/shape distribution from SAM's own instance masks --
  // null whenever segmentation refinement was off or SAM wasn't
  // available (the plain vegetation mask has no per-plant boundaries to
  // measure). A second axis of analysis beyond plant_count: two fields
  // with the same count and health score can still have very differently
  // distributed individual plants.
  plant_size_stats: {
    plant_count: number;
    mean_area_cm2: number;
    median_area_cm2: number;
    min_area_cm2: number;
    max_area_cm2: number;
    mean_aspect_ratio: number;
    size_uniformity_score: number;
  } | null;
  estimated_yield: number;
  // The per-plant yield the backend actually used to compute
  // estimated_yield (its own crop-specific baseline, health/coverage-
  // adjusted) -- not a value this app invents; persisted to history as-is.
  average_yield_per_plant_kg: number;
  confidence_score: number;
  detections: Detection[];
  // Pixel space `detections` coordinates are relative to -- the backend's
  // preprocessed/resized image, not necessarily the original upload's raw
  // dimensions. Needed to scale boxes onto the displayed photo correctly.
  image_width: number;
  image_height: number;
  segmentation_overlay: string;
  heatmap_overlay: string;
};

// Best-effort suggestions only -- see backend/app/services/schemas.py's
// InspectResult. crop_type is zero-shot CLIP (~69% accurate); never treat
// either field as authoritative.
type InspectResult = {
  crop_type: string;
  confidence: number;
  estimated_area_hectares: number | null;
  area_source: string;
};

type SessionMeta = {
  name: string;
  crop: string;
  date: string;
  image?: string;
  fieldAreaHectares: number;
  // Carried from the /inspect call straight into Results (see Phase M) so
  // a wrong AI guess is visible there, not just silently used -- null
  // means inspection never completed (e.g. network failure), which is
  // itself worth showing as "not verified" rather than hiding.
  cropConfidence: number | null;
  areaSource: string | null;
};

type Analysis = AnalysisResult & SessionMeta & { id: number };

// Row shape returned by the app's own D1-backed /api/analyses route (see
// db/schema.ts) -- this is what actually persists across a server restart,
// unlike the Python backend's per-process analyze() call.
type D1AnalysisRow = {
  id: number;
  createdAt: string;
  cropType: string;
  fieldSizeHectares: number;
  averageYieldPerPlantKg: number;
  plantCount: number;
  cropDensity: number;
  cropCoverage: number;
  vegetationScore: number;
  healthScore: number;
  estimatedYield: number;
  confidenceScore: number;
  imagePath: string | null;
  // Phases P/R/S -- nullable because rows written before this migration
  // (or by an older-shaped payload) genuinely don't have these, not
  // because they failed to compute.
  textureUniformityScore: number | null;
  texturePattern: string | null;
  tiltCorrected: boolean | null;
  plantSizeMeanAreaCm2: number | null;
  plantSizeUniformityScore: number | null;
  plantSizeMeanAspectRatio: number | null;
};

type HistoryRow = {
  id: number;
  date: string;
  createdAt: string;
  name: string;
  crop: string;
  plant_count: number;
  crop_density: number;
  crop_coverage: number;
  vegetation_score: number;
  health_score: number;
  estimated_yield: number;
  confidence_score: number;
  texture_pattern: string | null;
  tilt_corrected: boolean | null;
  plant_size_uniformity_score: number | null;
};

function toHistoryRow(row: D1AnalysisRow): HistoryRow {
  const createdAt = row.createdAt.replace(" ", "T") + "Z";
  return {
    id: row.id,
    date: new Date(createdAt).toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric",
    }),
    createdAt,
    name: `Analysis #${row.id}`,
    crop: row.cropType,
    plant_count: row.plantCount,
    crop_density: row.cropDensity,
    crop_coverage: row.cropCoverage,
    vegetation_score: row.vegetationScore,
    health_score: row.healthScore,
    estimated_yield: row.estimatedYield,
    confidence_score: row.confidenceScore,
    texture_pattern: row.texturePattern,
    tilt_corrected: row.tiltCorrected,
    plant_size_uniformity_score: row.plantSizeUniformityScore,
  };
}

type MonthBucket = { key: number; label: string; avgHealth: number };

// Months since epoch -- a single comparable integer, so buckets sort
// correctly across year boundaries without string-parsing tricks.
function monthKey(date: Date): number {
  return date.getFullYear() * 12 + date.getMonth();
}

// Only produces a bucket for months that actually have an analysis --
// months with no data are skipped entirely rather than shown as zero, so
// the trend line never implies a real (but unmeasured) health score.
function bucketHealthByMonth(records: HistoryRow[]): MonthBucket[] {
  const sums = new Map<number, { total: number; count: number; label: string }>();
  for (const r of records) {
    const d = new Date(r.createdAt);
    const key = monthKey(d);
    const entry = sums.get(key) ?? { total: 0, count: 0, label: d.toLocaleDateString("en-US", { month: "short" }).toUpperCase() };
    entry.total += r.health_score;
    entry.count += 1;
    sums.set(key, entry);
  }
  return Array.from(sums.entries())
    .map(([key, v]) => ({ key, label: v.label, avgHealth: v.total / v.count }))
    .sort((a, b) => a.key - b.key);
}

// Maps content coordinates (in contentWidth x contentHeight space) onto a
// container rendered with CSS object-fit:cover -- the content is scaled
// uniformly until it fills the container on one axis, then the overflow on
// the other axis is centered and clipped. Returns null until the container
// has been measured (avoids dividing by zero on first render).
function coverTransform(containerWidth: number, containerHeight: number, contentWidth: number, contentHeight: number) {
  if (!containerWidth || !containerHeight || !contentWidth || !contentHeight) return null;
  const contentAspect = contentWidth / contentHeight;
  const containerAspect = containerWidth / containerHeight;
  const renderedWidth = contentAspect > containerAspect ? containerHeight * contentAspect : containerWidth;
  const renderedHeight = contentAspect > containerAspect ? containerHeight : containerWidth / contentAspect;
  return {
    scaleX: renderedWidth / contentWidth,
    scaleY: renderedHeight / contentHeight,
    offsetX: (renderedWidth - containerWidth) / 2,
    offsetY: (renderedHeight - containerHeight) / 2,
  };
}

const API_BASE = "http://127.0.0.1:8000";

const NETWORK_ERROR =
  "Could not reach the analysis server. Start it with `uvicorn app.main:app --reload` from the backend directory, then try again.";

async function parseError(res: Response, fallback: string) {
  const body = await res.json().catch(() => null);
  return (body && typeof body.detail === "string" && body.detail) || fallback;
}

// Mirrors backend/app/services/yield_estimator.py's YIELD_PER_PLANT_KG
// keys -- used only to sanity-check CLIP's crop guess before trusting it
// as a display value. The actual per-plant yield now lives entirely on
// the backend (see AnalysisResult.average_yield_per_plant_kg); this app no
// longer keeps its own copy of that table.
const SUPPORTED_CROPS = ["Wheat", "Corn", "Rice", "Soybean", "Tomato"];

const nav = [
  ["dashboard", "⌂", "Overview"],
  ["analyze", "⌁", "New analysis"],
  ["history", "▤", "Field history"],
  ["compare", "⇄", "Compare flights"],
  ["settings", "⚙", "Settings"],
] as const;

export default function Home() {
  const [view, setView] = useState<View>("dashboard");
  const [records, setRecords] = useState<HistoryRow[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [crop, setCrop] = useState("Wheat");
  const [area, setArea] = useState("2");
  const [fieldName, setFieldName] = useState("West Field");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [latest, setLatest] = useState<Analysis | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [cropSuggested, setCropSuggested] = useState(false);
  const [areaSuggested, setAreaSuggested] = useState(false);
  // "unavailable" after a completed inspection means neither EXIF/XMP
  // altitude nor the row-spacing heuristic could place a scale on this
  // photo -- only then does the UI ask for the one last-resort field
  // (flight altitude) instead of silently keeping the stale default area.
  const [areaSource, setAreaSource] = useState<string | null>(null);
  const [manualAltitude, setManualAltitude] = useState("");
  // CLIP's own confidence in its crop-type guess (0-100) -- surfaced in
  // Results (Phase M) so a low-confidence guess is visibly flagged rather
  // than presented with the same certainty as a high-confidence one.
  const [cropConfidence, setCropConfidence] = useState<number | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const inputRef = useRef<HTMLInputElement>(null);
  const inspectRequestId = useRef(0);
  const inspectPromiseRef = useRef<Promise<void> | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading persisted preferences after mount, not during initial render
    setSettings(loadSettings());
  }, []);

  function saveSettings(next: Settings) {
    setSettings(next);
    persistSettings(next);
  }

  const totals = useMemo(() => ({
    plants: records.reduce((n, r) => n + r.plant_count, 0),
    yieldKg: records.reduce((n, r) => n + r.estimated_yield, 0),
    density: records.length ? Math.round(records.reduce((n, r) => n + r.crop_density, 0) / records.length) : 0,
  }), [records]);

  async function fetchHistory() {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch("/api/analyses");
      const body = await res.json().catch(() => null);
      if (!res.ok) throw new Error((body && body.error) || `Failed to load history (${res.status}).`);
      const rows: D1AnalysisRow[] = body;
      setRecords(rows.map(toHistoryRow));
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Failed to load history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load from the backend on mount
    fetchHistory();
  }, []);

  function applyCrop(value: string, suggested: boolean) {
    setCrop(value);
    setCropSuggested(suggested);
  }

  function applyArea(value: string, suggested: boolean) {
    setArea(value);
    setAreaSuggested(suggested);
  }

  function chooseFile(f?: File) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setCropSuggested(false);
    setAreaSuggested(false);
    setAreaSource(null);
    setManualAltitude("");
    setCropConfidence(null);
    inspectPromiseRef.current = inspectImage(f);
  }

  async function inspectImage(f: File, manualAltitudeM?: number) {
    const requestId = ++inspectRequestId.current;
    setInspecting(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const formData = new FormData();
      formData.append("image", f);
      if (manualAltitudeM) formData.append("manual_altitude_m", String(manualAltitudeM));
      const res = await fetch(`${API_BASE}/api/inspect`, { method: "POST", body: formData, signal: controller.signal });
      if (requestId !== inspectRequestId.current || !res.ok) return;
      const result: InspectResult = await res.json();
      if (requestId !== inspectRequestId.current) return;
      if (SUPPORTED_CROPS.includes(result.crop_type)) applyCrop(result.crop_type, true);
      if (result.estimated_area_hectares != null) applyArea(String(result.estimated_area_hectares), true);
      setAreaSource(result.area_source);
      setCropConfidence(result.confidence);
    } catch {
      // Best-effort only: network error, timeout/abort, or the backend
      // being down. Falls back to the last-known crop/area defaults so
      // /analyze still has values to submit -- there is no manual entry
      // to fall back to anymore.
    } finally {
      clearTimeout(timeout);
      if (requestId === inspectRequestId.current) setInspecting(false);
    }
  }

  // The one last-resort field: only surfaced after a completed inspection
  // came back with no usable ground scale from either EXIF/XMP or the
  // row-spacing heuristic. Re-runs inspection with the altitude supplied.
  function estimateWithManualAltitude() {
    const altitude = Number(manualAltitude);
    if (!file || !altitude || altitude <= 0) return;
    inspectPromiseRef.current = inspectImage(file, altitude);
  }

  async function runAnalysis() {
    if (!file) return;
    // AI inspection (crop type + field area) is the authoritative input to
    // /analyze now -- wait for whatever inspection pass is in flight for
    // the current file so we never submit stale/default values while a
    // real prediction is still loading.
    if (inspectPromiseRef.current) await inspectPromiseRef.current;
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("crop_type", crop);
      formData.append("field_size_hectares", area);
      // average_yield_per_plant_kg is intentionally NOT sent -- the backend
      // resolves its own crop-specific, health/coverage-adjusted per-plant
      // yield (see YieldEstimator) rather than this app supplying one.
      formData.append("enhance", String(settings.enhancement));
      formData.append("refine_segmentation", String(settings.segmentationRefinement));
      formData.append("correct_tilt", String(settings.perspectiveCorrection));
      formData.append("conf_threshold", String(MODEL_PROFILE_CONF[settings.modelProfile]));
      const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await parseError(res, `Analysis failed (${res.status}).`));
      const result: AnalysisResult = await res.json();
      const meta: SessionMeta = {
        name: fieldName || "Untitled field",
        crop,
        date: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
        image: preview,
        fieldAreaHectares: Number(area),
        cropConfidence,
        areaSource,
      };
      const analysis: Analysis = { id: Date.now(), ...result, ...meta };
      setLatest(analysis);
      setView("results");

      // Persist to the app's own D1-backed history so it survives a
      // server restart. Non-fatal if it fails -- the result above is
      // already shown either way, and history can be retried on its page.
      try {
        await fetch("/api/analyses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop_type: crop,
            field_size_hectares: Number(area),
            average_yield_per_plant_kg: result.average_yield_per_plant_kg,
            plant_count: result.plant_count,
            crop_density: result.crop_density,
            crop_coverage: result.crop_coverage,
            vegetation_score: result.vegetation_score,
            health_score: result.health_score,
            estimated_yield: result.estimated_yield,
            confidence_score: result.confidence_score,
            image_path: file.name,
            texture_uniformity_score: result.texture_uniformity_score,
            texture_pattern: result.texture_pattern,
            tilt_corrected: result.tilt_corrected,
            plant_size_mean_area_cm2: result.plant_size_stats?.mean_area_cm2 ?? null,
            plant_size_uniformity_score: result.plant_size_stats?.size_uniformity_score ?? null,
            plant_size_mean_aspect_ratio: result.plant_size_stats?.mean_aspect_ratio ?? null,
          }),
        });
        fetchHistory();
      } catch {
        // History persistence is best-effort; ignore.
      }
    } catch (err) {
      setAnalyzeError(err instanceof TypeError ? NETWORK_ERROR : err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setView("dashboard")}><span className="brandmark">A</span><span>AGRISIGHT<small>INTELLIGENCE</small></span></button>
        <nav>
          <p className="nav-label">WORKSPACE</p>
          {nav.map(([id, icon, label]) => <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}><i>{icon}</i>{label}</button>)}
        </nav>
        <div className="sidebar-foot">
          <div className="system"><span className="pulse" /><span><b>AI systems operational</b><small>YOLO · SAM · OpenCV</small></span></div>
          <div className="profile"><div>MK</div><span><b>Dr. May Khine</b><small>Agricultural Researcher</small></span><button>⋮</button></div>
        </div>
      </aside>

      <section className="content">
        <header><div><span className="eyebrow">DRONE CROP INTELLIGENCE</span><h1>{view === "dashboard" ? "Field overview" : view === "analyze" ? "Analyze a field" : view === "results" ? "Analysis complete" : view === "history" ? "Field history" : view === "compare" ? "Compare flights" : "System settings"}</h1></div><div className="header-actions"><button className="icon-btn">?</button><button className="icon-btn">♢</button><button className="primary compact" onClick={() => setView("analyze")}>＋ New analysis</button></div></header>

        {view === "dashboard" && <Dashboard records={records} totals={totals} onAnalyze={() => setView("analyze")} onHistory={() => setView("history")} onCompare={() => setView("compare")} />}
        {view === "analyze" && <Upload file={file} preview={preview} fieldName={fieldName} crop={crop} area={area} analyzing={analyzing} error={analyzeError} inspecting={inspecting} cropSuggested={cropSuggested} areaSuggested={areaSuggested} areaSource={areaSource} manualAltitude={manualAltitude} setManualAltitude={setManualAltitude} onEstimateAltitude={estimateWithManualAltitude} areaUnit={settings.areaUnit} inputRef={inputRef} onFile={chooseFile} setFieldName={setFieldName} run={runAnalysis} />}
        {view === "results" && latest && <Results data={latest} file={file} areaUnit={settings.areaUnit} onNew={() => setView("analyze")} onAdjust={(patch) => setLatest((prev) => prev ? { ...prev, ...patch } : prev)} />}
        {view === "history" && <History records={records} loading={historyLoading} error={historyError} onRetry={fetchHistory} areaUnit={settings.areaUnit} />}
        {view === "compare" && <Compare />}
        {view === "settings" && <Settings settings={settings} onSave={saveSettings} />}
      </section>
    </main>
  );
}

function Dashboard({ records, totals, onAnalyze, onHistory, onCompare }: { records: HistoryRow[]; totals: { plants: number; yieldKg: number; density: number }; onAnalyze: () => void; onHistory: () => void; onCompare: () => void }) {
  const [period, setPeriod] = useState<"3" | "6" | "12" | "all">("6");

  const thisMonth = monthKey(new Date());
  const lastMonth = thisMonth - 1;
  const farmsThisMonth = records.filter((r) => monthKey(new Date(r.createdAt)) === thisMonth).length;
  const plantsThisMonth = records.filter((r) => monthKey(new Date(r.createdAt)) === thisMonth).reduce((n, r) => n + r.plant_count, 0);
  const plantsLastMonth = records.filter((r) => monthKey(new Date(r.createdAt)) === lastMonth).reduce((n, r) => n + r.plant_count, 0);
  // Only a real baseline (at least one analysis last month) makes a % change
  // meaningful -- otherwise it's undefined, not zero, so it's left off
  // rather than shown as a fabricated "+0%" or "+12.4%".
  const plantsDeltaPct = plantsLastMonth > 0 ? ((plantsThisMonth - plantsLastMonth) / plantsLastMonth) * 100 : null;

  const stats = [
    ["FARMS ANALYZED", String(records.length), `+${farmsThisMonth} this month`, "↗"],
    ["PLANTS DETECTED", totals.plants.toLocaleString(), plantsDeltaPct === null ? "" : `${plantsDeltaPct >= 0 ? "+" : ""}${plantsDeltaPct.toFixed(1)}% vs last month`, "⌁"],
    ["AVG. CROP DENSITY", totals.density.toLocaleString(), "plants / hectare", "▦"],
    ["ESTIMATED HARVEST", `${(totals.yieldKg / 1000).toFixed(1)} t`, "Across all fields", "◒"],
  ];

  // Records written before Phases P/R/S (or by an older backend build)
  // genuinely have none of this -- the section is hidden entirely rather
  // than showing zeroed-out tiles that would misread as "nothing found".
  const withTexture = records.filter((r) => r.texture_pattern);
  const patchyCount = withTexture.filter((r) => r.texture_pattern === "patchy").length;
  const tiltCorrectedCount = records.filter((r) => r.tilt_corrected).length;
  const withSize = records.filter((r) => r.plant_size_uniformity_score != null);
  const avgSizeUniformity = withSize.length
    ? withSize.reduce((n, r) => n + (r.plant_size_uniformity_score ?? 0), 0) / withSize.length
    : null;
  const hasAiSignals = withTexture.length > 0 || tiltCorrectedCount > 0 || withSize.length > 0;
  const aiSignalStats = [
    ["PATCHY TEXTURE FLAGGED", withTexture.length ? `${patchyCount} / ${withTexture.length}` : "—", "possible disease/pest pattern", "◆"],
    ["PERSPECTIVE CORRECTED", String(tiltCorrectedCount), "non-nadir photos straightened", "⇕"],
    ["AVG. SIZE UNIFORMITY", avgSizeUniformity == null ? "—" : `${Math.round(avgSizeUniformity)}/100`, "stand establishment evenness", "⊞"],
  ];

  const allBuckets = bucketHealthByMonth(records);
  const cutoff = period === "all" ? -Infinity : thisMonth - (Number(period) - 1);
  const buckets = allBuckets.filter((b) => b.key >= cutoff);
  const points = buckets.map((b, i) => ({
    x: buckets.length > 1 ? (i / (buckets.length - 1)) * 100 : 50,
    y: 100 - b.avgHealth,
    label: b.label,
  }));
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x},${p.y}`).join(" ");
  const areaPath = points.length > 1
    ? `M ${points[0].x},100 L ${points.map((p) => `${p.x},${p.y}`).join(" L ")} L ${points[points.length - 1].x},100 Z`
    : "";

  return <div className="page">
    <section className="hero">
      <div><span className="tag"><span className="pulse" /> LIVE CROP MONITORING</span><h2>Turn aerial imagery into<br /><em>actionable crop insight.</em></h2><p>Analyze plant populations, vegetation health, crop coverage and projected harvests—powered by computer vision.</p><div><button className="primary" onClick={onAnalyze}>Analyze drone imagery <span>→</span></button><button className="ghost" onClick={onHistory}>View recent fields</button></div><button className="hero-compare-link" onClick={onCompare}>⇄ New: compare two flights to see what changed <span>→</span></button></div>
      <div className="field-card"><div className="field-image"><div className="scanline" /><span className="bbox b1">48</span><span className="bbox b2">92</span><span className="bbox b3">71</span><div className="map-label">◉ LIVE MODEL VIEW</div></div><div className="field-caption"><span><small>MODEL CONFIDENCE</small><b>94.8%</b></span><span><small>VEGETATION SIGNAL</small><b className="green">Strong</b></span></div></div>
    </section>
    <div className="section-title"><span>Portfolio performance</span><small>Updated moments ago</small></div>
    <section className="stat-grid">{stats.map(([label, value, sub, icon]) => <article className="stat" key={label}><div className="stat-icon">{icon}</div><small>{label}</small><strong>{value}</strong><p>{sub}</p></article>)}</section>
    {hasAiSignals && <><div className="section-title"><span>AI signal coverage</span><small>From texture, tilt, and per-plant size analysis</small></div>
    <section className="stat-grid ai-signal-grid">{aiSignalStats.map(([label, value, sub, icon]) => <article className="stat" key={label}><div className="stat-icon">{icon}</div><small>{label}</small><strong>{value}</strong><p>{sub}</p></article>)}</section></>}
    <section className="lower-grid">
      <article className="panel chart-panel"><div className="panel-head"><div><h3>Crop health trend</h3><p>Average health score across analyzed fields</p></div><select aria-label="Time period" value={period} onChange={(e) => setPeriod(e.target.value as "3" | "6" | "12" | "all")}><option value="3">Last 3 months</option><option value="6">Last 6 months</option><option value="12">Last 12 months</option><option value="all">All time</option></select></div>{points.length === 0 ? <p style={{fontSize: 10, color: "#7a837c", margin: "20px 0"}}>No analyses yet — run one to start tracking health trends.</p> : <div className="chart"><div className="yaxis"><span>100</span><span>75</span><span>50</span><span>25</span></div><div className="chart-area"><div className="gridlines" /><div className="trend-chart"><svg viewBox="0 0 100 100" preserveAspectRatio="none">{areaPath && <path d={areaPath} fill="#8cbc7640" stroke="none" />}{points.length > 1 && <path d={linePath} fill="none" stroke="#397452" strokeWidth={2} vectorEffect="non-scaling-stroke" />}</svg>{points.map((p, i) => <div key={i} className="point" style={{left: `${p.x}%`, top: `${p.y}%`}} />)}</div><div className="xaxis">{points.map((p, i) => <span key={i}>{p.label}</span>)}</div></div></div>}</article>
      <article className="panel recent"><div className="panel-head"><div><h3>Recent analyses</h3><p>Latest processed imagery</p></div><button onClick={onHistory}>View all →</button></div>{records.length === 0 && <p style={{fontSize: 10, color: "#7a837c"}}>No analyses yet.</p>}{records.slice(0, 3).map((r, i) => <div className="recent-row" key={r.id}><div className={`thumb t${i + 1}`} /><span><b>{r.name}</b><small>{r.crop} · {r.date}</small></span><div className="health-ring" style={{"--score": `${r.health_score * 3.6}deg`} as React.CSSProperties}>{Math.round(r.health_score)}</div></div>)}</article>
    </section>
  </div>;
}

function Upload(p: any) {
  // area (p.area) is always stored in hectares, matching the backend
  // contract -- only the displayed number and unit label switch with the
  // preference. Both crop and area are set by AI inspection (see
  // inspectImage/applyCrop/applyArea in Home) -- there is no manual input
  // for either anymore, this is read-only display of the prediction.
  const unit = p.areaUnit === "acres" ? "ac" : "ha";
  const toDisplayArea = (ha: string) =>
    ha !== "" && !Number.isNaN(Number(ha))
      ? String(Math.round(Number(ha) * (p.areaUnit === "acres" ? ACRES_PER_HA : 1) * 100) / 100)
      : ha;

  return <div className="page narrow"><div className="steps steps-2"><span className="done">1</span><i /><span>2</span><small>Upload imagery</small><small>AI analysis</small></div>
    <section className="upload-grid"><article className="panel upload-panel"><h3>Drone imagery</h3><p>Upload a high-resolution aerial image of your field.</p><div className={`dropzone ${p.preview ? "has-image" : ""}`} onClick={() => p.inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); p.onFile(e.dataTransfer.files[0]); }}>{p.preview ? <img src={p.preview} alt="Selected aerial field" /> : <><div className="upload-icon">↥</div><b>Drop your drone image here</b><span>or click to browse your files</span><small>JPG or PNG · Maximum 20 MB</small></>}{p.inspecting && <span className="inspect-status"><span className="spinner" /> AI is reading crop type and field area…</span>}<input ref={p.inputRef} hidden type="file" accept="image/png,image/jpeg" onChange={(e) => p.onFile(e.target.files?.[0])} /></div>{p.file && <div className="file-pill"><span>✓</span><div><b>{p.file.name}</b><small>{(p.file.size / 1048576).toFixed(2)} MB · Ready for analysis</small></div><button onClick={(e) => { e.stopPropagation(); p.inputRef.current?.click(); }}>Replace</button></div>}</article>
    <article className="panel form-panel"><h3>AI-assisted analysis</h3><p>No setup needed — crop type, field area and yield are predicted from the photo itself.</p><label>FIELD NAME (OPTIONAL)<input value={p.fieldName} onChange={(e) => p.setFieldName(e.target.value)} placeholder="e.g. West Field" /></label>{p.file && <div className="ai-predict-summary">{p.inspecting ? <span><span className="spinner" /> Detecting crop type and field area…</span> : <><span>AI detected: <b>{p.crop}</b> · <b>{toDisplayArea(p.area)} {unit}</b> (estimated)</span><small>Not sure this is right? You can adjust it from the Results screen after analysis.</small></>}</div>}{p.file && !p.inspecting && p.areaSource === "unavailable" && <div className="parameter-note"><b>Couldn't measure ground area from this photo</b><span>No altitude data and no visible row pattern to measure against. If you know the drone's flight altitude, enter it below for a more accurate area — otherwise a default area estimate is used.</span><div className="altitude-inline"><input type="number" min="1" step="1" placeholder="Flight altitude (m)" value={p.manualAltitude} onChange={(e) => p.setManualAltitude(e.target.value)} /><button type="button" className="ghost" onClick={p.onEstimateAltitude} disabled={!p.manualAltitude || Number(p.manualAltitude) <= 0}>Estimate</button></div></div>}<div className="tech-note"><b>⌁ RGB health analysis</b><span>The uploaded image is measured for green vegetation coverage using three independent RGB indices (Excess Green, VARI, ExGR) that must agree before a pixel counts as vegetation. Non-green areas are treated as bare soil or stressed ground, not classified by color. Results are visual indicators—not a laboratory diagnosis.</span></div>{p.error && <div className="tech-note error-note"><b>⚠ Analysis failed</b><span>{p.error}</span></div>}<button className="primary full" disabled={!p.file || p.analyzing || p.inspecting} onClick={p.run}>{p.analyzing ? <><span className="spinner" /> Running computer-vision pipeline…</> : <>Run image analysis <span>→</span></>}</button></article></section>
  </div>;
}

// Human-readable label + confidence tier for each area_source the backend
// can return (see backend/app/services/field_area_estimator.py) -- shown
// in Results so a fallback/default estimate reads differently from a real
// measurement, instead of both looking equally authoritative.
const AREA_SOURCE_INFO: Record<string, { label: string; tier: "high" | "medium" | "low" }> = {
  exif_gps_altitude: { label: "measured from the photo's GPS altitude", tier: "high" },
  exif_gps_altitude_default_focal: { label: "measured from GPS altitude (assumed lens)", tier: "medium" },
  xmp_relative_altitude: { label: "measured from the drone's recorded flight altitude", tier: "high" },
  xmp_relative_altitude_default_focal: { label: "measured from flight altitude (assumed lens)", tier: "medium" },
  manual_altitude: { label: "calculated from the altitude you entered", tier: "high" },
  manual_altitude_default_focal: { label: "calculated from your altitude (assumed lens)", tier: "medium" },
  row_spacing_estimate: { label: "estimated from visible crop-row spacing in the photo", tier: "medium" },
  // Not a real area_source from the backend -- set locally when the user
  // corrects the AI's guess via the Results "Adjust" panel (Phase N).
  manual_override: { label: "manually corrected by you", tier: "high" },
};

function confidenceTier(confidence: number | null): "high" | "medium" | "low" {
  if (confidence == null) return "low";
  if (confidence >= 70) return "high";
  if (confidence >= 40) return "medium";
  return "low";
}

type ResultsPatch = Pick<Analysis, "crop" | "fieldAreaHectares" | "crop_density" | "estimated_yield" | "average_yield_per_plant_kg" | "cropConfidence" | "areaSource">;

function Results({ data, file, areaUnit, onNew, onAdjust }: { data: Analysis; file: File | null; areaUnit: "ha" | "acres"; onNew: () => void; onAdjust: (patch: ResultsPatch) => void }) {
  const cropTier = confidenceTier(data.cropConfidence);
  const areaInfo = data.areaSource ? AREA_SOURCE_INFO[data.areaSource] : undefined;
  // No entry in AREA_SOURCE_INFO means "unavailable" (or inspection never
  // completed) -- area is a plain default, never measured from this photo.
  const areaTier = areaInfo?.tier ?? "low";
  const areaLabel = areaInfo?.label ?? "not measured from this photo — default estimate";
  const [adjustOpen, setAdjustOpen] = useState(false);
  const [draftCrop, setDraftCrop] = useState(data.crop);
  const [draftArea, setDraftArea] = useState(String(data.fieldAreaHectares));
  const [adjusting, setAdjusting] = useState(false);
  const [adjustError, setAdjustError] = useState<string | null>(null);

  function openAdjust() {
    setDraftCrop(data.crop);
    setDraftArea(String(data.fieldAreaHectares));
    setAdjustError(null);
    setAdjustOpen(true);
  }

  async function saveAdjust() {
    const areaHa = Number(draftArea);
    if (!draftCrop || !areaHa || areaHa <= 0) return;
    setAdjusting(true);
    setAdjustError(null);
    try {
      const formData = new FormData();
      formData.append("plant_count", String(data.plant_count));
      formData.append("crop_type", draftCrop);
      formData.append("field_size_hectares", String(areaHa));
      formData.append("coverage", String(data.crop_coverage));
      formData.append("health", String(data.health_score));
      const res = await fetch(`${API_BASE}/api/recompute`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await parseError(res, `Recalculation failed (${res.status}).`));
      const result: { crop_density: number; estimated_yield: number; average_yield_per_plant_kg: number } = await res.json();
      onAdjust({
        crop: draftCrop,
        fieldAreaHectares: areaHa,
        crop_density: result.crop_density,
        estimated_yield: result.estimated_yield,
        average_yield_per_plant_kg: result.average_yield_per_plant_kg,
        // A user-entered correction is ground truth as far as this app is
        // concerned -- shown at full confidence rather than re-running
        // CLIP/area estimation, which weren't wrong about this photo, the
        // user just knows better.
        cropConfidence: 100,
        areaSource: "manual_override",
      });
      setAdjustOpen(false);
    } catch (err) {
      setAdjustError(err instanceof TypeError ? NETWORK_ERROR : err instanceof Error ? err.message : "Recalculation failed.");
    } finally {
      setAdjusting(false);
    }
  }
  const densityValue = areaUnit === "acres" ? data.crop_density / ACRES_PER_HA : data.crop_density;
  const densityLabel = areaUnit === "acres" ? "plants / acre" : "plants / hectare";
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [tab, setTab] = useState<"detection" | "segmentation" | "heatmap">("detection");
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = imageContainerRef.current;
    if (!el) return;
    // Measure synchronously on mount rather than waiting on the observer's
    // first callback -- ResizeObserver callbacks are tied to the browser's
    // paint/composite cycle, which can be delayed (or, in some embedded/
    // headless contexts, never fire) even though layout itself is already
    // settled. clientWidth/clientHeight reflect layout immediately.
    const measure = () => setContainerSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const transform = coverTransform(containerSize.width, containerSize.height, data.image_width, data.image_height);
  // Per-box text labels get unreadable once a field has many detections --
  // above this, show outlines only and rely on the YOLO DETECTIONS badge
  // for the count.
  const showBoxLabels = data.detections.length <= 15;

  const healthLabel = data.health_score >= 80 ? "Healthy vegetation" : data.health_score >= 55 ? "Moderate / mixed health" : "Poor or stressed vegetation";
  const healthCopy = data.health_score >= 80
    ? "The image contains a strong, continuous green canopy."
    : data.health_score >= 55
      ? "Green crop is present, but gaps or stressed vegetation reduce the score."
      : "Low green coverage was detected, which can indicate exposed soil, sparse growth, or discolored vegetation.";
  // Texture only changes the recommendation once health has already
  // dropped -- a healthy field's texture pattern isn't actionable either
  // way. Below that: patchy texture (disease/pest-shaped) and uniform
  // texture (drought/nutrient-shaped) point at genuinely different causes
  // for the same color-based health score, which is the whole reason
  // texture analysis exists here rather than color alone.
  const recommendation = data.health_score >= 80
    ? "Maintain current management and compare the next flight for emerging changes."
    : data.texture_pattern === "patchy"
      ? "The patchy texture pattern in the affected areas is more consistent with disease or pest damage than a uniform stress -- inspect those irregular zones specifically, not just the low-vegetation areas overall."
      : data.texture_pattern === "uniform"
        ? "The texture stays uniform despite the reduced color health, which points toward a broad, even cause -- water stress, nutrient deficiency, or crop stage -- rather than disease. Check irrigation and soil conditions across the whole field."
        : data.health_score >= 55
          ? "Inspect bare or discolored zones for water stress, pests, disease, or establishment problems."
          : "Prioritize a field inspection. Confirm whether the image shows drought, lodging, harvest residue, disease, or bare soil.";

  async function exportReport() {
    if (!file) return;
    setExporting(true);
    setExportError(null);
    try {
      const resultPayload: AnalysisResult = {
        plant_count: data.plant_count,
        crop_density: data.crop_density,
        crop_coverage: data.crop_coverage,
        vegetation_score: data.vegetation_score,
        health_score: data.health_score,
        texture_uniformity_score: data.texture_uniformity_score,
        texture_pattern: data.texture_pattern,
        tilt_corrected: data.tilt_corrected,
        tilt_correction_note: data.tilt_correction_note,
        analyzed_image: data.analyzed_image,
        plant_size_stats: data.plant_size_stats,
        estimated_yield: data.estimated_yield,
        average_yield_per_plant_kg: data.average_yield_per_plant_kg,
        confidence_score: data.confidence_score,
        detections: data.detections,
        image_width: data.image_width,
        image_height: data.image_height,
        segmentation_overlay: data.segmentation_overlay,
        heatmap_overlay: data.heatmap_overlay,
      };
      const formData = new FormData();
      // The PDF draws detection boxes at result.detections' coordinates,
      // which are relative to the corrected frame once tilt correction
      // ran -- the original upload no longer matches that geometry, so
      // the corrected image has to go to the report generator instead.
      const reportImage = data.tilt_corrected && data.analyzed_image
        ? await (await fetch(data.analyzed_image)).blob()
        : file;
      formData.append("image", reportImage, file.name);
      formData.append("result", JSON.stringify(resultPayload));
      formData.append("field_name", data.name);
      formData.append("crop_type", data.crop);
      formData.append("field_area_hectares", String(data.fieldAreaHectares));
      formData.append("analysis_date", data.date);
      formData.append("health_label", healthLabel);
      formData.append("health_copy", healthCopy);
      formData.append("recommendation", recommendation);
      const res = await fetch(`${API_BASE}/api/report`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await parseError(res, `Report generation failed (${res.status}).`));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${data.name || "field"}-${data.date}.pdf`.replace(/\s+/g, "-");
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err instanceof TypeError ? NETWORK_ERROR : err instanceof Error ? err.message : "Report generation failed.");
    } finally {
      setExporting(false);
    }
  }

  return <div className="page"><div className="result-top"><div><span className="success">✓ ANALYSIS SUCCESSFUL</span><p>{data.name} · {data.crop} · {data.date}</p></div><div><button className="ghost" onClick={exportReport} disabled={exporting || !file}>{exporting ? <><span className="spinner spinner-dark" /> Generating…</> : "Export report"}</button><button className="primary compact" onClick={onNew}>Analyze another</button></div></div>
    {exportError && <div className="tech-note error-note"><b>⚠ Export failed</b><span>{exportError}</span></div>}
    <div className="ai-confidence-row">
      <div className={`confidence-badge tier-${cropTier}`}>
        <span className="confidence-dot" />
        <span><b>Crop type: {data.crop}</b><small>{data.cropConfidence == null ? "not verified — inspection didn't complete" : `${data.cropConfidence.toFixed(0)}% AI confidence`}</small></span>
      </div>
      <div className={`confidence-badge tier-${areaTier}`}>
        <span className="confidence-dot" />
        <span><b>Field area: {data.fieldAreaHectares.toLocaleString(undefined, {maximumFractionDigits: 2})} ha</b><small>{areaLabel}</small></span>
      </div>
    </div>
    {!adjustOpen && <button type="button" className="adjust-link" onClick={openAdjust}>Not sure this is right? Adjust crop type or area →</button>}
    {adjustOpen && <div className="adjust-panel">
      <b>Correct the AI's guess</b>
      <span>Only crop density and estimated yield are recalculated — detection, coverage, and health scores come from the image itself and don't change.</span>
      <div className="adjust-row">
        <label>CROP TYPE<select value={draftCrop} onChange={(e) => setDraftCrop(e.target.value)}>{SUPPORTED_CROPS.map((c) => <option key={c}>{c}</option>)}</select></label>
        <label>FIELD AREA (HA)<input type="number" min=".01" step=".01" value={draftArea} onChange={(e) => setDraftArea(e.target.value)} /></label>
      </div>
      {adjustError && <div className="tech-note error-note"><b>⚠ Recalculation failed</b><span>{adjustError}</span></div>}
      <div className="adjust-actions">
        <button type="button" className="ghost" onClick={() => setAdjustOpen(false)} disabled={adjusting}>Cancel</button>
        <button type="button" className="primary compact" onClick={saveAdjust} disabled={adjusting || !draftCrop || Number(draftArea) <= 0}>{adjusting ? <><span className="spinner" /> Recalculating…</> : "Save correction"}</button>
      </div>
    </div>}
    <section className="result-metrics">{[["PLANT COUNT", data.plant_count.toLocaleString(), "detected plants"], ["CROP DENSITY", densityValue.toLocaleString(undefined, {maximumFractionDigits: 2}), densityLabel], ["CROP COVERAGE", `${data.crop_coverage}%`, "segmented area"], ["HEALTH SCORE", `${Math.round(data.health_score)}/100`, "strong vegetation"], ["EST. HARVEST", `${data.estimated_yield.toLocaleString()} kg`, `${(data.estimated_yield / 1000).toFixed(2)} metric tons`]].map(([a,b,c]) => <article key={a}><small>{a}</small><b>{b}</b><span>{c}</span></article>)}</section>
    <section className="vision-grid"><article className="panel vision-panel"><div className="panel-head"><div><h3>Computer vision output</h3><p>Detection and segmentation layers{data.tilt_corrected ? " · perspective corrected" : ""}</p></div><div className="seg-tabs"><button className={tab === "detection" ? "selected" : ""} onClick={() => setTab("detection")}>Detection</button><button className={tab === "segmentation" ? "selected" : ""} onClick={() => setTab("segmentation")}>Segmentation</button><button className={tab === "heatmap" ? "selected" : ""} onClick={() => setTab("heatmap")}>Heatmap</button></div></div><div className="result-image" ref={imageContainerRef}>{tab === "detection" && (data.analyzed_image ?? data.image) && <img src={data.analyzed_image ?? data.image} alt="Analyzed crop field" />}{tab === "segmentation" && <img src={data.segmentation_overlay} alt="Segmentation mask overlay" />}{tab === "heatmap" && <img src={data.heatmap_overlay} alt="Vegetation density heatmap" />}{tab === "detection" && transform && data.detections.map((d, i) => <span key={i} className="bbox" style={{ left: d.x1 * transform.scaleX - transform.offsetX, top: d.y1 * transform.scaleY - transform.offsetY, width: (d.x2 - d.x1) * transform.scaleX, height: (d.y2 - d.y1) * transform.scaleY }}>{showBoxLabels ? `${d.label} .${Math.round(d.confidence * 100)}` : ""}</span>)}{tab === "detection" && <div className="model-badge">YOLO DETECTIONS · {data.plant_count.toLocaleString()}</div>}{tab === "detection" && data.tilt_corrected && <div className="model-badge tilt-badge">⇕ TILT CORRECTED</div>}</div></article>
    <article className={`panel insights health-${data.health_score >= 80 ? "good" : data.health_score >= 55 ? "mixed" : "poor"}`}><h3>Field intelligence</h3><p>Interpreted from visible RGB vegetation signals.</p><div className="score"><div className="score-ring" style={{"--score": `${data.health_score * 3.6}deg`} as React.CSSProperties}><span><b>{Math.round(data.health_score)}</b><small>/ 100</small></span></div><div><b>{healthLabel}</b><p>{healthCopy}</p></div></div><hr /><div className="insight-row"><span>GREEN VEGETATION RATIO</span><b>{data.vegetation_score.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${data.vegetation_score}%`}} /></div><div className="insight-row"><span>TEXTURE PATTERN</span><b>{data.texture_pattern} · {data.texture_uniformity_score.toFixed(0)}%</b></div><div className={`bar texture-${data.texture_pattern}`}><i style={{width: `${data.texture_uniformity_score}%`}} /></div><div className="insight-row"><span>AVG. DETECTION CONFIDENCE</span><b>{data.confidence_score.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${data.confidence_score}%`}} /></div><div className="method-warning"><b>RGB screening result</b><span>Color analysis can flag suspicious areas, but cannot distinguish disease from drought, mature crops, harvest residue, shadows, or soil without field context.</span></div><div className="method-warning"><b>Texture screening result</b><span>GLCM/Haralick texture on the segmented canopy separates uniform condition changes (drought, nutrient stress) from patchy ones (disease, pest damage) -- it flags a pattern, not a diagnosis.</span></div><div className="recommend"><b>Recommendation</b><p>{recommendation}</p></div></article></section>
    {data.plant_size_stats && <section className="panel plant-size-panel"><h3>Per-plant size &amp; shape</h3><p>Measured from SAM's own per-plant masks, not just plant count -- a second axis of analysis independent of health/texture.</p>
      <div className="plant-size-grid">
        <article><small>MEAN CANOPY AREA</small><b>{data.plant_size_stats.mean_area_cm2.toLocaleString(undefined, {maximumFractionDigits: 0})} cm²</b><span>median {data.plant_size_stats.median_area_cm2.toLocaleString(undefined, {maximumFractionDigits: 0})} cm²</span></article>
        <article><small>SIZE RANGE</small><b>{data.plant_size_stats.min_area_cm2.toLocaleString(undefined, {maximumFractionDigits: 0})}–{data.plant_size_stats.max_area_cm2.toLocaleString(undefined, {maximumFractionDigits: 0})} cm²</b><span>smallest to largest plant</span></article>
        <article><small>SIZE UNIFORMITY</small><b>{data.plant_size_stats.size_uniformity_score.toFixed(0)}/100</b><span>{data.plant_size_stats.size_uniformity_score >= 70 ? "even stand establishment" : data.plant_size_stats.size_uniformity_score >= 40 ? "moderately uneven sizing" : "highly uneven sizing"}</span></article>
        <article><small>MEAN ELONGATION</small><b>{data.plant_size_stats.mean_aspect_ratio.toFixed(2)}×</b><span>{data.plant_size_stats.mean_aspect_ratio <= 1.3 ? "compact / round canopy" : "narrow / elongated canopy"}</span></article>
      </div>
      <div className="method-warning"><b>Size/shape screening result</b><span>Low size uniformity across an otherwise healthy-looking field can point at uneven emergence timing, plant competition, or patchy stress that an averaged health score alone wouldn't show.</span></div>
    </section>}
  </div>;
}

function History({ records, loading, error, onRetry, areaUnit }: { records: HistoryRow[]; loading: boolean; error: string | null; onRetry: () => void; areaUnit: "ha" | "acres" }) {
  const [search, setSearch] = useState("");
  const [cropFilter, setCropFilter] = useState("all");
  const densityHeader = areaUnit === "acres" ? "DENSITY / ACRE" : "DENSITY / HA";
  const toDisplayDensity = (perHa: number) => (areaUnit === "acres" ? perHa / ACRES_PER_HA : perHa).toLocaleString(undefined, {maximumFractionDigits: 2});

  const cropOptions = Array.from(new Set(records.map((r) => r.crop))).sort();
  const query = search.trim().toLowerCase();
  const filtered = records.filter((r) =>
    (cropFilter === "all" || r.crop === cropFilter) &&
    (!query || r.name.toLowerCase().includes(query) || r.crop.toLowerCase().includes(query))
  );
  const isFiltered = query !== "" || cropFilter !== "all";

  return <div className="page"><section className="panel history"><div className="panel-head"><div><h3>Analysis archive</h3><p>{loading ? "Loading…" : error ? "Unable to load history" : isFiltered ? `${filtered.length} of ${records.length} processed field surveys` : `${records.length} processed field surveys`}</p></div><div><input placeholder="Search fields…" value={search} onChange={(e) => setSearch(e.target.value)} /><select aria-label="Filter by crop" value={cropFilter} onChange={(e) => setCropFilter(e.target.value)}><option value="all">All crops</option>{cropOptions.map((c) => <option key={c} value={c}>{c}</option>)}</select></div></div>
    {error && <div className="tech-note error-note"><b>⚠ {error}</b><span><button className="ghost" onClick={onRetry}>Retry</button></span></div>}
    {!error && <div className="table"><div className="tr th"><span>FIELD</span><span>DATE</span><span>PLANTS</span><span>{densityHeader}</span><span>HEALTH</span><span>EST. YIELD</span><span>SIGNALS</span></div>
    {loading && <div className="tr"><span>Loading analysis history…</span></div>}
    {!loading && records.length === 0 && <div className="tr"><span>No analyses yet. Run one from “New analysis”.</span></div>}
    {!loading && records.length > 0 && filtered.length === 0 && <div className="tr"><span>No analyses match your search or filter.</span></div>}
    {!loading && filtered.map((r, i) => <div className="tr" key={r.id}><span className="field-cell"><i className={`thumb t${i % 3 + 1}`} /><span><b>{r.name}</b><small>{r.crop}</small></span></span><span>{r.date}</span><span>{r.plant_count.toLocaleString()}</span><span>{toDisplayDensity(r.crop_density)}</span><span><em className="health-pill">● {Math.round(r.health_score)}</em></span><span><b>{r.estimated_yield.toLocaleString()} kg</b></span><span className="signals-cell">{r.texture_pattern && <em className={`signal-chip texture-${r.texture_pattern}`} title={`Texture: ${r.texture_pattern}`}>{r.texture_pattern}</em>}{r.tilt_corrected && <em className="signal-chip tilt" title="Perspective corrected">⇕</em>}{r.plant_size_uniformity_score != null && <em className="signal-chip size" title={`Size uniformity: ${Math.round(r.plant_size_uniformity_score)}/100`}>⊞ {Math.round(r.plant_size_uniformity_score)}</em>}{!r.texture_pattern && !r.tilt_corrected && r.plant_size_uniformity_score == null && <small className="field-help" style={{margin: 0}}>—</small>}</span></div>)}</div>}
  </section></div>;
}

type ComparisonResult = {
  alignment_ok: boolean;
  keypoints_matched: number;
  inlier_ratio: number;
  growth_percent: number;
  loss_percent: number;
  unchanged_percent: number;
  diff_overlay: string;
  warning: string | null;
};

function CompareDropzone({ label, hint, preview, onFile }: { label: string; hint: string; preview: string; onFile: (f?: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  return <article className="panel upload-panel compare-slot"><h3>{label}</h3><p>{hint}</p>
    <div className={`dropzone ${preview ? "has-image" : ""}`} onClick={() => inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0]); }}>
      {preview ? <img src={preview} alt={`${label} field photo`} /> : <><div className="upload-icon">↥</div><b>Drop photo here</b><span>or click to browse</span></>}
      <input ref={inputRef} hidden type="file" accept="image/png,image/jpeg" onChange={(e) => onFile(e.target.files?.[0])} />
    </div>
  </article>;
}

function Compare() {
  const [before, setBefore] = useState<File | null>(null);
  const [beforePreview, setBeforePreview] = useState("");
  const [after, setAfter] = useState<File | null>(null);
  const [afterPreview, setAfterPreview] = useState("");
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResult | null>(null);

  function chooseBefore(f?: File) {
    if (!f || !f.type.startsWith("image/")) return;
    setBefore(f);
    setBeforePreview(URL.createObjectURL(f));
    setResult(null);
  }
  function chooseAfter(f?: File) {
    if (!f || !f.type.startsWith("image/")) return;
    setAfter(f);
    setAfterPreview(URL.createObjectURL(f));
    setResult(null);
  }

  async function runCompare() {
    if (!before || !after) return;
    setComparing(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("image_before", before);
      formData.append("image_after", after);
      const res = await fetch(`${API_BASE}/api/compare`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await parseError(res, `Comparison failed (${res.status}).`));
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof TypeError ? NETWORK_ERROR : err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setComparing(false);
    }
  }

  return <div className="page narrow">
    <section className="upload-grid compare-grid">
      <CompareDropzone label="Before" hint="An earlier flight over the same field." preview={beforePreview} onFile={chooseBefore} />
      <CompareDropzone label="After" hint="A more recent flight over the same field." preview={afterPreview} onFile={chooseAfter} />
    </section>
    <div className="tech-note"><b>⇄ How this works</b><span>ORB feature matching finds points the two photos share, then a homography aligns the "before" photo onto the "after" photo's frame -- correcting for the drone not flying the exact same path twice. The aligned vegetation masks are then diffed directly: green marks new growth, red marks vegetation lost between the two flights.</span></div>
    {error && <div className="tech-note error-note"><b>⚠ Comparison failed</b><span>{error}</span></div>}
    <button className="primary full" disabled={!before || !after || comparing} onClick={runCompare}>{comparing ? <><span className="spinner" /> Aligning photos and diffing vegetation…</> : <>Compare flights <span>→</span></>}</button>

    {result && !result.alignment_ok && <div className="tech-note error-note" style={{marginTop: 18}}><b>⚠ Couldn't align these photos</b><span>{result.warning ?? "The two photos could not be confidently aligned."} Try two photos that clearly show the same field from a similar angle.</span></div>}

    {result && result.alignment_ok && <section className="compare-results">
      <article className="panel vision-panel"><div className="panel-head"><div><h3>Change since last flight</h3><p>{result.keypoints_matched} matched features · {result.inlier_ratio.toFixed(0)}% alignment confidence</p></div></div>
        <div className="result-image"><img src={result.diff_overlay} alt="Vegetation change diff overlay" /><div className="model-badge">◈ CHANGE OVERLAY</div></div>
        <div className="compare-legend"><span><i className="legend-dot grow" /> New growth</span><span><i className="legend-dot lose" /> Vegetation lost</span><span><i className="legend-dot dim" /> Outside overlap (not compared)</span></div>
      </article>
      <article className="panel insights"><h3>Change summary</h3><p>Measured within the region visible in both photos after alignment.</p><hr />
        <div className="insight-row"><span>NEW GROWTH</span><b>{result.growth_percent.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${result.growth_percent}%`, background: "#2f704c"}} /></div>
        <div className="insight-row"><span>VEGETATION LOST</span><b>{result.loss_percent.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${result.loss_percent}%`, background: "#b64c3c"}} /></div>
        <div className="insight-row"><span>UNCHANGED</span><b>{result.unchanged_percent.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${result.unchanged_percent}%`}} /></div>
        <div className="method-warning"><b>Alignment method</b><span>ORB keypoints + RANSAC homography. {result.keypoints_matched} inlier matches at {result.inlier_ratio.toFixed(0)}% agreement -- higher agreement means the alignment (and therefore the diff) is more trustworthy.</span></div>
      </article>
    </section>}
  </div>;
}

function Settings({ settings, onSave }: { settings: Settings; onSave: (next: Settings) => void }) {
  const [draft, setDraft] = useState<Settings>(settings);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- resyncs the draft if settings load asynchronously after this view is already mounted
    setDraft(settings);
  }, [settings]);

  const isDirty = JSON.stringify(draft) !== JSON.stringify(settings);

  return <div className="page narrow"><section className="panel settings"><h3>Analysis defaults</h3><p>Configure how new field surveys are interpreted.</p><label>DEFAULT AREA UNIT<select value={draft.areaUnit} onChange={(e) => setDraft({...draft, areaUnit: e.target.value as Settings["areaUnit"]})}><option value="ha">Hectares (ha)</option><option value="acres">Acres</option></select></label><label>MODEL PROFILE<select value={draft.modelProfile} onChange={(e) => setDraft({...draft, modelProfile: e.target.value as Settings["modelProfile"]})}><option value="balanced">Balanced · Recommended</option><option value="sensitive">High sensitivity</option><option value="precise">High precision</option></select><small className="field-help">Sensitivity trades false negatives for false positives -- high sensitivity flags more candidate plants but with more mistakes; high precision is stricter and undercounts sparse fields.</small></label><div className="toggle-row"><span><b>Automatic image enhancement</b><small>Apply denoising and histogram normalization before detection.</small></span><button className={draft.enhancement ? "toggle on" : "toggle"} aria-label="Toggle enhancement" onClick={() => setDraft({...draft, enhancement: !draft.enhancement})}><i /></button></div><div className="toggle-row"><span><b>Segmentation refinement</b><small>Use SAM masks to refine crop coverage estimates.</small></span><button className={draft.segmentationRefinement ? "toggle on" : "toggle"} aria-label="Toggle segmentation" onClick={() => setDraft({...draft, segmentationRefinement: !draft.segmentationRefinement})}><i /></button></div><div className="toggle-row"><span><b>Perspective correction</b><small>Straighten non-nadir (tilted/handheld) photos using detected row lines before analysis.</small></span><button className={draft.perspectiveCorrection ? "toggle on" : "toggle"} aria-label="Toggle perspective correction" onClick={() => setDraft({...draft, perspectiveCorrection: !draft.perspectiveCorrection})}><i /></button></div><button className="primary compact" disabled={!isDirty} onClick={() => onSave(draft)}>{isDirty ? "Save preferences" : "Saved ✓"}</button></section></div>;
}
