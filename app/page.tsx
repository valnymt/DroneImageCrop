"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type View = "dashboard" | "analyze" | "results" | "history" | "settings";

type Detection = { x1: number; y1: number; x2: number; y2: number; confidence: number; label: string };

type AnalysisResult = {
  plant_count: number;
  crop_density: number;
  crop_coverage: number;
  vegetation_score: number;
  health_score: number;
  estimated_yield: number;
  confidence_score: number;
  detections: Detection[];
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

type SessionMeta = { name: string; crop: string; date: string; image?: string };

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
};

type HistoryRow = {
  id: number;
  date: string;
  name: string;
  crop: string;
  plant_count: number;
  crop_density: number;
  crop_coverage: number;
  vegetation_score: number;
  health_score: number;
  estimated_yield: number;
  confidence_score: number;
};

function toHistoryRow(row: D1AnalysisRow): HistoryRow {
  return {
    id: row.id,
    date: new Date(row.createdAt.replace(" ", "T") + "Z").toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric",
    }),
    name: `Analysis #${row.id}`,
    crop: row.cropType,
    plant_count: row.plantCount,
    crop_density: row.cropDensity,
    crop_coverage: row.cropCoverage,
    vegetation_score: row.vegetationScore,
    health_score: row.healthScore,
    estimated_yield: row.estimatedYield,
    confidence_score: row.confidenceScore,
  };
}

const API_BASE = "http://127.0.0.1:8000";

const NETWORK_ERROR =
  "Could not reach the analysis server. Start it with `uvicorn app.main:app --reload` from the backend directory, then try again.";

async function parseError(res: Response, fallback: string) {
  const body = await res.json().catch(() => null);
  return (body && typeof body.detail === "string" && body.detail) || fallback;
}

const yieldDefaults: Record<string, number> = {
  Wheat: 0.02, Corn: 0.18, Rice: 0.025, Soybean: 0.015, Tomato: 3,
};

const nav = [
  ["dashboard", "⌂", "Overview"],
  ["analyze", "⌁", "New analysis"],
  ["history", "▤", "Field history"],
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
  const [avgYield, setAvgYield] = useState(String(yieldDefaults.Wheat));
  const [fieldName, setFieldName] = useState("West Field");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [latest, setLatest] = useState<Analysis | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [cropSuggested, setCropSuggested] = useState(false);
  const [areaSuggested, setAreaSuggested] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const inspectRequestId = useRef(0);

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
    setAvgYield(String(yieldDefaults[value]));
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
    inspectImage(f);
  }

  async function inspectImage(f: File) {
    const requestId = ++inspectRequestId.current;
    setInspecting(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const formData = new FormData();
      formData.append("image", f);
      const res = await fetch(`${API_BASE}/api/inspect`, { method: "POST", body: formData, signal: controller.signal });
      if (requestId !== inspectRequestId.current || !res.ok) return;
      const result: InspectResult = await res.json();
      if (requestId !== inspectRequestId.current) return;
      if (result.crop_type in yieldDefaults) applyCrop(result.crop_type, true);
      if (result.estimated_area_hectares != null) applyArea(String(result.estimated_area_hectares), true);
    } catch {
      // Best-effort only: network error, timeout/abort, or the backend
      // being down. The form works fine with manual entry -- fail silent,
      // no error surfaced (this call never blocks /analyze).
    } finally {
      clearTimeout(timeout);
      if (requestId === inspectRequestId.current) setInspecting(false);
    }
  }

  async function runAnalysis() {
    if (!file) return;
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("crop_type", crop);
      formData.append("field_size_hectares", area);
      formData.append("average_yield_per_plant_kg", avgYield);
      const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await parseError(res, `Analysis failed (${res.status}).`));
      const result: AnalysisResult = await res.json();
      const meta: SessionMeta = {
        name: fieldName || "Untitled field",
        crop,
        date: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
        image: preview,
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
            average_yield_per_plant_kg: Number(avgYield),
            plant_count: result.plant_count,
            crop_density: result.crop_density,
            crop_coverage: result.crop_coverage,
            vegetation_score: result.vegetation_score,
            health_score: result.health_score,
            estimated_yield: result.estimated_yield,
            confidence_score: result.confidence_score,
            image_path: file.name,
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
        <header><div><span className="eyebrow">DRONE CROP INTELLIGENCE</span><h1>{view === "dashboard" ? "Field overview" : view === "analyze" ? "Analyze a field" : view === "results" ? "Analysis complete" : view === "history" ? "Field history" : "System settings"}</h1></div><div className="header-actions"><button className="icon-btn">?</button><button className="icon-btn">♢</button><button className="primary compact" onClick={() => setView("analyze")}>＋ New analysis</button></div></header>

        {view === "dashboard" && <Dashboard records={records} totals={totals} onAnalyze={() => setView("analyze")} onHistory={() => setView("history")} />}
        {view === "analyze" && <Upload file={file} preview={preview} fieldName={fieldName} crop={crop} area={area} avgYield={avgYield} analyzing={analyzing} error={analyzeError} inspecting={inspecting} cropSuggested={cropSuggested} areaSuggested={areaSuggested} inputRef={inputRef} onFile={chooseFile} setFieldName={setFieldName} setCrop={(value: string) => applyCrop(value, false)} setArea={(value: string) => applyArea(value, false)} setAvgYield={setAvgYield} run={runAnalysis} />}
        {view === "results" && latest && <Results data={latest} onNew={() => setView("analyze")} />}
        {view === "history" && <History records={records} loading={historyLoading} error={historyError} onRetry={fetchHistory} />}
        {view === "settings" && <Settings />}
      </section>
    </main>
  );
}

function Dashboard({ records, totals, onAnalyze, onHistory }: { records: HistoryRow[]; totals: { plants: number; yieldKg: number; density: number }; onAnalyze: () => void; onHistory: () => void }) {
  const stats = [
    ["FARMS ANALYZED", String(records.length), "+1 this month", "↗"],
    ["PLANTS DETECTED", totals.plants.toLocaleString(), "+12.4% vs last period", "⌁"],
    ["AVG. CROP DENSITY", totals.density.toLocaleString(), "plants / hectare", "▦"],
    ["ESTIMATED HARVEST", `${(totals.yieldKg / 1000).toFixed(1)} t`, "Across all fields", "◒"],
  ];
  return <div className="page">
    <section className="hero">
      <div><span className="tag"><span className="pulse" /> LIVE CROP MONITORING</span><h2>Turn aerial imagery into<br /><em>actionable crop insight.</em></h2><p>Analyze plant populations, vegetation health, crop coverage and projected harvests—powered by computer vision.</p><div><button className="primary" onClick={onAnalyze}>Analyze drone imagery <span>→</span></button><button className="ghost" onClick={onHistory}>View recent fields</button></div></div>
      <div className="field-card"><div className="field-image"><div className="scanline" /><span className="bbox b1">48</span><span className="bbox b2">92</span><span className="bbox b3">71</span><div className="map-label">◉ LIVE MODEL VIEW</div></div><div className="field-caption"><span><small>MODEL CONFIDENCE</small><b>94.8%</b></span><span><small>VEGETATION SIGNAL</small><b className="green">Strong</b></span></div></div>
    </section>
    <div className="section-title"><span>Portfolio performance</span><small>Updated moments ago</small></div>
    <section className="stat-grid">{stats.map(([label, value, sub, icon]) => <article className="stat" key={label}><div className="stat-icon">{icon}</div><small>{label}</small><strong>{value}</strong><p>{sub}</p></article>)}</section>
    <section className="lower-grid">
      <article className="panel chart-panel"><div className="panel-head"><div><h3>Crop health trend</h3><p>Average health score across analyzed fields</p></div><select aria-label="Time period"><option>Last 6 months</option></select></div><div className="chart"><div className="yaxis"><span>100</span><span>75</span><span>50</span><span>25</span></div><div className="chart-area"><div className="gridlines" /><div className="area-line" /><div className="point p1" /><div className="point p2" /><div className="point p3" /><div className="point p4" /><div className="point p5" /><div className="xaxis"><span>FEB</span><span>MAR</span><span>APR</span><span>MAY</span><span>JUN</span><span>JUL</span></div></div></div></article>
      <article className="panel recent"><div className="panel-head"><div><h3>Recent analyses</h3><p>Latest processed imagery</p></div><button onClick={onHistory}>View all →</button></div>{records.length === 0 && <p style={{fontSize: 10, color: "#7a837c"}}>No analyses yet.</p>}{records.slice(0, 3).map((r, i) => <div className="recent-row" key={r.id}><div className={`thumb t${i + 1}`} /><span><b>{r.name}</b><small>{r.crop} · {r.date}</small></span><div className="health-ring" style={{"--score": `${r.health_score * 3.6}deg`} as React.CSSProperties}>{Math.round(r.health_score)}</div></div>)}</article>
    </section>
  </div>;
}

function Upload(p: any) {
  return <div className="page narrow"><div className="steps"><span className="done">1</span><i /><span>2</span><i /><span>3</span><small>Upload imagery</small><small>Field details</small><small>AI analysis</small></div>
    <section className="upload-grid"><article className="panel upload-panel"><h3>Drone imagery</h3><p>Upload a high-resolution aerial image of your field.</p><div className={`dropzone ${p.preview ? "has-image" : ""}`} onClick={() => p.inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); p.onFile(e.dataTransfer.files[0]); }}>{p.preview ? <img src={p.preview} alt="Selected aerial field" /> : <><div className="upload-icon">↥</div><b>Drop your drone image here</b><span>or click to browse your files</span><small>JPG or PNG · Maximum 20 MB</small></>}{p.inspecting && <span className="inspect-status"><span className="spinner" /> Checking photo details…</span>}<input ref={p.inputRef} hidden type="file" accept="image/png,image/jpeg" onChange={(e) => p.onFile(e.target.files?.[0])} /></div>{p.file && <div className="file-pill"><span>✓</span><div><b>{p.file.name}</b><small>{(p.file.size / 1048576).toFixed(2)} MB · Ready for analysis</small></div><button onClick={(e) => { e.stopPropagation(); p.inputRef.current?.click(); }}>Replace</button></div>}</article>
    <article className="panel form-panel"><h3>Field parameters</h3><p>These details calibrate density and yield estimates.</p><label>FIELD NAME<input value={p.fieldName} onChange={(e) => p.setFieldName(e.target.value)} /></label><div className="form-row"><label>CROP TYPE<select value={p.crop} onChange={(e) => p.setCrop(e.target.value)}><option>Wheat</option><option>Corn</option><option>Rice</option><option>Soybean</option><option>Tomato</option></select>{p.cropSuggested && <small className="suggested-tag">Suggested: <b>{p.crop}</b> — confirm or change</small>}</label><label>FIELD AREA<div className="input-unit"><input type="number" min=".01" step=".01" value={p.area} onChange={(e) => p.setArea(e.target.value)} /><span>ha</span></div>{p.areaSuggested ? <small className="suggested-tag">Suggested: <b>{p.area} ha</b> — confirm or change</small> : <small className="field-help">Total ground area visible in this image. 1 hectare = 10,000 m² = 2.47 acres.</small>}</label></div><div className="quick-values"><span>QUICK AREA:</span>{["0.5","1","2","5"].map((value) => <button type="button" className={p.area === value ? "chosen" : ""} key={value} onClick={() => p.setArea(value)}>{value} ha</button>)}</div><label>AVERAGE YIELD PER PLANT<div className="input-unit"><input type="number" min=".001" step=".001" value={p.avgYield} onChange={(e) => p.setAvgYield(e.target.value)} /><span>kg</span></div><small className="field-help">Expected harvested weight from one plant. A typical {p.crop.toLowerCase()} starting value has been filled automatically; use your farm records when available.</small></label><div className="parameter-note"><b>Not sure about field area?</b><span>Use the mapped plot size from your drone flight plan, GPS survey, Google Earth measurement, or farm record. A normal photo cannot reveal physical area without altitude or ground-scale data.</span></div><div className="tech-note"><b>⌁ RGB health analysis</b><span>The uploaded image is measured for green vegetation coverage using three independent RGB indices (Excess Green, VARI, ExGR) that must agree before a pixel counts as vegetation. Non-green areas are treated as bare soil or stressed ground, not classified by color. Results are visual indicators—not a laboratory diagnosis.</span></div>{p.error && <div className="tech-note error-note"><b>⚠ Analysis failed</b><span>{p.error}</span></div>}<button className="primary full" disabled={!p.file || p.analyzing || Number(p.area) <= 0 || Number(p.avgYield) <= 0} onClick={p.run}>{p.analyzing ? <><span className="spinner" /> Running computer-vision pipeline…</> : <>Run image analysis <span>→</span></>}</button></article></section>
  </div>;
}

function Results({ data, onNew }: { data: Analysis; onNew: () => void }) {
  const healthLabel = data.health_score >= 80 ? "Healthy vegetation" : data.health_score >= 55 ? "Moderate / mixed health" : "Poor or stressed vegetation";
  const healthCopy = data.health_score >= 80
    ? "The image contains a strong, continuous green canopy."
    : data.health_score >= 55
      ? "Green crop is present, but gaps or stressed vegetation reduce the score."
      : "Low green coverage was detected, which can indicate exposed soil, sparse growth, or discolored vegetation.";
  const recommendation = data.health_score >= 80
    ? "Maintain current management and compare the next flight for emerging changes."
    : data.health_score >= 55
      ? "Inspect bare or discolored zones for water stress, pests, disease, or establishment problems."
      : "Prioritize a field inspection. Confirm whether the image shows drought, lodging, harvest residue, disease, or bare soil.";
  return <div className="page"><div className="result-top"><div><span className="success">✓ ANALYSIS SUCCESSFUL</span><p>{data.name} · {data.crop} · {data.date}</p></div><div><button className="ghost">Export report</button><button className="primary compact" onClick={onNew}>Analyze another</button></div></div>
    <section className="result-metrics">{[["PLANT COUNT", data.plant_count.toLocaleString(), "detected plants"], ["CROP DENSITY", data.crop_density.toLocaleString(), "plants / hectare"], ["CROP COVERAGE", `${data.crop_coverage}%`, "segmented area"], ["HEALTH SCORE", `${Math.round(data.health_score)}/100`, "strong vegetation"], ["EST. HARVEST", `${data.estimated_yield.toLocaleString()} kg`, `${(data.estimated_yield / 1000).toFixed(2)} metric tons`]].map(([a,b,c]) => <article key={a}><small>{a}</small><b>{b}</b><span>{c}</span></article>)}</section>
    <section className="vision-grid"><article className="panel vision-panel"><div className="panel-head"><div><h3>Computer vision output</h3><p>Detection and segmentation layers</p></div><div className="seg-tabs"><button className="selected">Detection</button><button>Segmentation</button><button>Heatmap</button></div></div><div className="result-image">{data.image && <img src={data.image} alt="Analyzed crop field" />}{data.detections.slice(0, 3).map((d, i) => <span key={i} className={`bbox rb${i + 1}`}>{d.label} .{Math.round(d.confidence * 100)}</span>)}<div className="model-badge">YOLO DETECTIONS · {data.plant_count.toLocaleString()}</div></div></article>
    <article className={`panel insights health-${data.health_score >= 80 ? "good" : data.health_score >= 55 ? "mixed" : "poor"}`}><h3>Field intelligence</h3><p>Interpreted from visible RGB vegetation signals.</p><div className="score"><div className="score-ring" style={{"--score": `${data.health_score * 3.6}deg`} as React.CSSProperties}><span><b>{Math.round(data.health_score)}</b><small>/ 100</small></span></div><div><b>{healthLabel}</b><p>{healthCopy}</p></div></div><hr /><div className="insight-row"><span>GREEN VEGETATION RATIO</span><b>{data.vegetation_score.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${data.vegetation_score}%`}} /></div><div className="insight-row"><span>AVG. DETECTION CONFIDENCE</span><b>{data.confidence_score.toFixed(1)}%</b></div><div className="bar"><i style={{width: `${data.confidence_score}%`}} /></div><div className="method-warning"><b>RGB screening result</b><span>Color analysis can flag suspicious areas, but cannot distinguish disease from drought, mature crops, harvest residue, shadows, or soil without field context.</span></div><div className="recommend"><b>Recommendation</b><p>{recommendation}</p></div></article></section>
  </div>;
}

function History({ records, loading, error, onRetry }: { records: HistoryRow[]; loading: boolean; error: string | null; onRetry: () => void }) {
  return <div className="page"><section className="panel history"><div className="panel-head"><div><h3>Analysis archive</h3><p>{loading ? "Loading…" : error ? "Unable to load history" : `${records.length} processed field surveys`}</p></div><div><input placeholder="Search fields…" /><button className="ghost">Filter</button></div></div>
    {error && <div className="tech-note error-note"><b>⚠ {error}</b><span><button className="ghost" onClick={onRetry}>Retry</button></span></div>}
    {!error && <div className="table"><div className="tr th"><span>FIELD</span><span>DATE</span><span>PLANTS</span><span>DENSITY / HA</span><span>HEALTH</span><span>EST. YIELD</span></div>
    {loading && <div className="tr"><span>Loading analysis history…</span></div>}
    {!loading && records.length === 0 && <div className="tr"><span>No analyses yet. Run one from “New analysis”.</span></div>}
    {!loading && records.map((r, i) => <div className="tr" key={r.id}><span className="field-cell"><i className={`thumb t${i % 3 + 1}`} /><span><b>{r.name}</b><small>{r.crop}</small></span></span><span>{r.date}</span><span>{r.plant_count.toLocaleString()}</span><span>{r.crop_density.toLocaleString()}</span><span><em className="health-pill">● {Math.round(r.health_score)}</em></span><span><b>{r.estimated_yield.toLocaleString()} kg</b></span></div>)}</div>}
  </section></div>;
}

function Settings() {
  return <div className="page narrow"><section className="panel settings"><h3>Analysis defaults</h3><p>Configure how new field surveys are interpreted.</p><label>DEFAULT AREA UNIT<select><option>Hectares (ha)</option><option>Acres</option></select></label><label>MODEL PROFILE<select><option>Balanced · Recommended</option><option>High sensitivity</option><option>High precision</option></select></label><div className="toggle-row"><span><b>Automatic image enhancement</b><small>Apply denoising and histogram normalization before detection.</small></span><button className="toggle on" aria-label="Toggle enhancement"><i /></button></div><div className="toggle-row"><span><b>Segmentation refinement</b><small>Use SAM masks to refine crop coverage estimates.</small></span><button className="toggle on" aria-label="Toggle segmentation"><i /></button></div><button className="primary compact">Save preferences</button></section></div>;
}
