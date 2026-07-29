"use client";

import { useMemo, useRef, useState } from "react";

type View = "dashboard" | "analyze" | "results" | "history" | "settings";
type Analysis = {
  id: number;
  name: string;
  crop: string;
  date: string;
  plants: number;
  density: number;
  coverage: number;
  health: number;
  yieldKg: number;
  vegetationScore?: number;
  stressScore?: number;
  image?: string;
};

const yieldDefaults: Record<string, number> = {
  Wheat: 0.02, Corn: 0.18, Rice: 0.025, Soybean: 0.015, Tomato: 3,
};

async function inspectImage(file: File) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 420 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Image analysis is not supported by this browser.");
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let green = 0, stressed = 0, usable = 0, greenStrength = 0;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    if (max < 22 || min > 248) continue;
    usable++;
    const excessGreen = 2 * g - r - b;
    const isGreen = g > 42 && g > r * 1.025 && g > b * 1.07 && excessGreen > 11;
    const isYellowBrown = r > 55 && g > 42 && r >= g * 1.015 &&
      g > b * 1.08 && r - b > 24 && max - min > 22;
    if (isGreen) {
      green++;
      greenStrength += Math.min(1, Math.max(0, excessGreen / 95));
    } else if (isYellowBrown) stressed++;
  }

  const greenPercent = 100 * green / Math.max(usable, 1);
  const stressPercent = 100 * stressed / Math.max(usable, 1);
  const coverage = Math.min(100, greenPercent + stressPercent * 0.45);
  const vigor = green / Math.max(green + stressed, 1);
  const colorVigor = greenStrength / Math.max(green, 1);
  // Health rewards vigorous green canopy and coverage, while yellow/brown
  // vegetation and exposed soil reduce the score.
  const health = Math.max(4, Math.min(98,
    30 * vigor + 15 * colorVigor + 55 * Math.min(1, coverage / 85)
  ));
  return {
    coverage: Number(coverage.toFixed(1)),
    vegetation: Number(greenPercent.toFixed(1)),
    stress: Number(stressPercent.toFixed(1)),
    health: Math.round(health),
  };
}

const seed: Analysis[] = [
  { id: 1, name: "North Field A", crop: "Wheat", date: "Jul 24, 2026", plants: 12480, density: 6240, coverage: 78.2, health: 91, yieldKg: 8736 },
  { id: 2, name: "East Plot", crop: "Corn", date: "Jul 19, 2026", plants: 8960, density: 4480, coverage: 72.6, health: 86, yieldKg: 7168 },
  { id: 3, name: "River Field", crop: "Soybean", date: "Jul 12, 2026", plants: 15820, density: 7910, coverage: 84.1, health: 94, yieldKg: 6328 },
];

const nav = [
  ["dashboard", "⌂", "Overview"],
  ["analyze", "⌁", "New analysis"],
  ["history", "▤", "Field history"],
  ["settings", "⚙", "Settings"],
] as const;

export default function Home() {
  const [view, setView] = useState<View>("dashboard");
  const [records, setRecords] = useState(seed);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [crop, setCrop] = useState("Wheat");
  const [area, setArea] = useState("2");
  const [avgYield, setAvgYield] = useState(String(yieldDefaults.Wheat));
  const [fieldName, setFieldName] = useState("West Field");
  const [analyzing, setAnalyzing] = useState(false);
  const [latest, setLatest] = useState<Analysis | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const totals = useMemo(() => ({
    plants: records.reduce((n, r) => n + r.plants, 0),
    yieldKg: records.reduce((n, r) => n + r.yieldKg, 0),
    density: Math.round(records.reduce((n, r) => n + r.density, 0) / records.length),
  }), [records]);

  function chooseFile(f?: File) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function runAnalysis() {
    if (!file) return;
    setAnalyzing(true);
    try {
      const imageMetrics = await inspectImage(file);
      await new Promise((resolve) => window.setTimeout(resolve, 650));
      const areaNumber = Math.max(Number(area), 0.01);
      const plants = Math.max(0, Math.round(areaNumber * 6200 * imageMetrics.coverage / 100));
      const result: Analysis = {
        id: Date.now(), name: fieldName || "Untitled field", crop,
        date: "Jul 29, 2026", plants,
        density: Math.round(plants / areaNumber),
        coverage: imageMetrics.coverage,
        health: imageMetrics.health,
        vegetationScore: imageMetrics.vegetation,
        stressScore: imageMetrics.stress,
        yieldKg: Math.round(plants * Number(avgYield || yieldDefaults[crop]) * (0.45 + 0.55 * imageMetrics.health / 100)),
        image: preview,
      };
      setLatest(result);
      setRecords((r) => [result, ...r]);
      setAnalyzing(false);
      setView("results");
    } catch {
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
        {view === "analyze" && <Upload file={file} preview={preview} fieldName={fieldName} crop={crop} area={area} avgYield={avgYield} analyzing={analyzing} inputRef={inputRef} onFile={chooseFile} setFieldName={setFieldName} setCrop={(value: string) => { setCrop(value); setAvgYield(String(yieldDefaults[value])); }} setArea={setArea} setAvgYield={setAvgYield} run={runAnalysis} />}
        {view === "results" && latest && <Results data={latest} onNew={() => setView("analyze")} />}
        {view === "history" && <History records={records} />}
        {view === "settings" && <Settings />}
      </section>
    </main>
  );
}

function Dashboard({ records, totals, onAnalyze, onHistory }: { records: Analysis[]; totals: { plants: number; yieldKg: number; density: number }; onAnalyze: () => void; onHistory: () => void }) {
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
      <article className="panel recent"><div className="panel-head"><div><h3>Recent analyses</h3><p>Latest processed imagery</p></div><button onClick={onHistory}>View all →</button></div>{records.slice(0, 3).map((r, i) => <div className="recent-row" key={r.id}><div className={`thumb t${i + 1}`} /><span><b>{r.name}</b><small>{r.crop} · {r.date}</small></span><div className="health-ring" style={{"--score": `${r.health * 3.6}deg`} as React.CSSProperties}>{r.health}</div></div>)}</article>
    </section>
  </div>;
}

function Upload(p: any) {
  return <div className="page narrow"><div className="steps"><span className="done">1</span><i /><span>2</span><i /><span>3</span><small>Upload imagery</small><small>Field details</small><small>AI analysis</small></div>
    <section className="upload-grid"><article className="panel upload-panel"><h3>Drone imagery</h3><p>Upload a high-resolution aerial image of your field.</p><div className={`dropzone ${p.preview ? "has-image" : ""}`} onClick={() => p.inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); p.onFile(e.dataTransfer.files[0]); }}>{p.preview ? <img src={p.preview} alt="Selected aerial field" /> : <><div className="upload-icon">↥</div><b>Drop your drone image here</b><span>or click to browse your files</span><small>JPG or PNG · Maximum 20 MB</small></>}<input ref={p.inputRef} hidden type="file" accept="image/png,image/jpeg" onChange={(e) => p.onFile(e.target.files?.[0])} /></div>{p.file && <div className="file-pill"><span>✓</span><div><b>{p.file.name}</b><small>{(p.file.size / 1048576).toFixed(2)} MB · Ready for analysis</small></div><button onClick={(e) => { e.stopPropagation(); p.inputRef.current?.click(); }}>Replace</button></div>}</article>
    <article className="panel form-panel"><h3>Field parameters</h3><p>These details calibrate density and yield estimates.</p><label>FIELD NAME<input value={p.fieldName} onChange={(e) => p.setFieldName(e.target.value)} /></label><div className="form-row"><label>CROP TYPE<select value={p.crop} onChange={(e) => p.setCrop(e.target.value)}><option>Wheat</option><option>Corn</option><option>Rice</option><option>Soybean</option><option>Tomato</option></select></label><label>FIELD AREA<div className="input-unit"><input type="number" min=".01" step=".01" value={p.area} onChange={(e) => p.setArea(e.target.value)} /><span>ha</span></div><small className="field-help">Total ground area visible in this image. 1 hectare = 10,000 m² = 2.47 acres.</small></label></div><div className="quick-values"><span>QUICK AREA:</span>{["0.5","1","2","5"].map((value) => <button type="button" className={p.area === value ? "chosen" : ""} key={value} onClick={() => p.setArea(value)}>{value} ha</button>)}</div><label>AVERAGE YIELD PER PLANT<div className="input-unit"><input type="number" min=".001" step=".001" value={p.avgYield} onChange={(e) => p.setAvgYield(e.target.value)} /><span>kg</span></div><small className="field-help">Expected harvested weight from one plant. A typical {p.crop.toLowerCase()} starting value has been filled automatically; use your farm records when available.</small></label><div className="parameter-note"><b>Not sure about field area?</b><span>Use the mapped plot size from your drone flight plan, GPS survey, Google Earth measurement, or farm record. A normal photo cannot reveal physical area without altitude or ground-scale data.</span></div><div className="tech-note"><b>⌁ RGB health analysis</b><span>The uploaded image is measured for green vegetation, canopy coverage, exposed soil, and yellow/brown stress. Results are visual indicators—not a laboratory diagnosis.</span></div><button className="primary full" disabled={!p.file || p.analyzing || Number(p.area) <= 0 || Number(p.avgYield) <= 0} onClick={p.run}>{p.analyzing ? <><span className="spinner" /> Measuring vegetation pixels…</> : <>Run image analysis <span>→</span></>}</button></article></section>
  </div>;
}

function Results({ data, onNew }: { data: Analysis; onNew: () => void }) {
  const healthLabel = data.health >= 80 ? "Healthy vegetation" : data.health >= 55 ? "Moderate / mixed health" : "Poor or stressed vegetation";
  const healthCopy = data.health >= 80
    ? "The image contains a strong, continuous green canopy."
    : data.health >= 55
      ? "Green crop is present, but gaps or stressed vegetation reduce the score."
      : "Low green coverage or substantial yellow/brown vegetation was detected.";
  const recommendation = data.health >= 80
    ? "Maintain current management and compare the next flight for emerging changes."
    : data.health >= 55
      ? "Inspect bare or discolored zones for water stress, pests, disease, or establishment problems."
      : "Prioritize a field inspection. Confirm whether the image shows drought, lodging, harvest residue, disease, or bare soil.";
  return <div className="page"><div className="result-top"><div><span className="success">✓ ANALYSIS SUCCESSFUL</span><p>{data.name} · {data.crop} · Jul 29, 2026</p></div><div><button className="ghost">Export report</button><button className="primary compact" onClick={onNew}>Analyze another</button></div></div>
    <section className="result-metrics">{[["PLANT COUNT", data.plants.toLocaleString(), "detected plants"], ["CROP DENSITY", data.density.toLocaleString(), "plants / hectare"], ["CROP COVERAGE", `${data.coverage}%`, "segmented area"], ["HEALTH SCORE", `${data.health}/100`, "strong vegetation"], ["EST. HARVEST", `${data.yieldKg.toLocaleString()} kg`, `${(data.yieldKg / 1000).toFixed(2)} metric tons`]].map(([a,b,c]) => <article key={a}><small>{a}</small><b>{b}</b><span>{c}</span></article>)}</section>
    <section className="vision-grid"><article className="panel vision-panel"><div className="panel-head"><div><h3>Computer vision output</h3><p>Detection and segmentation layers</p></div><div className="seg-tabs"><button className="selected">Detection</button><button>Segmentation</button><button>Heatmap</button></div></div><div className="result-image">{data.image && <img src={data.image} alt="Analyzed crop field" />}<span className="bbox rb1">crop .96</span><span className="bbox rb2">crop .93</span><span className="bbox rb3">crop .91</span><div className="model-badge">YOLO DETECTIONS · {data.plants.toLocaleString()}</div></div></article>
    <article className={`panel insights health-${data.health >= 80 ? "good" : data.health >= 55 ? "mixed" : "poor"}`}><h3>Field intelligence</h3><p>Interpreted from visible RGB vegetation signals.</p><div className="score"><div className="score-ring" style={{"--score": `${data.health * 3.6}deg`} as React.CSSProperties}><span><b>{data.health}</b><small>/ 100</small></span></div><div><b>{healthLabel}</b><p>{healthCopy}</p></div></div><hr /><div className="insight-row"><span>GREEN VEGETATION RATIO</span><b>{(data.vegetationScore ?? data.coverage).toFixed(1)}%</b></div><div className="bar"><i style={{width: `${data.vegetationScore ?? data.coverage}%`}} /></div><div className="insight-row"><span>YELLOW / BROWN SIGNAL</span><b>{(data.stressScore ?? 0).toFixed(1)}%</b></div><div className="bar stress-bar"><i style={{width: `${data.stressScore ?? 0}%`}} /></div><div className="method-warning"><b>RGB screening result</b><span>Color analysis can flag suspicious areas, but cannot distinguish disease from drought, mature crops, harvest residue, shadows, or soil without field context.</span></div><div className="recommend"><b>Recommendation</b><p>{recommendation}</p></div></article></section>
  </div>;
}

function History({ records }: { records: Analysis[] }) {
  return <div className="page"><section className="panel history"><div className="panel-head"><div><h3>Analysis archive</h3><p>{records.length} processed field surveys</p></div><div><input placeholder="Search fields…" /><button className="ghost">Filter</button></div></div><div className="table"><div className="tr th"><span>FIELD</span><span>DATE</span><span>PLANTS</span><span>DENSITY / HA</span><span>HEALTH</span><span>EST. YIELD</span></div>{records.map((r, i) => <div className="tr" key={r.id}><span className="field-cell"><i className={`thumb t${i % 3 + 1}`} /><span><b>{r.name}</b><small>{r.crop}</small></span></span><span>{r.date}</span><span>{r.plants.toLocaleString()}</span><span>{r.density.toLocaleString()}</span><span><em className="health-pill">● {r.health}</em></span><span><b>{r.yieldKg.toLocaleString()} kg</b></span></div>)}</div></section></div>;
}

function Settings() {
  return <div className="page narrow"><section className="panel settings"><h3>Analysis defaults</h3><p>Configure how new field surveys are interpreted.</p><label>DEFAULT AREA UNIT<select><option>Hectares (ha)</option><option>Acres</option></select></label><label>MODEL PROFILE<select><option>Balanced · Recommended</option><option>High sensitivity</option><option>High precision</option></select></label><div className="toggle-row"><span><b>Automatic image enhancement</b><small>Apply denoising and histogram normalization before detection.</small></span><button className="toggle on" aria-label="Toggle enhancement"><i /></button></div><div className="toggle-row"><span><b>Segmentation refinement</b><small>Use SAM masks to refine crop coverage estimates.</small></span><button className="toggle on" aria-label="Toggle segmentation"><i /></button></div><button className="primary compact">Save preferences</button></section></div>;
}
