import { desc } from "drizzle-orm";
import { getDb } from "../../../db";
import { analyses } from "../../../db/schema";

function toRouteErrorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected error";
  const detail =
    error instanceof Error && error.cause instanceof Error ? error.cause.message : "";
  const combined = `${message}\n${detail}`;

  if (combined.includes("no such table") || combined.includes('from "analyses"')) {
    return "The analyses table is unavailable. Generate the migration locally with `npm run db:generate`, then apply it to the local D1 database before using history.";
  }

  return message;
}

export async function GET() {
  try {
    const db = getDb();
    const rows = await db
      .select()
      .from(analyses)
      .orderBy(desc(analyses.id))
      .limit(200);

    return Response.json(rows);
  } catch (error) {
    return Response.json({ error: toRouteErrorMessage(error) }, { status: 500 });
  }
}

type AnalysisPayload = {
  crop_type?: string;
  field_size_hectares?: number;
  average_yield_per_plant_kg?: number;
  plant_count?: number;
  crop_density?: number;
  crop_coverage?: number;
  vegetation_score?: number;
  health_score?: number;
  estimated_yield?: number;
  confidence_score?: number;
  image_path?: string | null;
  // Phases P/R/S -- optional so older-shaped callers (or a backend that's
  // running an older build) still persist successfully; the fields the
  // history/dashboard views actually use are just absent, not an error.
  texture_uniformity_score?: number | null;
  texture_pattern?: string | null;
  tilt_corrected?: boolean | null;
  plant_size_mean_area_cm2?: number | null;
  plant_size_uniformity_score?: number | null;
  plant_size_mean_aspect_ratio?: number | null;
};

const REQUIRED_NUMBER_FIELDS = [
  "field_size_hectares",
  "average_yield_per_plant_kg",
  "plant_count",
  "crop_density",
  "crop_coverage",
  "vegetation_score",
  "health_score",
  "estimated_yield",
  "confidence_score",
] as const;

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as AnalysisPayload;

    if (!payload.crop_type?.trim()) {
      return Response.json({ error: "crop_type is required" }, { status: 400 });
    }
    for (const field of REQUIRED_NUMBER_FIELDS) {
      if (typeof payload[field] !== "number" || Number.isNaN(payload[field])) {
        return Response.json({ error: `${field} must be a number` }, { status: 400 });
      }
    }

    const db = getDb();
    const [row] = await db
      .insert(analyses)
      .values({
        cropType: payload.crop_type.trim(),
        fieldSizeHectares: payload.field_size_hectares!,
        averageYieldPerPlantKg: payload.average_yield_per_plant_kg!,
        plantCount: payload.plant_count!,
        cropDensity: payload.crop_density!,
        cropCoverage: payload.crop_coverage!,
        vegetationScore: payload.vegetation_score!,
        healthScore: payload.health_score!,
        estimatedYield: payload.estimated_yield!,
        confidenceScore: payload.confidence_score!,
        imagePath: payload.image_path ?? null,
        textureUniformityScore: payload.texture_uniformity_score ?? null,
        texturePattern: payload.texture_pattern ?? null,
        tiltCorrected: payload.tilt_corrected ?? null,
        plantSizeMeanAreaCm2: payload.plant_size_mean_area_cm2 ?? null,
        plantSizeUniformityScore: payload.plant_size_uniformity_score ?? null,
        plantSizeMeanAspectRatio: payload.plant_size_mean_aspect_ratio ?? null,
      })
      .returning();

    return Response.json(row, { status: 201 });
  } catch (error) {
    return Response.json({ error: toRouteErrorMessage(error) }, { status: 500 });
  }
}
