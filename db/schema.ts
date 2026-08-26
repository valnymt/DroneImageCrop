import { sql } from "drizzle-orm";
import { integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const analyses = sqliteTable("analyses", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  cropType: text("crop_type").notNull(),
  fieldSizeHectares: real("field_size_hectares").notNull(),
  averageYieldPerPlantKg: real("average_yield_per_plant_kg").notNull(),
  plantCount: integer("plant_count").notNull(),
  cropDensity: real("crop_density").notNull(),
  cropCoverage: real("crop_coverage").notNull(),
  vegetationScore: real("vegetation_score").notNull(),
  healthScore: real("health_score").notNull(),
  estimatedYield: real("estimated_yield").notNull(),
  confidenceScore: real("confidence_score").notNull(),
  imagePath: text("image_path"),
  // Phases P/R/S -- previously computed by the backend but discarded after
  // the Results screen closed, so History/Dashboard never showed any of
  // it. texture fields are always present on a real analysis (not
  // nullable in the backend response); tiltCorrected likewise. Plant size
  // stats are genuinely absent whenever SAM wasn't available/refinement
  // was off, so those stay nullable rather than defaulting to 0.
  textureUniformityScore: real("texture_uniformity_score"),
  texturePattern: text("texture_pattern"),
  tiltCorrected: integer("tilt_corrected", { mode: "boolean" }),
  plantSizeMeanAreaCm2: real("plant_size_mean_area_cm2"),
  plantSizeUniformityScore: real("plant_size_uniformity_score"),
  plantSizeMeanAspectRatio: real("plant_size_mean_aspect_ratio"),
});

export const users = sqliteTable("users", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  email: text("email").notNull().unique(),
  username: text("username").notNull().default("Farmer"),
  occupation: text("occupation").notNull().default("Agricultural Researcher"),
  passwordHash: text("password_hash").notNull(),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});

export const sessions = sqliteTable("sessions", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  token: text("token").notNull().unique(),
  userId: integer("user_id").notNull().references(() => users.id),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});
